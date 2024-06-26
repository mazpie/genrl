import re

import numpy as np

import tools.utils as utils
import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F

Module = nn.Module 

def symlog(x):
  return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
  return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)

class SampleDist:
    def __init__(self, dist: D.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        sample = self._dist.rsample((self._samples,))
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        sample = self._dist.rsample((self._samples,))
        logprob = self._dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.rsample()

class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    @property
    def mean(self):
        return self._mode

    def mode(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss

class SymlogDist:

  def __init__(self, mode, dims, dist='mse', agg='sum', tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._dist = dist
    self._agg = agg
    self._tol = tol
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    if self._dist == 'mse':
      distance = (self._mode - symlog(value)) ** 2
      distance = torch.where(distance < self._tol, torch.tensor([0.], dtype=distance.dtype, device=distance.device), distance)
    elif self._dist == 'abs':
      distance = torch.abs(self._mode - symlog(value))
      distance = torch.where(distance < self._tol, torch.tensor([0.], dtype=distance.dtype, device=distance.device), distance)
    else:
      raise NotImplementedError(self._dist)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss

class TwoHotDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
    ):
        assert logits.shape[-1] == 255
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(logits.device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    @property
    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    @property
    def mode(self):
        return self.mean

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)

class OneHotDist(D.OneHotCategorical):

  def __init__(self, logits=None, probs=None, unif_mix=0.99):
    super().__init__(logits=logits, probs=probs)
    probs = super().probs
    probs = unif_mix * probs + (1 - unif_mix) * torch.ones_like(probs, device=probs.device) / probs.shape[-1]
    super().__init__(probs=probs)

  def mode(self):
    _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
    return _mode.detach() + super().logits - super().logits.detach()

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError('need to check')
    sample = super().sample(sample_shape)
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach() # ST-gradients
    return sample

class BernoulliDist(D.Bernoulli):
  def __init__(self, logits=None, probs=None):
    super().__init__(logits=logits, probs=probs)

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError('need to check')
    sample = super().sample(sample_shape)
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach() # ST-gradients
    return sample

def static_scan_for_lambda_return(fn, inputs, start):
  last = start
  indices = range(inputs[0].shape[0])
  indices = reversed(indices)
  flag = True
  for index in indices:
    inp = lambda x: (_input[x].unsqueeze(0) for _input in inputs)
    last = fn(last, *inp(index))
    if flag:
      outputs = last
      flag = False
    else:
      outputs = torch.cat([last, outputs], dim=0) 
  return outputs

def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  #assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * torch.ones_like(reward, device=reward.device)
  dims = list(range(len(reward.shape)))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = reward.permute(dims)
    value = value.permute(dims)
    pcont = pcont.permute(dims)
  if bootstrap is None:
    bootstrap = torch.zeros_like(value[-1], device=reward.device)
  if len(bootstrap.shape) < len(value.shape):
    bootstrap = bootstrap[None]
  next_values = torch.cat([value[1:], bootstrap], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan_for_lambda_return(
      lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
      (inputs, pcont), bootstrap)
  if axis != 0:
    returns = returns.permute(dims)
  return returns

def static_scan(fn, inputs, start, reverse=False, unpack=False):
  last = start
  indices = range(inputs[0].shape[0])
  flag = True
  for index in indices:
    inp = lambda x: (_input[x] for _input in inputs)
    if unpack:
      last = fn(last, *[inp[index] for inp in inputs]) 
    else:
      last = fn(last, inp(index)) 
    if flag:
      if type(last) == type({}):
        outputs = {key: [value] for key, value in last.items()}
      else:
        outputs = []
        for _last in last:
          if type(_last) == type({}):
            outputs.append({key: [value] for key, value in _last.items()})
          else:
            outputs.append([_last])
      flag = False
    else:
      if type(last) == type({}):
        for key in last.keys():
          outputs[key].append(last[key]) 
      else:
        for j in range(len(outputs)):
          if type(last[j]) == type({}):
            for key in last[j].keys():
              outputs[j][key].append(last[j][key]) 
          else:
            outputs[j].append(last[j]) 
  # Stack everything at the end
  if type(last) == type({}):
    for key in last.keys():
      outputs[key] = torch.stack(outputs[key], dim=0)
  else:
    for j in range(len(outputs)):
      if type(last[j]) == type({}):
        for key in last[j].keys():
          outputs[j][key] = torch.stack(outputs[j][key], dim=0)
      else:
        outputs[j] = torch.stack(outputs[j], dim=0)
  if type(last) == type({}):
    outputs = [outputs]
  return outputs

class EnsembleRSSM(Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='SiLU', norm='none', std_act='softplus', min_std=0.1, action_dim=None, embed_dim=1536, device='cuda', 
      single_obs_posterior=False, cell_input='stoch', cell_type='gru',):
    super().__init__()
    assert action_dim is not None 
    self.device = device
    self._embed_dim = embed_dim
    self._action_dim = action_dim
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._cell_type = cell_type
    self.cell_input = cell_input
    if cell_type == 'gru':
      self._cell = GRUCell(self._hidden, self._deter, norm=True, device=self.device)
    else:
      raise NotImplementedError(f"{cell_type} not implemented")
    self.single_obs_posterior = single_obs_posterior

    if discrete:
      self._ensemble_img_dist = nn.ModuleList([ nn.Linear(hidden, stoch*discrete) for _ in range(ensemble)])
      self._obs_dist = nn.Linear(hidden, stoch*discrete)
    else:
      self._ensemble_img_dist = nn.ModuleList([ nn.Linear(hidden, 2*stoch) for _ in range(ensemble)])
      self._obs_dist = nn.Linear(hidden, 2*stoch)

    # Layer that projects (stoch, input) to cell_state space
    cell_state_input_size = getattr(self, f'get_{self.cell_input}_size')()
    self._img_in = nn.Sequential(nn.Linear(cell_state_input_size + action_dim, hidden, bias=norm != 'none'), NormLayer(norm, hidden))
    # Layer that project deter -> hidden [before projecting hidden -> stoch]
    self._ensemble_img_out = nn.ModuleList([ nn.Sequential(nn.Linear(self.get_deter_size(), hidden, bias=norm != 'none'), NormLayer(norm, hidden)) for _ in range(ensemble)])

    if self.single_obs_posterior:
      self._obs_out = nn.Sequential(nn.Linear(embed_dim, hidden, bias=norm != 'none'), NormLayer(norm, hidden))
    else:
      self._obs_out = nn.Sequential(nn.Linear(deter + embed_dim, hidden, bias=norm != 'none'), NormLayer(norm, hidden))

  def initial(self, batch_size):
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete], device=self.device), 
          stoch=torch.zeros([batch_size, self._stoch, self._discrete], device=self.device), 
          deter=self._cell.get_initial_state(None, batch_size)) 
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch], device=self.device), 
          std=torch.zeros([batch_size, self._stoch], device=self.device),
          stoch=torch.zeros([batch_size, self._stoch], device=self.device), 
          deter=self._cell.get_initial_state(None, batch_size)) 
    return state

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None: state = self.initial(action.shape[0])

    post, prior = static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(is_first)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None, sample=True):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = static_scan(self.img_step, [action, float(sample) + torch.zeros(action.shape[0])], state, unpack=True)[0] 
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_stoch_size(self,):
    if self._discrete:
      return self._stoch * self._discrete
    else:
      return self._stoch

  def get_deter_size(self,):
      return self._cell.state_size

  def get_feat_size(self,):
    return self.get_deter_size() + self.get_stoch_size()

  def get_stoch(self, state):
    stoch = state['stoch'] 
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return stoch

  def get_deter(self, state):
      return state['deter']

  def get_feat(self, state):
    deter = self.get_deter(state)
    stoch = self.get_stoch(state)
    return torch.cat([stoch, deter], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      dist = D.Independent(OneHotDist(logit.float()), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = D.Independent(D.Normal(mean, std), 1)
      dist.sample = dist.rsample
    return dist

  def get_unif_dist(self, state):
    if self._discrete:
      logit = state['logit']
      dist = D.Independent(OneHotDist(torch.ones_like(logit, device=logit.device)), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = D.Independent(D.Normal(torch.zeros_like(mean, device=mean.device), torch.ones_like(std, device=std.device)), 1)
      dist.sample = dist.rsample
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first, should_sample=True):
    if is_first.any():
      prev_state = { k: torch.einsum('b,b...->b...', 1.0 - is_first.float(), x) for k, x in prev_state.items() }
      prev_action = torch.einsum('b,b...->b...', 1.0 - is_first.float(), prev_action)
    #
    prior = self.img_step(prev_state, prev_action, should_sample)
    stoch, stats = self.get_post_stoch(embed, prior, should_sample)
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def get_post_stoch(self, embed, prior, should_sample=True):
    if self.single_obs_posterior:
      x = embed
    else:
      x = torch.cat([prior['deter'], embed], -1)
    x = self._obs_out(x)
    x = self._act(x)
  
    bs = list(x.shape[:-1])
    x = x.reshape([-1, x.shape[-1]])
    stats = self._suff_stats_layer('_obs_dist', x)
    stats = { k: v.reshape( bs + list(v.shape[1:])) for k, v in stats.items()}
    
    dist = self.get_dist(stats)
    stoch = dist.sample() if should_sample else dist.mode() 
    return stoch, stats

  def img_step(self, prev_state, prev_action, sample=True,):
    prev_state_input = getattr(self, f'get_{self.cell_input}')(prev_state)
    x = torch.cat([prev_state_input, prev_action], -1)
    x = self._img_in(x)
    x = self._act(x)
    deter = prev_state['deter']
    if self._cell_type == 'gru':
      x, deter = self._cell(x, [deter])
      temp_state = {'deter' : deter[0] }
    else:
      raise NotImplementedError(f"no {self._cell_type} cell method")
    deter = deter[0]  # It's wrapped in a list.
    stoch, stats = self.get_stoch_stats_from_deter_state(temp_state, sample)
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def get_stoch_stats_from_deter_state(self, temp_state, sample=True):
    stats = self._suff_stats_ensemble(self.get_deter(temp_state))
    index = torch.randint(0, self._ensemble, ()) 
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    if sample:
      stoch = dist.sample()
    else:
      try:
        stoch = dist.mode()
      except:
        stoch = dist.mean
    return stoch, stats

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self._ensemble_img_out[k](inp)
      x = self._act(x)
      stats.append(self._suff_stats_layer('_ensemble_img_dist', x, k=k))
    stats = {
        k: torch.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x, k=None):
    layer = getattr(self, name)
    if k is not None:
      layer = layer[k]
    x = layer(x)
    if self._discrete:
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      mean, std = torch.chunk(x, 2, -1)
      std = {
          'softplus': lambda: F.softplus(std),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def vq_loss(self, post, prior, balance):
    dim_repr = prior['output'].shape[-1]
    # Vectors and codes are the same, but vectors have gradients
    dyn_loss = balance * F.mse_loss(prior['output'], post['vectors'].detach()) + (1 - balance) * F.mse_loss(prior['output'].detach(), post['vectors'])
    dyn_loss += balance * F.mse_loss(prior['output'], post['codes'].detach()) + (1 - balance) * F.mse_loss(prior['output'].detach(), post['codes']) 
    dyn_loss /= 2
    vq_loss = 0.25 * F.mse_loss(post['output'], post['codes'].detach()) + F.mse_loss(post['output'].detach(), post['codes'])

    loss = vq_loss + dyn_loss 
    return loss * dim_repr, dyn_loss * dim_repr
  
  def kl_loss(self, post, prior, forward, balance, free, free_avg,):
    kld = D.kl_divergence
    sg = lambda x: {k: v.detach() for k, v in x.items()} 
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    dtype = post['stoch'].dtype
    device = post['stoch'].device
    free_tensor = torch.tensor([free], dtype=dtype, device=device)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = torch.maximum(value, free_tensor).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = torch.maximum(value_lhs.mean(), free_tensor)
        loss_rhs = torch.maximum(value_rhs.mean(), free_tensor)
      else:
        loss_lhs = torch.maximum(value_lhs, free_tensor).mean()
        loss_rhs = torch.maximum(value_rhs, free_tensor).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class Encoder(Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='SiLU', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400], symlog_inputs=False,):
    super().__init__()
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Encoder CNN inputs:', list(self.cnn_keys))
    print('Encoder MLP inputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers
    self._symlog_inputs = symlog_inputs

    if len(self.cnn_keys) > 0:
      self._conv_model = []
      for i, kernel in enumerate(self._cnn_kernels):
        if i == 0:
          prev_depth = 3
        else:
          prev_depth = 2 ** (i-1) * self._cnn_depth  
        depth = 2 ** i * self._cnn_depth
        self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
        self._conv_model.append(ImgChLayerNorm(depth) if norm == 'layer' else NormLayer(norm,depth))
        self._conv_model.append(self._act)
      self._conv_model = nn.Sequential(*self._conv_model)
    if len(self.mlp_keys) > 0:
      self._mlp_model = []
      for i, width in enumerate(self._mlp_layers):
        if i == 0:
          prev_width = np.sum([shapes[k] for k in self.mlp_keys]) 
        else:
          prev_width = self._mlp_layers[i-1]
        self._mlp_model.append(nn.Linear(prev_width, width, bias=norm != 'none'))
        self._mlp_model.append(NormLayer(norm, width))
        self._mlp_model.append(self._act)
      if len(self._mlp_model) == 0:
        self._mlp_model.append(nn.Identity())
      self._mlp_model = nn.Sequential(*self._mlp_model)

  def forward(self, data):
    key, shape = list(self.shapes.items())[0]
    batch_dims = data[key].shape[:-len(shape)]
    data = {
        k: v.reshape((-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_keys:
      outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
    if self.mlp_keys:
      outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
    output = torch.cat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    x = torch.cat(list(data.values()), -1)
    x = self._conv_model(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = torch.cat(list(data.values()), -1)
    if self._symlog_inputs:
      x = symlog(x)
    x = self._mlp_model(x)
    return x


class Decoder(Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='SiLU', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400], embed_dim=1024, mlp_dist='mse', image_dist='mse'):
    super().__init__()
    self._embed_dim = embed_dim
    self._shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Decoder CNN outputs:', list(self.cnn_keys))
    print('Decoder MLP outputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers
    self.channels = {k: self._shapes[k][0] for k in self.cnn_keys}
    self._mlp_dist = mlp_dist
    self._image_dist = image_dist

    if len(self.cnn_keys) > 0:

      self._conv_in = nn.Sequential(nn.Linear(embed_dim, 32*self._cnn_depth))
      self._conv_model = []
      for i, kernel in enumerate(self._cnn_kernels):
        if i == 0:
          prev_depth = 32*self._cnn_depth
        else:
          prev_depth = 2 ** (len(self._cnn_kernels) - (i - 1) - 2) * self._cnn_depth
        depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
        act, norm = self._act, self._norm
        # Last layer is dist layer 
        if i == len(self._cnn_kernels) - 1:
          depth, act, norm = sum(self.channels.values()), nn.Identity(), 'none'
        self._conv_model.append(nn.ConvTranspose2d(prev_depth, depth, kernel, stride=2))
        self._conv_model.append(ImgChLayerNorm(depth) if norm == 'layer' else NormLayer(norm, depth))
        self._conv_model.append(act)
      self._conv_model = nn.Sequential(*self._conv_model)
    if len(self.mlp_keys) > 0:
      self._mlp_model = []
      for i, width in enumerate(self._mlp_layers):
        if i == 0:
          prev_width = embed_dim
        else:
          prev_width = self._mlp_layers[i-1]
        self._mlp_model.append(nn.Linear(prev_width, width, bias=self._norm != 'none'))
        self._mlp_model.append(NormLayer(self._norm, width))
        self._mlp_model.append(self._act)
      self._mlp_model = nn.Sequential(*self._mlp_model)
      for key, shape in { k : shapes[k] for k in self.mlp_keys }.items():
        self.add_module(f'dense_{key}', DistLayer(width, shape, dist=self._mlp_dist))

  def forward(self, features):
    outputs = {}
    
    if self.cnn_keys:
      outputs.update(self._cnn(features))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features):
    x = self._conv_in(features)
    x = x.reshape([-1, 32 * self._cnn_depth, 1, 1,])
    x = self._conv_model(x)
    x = x.reshape(list(features.shape[:-1]) + list(x.shape[1:])) 
    if len(x.shape) == 5:
      means = torch.split(x, list(self.channels.values()), 2)
    else:
      means = torch.split(x, list(self.channels.values()), 1)
    image_dist = dict(mse=lambda x : MSEDist(x), normal_unit_std=lambda x : D.Independent(D.Normal(x, 1.0), 3))[self._image_dist]
    dists = { key: image_dist(mean) for (key, shape), mean in zip(self.channels.items(), means)}
    return dists

  def _mlp(self, features):
    shapes = {k: self._shapes[k] for k in self.mlp_keys}
    x = features
    x = self._mlp_model(x)
    dists = {}
    for key, shape in shapes.items():
      dists[key] = getattr(self, f'dense_{key}')(x)
    return dists


class MLP(Module):

  def __init__(self, in_shape, shape, layers, units, act='SiLU', norm='none', **out):
    super().__init__()
    self._in_shape = in_shape
    if out['dist'] == 'twohot':
      shape = 255
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out
    
    last_units = in_shape
    for index in range(self._layers):
      self.add_module(f'dense{index}', nn.Linear(last_units, units, bias=norm != 'none'))
      self.add_module(f'norm{index}', NormLayer(norm, units))
      last_units = units
    self._out = DistLayer(units, shape, **out)

  def forward(self, features):
    x = features 
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = getattr(self, f'dense{index}')(x)
      x = getattr(self, f'norm{index}')(x)
      x = self._act(x)
    x = x.reshape(list(features.shape[:-1]) + [x.shape[-1]])
    return self._out(x)


class GRUCell(Module):

  def __init__(self, inp_size, size, norm=False, act='Tanh', update_bias=-1, device='cuda', **kwargs):
    super().__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self.device = device
    self._layer = nn.Linear(inp_size + size, 3 * size, bias=(not norm), **kwargs)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return torch.zeros((batch_size), self._size, device=self.device)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, deter_state):
    """
      inputs : non-linear combination of previous stoch and action 
      deter_state : prev hidden state of the cell
    """
    deter_state = deter_state[0]  # State is wrapped in a list.
    parts = self._layer(torch.cat([inputs, deter_state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.chunk(parts, 3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * deter_state
    return output, [output]

class DistLayer(Module):

  def __init__(
      self, in_dim, shape, dist='mse', min_std=0.1, max_std=1.0, init_std=0.0, bias=True):
    super().__init__()
    self._in_dim = in_dim
    self._shape = shape if type(shape) in [list,tuple] else [shape]
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._max_std = max_std
    self._out = nn.Linear(in_dim, int(np.prod(shape)) , bias=bias)
    if dist in ('normal', 'tanh_normal', 'trunc_normal'):
      self._std = nn.Linear(in_dim, int(np.prod(shape)) )

  def forward(self, inputs):
    out = self._out(inputs)
    out = out.reshape(list(inputs.shape[:-1]) + list(self._shape)) 
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self._std(inputs) 
      std = std.reshape(list(inputs.shape[:-1]) + list(self._shape)) 
    if self._dist == 'mse':
      return MSEDist(out,)
    if self._dist == 'normal_unit_std':
      dist = D.Normal(out, 1.0)
      dist.sample = dist.rsample
      return D.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      mean = torch.tanh(out)
      std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
      dist = D.Normal(mean, std)
      dist.sample = dist.rsample
      return D.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      out = torch.sigmoid(out)
      dist = BernoulliDist(out)
      return D.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * torch.tanh(out / 5)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = utils.SquashedNormal(mean, std)
      dist = D.Independent(dist, len(self._shape))
      return SampleDist(dist)
    if self._dist == 'trunc_normal':
      mean = torch.tanh(out)
      std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = utils.TruncatedNormal(mean, std)
      return D.Independent(dist, 1)
    if self._dist == 'onehot':
      return OneHotDist(out.float()) 
    if self._dist == 'twohot':
      return TwoHotDist(out.float())
    if self._dist == 'symlog_mse':
      return SymlogDist(out, len(self._shape), 'mse') 
    raise NotImplementedError(self._dist)


class NormLayer(Module):

  def __init__(self, name, dim=None):
    super().__init__()
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      assert dim != None
      self._layer = nn.LayerNorm(dim)
    else:
      raise NotImplementedError(name)

  def forward(self, features):
    if self._layer is None:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return nn.Identity()
  elif hasattr(nn, name):
    return getattr(nn, name)()
  else:
    raise NotImplementedError(name)


class Optimizer:

  def __init__(
      self, name, parameters, lr, eps=1e-4, clip=None, wd=None,
      opt='adam', wd_pattern=r'.*', use_amp=False):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: torch.optim.Adam(parameters, lr, eps=eps),
        'nadam': lambda: torch.optim.Nadam(parameters, lr, eps=eps),
        'adamax': lambda: torch.optim.Adamax(parameters, lr, eps=eps),
        'sgd': lambda: torch.optim.SGD(parameters, lr),
        'momentum': lambda: torch.optim.SGD(lr, momentum=0.9),
    }[opt]()
    self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    self._once = True

  def __call__(self, loss, params):
    params = list(params)
    assert len(loss.shape) == 0 or (len(loss.shape) == 1 and loss.shape[0] == 1), (self._name, loss.shape)
    metrics = {}

    # Count parameters.
    if self._once:
      count = sum(p.numel() for p in params if p.requires_grad) 
      print(f'Found {count} {self._name} parameters.')
      self._once = False

    # Check loss.
    metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()

    # Compute scaled gradient.
    self._scaler.scale(loss).backward()
    self._scaler.unscale_(self._opt)

    # Gradient clipping.
    if self._clip:
      norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
      metrics[f'{self._name}_grad_norm'] = norm.item()
  
    # Weight decay.
    if self._wd:
      self._apply_weight_decay(params)
    
    # # Apply gradients.
    self._scaler.step(self._opt)
    self._scaler.update()
    
    self._opt.zero_grad() 
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      raise NotImplementedError('Non trivial weight decay')
    else:
      for var in varibs:
        var.data = (1 - self._wd) * var.data

class StreamNorm:

  def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8, device='cuda'):
    # Momentum of 0 normalizes only based on the current batch.
    # Momentum of 1 disables normalization.
    self.device = device
    self._shape = tuple(shape)
    self._momentum = momentum
    self._scale = scale
    self._eps = eps
    self.mag = None # torch.ones(shape).to(self.device) 

    self.step = 0
    self.mean = None # torch.zeros(shape).to(self.device)
    self.square_mean = None # torch.zeros(shape).to(self.device)

  def reset(self):
    self.step = 0
    self.mag = None # torch.ones_like(self.mag).to(self.device)
    self.mean = None # torch.zeros_like(self.mean).to(self.device)
    self.square_mean = None # torch.zeros_like(self.square_mean).to(self.device)

  def __call__(self, inputs):
    metrics = {}
    self.update(inputs)
    metrics['mean'] = inputs.mean()
    metrics['std'] = inputs.std()
    outputs = self.transform(inputs)
    metrics['normed_mean'] = outputs.mean()
    metrics['normed_std'] = outputs.std()
    return outputs, metrics

  def update(self, inputs):
    self.step += 1
    batch = inputs.reshape((-1,) + self._shape)
    
    mag = torch.abs(batch).mean(0) 
    if self.mag is not None:
      self.mag.data = self._momentum * self.mag.data + (1 - self._momentum) * mag 
    else:
      self.mag =  mag.clone().detach()
    
    mean = torch.mean(batch)
    if self.mean is not None:
      self.mean.data = self._momentum * self.mean.data + (1 - self._momentum) * mean 
    else:
      self.mean = mean.clone().detach()
    
    square_mean = torch.mean(batch * batch)
    if self.square_mean is not None:
      self.square_mean.data = self._momentum * self.square_mean.data + (1 - self._momentum) * square_mean 
    else:
      self.square_mean = square_mean.clone().detach()

  def transform(self, inputs):
    if self._momentum == 1:
      return inputs
    values = inputs.reshape((-1,) + self._shape)
    values /= self.mag[None] + self._eps 
    values *= self._scale
    return values.reshape(inputs.shape)

  def corrected_mean_var_std(self,):
    corr = 1 # 1 - self._momentum ** self.step # NOTE: this led to exploding values for first few iterations
    corr_mean = self.mean / corr 
    corr_var = (self.square_mean / corr) - self.mean ** 2
    corr_std = torch.sqrt(torch.maximum(corr_var, torch.zeros_like(corr_var, device=self.device)) + self._eps)
    return corr_mean, corr_var, corr_std

class RequiresGrad:

  def __init__(self, model):
    self._model = model

  def __enter__(self):
    self._model.requires_grad_(requires_grad=True)

  def __exit__(self, *args):
    self._model.requires_grad_(requires_grad=False)

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x