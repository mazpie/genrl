import torch.nn as nn
import torch

import tools.utils as utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np

from tools.genrl_utils import *

def stop_gradient(x):
  return x.detach()

Module = nn.Module 

def env_reward(agent, seq):
  return agent.wm.heads['reward'](seq['feat']).mean

class DreamerAgent(Module):

  def __init__(self, 
                name, cfg, obs_space, act_spec, **kwargs):
    super().__init__()
    self.name = name
    self.cfg = cfg
    self.cfg.update(**kwargs)
    self.obs_space = obs_space
    self.act_spec = act_spec
    self._use_amp = (cfg.precision == 16)
    self.device = cfg.device
    self.act_dim = act_spec.shape[0]
    self.wm = WorldModel(cfg, obs_space, self.act_dim,)
    self.instantiate_acting_behavior()

    self.to(cfg.device)
    self.requires_grad_(requires_grad=False)

  def instantiate_acting_behavior(self,):
    self._acting_behavior = ActorCritic(self.cfg, self.act_spec, self.wm.inp_size).to(self.device)
    
  def act(self, obs, meta, step, eval_mode, state):
    if self.cfg.only_random_actions:
      return np.random.uniform(-1, 1, self.act_dim,).astype(self.act_spec.dtype), (None, None)
    obs = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
    else:
      latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
    feat = self.wm.rssm.get_feat(latent)
    if eval_mode:
      actor = self._acting_behavior.actor(feat)
      try:
        action = actor.mean 
      except:
        action = actor._mean
    else:
      actor = self._acting_behavior.actor(feat)
      action = actor.sample()
    new_state = (latent, action)
    return action.cpu().numpy()[0], new_state

  def update_wm(self, data, step):
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    outputs['is_terminal'] = data['is_terminal']
    metrics.update(mets)
    return state, outputs, metrics

  def update_acting_behavior(self, state=None, outputs=None, metrics={}, data=None, reward_fn=None):
    if self.cfg.only_random_actions:
      return {}, metrics
    if outputs is not None:
      post = outputs['post']
      is_terminal = outputs['is_terminal']
    else:
      data = self.wm.preprocess(data)
      embed = self.wm.encoder(data)
      post, _ = self.wm.rssm.observe(
          embed, data['action'], data['is_first'])
      is_terminal = data['is_terminal']
    #
    start = {k: stop_gradient(v) for k,v in post.items()}
    if reward_fn is None:
      acting_reward_fn = lambda seq: globals()[self.cfg.acting_reward_fn](self, seq) #.mode()
    else:
      acting_reward_fn = lambda seq: reward_fn(self, seq) #.mode()
    metrics.update(self._acting_behavior.update(self.wm, start, is_terminal, acting_reward_fn))
    return start, metrics

  def update(self, data, step):
    state, outputs, metrics = self.update_wm(data, step)
    start, metrics = self.update_acting_behavior(state, outputs, metrics, data)
    return state, metrics

  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key)
    for fn in getattr(self.cfg, 'additional_report_fns', []):
      call_fn = globals()[fn]
      additional_report = call_fn(self, data)
      report.update(additional_report)
    return report

  def get_meta_specs(self):
    return tuple()

  def init_meta(self):
    return OrderedDict()

  def update_meta(self, meta, global_step, time_step, finetune=False):
    return meta

class WorldModel(Module):
  def __init__(self, config, obs_space, act_dim,):
    super().__init__()
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.shapes = shapes
    self.cfg = config
    self.device = config.device
    self.encoder = common.Encoder(shapes, **config.encoder)
    # Computing embed dim
    with torch.no_grad():
      zeros = {k: torch.zeros( (1,) + v) for k, v in shapes.items()}
      outs = self.encoder(zeros)
      embed_dim = outs.shape[1]
    self.embed_dim = embed_dim
    self.rssm = common.EnsembleRSSM(**config.rssm, action_dim=act_dim, embed_dim=embed_dim, device=self.device,)
    self.heads = {}
    self._use_amp = (config.precision == 16)
    self.inp_size = self.rssm.get_feat_size()
    self.decoder_input_fn = getattr(self.rssm, f'get_{config.decoder_inputs}')
    self.decoder_input_size = getattr(self.rssm, f'get_{config.decoder_inputs}_size')()
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder, embed_dim=self.decoder_input_size, image_dist=config.image_dist)
    self.heads['reward'] = common.MLP(self.inp_size, (1,), **config.reward_head)
    # zero init
    with torch.no_grad():
      for p in self.heads['reward']._out.parameters():
        p.data = p.data * 0
    #
    if config.pred_discount:
      self.heads['discount'] = common.MLP(self.inp_size, (1,), **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.grad_heads = config.grad_heads
    self.heads = nn.ModuleDict(self.heads)
    self.model_opt = common.Optimizer('model', self.parameters(), **config.model_opt, use_amp=self._use_amp)
    self.e2e_update_fns = {}
    self.detached_update_fns = {}
    self.eval()

  def add_module_to_update(self, name, module, update_fn, detached=False):
    self.add_module(name, module)
    if detached:
      self.detached_update_fns[name] = update_fn
    else:
      self.e2e_update_fns[name] = update_fn
    self.model_opt = common.Optimizer('model', self.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)

  def update(self, data, state=None):
    self.train()
    with common.RequiresGrad(self):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        if getattr(self.cfg, "freeze_decoder", False):
          self.heads['decoder'].requires_grad_(False)
        if getattr(self.cfg, "freeze_post", False) or getattr(self.cfg, "freeze_model", False):
          self.heads['decoder'].requires_grad_(False)
          self.encoder.requires_grad_(False)
          # Updating only prior
          self.grad_heads = []
          self.rssm.requires_grad_(False)
          if not getattr(self.cfg, "freeze_model", False):
            self.rssm._ensemble_img_out.requires_grad_(True)
            self.rssm._ensemble_img_dist.requires_grad_(True)
        model_loss, state, outputs, metrics = self.loss(data, state)
        model_loss, metrics = self.update_additional_e2e_modules(data, outputs, model_loss, metrics)
      metrics.update(self.model_opt(model_loss, self.parameters())) 
    if len(self.detached_update_fns) > 0:
      detached_loss, metrics = self.update_additional_detached_modules(data, outputs, metrics)
    self.eval()
    return state, outputs, metrics

  def update_additional_detached_modules(self, data, outputs, metrics):
    # additional detached losses
    detached_loss = 0
    for k in self.detached_update_fns:
      detached_module = getattr(self, k)
      with common.RequiresGrad(detached_module):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
          add_loss, add_metrics = self.detached_update_fns[k](self, k, data, outputs, metrics)
          metrics.update(add_metrics)
          opt_metrics = self.model_opt(add_loss, detached_module.parameters())
          metrics.update({ f'{k}_{m}' : opt_metrics[m] for m in opt_metrics})
    return detached_loss, metrics

  def update_additional_e2e_modules(self, data, outputs, model_loss, metrics):
    # additional e2e losses
    for k in self.e2e_update_fns:
      add_loss, add_metrics = self.e2e_update_fns[k](self, k, data, outputs, metrics)
      model_loss += add_loss
      metrics.update(add_metrics)
    return model_loss, metrics

  def observe_data(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
    outs = dict(embed=embed, post=post, prior=prior, is_terminal=data['is_terminal'])
    return outs, { 'model_kl' : kl_value.mean() }

  def loss(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
    assert len(kl_loss.shape) == 0 or (len(kl_loss.shape) == 1 and kl_loss.shape[0] == 1), kl_loss.shape
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.grad_heads)
      if name == 'decoder':
        inp = self.decoder_input_fn(post)
      else:
        inp = feat
      inp = inp if grad_head else stop_gradient(inp)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = dist.log_prob(data[key]) 
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.cfg.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon, task_cond=None, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    inp = start['feat'] if task_cond is None else torch.cat([start['feat'], task_cond], dim=-1)
    policy_dist = policy(inp)
    start['action'] = torch.zeros_like(policy_dist.sample(), device=self.device) #.mode())
    seq = {k: [v] for k, v in start.items()}
    if task_cond is not None: seq['task'] = [task_cond]
    for _ in range(horizon):
      inp = seq['feat'][-1] if task_cond is None else torch.cat([seq['feat'][-1], task_cond], dim=-1)
      policy_dist = policy(stop_gradient(inp))
      action = policy_dist.sample() if not eval_policy else policy_dist.mean
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
      if task_cond is not None: seq['task'].append(task_cond)
    # shape will be (T, B, *DIMS)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal) 
        disc = torch.cat([true_first[None], disc[1:]], 0)
    else:
      disc = torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)
    seq['discount'] = disc * self.cfg.discount
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = torch.cumprod(torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0), 0)
    return seq

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype in [np.uint8, torch.uint8]:
        value = value / 255.0 - 0.5 
      obs[key] = value
    obs['reward'] = {
        'identity': nn.Identity(),
        'sign': torch.sign,
        'tanh': torch.tanh,
    }[self.cfg.clip_rewards](obs['reward'])
    obs['discount'] = (1.0 - obs['is_terminal'].float())
    if len(obs['discount'].shape) < len(obs['reward'].shape):
      obs['discount'] = obs['discount'].unsqueeze(-1)
    return obs

  def video_pred(self, data, key, nvid=8):
    decoder = self.heads['decoder'] # B, T, C, H, W
    truth = data[key][:nvid] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:nvid, :5], data['action'][:nvid, :5], data['is_first'][:nvid, :5])
    recon = decoder(self.decoder_input_fn(states))[key].mean[:nvid] # mode
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:nvid, 5:], init)
    prior_recon = decoder(self.decoder_input_fn(prior))[key].mean # mode
    model = torch.clip(torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1)
    error = (model - truth + 1) / 2
    video = torch.cat([truth, model, error], 3)
    B, T, C, H, W = video.shape
    return video 

class ActorCritic(Module):
  def __init__(self, config, act_spec, feat_size, name=''):
    super().__init__()
    self.name = name
    self.cfg = config
    self.act_spec = act_spec
    self._use_amp = (config.precision == 16)
    self.device = config.device
    
    if getattr(self.cfg, 'discrete_actions', False):
      self.cfg.actor.dist = 'onehot'

    self.actor_grad = getattr(self.cfg, f'{self.name}_actor_grad'.strip('_'))
    
    inp_size = feat_size
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0 # tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    
    if self.cfg.reward_ema:
        # register ema_vals to nn.Module for enabling torch.save and torch.load
        self.register_buffer("ema_vals", torch.zeros((2,)).to(self.device))
        self.reward_ema = common.RewardEMA(device=self.device)
        self.rewnorm = common.StreamNorm(momentum=1, scale=1.0, device=self.device)
    else:
        self.rewnorm = common.StreamNorm(**self.cfg.reward_norm, device=self.device)

    # zero init
    with torch.no_grad():
      for p in self.critic._out.parameters():
        p.data = p.data * 0
    # hard copy critic initial params
    for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
      d.data = s.data
    #


  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = world_model.imagine(self.actor, start, is_terminal, hor)
        reward = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        target, mets2, baseline = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target, baseline)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return { f'{self.name}_{k}'.strip('_') : v for k,v in metrics.items() }

  def actor_loss(self, seq, target, baseline): #, step):
    # Two state-actions are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(stop_gradient(seq['feat'][:-2])) # actions are the ones in [1:-1]

    metrics = {}
    if self.cfg.reward_ema:
      offset, scale = self.reward_ema(target, self.ema_vals)
      normed_target = (target - offset) / scale
      normed_baseline = (baseline - offset) / scale
      # adv = normed_target - normed_baseline
      metrics['normed_target_mean'] = normed_target.mean()
      metrics['normed_target_std'] = normed_target.std()
      metrics["reward_ema_005"] = self.ema_vals[0]
      metrics["reward_ema_095"] = self.ema_vals[1]
    else:
      normed_target = target
      normed_baseline = baseline
    
    if self.actor_grad == 'dynamics':
      objective = normed_target[1:]
    elif self.actor_grad == 'reinforce':
      advantage = normed_target[1:] - normed_baseline[1:]
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    else:
      raise NotImplementedError(self.actor_grad)
    
    ent = policy.entropy()[:,:,None]
    ent_scale = self.cfg.actor_ent
    objective += ent_scale * ent
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean() 
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    feat = seq['feat'][:-1]
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    dist = self.critic(feat)
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'critic': dist.mean.mean() } 
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward'] 
    disc = seq['discount'] 
    value = self._target_critic(seq['feat']).mean 
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics, value[:-1]

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1 