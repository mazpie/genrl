import torch
from agent.dreamer import DreamerAgent, ActorCritic, stop_gradient, env_reward
import agent.dreamer_utils as common
import agent.video_utils as video_utils
from tools.genrl_utils import *

def connector_update_fn(self, module_name, data, outputs, metrics):
    connector = getattr(self, module_name)
    n_frames = connector.n_frames
    B, T = data['observation'].shape[:2]

    # video embed are actions
    if getattr(self.cfg, "viclip_encode", False):
      video_embed = data['clip_video']
    else:
      # Obtaining video embed
      with torch.no_grad():
        viclip_model = getattr(self, 'viclip_model')
        processed_obs = viclip_model.preprocess_transf(data['observation'].reshape(B*T, *data['observation'].shape[2:]) / 255)
        reshaped_obs = processed_obs.reshape(B * (T // n_frames), n_frames, 3,224,224)
        video_embed = viclip_model.get_vid_features(reshaped_obs.to(viclip_model.device))
      
    # Get posterior states from original model
    wm_post = outputs['post']
    return connector.update(video_embed, wm_post)

class GenRLAgent(DreamerAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.n_frames = 8 # NOTE: this should become an hyperparam if changing the model
        self.viclip_emb_dim =  512 # NOTE: this should become an hyperparam if changing the model
        
        assert self.cfg.batch_length % self.n_frames == 0, "Fix batch length param"
        
        if 'clip_video' in self.obs_space:
          self.viclip_emb_dim = self.obs_space['clip_video'].shape[0]
        
        connector = video_utils.VideoSSM(**self.cfg.connector, **self.cfg.connector_rssm, connector_kl=self.cfg.connector_kl, 
                                          n_frames=self.n_frames, action_dim=self.viclip_emb_dim + self.n_frames, 
                                          clip_add_noise=self.cfg.clip_add_noise, clip_lafite_noise=self.cfg.clip_lafite_noise,
                                          device=self.device, cell_input='stoch') 
        
        connector.to(self.device)

        self.wm.add_module_to_update('connector', connector, connector_update_fn, detached=self.cfg.connector.detached_post)
    
        if getattr(self.cfg, 'imag_reward_fn', None) is not None:
          self.instantiate_imag_behavior()
    
    def instantiate_imag_behavior(self):
      self._imag_behavior = ActorCritic(self.cfg, self.act_spec, self.wm.inp_size, name='imag').to(self.device) 
      self._imag_behavior.rewnorm = common.StreamNorm(**self.cfg.imag_reward_norm, device=self.device)    
    
    def finetune_mode(self,):
      self._acting_behavior = self._imag_behavior
      self.wm.detached_update_fns = {}
      self.wm.e2e_update_fns = {}
      self.wm.grad_heads.append('reward')

    def update_wm(self, data, step):
      return super().update_wm(data, step)

    def report(self, data, key='observation', nvid=8):
      # Redefine data with trim
      n_frames = self.wm.connector.n_frames
      obs = data['observation'][:nvid, n_frames:]
      B, T = obs.shape[:2]

      report_data = super().report(data)
      wm = self.wm
      n_frames = wm.connector.n_frames
      
      # Init is same as Dreamer for reporting
      truth = data[key][:nvid] / 255
      decoder = wm.heads['decoder'] # B, T, C, H, W
      preprocessed_data = self.wm.preprocess(data)

      embed = wm.encoder(preprocessed_data)
      states, _ = wm.rssm.observe(embed[:nvid, :n_frames], data['action'][:nvid, :n_frames], data['is_first'][:nvid, :n_frames])
      recon = decoder(wm.decoder_input_fn(states))[key].mean[:nvid] # mode
      dreamer_init = {k: v[:, -1] for k, v in states.items()}

      # video embed are actions
      if getattr(self.cfg, "viclip_encode", False):
        video_embed = data['clip_video'][:nvid,n_frames*2-1::n_frames]
      else:
        # Obtain embed
        processed_obs = wm.viclip_model.preprocess_transf(obs.reshape(B*T, *obs.shape[2:]) / 255)
        reshaped_obs = processed_obs.reshape(B * (T // n_frames), n_frames, 3,224,224)
        video_embed = wm.viclip_model.get_vid_features(reshaped_obs.to(wm.viclip_model.device))
      
      video_embed = video_embed.to(self.device)

      # Get actions
      video_embed = video_embed.reshape(B, T // n_frames, -1).unsqueeze(2).repeat(1,1,n_frames, 1).reshape(B, T, -1)
      prior = wm.connector.video_imagine(video_embed, dreamer_init, reset_every_n_frames=False)
      prior_recon = decoder(wm.decoder_input_fn(prior))[key].mean # mode
      model = torch.clip(torch.cat([recon[:, :n_frames] + 0.5, prior_recon + 0.5], 1), 0, 1)
      error = (model - truth + 1) / 2
      
      # Add video to logs
      video = torch.cat([truth, model, error], 3)
      report_data['video_clip_pred'] = video
      
      return report_data

    def update_imag_behavior(self, state=None, outputs=None, metrics={}, seq_data=None,):
        if getattr(self.cfg, 'imag_reward_fn', None) is None:
           return outputs['post'], metrics
        if outputs is not None:
            post = outputs['post']
            is_terminal = outputs['is_terminal']
        else:
            seq_data = self.wm.preprocess(seq_data)
            embed = self.wm.encoder(seq_data)
            post, _ = self.wm.rssm.observe(
                embed, seq_data['action'], seq_data['is_first'])
            is_terminal = seq_data['is_terminal']
        #
        start = {k: stop_gradient(v) for k,v in post.items()}
        imag_reward_fn = lambda seq: globals()[self.cfg.imag_reward_fn](self, seq, **self.cfg.imag_reward_args)
        metrics.update(self._imag_behavior.update(self.wm, start, is_terminal, imag_reward_fn,))
        return start, metrics