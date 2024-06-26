import torch
import torch.nn as nn
import torch.nn.functional as F
import agent.dreamer_utils as common
from collections import defaultdict
import numpy as np

class ResidualLinear(nn.Module):
    def __init__(self, in_channels, out_channels, norm='layer', act='SiLU', prenorm=False):
        super().__init__()
        self.norm_layer = common.NormLayer(norm, in_channels if prenorm else out_channels)
        self.act = common.get_act(act)
        self.layer = nn.Linear(in_channels, out_channels)
        self.prenorm = prenorm
        self.res_proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        if self.prenorm:
            h = self.norm_layer(x)
            h = self.layer(h)
        else:            
            h = self.layer(x)
            h = self.norm_layer(h)
        h = self.act(h)
        return h + self.res_proj(x)

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels : int, mid_channels : int, n_layers : int, norm='layer', act= 'SiLU', ):
        super().__init__()
        out_channels = in_channels
        self.down = nn.ModuleList()
        for i in range(n_layers):
            if i == (n_layers - 1):
                self.down.append(ResidualLinear(in_channels, mid_channels, norm=norm, act=act))
            else:
                self.down.append(ResidualLinear(in_channels, in_channels, norm=norm, act=act))

        self.mid = nn.ModuleList()
        for i in range(n_layers):
            self.mid.append(ResidualLinear(mid_channels, mid_channels, norm=norm, act=act))
        
        self.up = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.up.append(ResidualLinear(mid_channels * 2, out_channels, norm='none', act='Identity'))
            else:
                self.up.append(ResidualLinear(out_channels * 2, out_channels, norm=norm, act=act))
        
    def forward(self, x):
        down_res = []
        for down_layer in self.down:
            x = down_layer(x)
            down_res.append(x)
        
        for mid_layer in self.mid:
            x = mid_layer(x)
        
        down_res.reverse()
        for up_layer, res in zip(self.up, down_res):
            x = up_layer(torch.cat([x, res], dim=-1))
        return x


class VideoSSM(common.EnsembleRSSM):
    def __init__(self, *args, 
                 connector_kl={}, temporal_embeds=False, detached_post=True, n_frames=8, 
                 token_dropout=0.,  loss_scale=1, clip_add_noise=0, clip_lafite_noise=0,
                 rescale_embeds=False, denoising_ae=False, learn_initial=True, **kwargs,):
        super().__init__(*args, **kwargs)
        #
        self.n_frames = n_frames
        # by default, adding the n_frames in actions (doesn't hurt and easier to test whether it's useful or not)
        self.viclip_emb_dim = kwargs['action_dim'] - self.n_frames
        #
        self.temporal_embeds = temporal_embeds
        self.detached_post = detached_post
        self.connector_kl = connector_kl
        self.token_dropout = token_dropout
        self.loss_scale = loss_scale
        self.rescale_embeds = rescale_embeds
        self.clip_add_noise = clip_add_noise
        self.clip_lafite_noise = clip_lafite_noise
        self.clip_const = np.sqrt(self.viclip_emb_dim).item()
        self.denoising_ae = denoising_ae
        if self.denoising_ae:
            self.aligner = UNetDenoiser(self.viclip_emb_dim, self.viclip_emb_dim // 2, n_layers=2, norm='layer', act='SiLU')
        self.learn_initial = learn_initial
        if self.learn_initial:
            self.initial_state_pred = nn.Sequential(
                nn.Linear(kwargs['action_dim'], kwargs['hidden']),
                common.NormLayer(kwargs['norm'],kwargs['hidden']), common.get_act('SiLU'),
                nn.Linear(kwargs['hidden'], kwargs['hidden']),
                common.NormLayer(kwargs['norm'],kwargs['hidden']), common.get_act('SiLU'),
                nn.Linear(kwargs['hidden'], kwargs['deter'])
            )
        # Deleting non-useful models
        del self._obs_out
        del self._obs_dist

    def initial(self, batch_size, init_embed=None, ignore_learned=False):
        init = super().initial(batch_size)
        if self.learn_initial and not ignore_learned and hasattr(self, 'initial_state_pred'):
            assert init_embed is not None
            # patcher to avoid edge cases
            if init_embed.shape[-1] == self.viclip_emb_dim:
                patcher = torch.zeros((*init_embed.shape[:-1], 8), device=self.device)
                init_embed = torch.cat([init_embed, patcher], dim=-1)
            init['deter'] = self.initial_state_pred(init_embed)
            stoch, stats = self.get_stoch_stats_from_deter_state(init)
            init['stoch'] = stoch
            init.update(stats)
        return init

    def get_action(self, video_embed):
        n_frames = self.n_frames 
        B, T = video_embed.shape[:2]

        if self.rescale_embeds:
            video_embed = video_embed * self.clip_const
        
        temporal_embeds = F.one_hot(torch.arange(T).to(video_embed.device) % n_frames, n_frames).reshape(1, T, n_frames,).repeat(B, 1, 1,)
        if not self.temporal_embeds:
            temporal_embeds *= 0
        
        return torch.cat([video_embed, temporal_embeds],dim=-1)

    def update(self, video_embed, wm_post):
        n_frames = self.n_frames 
        B, T = video_embed.shape[:2]
        loss = 0
        metrics = {}

        # NOVEL
        video_embed = video_embed[:,n_frames-1::n_frames] # tested
        video_embed = video_embed.to(self.device)
        video_embed = video_embed.reshape(B, T // n_frames, 1, -1).repeat(1,1, n_frames, 1).reshape(B, T, -1)

        orig_video_embed = video_embed

        if self.clip_add_noise > 0:
            video_embed = video_embed  + torch.randn_like(video_embed, device=video_embed.device) * self.clip_add_noise
            video_embed = nn.functional.normalize(video_embed, dim=-1)
        if self.clip_lafite_noise > 0:
            normed_noise = F.normalize(torch.randn_like(video_embed, device=video_embed.device), dim=-1)
            video_embed = (1 - self.clip_lafite_noise) * video_embed  + self.clip_lafite_noise * normed_noise
            video_embed = nn.functional.normalize(video_embed, dim=-1)

        if self.denoising_ae:
            assert (self.clip_lafite_noise + self.clip_add_noise) > 0, "Nothing to denoise"
            denoised_embed = self.aligner(video_embed)
            denoised_embed = F.normalize(denoised_embed, dim=-1)
            denoising_loss = 1 - F.cosine_similarity(denoised_embed, orig_video_embed, dim=-1).mean() # works same as F.mse_loss(denoised_embed, orig_video_embed).mean()
            loss += denoising_loss
            metrics['aligner_cosine_distance'] = denoising_loss
            # if using a denoiser, it's the denoiser's duty to denoise the video embed
            video_embed = orig_video_embed # could also be denoised_embed for e2e training

        embed_actions = self.get_action(video_embed)
        
        if self.detached_post:
            wm_post = { k : v.reshape(B, T, *v.shape[2:]).detach() for k,v in wm_post.items() } 
        else:
            wm_post = { k : v.reshape(B, T, *v.shape[2:]) for k,v in wm_post.items() } 
        
        # Get prior states
        prior_states = defaultdict(list)
        for t in range(T):
            # Get video action
            action = embed_actions[:, t]

            if t == 0: 
                prev_state = self.initial(batch_size=wm_post['stoch'].shape[0], init_embed=action)
            else:
                # Get deter from prior, get stoch from wm_post
                prev_state = prior
                prev_state[self.cell_input] = wm_post[self.cell_input][:, t-1]

            if self.token_dropout > 0:
                prev_state['stoch'] = torch.einsum('b...,b->b...', prev_state['stoch'], (torch.rand(B, device=action.device) > self.token_dropout).float() )

            prior = self.img_step(prev_state, action)
            for k in prior: 
                prior_states[k].append(prior[k])

        # Aggregate
        for k in prior_states:
            prior_states[k] = torch.stack(prior_states[k], dim=1)

        # Compute loss
        prior = prior_states
            
        kl_loss, kl_value = self.kl_loss(wm_post, prior, **self.connector_kl)
        video_loss = self.loss_scale * kl_loss 
        metrics['connector_kl'] = kl_value.mean()
        loss += video_loss
        
        # Compute initial KL
        video_embed = video_embed.reshape(B, T // n_frames, n_frames, -1)[:,1:,0].reshape(B * (T//n_frames-1), 1, -1) # taking only one (0) and skipping first temporal step
        embed_actions = self.get_action(video_embed)
        wm_post = { k : v.reshape(B, T // n_frames, n_frames, *v.shape[2:])[:,1:,0].reshape(B * (T//n_frames-1), *v.shape[2:]) for k,v in wm_post.items() } 
        action = embed_actions[:, 0]
        prev_state = self.initial(batch_size=wm_post['stoch'].shape[0], init_embed=action)
        prior = self.img_step(prev_state, action)
        kl_loss, kl_value = self.kl_loss(wm_post, prior, **self.connector_kl)
        metrics['connector_initial_kl'] = kl_value.mean()

        return loss, metrics
            
    def video_imagine(self, video_embed, dreamer_init=None, sample=True, reset_every_n_frames=True, denoise=False):
        n_frames = self.n_frames 
        B, T = video_embed.shape[:2]

        if self.denoising_ae and denoise:
            denoised_embed = self.aligner(video_embed)
            video_embed = F.normalize(denoised_embed, dim=-1)

        action = self.get_action(video_embed)
        # Imagine
        init = self.initial(batch_size=B, init_embed=action[:, 0]) # -> this ensures only stoch is used from the current frame
        if dreamer_init is not None:
            init[self.cell_input] = dreamer_init[self.cell_input]

        if reset_every_n_frames:
            prior_states = defaultdict(list)
            for action_chunk in torch.chunk(action, T // n_frames, dim=1):
                prior = self.imagine(action_chunk, init, sample=sample)
                for k in prior: 
                    prior_states[k].append(prior[k])
        
                # -> this ensures only stoch is used from the current frame
                init = self.initial(batch_size=B, ignore_learned=True) 
                init[self.cell_input] = prior[self.cell_input][:, -1]
            
            # Agg
            for k in prior_states:
                prior_states[k] = torch.cat(prior_states[k], dim=1)
            prior = prior_states
        else:
            prior = self.imagine(action, init, sample=sample)
        return prior