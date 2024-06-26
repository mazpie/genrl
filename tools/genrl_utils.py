import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path

MODELS_ROOT_PATH = Path(__file__).parent.parent / 'models'
INTERNVIDEO_PATH = Path(__file__).parent.parent / 'third_party' / 'InternVideo'

DOMAIN2PREDICATES = {
    'walker' : ['taking a walk', 'standing up vertically on both feet', 'single-leg balancing', "standing upside down",  'high kick', 'walking', 'stepping forward', 'running fast', 
                'standing on one bended knee', 'lying down on the back with one raised leg', 'sitting on the knees', 'dog yoga pose',  'lying down horizontally', ],
    'stickman' : ['taking a walk', 'standing up vertically', 'one leg balancing',  'high kick', 'walking', 'running fast', 
                'praying', 'lying down with one raised leg', 'dog yoga pose',  'lying down horizontally', 'punching', 'raised hands' ],
    'cheetah' : ['jumping', 'crawling', 'running', 'flipping', 'standing up', 'hopping', 'lying down', 'falling', 
                 'standing on the knees'],
    'quadruped' : ['jumping', 'crawling', 'walking', 'standing up', 
                   'hopping', 'lying down', 'falling', 'standing on the knees'],
    'finger' : ['spin', 'touch', 'rotate', 'horizontal', 'vertical', "not moving", "is not touching", "staying far away", "staying still"],
    'pendulum' : ['horizontal', 'vertical', 'left', 'right', 
                  'swingup', 'balance'],
    'hopper' : ['jumping', 'crawling', 'walking', 'standing up', 
                'hopping', 'lying down', 'falling', 'standing on the knees'],
    'reacher' : ['horizontal', 'vertical', 'ball on the left', 'ball on the right', 'touch the ball with the elbow', 'touch the ball with the tip', 'arm reaches the sphere', 'rotating', 'bending', 'keeping straight', "not moving", "is not touching"],
    'jaco' : ['horizontal', 'vertical', 'left', 'right', 'spin', 'touch', 'rotate', 'bend', 'straight', "is not touching"],
    'kitchen' : [ "touch", "pick up", "lift", "grasp", "hold", "pull", "open", "close", 
                "push", "sweep", "slide"] + ['switch light on', 'open the microwave', 'move the kettle', 'turn on the burner'],
}

TASK2PROMPT = {
    "quadruped_run" : 'spider running fast',
    "quadruped_walk" : 'spider walking fast',
    "quadruped_stand" : 'spider standing',
    "quadruped_jump" : 'spider jumping',

    "quadruped_two_legs" : 'on two legs',
    "quadruped_lie_down" : 'lying down',
    
    "cheetah_run" : 'running like a quadruped',
    
    "cheetah_flipping" : 'quadruped rotating flips',
    "cheetah_standing" : 'standing like a human',
    "cheetah_lying_down" : 'lying down',

    'stickman_walk' : 'robot walk fast clean',
    'stickman_run' : 'robot run fast clean',
    'stickman_stand' : 'standing',
    'stickman_urlb_flip' : 'doing flips',

    'stickman_flip' : 'doing flips',
    'stickman_flipping' : 'doing flips',
    'stickman_backflip' : 'doing backflips',
    'stickman_one_foot' : 'stand on one foot',
    'stickman_high_kick' : 'stand up and kick', 
    'stickman_lying_down' : 'lying down horizontally',
    'stickman_legs_up' : 'lying down with feet up',
    'stickman_sit_knees' : 'praying',
    'stickman_lunge_pose' : 'lunge_pose',
    'stickman_headstand' : 'headstand',
    'stickman_boxing' : 'punch',
    'stickman_hands_up' : 'standing with the hands up',

    'walker_walk' : 'walk fast clean',
    'walker_run' : 'run fast clean',
    'walker_stand' : 'standing up straight',
    'walker_urlb_flip' : 'doing backflips',

    'walker_flip' : 'doing flips',
    'walker_flipping' : 'doing backflips',
    'walker_backflip' : 'doing backflips',
    'walker_one_foot' : 'stand on one foot',
    'walker_high_kick' : 'stand up and kick', 
    'walker_lying_down' : 'lying down horizontally',
    'walker_arabesque' : 'arabesque position', 
    'walker_legs_up' : 'lying down with feet up',
    'walker_sit_knees' : 'praying',
    'walker_lunge_pose' : 'lunge_pose',
    'walker_headstand' : 'headstand',

    'kitchen_microwave' : 'opening the microwave fully open',
    'kitchen_light' : 'activate the light',
    'kitchen_burner' : 'the burner becomes red',
    'kitchen_slide' : 'slide cabinet above the knobs',

    'kitchen_kettle' : 'pushing up the kettle',

    'jaco_reach_top_left' : 'robot grasp the red cube',
    'jaco_reach_bottom_left' : 'robot grasp the red cube',
    'jaco_reach_top_right' : 'robot grasp the red cube',
    'jaco_reach_bottom_right' : 'robot grasp the red cube',
}

class ViCLIPGlobalInstance:
    def __init__(self, model='internvideo2'):
        self._instantiated = False
        self._model = model
    
    def instantiate(self, device='cuda'):
        from torchvision.transforms import transforms as vision_transf
        import sys

        self._instantiated = True
        
        if self._model =='internvideo2':
            sys.path.insert(0, str(INTERNVIDEO_PATH / 'InternVideo2/multi_modality/demo/'))
            sys.path.insert(0, str(INTERNVIDEO_PATH / 'InternVideo2/multi_modality'))
            import numpy as np
            from small_config import (Config, eval_dict_leaf)
            from small_utils import setup_internvideo2 
            config = Config.from_file(INTERNVIDEO_PATH / 'InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
            config = eval_dict_leaf(config)
            config.model.vision_encoder.num_frames = 8
            config.num_frames = 8
            config.num_frames_test = 8
            # # >> can be configured in case the bert model doesn't load
            # config.model.text_encoder.pretrained = str(MODELS_ROOT_PATH / 'bert-large-uncased')
            config.model.text_encoder.config = str(INTERNVIDEO_PATH / 'InternVideo2/multi_modality') + "/" + config.model.text_encoder.config
            model_pth = str(MODELS_ROOT_PATH / 'InternVideo2-stage2_1b-224p-f4.pt')
            config.pretrained_path = model_pth
            config['model']['vision_encoder']['pretrained'] = model_pth
            intern_model, tokenizer = setup_internvideo2(config)
            self.viclip_tokenizer = tokenizer
            self.viclip = intern_model
            self.viclip.device = device 
            self.viclip.to(self.viclip.device)
            self.viclip.eval()
            self.viclip.n_frames = 8
            self.viclip.preprocess_transf = vision_transf.Compose([
                vision_transf.Resize(size=(224, 224), interpolation=vision_transf.InterpolationMode.BILINEAR), 
                vision_transf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            sys.path.pop(0)
            sys.path.pop(0)
        else:
            raise NotImplementedError(f"Model {self._model} not implemented")
    
        vid_feat = self.viclip.get_vid_features(torch.zeros(1,self.viclip.n_frames,3,224,224, device=self.viclip.device))
        self.viclip_emb_dim = vid_feat.shape[1]


def report_text2video(agent, data,):
    report = {}

    domain = agent.cfg.task.split('_')[0]
    labels_list = DOMAIN2PREDICATES[domain]

    wm = world_model = agent.wm
    decoder = world_model.heads['decoder'] # B, T, C, H, W
    connector = agent.wm.connector
    n_frames = connector.n_frames

    if hasattr(world_model, 'viclip_model'):
        clip = world_model.viclip_model
    else:
        # Get ViCLIP
        viclip_global_instance = globals()['viclip_global_instance']
        if not viclip_global_instance._instantiated:
            viclip_global_instance.instantiate()
        clip = viclip_global_instance.viclip

    # Get text(video) embed
    text_feat = []
    for text in labels_list:
        with torch.no_grad():
            text_feat.append(clip.get_txt_feat(text,))
    text_feat = torch.stack(text_feat, dim=0)
    # Check device is right
    video_embed = text_feat.to(agent.device)
    B = video_embed.shape[0]

    # Get actions
    video_embed = video_embed.repeat(1,n_frames, 1)
    # Imagine
    prior = wm.connector.video_imagine(video_embed, dreamer_init=None, sample=False, reset_every_n_frames=False, denoise=True)
    prior_recon = decoder(wm.decoder_input_fn(prior))['observation'].mean + 0.5
    report[f'text_to_video'] = prior_recon
    return report    

def max_cosine_similarity(u, v, dim=-1):
    max_norm = torch.max(torch.norm(u, dim=dim), torch.norm(v, dim=dim)).unsqueeze(-1)
    return torch.sum((u / max_norm) * (v / max_norm), dim=dim)

def neg_mse_fn(a, b, dim=-1, scale=True):
    dist = - torch.norm(a - b, dim=dim)
    if scale:
        dist = dist / np.sqrt(a.shape[-1]).item()
    return dist

def compute_reward(agent, agent_seq, target_seq, score_fn='cosine',):
    if score_fn in ['cosine', 'max_cosine', 'neg_mse', 'exp_neg_mse']:
        distance_fn = dict(cosine=F.cosine_similarity, max_cosine=max_cosine_similarity, neg_mse=neg_mse_fn, exp_neg_mse=neg_mse_fn)[score_fn]
        target_stoch = agent.wm.connector.get_stoch( target_seq )
        agent_stoch = agent.wm.rssm.get_stoch( agent_seq )
        conv_target = agent.wm.heads['decoder']._conv_in[0](target_stoch)
        conv_agent = agent.wm.heads['decoder']._conv_in[0](agent_stoch)
        reward = distance_fn(conv_target, conv_agent, dim=-1)
        if score_fn == 'exp_neg_mse':
            reward = torch.exp(reward)
    elif score_fn == 'neg_kl':
        agent_dist = agent.wm.rssm.get_dist( agent_seq )
        target_dist = agent.wm.connector.get_dist( target_seq )
        reward =  -torch.distributions.kl_divergence(agent_dist, target_dist,)
        # scaling factor ( x log x w.r.t. to classes, or just x)
        if 'logit' in target_seq:
            reward = reward / ( np.log(target_seq['logit'].shape[-1]) * target_seq['logit'].shape[-2] )
        else:
            reward = reward / target_seq['mean'].shape[-1]
    elif score_fn == 'max_like':
        agent_dist = agent.wm.rssm.get_dist( agent_seq )
        target_sample = target_seq['stoch']
        reward =  agent_dist.log_prob(target_sample)
    elif score_fn == 'combo':
        return compute_reward(agent, agent_seq, target_seq, 'cosine') + compute_reward(agent, agent_seq, target_seq, 'neg_kl')        
    else:
        raise NotImplementedError(f"{score_fn} reward not implemented")
    return reward

def video_text_reward(agent, seq, score_fn='cosine', 
                      sample_for_target=False, weighted_align=False, align_initial=False, align_sequence=False, 
                      task_prompt='', skip_first_target=False, **kwargs):
    wm = world_model = agent.wm
    connector = agent.wm.connector
    n_frames = connector.n_frames

    T, B = seq['deter'].shape[:2]
    imagined_steps = T

    if not hasattr(agent, 'unconditional_target'):
        if hasattr(world_model, 'viclip_model'):
            clip = world_model.viclip_model
        else:
            # Get ViCLIP
            viclip_global_instance = globals()['viclip_global_instance']
            if not viclip_global_instance._instantiated:
                viclip_global_instance.instantiate()
            clip = viclip_global_instance.viclip

        if task_prompt != '':
            task = [task_prompt]
        else:
            task = [ TASK2PROMPT[agent.cfg.task] ]
            
        # Get text(video) embed
        with torch.no_grad():
            text_feat = clip.get_txt_feat(task[0],)
        # Check device is right
        video_embed = text_feat.to(agent.device)

        # Unconditional gen
        if skip_first_target:
            video_embed = video_embed.reshape(1, 1, -1).repeat(B, imagined_steps + 1, 1)
            unconditional_stats = wm.connector.video_imagine(video_embed, dreamer_init=None, sample=sample_for_target, reset_every_n_frames=False, denoise=True) 
            unconditional_stats = { k: v[:,1:].permute([1,0] + list(range(2, len(v.shape)))) for k,v in unconditional_stats.items() }
        else:
            video_embed = video_embed.reshape(1, 1, -1).repeat(B, imagined_steps, 1)
            unconditional_stats = wm.connector.video_imagine(video_embed, dreamer_init=None, sample=sample_for_target, reset_every_n_frames=False, denoise=True) 
            unconditional_stats = { k: v.permute([1,0] + list(range(2, len(v.shape)))) for k,v in unconditional_stats.items() }
        agent.unconditional_target = unconditional_stats
    else:
        unconditional_stats = agent.unconditional_target
        
    agent_seq = seq
    target_seq = unconditional_stats
    if align_initial:
        assert not align_sequence, 'Cannot align initial and sequence at the same time'
        init_seq = { k: v[0] for k,v in target_seq.items() }
        init_score = compute_reward(agent, agent_seq, init_seq, score_fn=score_fn,)
        if weighted_align:
            w = 0.99 * torch.ones_like(init_score, device=init_score.device)
            w = torch.cumprod(w, dim=1)
            init_score = w * init_score
        # 
        best_indexes_one_hot = F.one_hot(torch.argmax(init_score, dim=0), num_classes=target_seq['stoch'].shape[0])
        ts_idx = torch.clip(torch.cumsum(torch.cumsum(best_indexes_one_hot, dim=1), dim=1) - 1, min=0).T
        new_target_seq = {}
        for k,v in target_seq.items():
            if len(v.shape) == 4:
                new_ts = ts_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1, v.shape[-2], v.shape[-1])
            else:
                new_ts = ts_idx.unsqueeze(-1).repeat(1,1, v.shape[-1])
            new_target_seq[k] = torch.gather(v, 0, new_ts) # out[i][j][k] = input[index[i][j][k]][j][k]
        return compute_reward(agent, agent_seq, new_target_seq, score_fn=score_fn,).unsqueeze(-1)
    elif align_sequence:
        align_score = []
        get_prev_a_b = lambda d, a, b : { k : v[a:b] for k,v in d.items() } 
        shorter_target_seq = get_prev_a_b(unconditional_stats, 0, n_frames)
        for t in range(T-n_frames):
            cur_agent_seq = get_prev_a_b(seq, t, t+n_frames)
            score = compute_reward(agent, cur_agent_seq, shorter_target_seq, score_fn=score_fn,).mean(dim=0) # 0 is time dimension
            align_score.append(score) 
        align_score = torch.stack(align_score, dim=0)
        if weighted_align:
            w = 0.99 * torch.ones_like(align_score, device=align_score.device)
            w = torch.cumprod(w, dim=1)
            align_score = w * align_score
        best_indexes_one_hot = F.one_hot(torch.argmax(align_score, dim=0), num_classes=target_seq['stoch'].shape[0])
        ts_idx = torch.clip(torch.cumsum(torch.cumsum(best_indexes_one_hot, dim=1), dim=1) - 1, min=0).T
        new_target_seq = {}
        for k,v in target_seq.items():
            if len(v.shape) == 4:
                new_ts = ts_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1, v.shape[-2], v.shape[-1])
            else:
                new_ts = ts_idx.unsqueeze(-1).repeat(1,1, v.shape[-1])
            new_target_seq[k] = torch.gather(v, 0, new_ts) # out[i][j][k] = input[index[i][j][k]][j][k]
        return compute_reward(agent, agent_seq, new_target_seq, score_fn=score_fn,).unsqueeze(-1)
    else:
        neg_kl = compute_reward(agent, agent_seq, target_seq, score_fn=score_fn,)
    
    return neg_kl.unsqueeze(-1)

global viclip_global_instance
viclip_global_instance = ViCLIPGlobalInstance()