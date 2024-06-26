import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from pathlib import Path
from collections import defaultdict

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import tools.utils as utils
from tools.logger import Logger
from tools.replay import ReplayBuffer, make_replay_loader

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_dreamer_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    if hasattr(cur_config, 'agent'):
        del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None,):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        self.cfg = cfg
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        self.task = task = cfg.task
        img_size = cfg.img_size

        import envs.main as envs
        self.train_env = envs.make(task, cfg.obs_type, cfg.action_repeat, cfg.seed, img_size=img_size,  viclip_encode=cfg.viclip_encode, clip_hd_rendering=cfg.clip_hd_rendering) 


        # # create agent 
        sample_agent = make_dreamer_agent(self.train_env.obs_space, self.train_env.act_space['action'], cfg, cfg.agent)

        # create replay buffer
        data_specs = (self.train_env.obs_space,
                      self.train_env.act_space,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        if cfg.train_from_data:
            # Loading replay buffer
            if cfg.replay_from_wandb_project is not None:
                api = wandb.Api()
                project_name = cfg.replay_from_wandb_project
                params2search = {
                    "task" : cfg.task if cfg.task_snapshot is None else cfg.task_snapshot,
                    "seed" : cfg.seed if cfg.seed_snapshot is None else cfg.seed_snapshot,
                }
                runs = api.runs(f"PUT_YOUR_USER_HERE/{project_name}")
                found = False
                for run in runs:
                    if np.all([ v == run.config.get(k, None) for k,v in params2search.items()]):
                        found = True
                        found_path = Path(run.config['workdir'].replace('/code', ''))
                        break
                if not found:
                    raise Exception("Replay from wandb buffer not found")

                replay_dir = found_path / 'code' / 'buffer'
            else:
                replay_dir = Path(cfg.replay_load_dir)

            # create data storage
            self.replay_storage = ReplayBuffer(data_specs, [],
                                                    replay_dir,
                                                    length=cfg.batch_length, **cfg.replay,
                                                    device=cfg.device, ignore_extra_keys=True, load_recursive=True)
            print('Loaded ', self.replay_storage._loaded_episodes, 'episodes from ', str(replay_dir))

            # create replay buffer
            self.replay_loader = make_replay_loader(self.replay_storage,
                                                    cfg.batch_size,)
            self._replay_iter = None

        # Loading snapshot
        if cfg.snapshot_from_wandb_project is not None:
            api = wandb.Api()
            project_name = cfg.snapshot_from_wandb_project
            params2search = {
                "task" : cfg.task if cfg.task_snapshot is None else cfg.task_snapshot,
                "agent_name" : cfg.agent.name if cfg.agent_name_snapshot is None else cfg.agent_name_snapshot,
                "seed" : cfg.seed if cfg.seed_snapshot is None else cfg.seed_snapshot,
            }
            if cfg.agent.clip_lafite_noise > 0.:
                params2search['clip_lafite_noise'] = cfg.agent.clip_lafite_noise
            if cfg.agent.clip_add_noise > 0.:
                params2search['clip_add_noise'] = cfg.agent.clip_add_noise
            if cfg.reset_connector:
                del params2search['clip_add_noise']
            runs = api.runs(f"PUT_YOUR_USER_HERE/{project_name}")
            found = False
            for run in runs:
                if np.all([ v == run.config.get(k, None) for k,v in params2search.items()]):
                    found = True
                    found_path = Path(run.config['workdir'].replace('/code', ''))
                    break
            if not found:
                raise Exception("Snapshot from wandb not found")

            if cfg.snapshot_step is None:
                snapshot_dir = found_path / 'code' / 'last_snapshot.pt'
            else:
                snapshot_dir = found_path / 'code' / f'snapshot_{cfg.snapshot_step}.pt'
        elif cfg.snapshot_load_dir is not None:
            snapshot_dir = Path(cfg.snapshot_load_dir)
        else:
            snapshot_dir = None

        if snapshot_dir is not None:        
            self.load_snapshot(snapshot_dir, resume=False)
            if self.cfg.reset_world_model:
                self.agent.wm = sample_agent.wm 
                # To reset optimization
                from agent import dreamer_utils as common
                self.agent.wm.model_opt = common.Optimizer('model', self.agent.wm.parameters(), **self.agent.wm.cfg.model_opt, use_amp=self.agent.wm._use_amp)
            if self.cfg.reset_connector:
                self.agent.wm.connector = sample_agent.wm.connector
                # To reset optimization
                from agent import dreamer_utils as common
                self.agent.wm.model_opt = common.Optimizer('model', self.agent.wm.parameters(), **self.agent.wm.cfg.model_opt, use_amp=self.agent.wm._use_amp)

            # overwriting cfg
            self.agent.cfg = sample_agent.cfg
            self.agent.wm.cfg = sample_agent.wm.cfg 
            
            if self.cfg.reset_imag_behavior:
                self.agent.instantiate_imag_behavior()
        else:
            self.agent = sample_agent

        self.eval_env = envs.make(self.task, self.cfg.obs_type, self.cfg.action_repeat, self.cfg.seed, img_size=64, )
        if hasattr(self.eval_env, 'eval_mode'):
            self.eval_env.eval_mode()
        eval_specs = (self.eval_env.obs_space,
                        self.eval_env.act_space,
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))
        self.eval_storage = ReplayBuffer(eval_specs, {},
                                                self.workdir / 'eval_buffer',
                                                length=cfg.batch_length, **cfg.replay,
                                                device=cfg.device, ignore_extra_keys=True,)
        self.eval_storage._minlen = 1

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        import envs.main as envs
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        episode_reward = []
        while eval_until_episode(len(episode_reward)):
            if len(episode_reward) > 0 and self.global_step == 0:
                return
            episode_reward.append(0)
            step, episode = 0, defaultdict(list)
            meta = self.agent.init_meta()
            time_step, dreamer_obs = self.eval_env.reset()
            data = dreamer_obs
            if 'clip_video' in data:
                del data['clip_video']
            self.eval_storage.add(data, meta) 
            agent_state = None
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(dreamer_obs, 
                                                meta,
                                                self.global_step,
                                                eval_mode=True,
                                                state=agent_state)
                time_step, dreamer_obs = self.eval_env.step(action)
                for k in dreamer_obs:
                    episode[k].append(dreamer_obs[k])
                episode_reward[-1] += time_step.reward
                if time_step.last():
                    if episode_reward[-1] == np.max(episode_reward):
                        best_episode = {**episode}
                    if episode_reward[-1] == np.min(episode_reward):
                        worst_episode = {**episode}
                data = dreamer_obs
                if 'clip_video' in data:
                    del data['clip_video']
                self.eval_storage.add(data, meta) 
                step += 1

        if self.global_step > 0 and self.global_frame % self.cfg.log_episodes_every_frames == 0:
            # B, T, C, H, W = video.shape
            videos = {'best_episode' : np.stack(best_episode['observation'], axis=0),
                    'worst_episode' : np.stack(worst_episode['observation'], axis=0),}
            self.logger.log_visual(videos, self.global_frame)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', np.mean(episode_reward))
            log('episode_length', step * self.cfg.action_repeat)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def eval_imag_behavior(self,):
        self.agent._backup_acting_behavior = self.agent._acting_behavior
        self.agent._acting_behavior = self.agent._imag_behavior
        self.eval()
        self.agent._acting_behavior = self.agent._backup_acting_behavior

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, 1)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, 1)
        should_save_model = utils.Every(self.cfg.save_every_frames, 1)
        should_log_visual = utils.Every(self.cfg.visual_every_frames, 1)
        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                if self.cfg.eval_modality == 'task':
                    self.eval()
                if self.cfg.eval_modality == 'task_imag':
                    self.eval_imag_behavior()
                if self.cfg.eval_modality == 'from_text':
                    self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                    self.eval_from_text()

            if self.cfg.train_from_data:
                # Sampling data
                batch_data = next(self.replay_iter)
                if self.cfg.train_world_model:
                    state, outputs, metrics = self.agent.update_wm(batch_data, self.global_step)
                else:
                    with torch.no_grad():
                        outputs, metrics = self.agent.wm.observe_data(batch_data,)
                if self.cfg.train_connector:
                    _, metrics = self.agent.wm.update_additional_detached_modules(batch_data, outputs, metrics)
            else:
                imag_warmup_steps = self.cfg.imag_warmup_steps
                metrics, batch_data = {}, None
                with torch.no_grad():
                    # fake actions 
                    mix = self.cfg.mix_random_actions
                    random = False
                    # num warmup steps

                    if mix:
                        init = self.agent.wm.rssm.initial(self.cfg.batch_size * (self.cfg.batch_length // 2))
                    else:                        
                        init = self.agent.wm.rssm.initial(self.cfg.batch_size * self.cfg.batch_length)


                    unif_dist = self.agent.wm.rssm.get_unif_dist(init)
                    if 'logit' in init:
                        init['logit'] = unif_dist.mean
                    else:
                        init['mean'] = unif_dist.mean 
                        init['std'] = unif_dist.std 
                    init['stoch'] = unif_dist.sample()
                    
                    if self.cfg.start_from_video in [True, 'mix']:
                        T = self.agent.wm.connector.n_frames * 2 # should this be an hyperparam?
                        B = init['deter'].shape[0] // T 
                        text_feat_dim = self.agent.wm.connector.viclip_emb_dim
                        video_embed = torch.randn((B, T, text_feat_dim), device=self.agent.device)
                        video_embed = torch.nn.functional.normalize(video_embed, dim=-1)
                        # Get initial state
                        video_init = self.agent.wm.connector.video_imagine(video_embed, dreamer_init=None, sample=True, reset_every_n_frames=False, denoise=True)
                        video_init = { k : v.reshape(B * T, *v.shape[2:]) for k, v in video_init.items()}
                        if self.cfg.start_from_video == 'mix':
                            probs = torch.rand((B * T, 1,1), device=init['stoch'].device) > 0.5  # should this be an hyperparam?
                            init['stoch'] = (probs * init['stoch']) + ( (~probs) * video_init['stoch'] )
                        else:
                            init['stoch'] = video_init['stoch']
                    
                    if random:
                        fake_action = torch.rand(self.cfg.batch_size * self.cfg.batch_length, imag_warmup_steps, self.agent.act_dim, device=self.agent.device) * 2 - 1
                        post = self.agent.wm.rssm.imagine(fake_action, init, sample=True)
                        post = { k : v[:, -1].reshape([self.cfg.batch_size, self.cfg.batch_length, ] + list(v.shape[2:])) for k,v in post.items() }
                    elif mix:
                        fake_action = torch.rand(self.cfg.batch_size * self.cfg.batch_length // 2, imag_warmup_steps, self.agent.act_dim, device=self.agent.device) * 2 - 1
                        post1 = self.agent.wm.rssm.imagine(fake_action, init, sample=True)
                        post1 = { k : v[:, -1].reshape([self.cfg.batch_size, self.cfg.batch_length // 2, ] + list(v.shape[2:])) for k,v in post1.items() }

                        init2 = { k : v.reshape([self.cfg.batch_size, self.cfg.batch_length // 2, ] + list(v.shape[1:])) for k,v in init.items() }
                        post2 = self.agent.wm.imagine(self.agent._imag_behavior.actor, init2, None, imag_warmup_steps) 
                        post2 = { k : v[-1, :].reshape([self.cfg.batch_size, self.cfg.batch_length // 2, ] + list(v.shape[2:])) for k,v in post2.items() }
                        post = { k: torch.cat([post1[k], post2[k]], dim=1) for k in post1 }
                    else:
                        init = { k : v.reshape([self.cfg.batch_size, self.cfg.batch_length, ] + list(v.shape[1:])) for k,v in init.items() }
                        post = self.agent.wm.imagine(self.agent._imag_behavior.actor, init, None, imag_warmup_steps) 
                        post = { k : v[-1, :].reshape([self.cfg.batch_size, self.cfg.batch_length, ] + list(v.shape[2:])) for k,v in post.items() }

                is_terminal = torch.zeros(self.cfg.batch_size, self.cfg.batch_length, device=self.agent.device)
                outputs = dict(post=post, is_terminal=is_terminal)
            if getattr(self.cfg.agent, 'imag_reward_fn', None) is not None:
                metrics.update(self.agent.update_imag_behavior(state=None, outputs=outputs, metrics=metrics, seq_data=batch_data,)[1])

            if self.global_step > 0:
                if should_log_scalars(self.global_step):
                    if hasattr(self, 'replay_storage'):
                        metrics.update(self.replay_storage.stats)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if should_log_visual(self.global_step) and self.cfg.train_from_data and hasattr(self.agent, 'report'):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        videos = self.agent.report(next(self.replay_iter))
                        self.logger.log_visual(videos, self.global_frame)
                if should_log_scalars(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', self.cfg.log_every_frames / elapsed_time)
                        log('step', self.global_step)
                        if 'model_loss' in metrics: 
                            log('episode_reward', metrics['model_loss'].item())
                    
                # save last model
                if should_save_model(self.global_step):
                    self.save_last_model()

            self._global_step += 1
            # == 1000 is to make sure everything is going well since the start
            if (self.global_frame == 1000) or (self.global_frame % self.cfg.snapshot_every_frames == 0):
                self.save_snapshot()

    @utils.retry
    def save_snapshot(self):
        snapshot = self.root_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = '_'.join([
            cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
            str(cfg.seed)
        ])
        wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name)
        flat_cfg = utils.flatten_dict(cfg)
        wandb.config.update(flat_cfg)
        self.wandb_run_id = wandb.run.id

    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if self.cfg.use_wandb: 
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    @utils.retry
    def load_snapshot(self, snapshot_dir, resume=True):
        print('Loading snapshot from: ', str(snapshot_dir))
        try:
            snapshot = snapshot_dir / 'last_snapshot.pt' if resume else snapshot_dir
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = Path(str(snapshot_dir).replace('last_snapshot', 'second_last_snapshot'))
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        if type(payload) != dict:
            self.agent = payload
            self.agent.requires_grad_(requires_grad=False)
            return
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id' and resume:
                assert wandb.run is None
                cfg = self.cfg
                exp_name = '_'.join([
                    cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
                    str(cfg.seed)
                ])
                wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name, id=v, resume="must")

    def get_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir 

def start_training(cfg, savedir, workdir):
    from train import Workspace as W
    root_dir = Path.cwd()
    cfg.workdir = str(root_dir)
    workspace = W(cfg, savedir, workdir)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(workspace.root_dir)
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    workspace.train()

@hydra.main(config_path='.', config_name='train')
def main(cfg):
    start_training(cfg, None, None)

if __name__ == '__main__':
    main()
