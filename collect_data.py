import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import tools.utils as utils
from tools.logger import Logger
from tools.replay import ReplayBuffer, make_replay_loader

torch.backends.cudnn.benchmark = True

# os.environ['WANDB_API_KEY'] = 'local-1b6c1e2a2fd8d4c98b8c049eb2914dbceccd4b7c' # local-1b6c1e2a2fd8d4c98b8c049eb2914dbceccd4b7c
# os.environ['WANDB_BASE_URL'] = 'https://192.168.170.90:443'
# os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_dreamer_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None):
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
        self.agent = make_dreamer_agent(self.train_env.obs_space, self.train_env.act_space['action'], cfg, cfg.agent)
        
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.obs_space,
                      self.train_env.act_space,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBuffer(data_specs, meta_specs,
                                                  self.workdir / 'buffer',
                                                  length=cfg.batch_length, **cfg.replay,
                                                  device=cfg.device)

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.batch_size,)
        self._replay_iter = None

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
        eval_env = envs.make(self.task, self.cfg.obs_type, self.cfg.action_repeat, self.cfg.seed, img_size=64,)
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step, dreamer_obs = eval_env.reset()
            agent_state = None
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(dreamer_obs, 
                                                meta,
                                                self.global_step,
                                                eval_mode=True,
                                                state=agent_state)
                time_step, dreamer_obs = eval_env.step(action)
                total_reward += time_step.reward
                step += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def eval_imag_behavior(self,):
        self.agent._backup_acting_behavior = self.agent._acting_behavior
        self.agent._acting_behavior = self.agent._imag_behavior
        self.eval()
        self.agent._acting_behavior = self.agent._backup_acting_behavior

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        train_every_n_steps = max(self.cfg.train_every_actions // self.cfg.action_repeat, 1) 
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat, self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, self.cfg.action_repeat)
        should_log_visual = utils.Every(self.cfg.visual_every_frames, self.cfg.action_repeat)
        should_save_model = utils.Every(self.cfg.save_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step, dreamer_obs = self.train_env.reset()
        agent_state = None
        meta = self.agent.init_meta()
        data = dreamer_obs
        self.replay_storage.add(data, meta) 
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                    ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage)) 
                        log('step', self.global_step)
                if should_save_model(self.global_step):
                    # save last model
                    self.save_last_model()

                # reset env
                time_step, dreamer_obs = self.train_env.reset()
                # Updating agent
                agent_state = None # Resetting agent's latent state
                meta = self.agent.init_meta()
                data = dreamer_obs
                self.replay_storage.add(data, meta) 
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                if self.cfg.eval_modality == 'task':
                    self.eval()
                if self.cfg.eval_modality == 'task_imag':
                    self.eval_imag_behavior()
                if self.cfg.eval_modality == 'from_text':
                    self.logger.log('eval_total_time', self.timer.total_time(),
                                    self.global_frame)
                    self.eval_from_text()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if  seed_until_step(self.global_step):
                    action =  self.train_env.act_space['action'].sample()
                    if getattr(self.cfg, 'discrete_actions', False):
                        action = (action == np.max(action)).astype(np.float32) # one-hot
                else:
                    action, agent_state = self.agent.act(dreamer_obs, # time_step.observation
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if should_train_step(self.global_step):
                    # prof.step()
                    # Sampling data
                    batch_data = next(self.replay_iter)
                    if hasattr(self.agent, ' update_wm'):
                        state, outputs, metrics = self.agent.update_wm(batch_data, self.global_step)
                        if hasattr(self.agent, "update_acting_behavior"):
                            metrics = self.agent.update_acting_behavior(state=state, outputs=outputs, metrics=metrics, data=batch_data)[1]
                        if hasattr(self.agent, "update_imag_behavior"):
                            metrics.update(self.agent.update_imag_behavior(state=state, outputs=outputs, metrics=metrics, seq_data=batch_data,)[1])
                    else:
                        outputs, metrics = self.agent.update(batch_data, self.global_step)

                if should_log_scalars(self.global_step):
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if self.global_step > 0 and should_log_visual(self.global_step):
                    if hasattr(self.agent, 'report'):
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            videos = self.agent.report(next(self.replay_iter))
                            self.logger.log_visual(videos, self.global_frame)

            # take env step
            time_step, dreamer_obs = self.train_env.step(action)
            episode_reward += time_step.reward
            data = dreamer_obs
            if time_step.last():
                if getattr(self.train_env, "accumulate", False):
                    assert not self.replay_storage._ongoing
                    # NOTE: this is ok as it comes right after adding to the repl
                    accumulated_data, accumulated_key = self.train_env.process_accumulate()
                    data[accumulated_key] = accumulated_data[-1]
                    self.replay_storage._ongoing_eps[0][accumulated_key][-len(accumulated_data[:-1]):] = accumulated_data[:-1]
            self.replay_storage.add(data, meta) 
            episode_step += 1
            self._global_step += 1

    @utils.retry
    def save_snapshot(self):
        snapshot = self.get_snapshot_dir() / f'snapshot_{self.global_frame}.pt'
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

    def load_snapshot(self, snapshot_dir):
        try:
            snapshot = snapshot_dir / 'last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = snapshot_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id':
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

@hydra.main(config_path='.', config_name='collect_data')
def main(cfg):
    from collect_data import Workspace as W
    root_dir = Path.cwd()
    cfg.workdir = str(root_dir)
    workspace = W(cfg)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(workspace.root_dir)
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()   
    workspace.train()

if __name__ == '__main__':
    main()
