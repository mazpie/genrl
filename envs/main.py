from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os

import dm_env
import numpy as np
from dm_env import StepType, specs

import gym
import torch

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FramesWrapper(dm_env.Environment):
    def __init__(self, env, num_frames=1, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class DMC:
  def __init__(self, env):
    self._env = env 
    self._ignored_keys = []

  def step(self, action):
    time_step = self._env.step(action)
    assert time_step.discount in (0, 1)
    obs = {
        'reward': time_step.reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'observation': time_step.observation,
        'action' : action,
        'discount': time_step.discount
    }
    return time_step, obs 

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'observation': time_step.observation,
        'action' : np.zeros_like(self.act_space['action'].sample()),
        'discount': time_step.discount
    }
    return time_step, obs

  def __getattr__(self, name):
    if name == 'obs_space':
        obs_spaces = {
            'observation': self._env.observation_spec(), 
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return obs_spaces
    if name == 'act_space':
        spec = self._env.action_spec()
        action = gym.spaces.Box((spec.minimum)*spec.shape[0], (spec.maximum)*spec.shape[0], shape=spec.shape, dtype=np.float32)
        act_space = {'action': action}
        return act_space
    return getattr(self._env, name)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

class KitchenWrapper:
    def __init__(
        self,
        name,
        seed=0,
        action_repeat=1,
        size=(64, 64),
    ):
        import envs.kitchen_extra as kitchen_extra
        self._env  = {
            'microwave' : kitchen_extra.KitchenMicrowaveV0,
            'kettle' : kitchen_extra.KitchenKettleV0,
            'burner' : kitchen_extra.KitchenBurnerV0,
            'light'  : kitchen_extra.KitchenLightV0,
            'hinge'  : kitchen_extra.KitchenHingeV0,
            'slide'  : kitchen_extra.KitchenSlideV0,
            'top_burner' : kitchen_extra.KitchenTopBurnerV0,
        }[name]()
            
        self._size = size
        self._action_repeat = action_repeat
        self._seed = seed
        self._eval = False

    def eval_mode(self,):
        self._env.dense = False
        self._eval = True

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        # assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action.copy())
            reward += rew 
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "observation": info['images'].transpose(2, 0, 1).copy(),
            "state": state.astype(np.float32),
            'action' : action,
            'discount' : 1
        }
        if self._eval:
            obs['reward'] = min(obs['reward'], 1)
            if obs['reward'] > 0:
                obs['is_last'] = True
        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID if not obs['is_last'] else dm_env.StepType.LAST, 
                reward=obs['reward'],
                discount=1,
                observation=obs['observation']), obs

    def reset(self,):
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": self.get_visual_obs(self._size),
            "state": state.astype(np.float32),
            'action' : np.zeros_like(self.act_space['action'].sample()),
            'discount' : 1
        }
        return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=obs['observation']), obs

    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)
    
    def get_visual_obs(self, resolution):
        img = self._env.render(resolution=resolution,).transpose(2, 0, 1).copy()
        return img

class ViClipWrapper:
    def __init__(self, env, hd_rendering=False, device='cuda'):
        self._env = env
        try:
            from tools.genrl_utils import viclip_global_instance
        except:
            from tools.genrl_utils import ViCLIPGlobalInstance
            viclip_global_instance = ViCLIPGlobalInstance()

        if not viclip_global_instance._instantiated:
            viclip_global_instance.instantiate(device)
        self.viclip_model = viclip_global_instance.viclip
        self.n_frames = self.viclip_model.n_frames
        self.viclip_emb_dim = viclip_global_instance.viclip_emb_dim
        self.n_frames = self.viclip_model.n_frames
        self.buffer = deque(maxlen=self.n_frames)
        # NOTE: these are hardcoded for now, as they are the best settings
        self.accumulate = True
        self.accumulate_buffer = []
        self.anticipate_conv1 = False
        self.hd_rendering = hd_rendering

    def hd_render(self, obs):
        if not self.hd_rendering:
            return obs['observation']
        if self._env._domain_name in ['mw', 'kitchen', 'mujoco']:
            return self.get_visual_obs((224,224,))
        else:
            render_kwargs = {**getattr(self, '_render_kwargs', {})}
            render_kwargs.update({'width' : 224, 'height' : 224})
            return self._env.physics.render(**render_kwargs).transpose(2,0,1)

    def preprocess(self, x):
        return x

    def process_accumulate(self, process_at_once=4): # NOTE: this could be varied for increasing FPS, depending on the size of the GPU
        self.accumulate = False
        x = np.stack(self.accumulate_buffer, axis=0)
        # Splitting in chunks
        chunks = []
        chunk_idxs = list(range(0, x.shape[0] + 1, process_at_once))
        if chunk_idxs[-1] != x.shape[0]:
            chunk_idxs.append(x.shape[0])
        start = 0
        for end in chunk_idxs[1:]:
            embeds = self.clip_process(x[start:end], bypass=True)
            chunks.append(embeds.cpu())
            start = end
        embeds = torch.cat(chunks, dim=0)
        assert embeds.shape[0] == len(self.accumulate_buffer)
        self.accumulate = True
        self.accumulate_buffer = []
        return [*embeds.cpu().numpy()], 'clip_video'
    
    def process_episode(self, obs, process_at_once=8):
        self.accumulate = False
        sequences = []
        for j in range(obs.shape[0] - self.n_frames + 1):
            sequences.append(obs[j:j+self.n_frames].copy())
        sequences = np.stack(sequences, axis=0)

        idx_start = 0
        clip_vid = []
        for idx_end in range(process_at_once, sequences.shape[0] + process_at_once, process_at_once):
            x = sequences[idx_start:idx_end]
            with torch.no_grad(): # , torch.cuda.amp.autocast():
                x = self.clip_process(x, bypass=True) 
            clip_vid.append(x)
            idx_start = idx_end
        if len(clip_vid) == 1: # process all at once
            embeds = clip_vid[0]
        else:
            embeds = torch.cat(clip_vid, dim=0)
        pad = torch.zeros( (self.n_frames - 1, *embeds.shape[1:]), device=embeds.device, dtype=embeds.dtype)
        embeds = torch.cat([pad, embeds], dim=0)
        assert embeds.shape[0] == obs.shape[0], f"Shapes are different {embeds.shape[0]} {obs.shape[0]}"
        return embeds.cpu().numpy()

    def get_sequence(self,):
        return np.expand_dims(np.stack(self.buffer, axis=0), axis=0)
    
    def clip_process(self, x, bypass=False):
        if len(self.buffer) == self.n_frames or bypass:
            if self.accumulate:
                self.accumulate_buffer.append(self.preprocess(x)[0])
                return torch.zeros(self.viclip_emb_dim)
            with torch.no_grad():
                B, n_frames, C, H, W = x.shape
                obs = torch.from_numpy(x.copy().reshape(B * n_frames, C, H, W)).to(self.viclip_model.device)
                processed_obs = self.viclip_model.preprocess_transf(obs / 255)
                reshaped_obs = processed_obs.reshape(B, n_frames, 3,processed_obs.shape[-2],processed_obs.shape[-1])
                video_embed = self.viclip_model.get_vid_features(reshaped_obs)
            return video_embed.detach()
        else:
            return torch.zeros(self.viclip_emb_dim)

    def step(self, action):
        ts, obs = self._env.step(action)
        self.buffer.append(self.hd_render(obs))
        obs['clip_video'] = self.clip_process(self.get_sequence()).cpu().numpy()
        return ts, obs

    def reset(self,):
        # Important to reset the buffer        
        self.buffer = deque(maxlen=self.n_frames)

        ts, obs = self._env.reset()
        self.buffer.append(self.hd_render(obs))
        obs['clip_video'] = self.clip_process(self.get_sequence()).cpu().numpy()
        return ts, obs

    def __getattr__(self, name):
        if name == 'obs_space':
            space = self._env.obs_space
            space['clip_video'] = gym.spaces.Box(-np.inf, np.inf, (self.viclip_emb_dim,), dtype=np.float32)  
            return space
        return getattr(self._env, name)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    ts, obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      ts = dm_env.TimeStep(dm_env.StepType.LAST, ts.reward, ts.discount, ts.observation)
      obs['is_last'] = True
      self._step = None
    return ts, obs

  def reset(self):
    self._step = 0
    return self._env.reset()

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)
  
class ClipActionWrapper:

  def __init__(self, env, low=-1.0, high=1.0):
    self._env = env
    self._low = low
    self._high = high

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def step(self, action):
    clipped_action = np.clip(action, self._low, self._high)
    return self._env.step(clipped_action)

  def reset(self):
    self._step = 0
    return self._env.reset()

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)

class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})

def _make_jaco(obs_type, domain, task, action_repeat, seed, img_size,):
    import envs.custom_dmc_tasks as cdmc
    env = cdmc.make_jaco(task, obs_type, seed, img_size,)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(obs_type, domain, task, action_repeat, seed, img_size,):
    visualize_reward = False
    from dm_control import manipulation, suite
    import envs.custom_dmc_tasks as cdmc

    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        from dm_control.suite.wrappers import pixels
        # zoom in camera for quadruped
        camera_id = dict(locom_rodent=1,quadruped=2).get(domain, 0)
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(name, obs_type, action_repeat, seed, img_size=64, viclip_encode=False, clip_hd_rendering=False, device='cuda'):
    assert obs_type in ['states', 'pixels']
    domain, task = name.split('_', 1)
    if domain == 'kitchen':
        env = TimeLimit(KitchenWrapper(task, seed=seed, action_repeat=action_repeat, size=(img_size,img_size)), 280 // action_repeat)
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl' 
        os.environ['MUJOCO_GL'] = 'egl'

        domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)

        make_fn = _make_jaco if domain == 'jaco' else _make_dmc
        env = make_fn(obs_type, domain, task, action_repeat, seed, img_size,)

        if obs_type == 'pixels':
            env = FramesWrapper(env,)
        else:
            env = ObservationDTypeWrapper(env, np.float32)

        from dm_control.suite.wrappers import action_scale
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        env = ExtendedTimeStepWrapper(env)

        env =  DMC(env)
    env._domain_name = domain
    
    if isinstance(env.act_space['action'], gym.spaces.Box):
        env = ClipActionWrapper(env,)

    if viclip_encode:
        env = ViClipWrapper(env, hd_rendering=clip_hd_rendering, device=device)
    return env
