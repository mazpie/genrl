import collections
import datetime
import io
import pathlib
import uuid
import os

import numpy as np
from gym.spaces import Dict
import random
from torch.utils.data import IterableDataset, DataLoader
import torch
import tools.utils as utils
import traceback
from pathlib import Path
from tqdm import tqdm

SIG_FAILURE = -1

def get_length(filename):
  if "-" in str(filename):
    length = int(str(filename).split('-')[-1])
  else:
    length = int(str(filename).split('_')[-1])
  return length

def get_idx(filename):
  if "-" in str(filename):
    length = int(str(filename).split('-')[0])
  else:
    length = int(str(filename).split('_')[0])
  return length

def on_fn(): return collections.defaultdict(list) # this function is to avoid lambdas

class ReplayBuffer(IterableDataset):

  def __init__(
      self, data_specs, meta_specs, directory, length=20, capacity=0, ongoing=False, minlen=1, maxlen=0,
      prioritize_ends=False, device='cuda', load_first=False, save_episodes=True, ignore_extra_keys=False, load_recursive=False, min_t_sampling=0, **kwargs):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._capacity = capacity
    self._ongoing = ongoing
    self._minlen = minlen
    self._maxlen = maxlen
    self._prioritize_ends = prioritize_ends
    self._ignore_extra_keys = ignore_extra_keys
    self._min_t_sampling = min_t_sampling
    # self._random = np.random.RandomState()
    # filename -> key -> value_sequence
    
    self._save_episodes = save_episodes
    self._last_added_idx = 0

    self._episode_lens = np.array([])
    self._complete_eps = {} 
    self._data_specs = data_specs
    self._meta_specs = meta_specs    
    for spec_group in [data_specs, meta_specs]: 
      for spec in spec_group:
        if type(spec) in [dict, Dict]:
          for k,v in spec.items():
            self._complete_eps[k] = [] 
        else:
            self._complete_eps[spec.name] = [] 

    # load episodes
    if type(directory) == str:
      directory = Path(directory)
    self._loaded_episodes = 0 
    self._loaded_steps = 0 
    for f in tqdm(load_filenames(self._directory, capacity, minlen, load_first=load_first, load_recursive=load_recursive)):
      self.store_episode(filename=f)
    try:
      self._total_episodes, self._total_steps = count_episodes(directory)
    except:
      print("Couldn't count episodes")
      print("Loaded episodes: ", self._loaded_episodes)
      print("Loaded steps: ", self._loaded_steps)
      self._total_episodes, self._total_steps = self._loaded_episodes, self._loaded_steps
    
    # worker -> key -> value_sequence
    self._length = length
    self._ongoing_eps = collections.defaultdict(on_fn)
    self.device = device
    try:
      assert self._minlen <= self._length <= self._maxlen
    except:
      print("Sampling sequences with fixed length ", length)
      self._minlen = self._maxlen = self._length = length

  def __len__(self):
    return self._total_steps

  def preallocate_memory(self, max_size):
      self._preallocated_mem = collections.defaultdict(list)
      for spec in self._data_specs:
        if type(spec) in [dict, Dict]:
          for k,v in spec.items():
            for _ in range(max_size):
              self._preallocated_mem[k].append(np.empty(list(v.shape), v.dtype))
              self._preallocated_mem[k][-1].fill(0.)
        else:
          for _ in range(max_size):
            self._preallocated_mem[spec.name].append(np.empty(list(v.shape), v.dtype))
            self._preallocated_mem[spec.name][-1].fill(0.)

  @property
  def stats(self):
    return {
        'total_steps': self._total_steps,
        'total_episodes': self._total_episodes,
        'loaded_steps': self._loaded_steps,
        'loaded_episodes': self._loaded_episodes,
    }

  def add(self, time_step, meta, idx=0):
    ### Useful if there was any failure in the environment
    if time_step == SIG_FAILURE:
      episode = self._ongoing_eps[idx]
      episode.clear()
      print("Discarding episode from process", idx)
      return
    #### 
      
    episode = self._ongoing_eps[idx]
    
    def add_to_episode(name, data, spec):
        value = data[name]
        if np.isscalar(value):
            value = np.full(spec.shape, value, spec.dtype)
        assert spec.shape == value.shape and spec.dtype == value.dtype, f"for ({name}) expected {spec.dtype, spec.shape, }), received ({value.dtype, value.shape, })"
        ### Deallocate preallocated memory
        if getattr(self, '_preallocated_mem', False):
          if len(self._preallocated_mem[name]) > 0:
            tmp = self._preallocated_mem[name].pop() 
            del tmp
          else:
            # Out of pre-allocated memory
            del self._preallocated_mem
        ###
        episode[name].append(value)
    
    for spec in self._data_specs:
        if type(spec) in [dict, Dict]:
          for k,v in spec.items():
            add_to_episode(k, time_step, v)
        else:
          add_to_episode(spec.name, time_step, spec)
    for spec in self._meta_specs:
        if type(spec) in [dict, Dict]:
          for k,v in spec.items():
            add_to_episode(k, meta, v)
        else:
          add_to_episode(spec.name, meta, spec)
    if type(time_step) in [dict, Dict]:
      if time_step['is_last']: 
        self.add_episode(episode)
        episode.clear()
    else:
      if time_step.last(): 
        self.add_episode(episode)
        episode.clear()

  def add_episode(self, episode):
    length = eplen(episode)
    if length < self._minlen:
      print(f'Skipping short episode of length {length}.')
      return
    self._total_steps += length
    self._total_episodes += 1
    episode = {key: convert(value) for key, value in episode.items()}
    if self._save_episodes:
      filename = self.save_episode(self._directory, episode)
    self.store_episode(episode=episode)

  def store_episode(self, filename=None, episode=None, run_checks=True):
    if filename is not None:
      episode = load_episode(filename)
      if len(episode['reward'].shape) == 1:
        episode['reward'] = episode['reward'].reshape(-1, 1)
      if 'discount' not in episode:
        episode['discount'] = (1 - episode['is_terminal']).reshape(-1, 1).astype(np.float32)
      #
      if run_checks:
        for spec_set in [self._data_specs, self._meta_specs]: 
          for spec in spec_set:
            if type(spec) in [dict, Dict]:
              for k,v in spec.items():
                  value = episode[k][0]
                  assert v.shape == value.shape and v.dtype == value.dtype, f"for ({k}) expected {v.dtype, v.shape, }), received ({value.dtype, value.shape, })"
            else:
              value = episode[spec.name][0]
              assert spec.shape == value.shape and spec.dtype == value.dtype, f"for ({spec.name}) expected {spec.dtype, spec.shape, }), received ({value.dtype, value.shape, })"
    if not episode:
      return False
    length = eplen(episode)
    if run_checks:
      for k in episode:
        assert len(episode[k]) == length, f'Found {episode[k].shape} VS eplen: {length}'

    # Enforce limit
    while self._loaded_steps + length > self._capacity:
      for k in self._complete_eps:
        self._complete_eps[k].pop(0)
      removed_len, self._episode_lens = self._episode_lens[0], self._episode_lens[1:]
      self._loaded_steps -= removed_len
      self._loaded_episodes -= 1

    # add episode
    for k,v in episode.items():
      if k not in self._complete_eps:
        if self._ignore_extra_keys: continue
        else: raise KeyError("Extra key ", k)
      self._complete_eps[k].append(v)
    self._episode_lens = np.append(self._episode_lens, length)
    self._loaded_steps += length
    self._loaded_episodes += 1

    return True

  def __iter__(self):
    while True:
      sequences, batch_size, batch_length = self._loaded_episodes, self.batch_size, self._length

      b_indices = np.random.randint(0, sequences, size=batch_size)
      t_indices = np.random.randint(np.zeros(batch_size) + self._min_t_sampling, self._episode_lens[b_indices]-batch_length+1, size=batch_size)
      t_ranges = np.repeat( np.expand_dims(np.arange(0, batch_length,), 0), batch_size, axis=0) + np.expand_dims(t_indices, 1)

      chunk = {}
      for k in self._complete_eps:
        chunk[k] = np.stack([self._complete_eps[k][b][t] for b,t in zip(b_indices, t_ranges)])
      for k in chunk: 
        chunk[k] = torch.as_tensor(chunk[k], device=self.device)
      yield chunk

  @utils.retry
  def save_episode(self, directory, episode):
    idx = self._total_episodes
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{idx}-{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    return filename

def load_episode(filename):
  try:
    with filename.open('rb') as f:
      episode = np.load(f, allow_pickle=True)
      episode = {k: episode[k] for k in episode.keys()}
  except Exception as e:
    print(f'Could not load episode {str(filename)}: {e}')
    return False
  return episode

def count_episodes(directory):
  filenames = list(directory.glob('*.npz'))
  num_episodes = len(filenames)
  if num_episodes == 0 : return 0, 0
  if len(filenames) > 0 and "-" in str(filenames[0]):
    num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
    last_episode = sorted(list(int(n.stem.split('-')[0]) for n in filenames))[-1]
  else:
    num_steps = sum(int(str(n).split('_')[-1][:-4]) - 1 for n in filenames)
    last_episode = sorted(list(int(n.stem.split('_')[0]) for n in filenames))[-1]
  return last_episode, num_steps

def load_filenames(directory, capacity=None, minlen=1, load_first=False, load_recursive=False):
  # The returned directory from filenames to episodes is guaranteed to be in
  # temporally sorted order.
  if load_recursive:
    filenames = sorted(directory.glob('**/*.npz'))
  else:
    filenames = sorted(directory.glob('*.npz'))
  if capacity:
    num_steps = 0
    num_episodes = 0
    ordered_filenames = filenames if load_first else reversed(filenames)
    for filename in ordered_filenames:
      if "-" in str(filename):
        length = int(str(filename).split('-')[-1][:-4])
      else:
        length = int(str(filename).split('_')[-1][:-4])
      num_steps += length
      num_episodes += 1
      if num_steps >= capacity:
        break
    if load_first:
      filenames = filenames[:num_episodes]
    else:
      filenames = filenames[-num_episodes:]
  return filenames

def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value

def eplen(episode):
  return len(episode['action'])

def make_replay_loader(buffer, batch_size,):
    buffer.batch_size = batch_size
    return DataLoader(buffer,
                      batch_size=None,
                      # NOTE: do not use any workers, 
                      # as they don't get copies of the replay buffer (requires different implementation)
          )