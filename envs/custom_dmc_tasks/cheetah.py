# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cheetah Domain."""

import collections
import os

from dm_control.suite import cheetah
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

_DOWN_HEIGHT = 0.15
_HIGH_HEIGHT = 1.00
_MID_HEIGHT = 0.45


# Running speed above which reward is 1.
_RUN_SPEED = 10
_SPIN_SPEED = 5

def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, 'custom_dmc_tasks', 'cheetah.xml'))
    return xml, common.ASSETS


@cheetah.SUITE.add('custom')
def flipping(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random, goal='flipping')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@cheetah.SUITE.add('custom')
def standing(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random, goal='standing')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def lying_down(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random, goal='lying_down')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_backward(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random, goal='run_backward')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def flip(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=True, flip=True, random=random, goal='flip')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def flip_backward(time_limit=_DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=True, random=random, goal='flip_backward')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""
    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""
    def __init__(self, goal=None, forward=True, flip=False, random=None):
        self._forward = 1 if forward else -1
        self._flip = flip
        self._goal = goal
        super(Cheetah, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        for _ in range(200):
            physics.step()

        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def _get_lying_down_reward(self, physics):
        torso = physics.named.data.xpos['torso', 'z']
        
        torso_down = rewards.tolerance(torso,
                                        bounds=(-float('inf'), _DOWN_HEIGHT),
                                        margin=_DOWN_HEIGHT * 1.5,)

        feet = physics.named.data.xpos['bfoot', 'z'] + physics.named.data.xpos['ffoot', 'z']

        feet_up = rewards.tolerance(feet,
                                        bounds=(_MID_HEIGHT, float('inf')),
                                        margin=_MID_HEIGHT / 2,)
        return (torso_down + feet_up) / 2

    def _get_standing_reward(self, physics):
        bfoot = physics.named.data.xpos['bfoot', 'z']
        ffoot = physics.named.data.xpos['ffoot', 'z']
        max_foot = bfoot if bfoot > ffoot else ffoot
        min_foot = bfoot if bfoot <= ffoot else ffoot
        
        low_foot_low = rewards.tolerance(min_foot,
                                        bounds=(-float('inf'), _DOWN_HEIGHT),
                                        margin=_DOWN_HEIGHT * 1.5,)

        high_foot_high = rewards.tolerance(max_foot,
                                        bounds=(_HIGH_HEIGHT, float('inf')),
                                        margin=_HIGH_HEIGHT / 2,)
        return high_foot_high * low_foot_low

    def _get_flip_reward(self, physics):
        return rewards.tolerance(self._forward * physics.angmomentum(),
                                        bounds=(_SPIN_SPEED, float('inf')),
                                        margin=_SPIN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
    
    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if self._goal in ['run', 'flip', 'run_backward', 'flip_backward']:
            if self._flip:
                return self._get_flip_reward(physics)
            else:
                reward = rewards.tolerance(self._forward * physics.speed(),
                                        bounds=(_RUN_SPEED, float('inf')),
                                        margin=_RUN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
                return reward
        elif self._goal == 'lying_down':
            return self._get_lying_down_reward(physics)
        elif self._goal == 'flipping':
            self._forward = True
            fwd_reward = self._get_flip_reward(physics)
            self._forward = False
            back_reward = self._get_flip_reward(physics)
            return max(fwd_reward, back_reward)
        elif self._goal == 'standing':
            return self._get_standing_reward(physics)
        else:
            raise NotImplementedError(self._goal)
