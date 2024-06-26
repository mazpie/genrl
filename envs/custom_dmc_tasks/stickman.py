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
"""Stickman Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import types

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control import suite

class StickmanYogaPoses:
    lie_back   = [ -1.2 ,  0. ,  -1.57,  0, 0. , 0.0, 0, -0.,  0.0]
    lie_front  = [-1.2, -0, 1.57, 0, 0, 0, 0, 0., 0.]
    legs_up    = [ -1.24 ,  0. ,  -1.57,  1.57, 0. , 0.0,  1.57, -0.,  0.0]

    kneel      = [ -0.5 ,  0. ,  0,  0, -1.57, -0.8,  1.57, -1.57,  0.0]
    side_angle = [ -0.3 ,  0. ,  0.9,  0, 0, -0.7,  1.87, -1.07,  0.0]
    stand_up   = [-0.15, 0., 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1]

    lean_back  = [-0.27, 0., -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]
    boat       = [ -1.04 ,  0. ,  -0.8,  1.6, 0. , 0.0, 1.6, -0.,  0.0]
    bridge     = [-1.1, 0., -2.2, -0.3, -1.5, 0., -0.3, -0.8, -0.4]

    head_stand = [-1, 0., -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]
    one_feet   = [-0.2, 0., 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]
    arabesque  = [-0.34, 0., 1.57, 1.57, 0, 0., 0, -0., 0.]

    # new
    high_kick = [-0.165, 3.3  , 5.55 , 1.35 ,-0, +0.5 , -0.7, 0. , 0.2,]
    splits    = [-0.7, 0., 0.5, -0.7, -1. , 0, 1.75, 0., -0.45 ]
    sit_knees = [-0.6, -0.2,  0.2, 0.95, -2.5, 0 , 0.95, -2.5, 0 ]


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.15

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8

# Copied from walker:
_YOGA_HANDS_UP_HEIGHT = 1.75
_YOGA_STAND_HEIGHT = 1.0 # lower than stan height = 1.2
_YOGA_LIE_DOWN_HEIGHT = 0.1
_YOGA_LEGS_UP_HEIGHT = 1.1

_YOGA_FEET_UP_HEIGHT = 0.5
_YOGA_FEET_UP_LIE_DOWN_HEIGHT = 0.35

_YOGA_KNEE_HEIGHT = 0.25
_YOGA_KNEESTAND_HEIGHT = 0.75

_YOGA_SITTING_HEIGHT = 0.55
_YOGA_SITTING_LEGS_HEIGHT = 0.15

# speed from: https://github.com/rll-research/url_benchmark/blob/710c3eb/custom_dmc_tasks/py
_SPIN_SPEED = 5.0
#
_PUNCH_SPEED = 5.0
_PUNCH_DIST = 0.29


SUITE = containers.TaggedTasks()

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
    xml = resources.GetResource(os.path.join(root_dir, 'custom_dmc_tasks', 'stickman.xml'))
    return xml, common.ASSETS

@SUITE.add('custom')
def hands_up(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the hands_up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='hands_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def boxing(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the boxing task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='boxing', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def arabesque(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Arabesque task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='arabesque', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def lying_down(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Lie Down task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='lying_down', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def legs_up(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Legs Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='legs_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def high_kick(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the High Kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='high_kick', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def one_foot(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the High Kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='one_foot', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def lunge_pose(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the High Kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='lunge_pose', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def sit_knees(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the High Kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='sit_knees', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def headstand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Headstand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='flip', move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def urlb_flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='urlb_flip', move_speed=_SPIN_SPEED, random=random) 
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def flipping(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flipping task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='flipping', move_speed=2 * _RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='flip', move_speed=2 * _RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def backflip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Backflip task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(goal='flip', move_speed=-2 * _RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('custom')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(move_speed=0, goal='stand', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(move_speed=_WALK_SPEED, goal='walk', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('custom')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Stickman(move_speed=_RUN_SPEED, goal='run', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the stickman domain."""
    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

    def angmomentum(self):
        """Returns the angular momentum of torso of the stickman about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Stickman(base.Task):
    """A planar stickman task."""
    def __init__(self, move_speed=0., goal='walk', forward=True, random=None):
        """Initializes an instance of `Stickman`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._goal = goal
        super().__init__(random=random)

    def _hands_up_reward(self, physics):
        standing = self._stand_reward(physics)
        left_hand_height = physics.named.data.xpos['left_hand', 'z']
        right_hand_height = physics.named.data.xpos['right_hand', 'z']
        
        hand_height = (left_hand_height + right_hand_height) / 2

        hands_up = rewards.tolerance(hand_height,
                                bounds=(_YOGA_HANDS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_HANDS_UP_HEIGHT/2)
        return standing * hands_up

    def _boxing_reward(self, physics):
        # torso up, but lower than standing
        # foot up, higher than torso
        # foot down
        standing = self._stand_reward(physics)

        left_hand_velocity = abs(physics.named.data.subtree_linvel['left_hand'][0])
        right_hand_velocity = abs(physics.named.data.subtree_linvel['right_hand'][0])
        punch_reward = rewards.tolerance(
                max(left_hand_velocity, right_hand_velocity),
                bounds=(_PUNCH_SPEED, float('inf')),
                margin=_PUNCH_SPEED / 2,
                value_at_margin=0.5,
                sigmoid='linear')

        # left_hand_dist = physics.named.data.xpos['left_hand', 'x'] - physics.named.data.xpos['torso', 'x']
        # right_hand_dist = physics.named.data.xpos['right_hand', 'x'] - physics.named.data.xpos['torso', 'x']
        # punch_reward = rewards.tolerance(
        #         max(left_hand_dist, right_hand_dist),
        #         bounds=(_PUNCH_DIST, float('inf')),
        #         margin=_PUNCH_DIST / 2,)

        return standing * punch_reward

    def _arabesque_reward(self, physics):
        # standing horizontal
        # one foot up, same height as torso
        # one foot down
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        
        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        
        max_foot = 'right_foot' if right_foot_height > left_foot_height else 'left_foot'
        min_foot = 'right_foot' if right_foot_height <= left_foot_height else 'left_foot'

        min_foot_height = physics.named.data.xpos[min_foot, 'z']
        max_foot_height = physics.named.data.xpos[max_foot, 'z']

        min_foot_down = rewards.tolerance(min_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        max_foot_up = rewards.tolerance(max_foot_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        
        min_foot_x = physics.named.data.xpos[min_foot, 'x']
        max_foot_x = physics.named.data.xpos[max_foot, 'x']
        
        correct_foot_pose = 0.1 if max_foot_x > min_foot_x else 1.0
 
        feet_pose = (min_foot_down + max_foot_up * 2) / 3
        return standing * feet_pose * correct_foot_pose
    
    def _lying_down_reward(self, physics):
        # torso down and horizontal
        # thigh and feet down
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        horizontal = 1 - abs(physics.torso_upright())
        
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        leg_height = (physics.named.data.xpos['left_leg', 'z'] + physics.named.data.xpos['right_leg', 'z']) / 2
        leg_down = rewards.tolerance(leg_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        return (3*torso_down + horizontal + thigh_down + feet_down + leg_down) / 7
    
    def _legs_up_reward(self, physics):
        # torso down and horizontal
        # legs up with thigh down
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        horizontal = 1 - abs(physics.torso_upright())
        torso_down = (3*torso_down +horizontal) / 4
        
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_FEET_UP_LIE_DOWN_HEIGHT, float('inf')),
                                margin=_YOGA_FEET_UP_LIE_DOWN_HEIGHT/2)

        return torso_down * feet_up
    
    def _high_kick_reward(self, physics):
        # torso up, but lower than standing
        # foot up, higher than torso
        # foot down
        standing = rewards.tolerance(physics.torso_height(),
                                    bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                    margin=_YOGA_STAND_HEIGHT/2)

        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        
        min_foot_height = min(left_foot_height, right_foot_height)
        max_foot_height = max(left_foot_height, right_foot_height)

        min_foot_down = rewards.tolerance(min_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        max_foot_up = rewards.tolerance(max_foot_height,
                                bounds=(_STAND_HEIGHT, float('inf')),
                                margin=_STAND_HEIGHT/2)
        
        feet_pose = (3 * max_foot_up + min_foot_down) / 4

        return standing * feet_pose
    
    def _one_foot_reward(self, physics):
        # torso up, standing
        # foot up higher than foot down
        standing = rewards.tolerance(physics.torso_height(),
                                    bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                    margin=_YOGA_STAND_HEIGHT/2)

        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        
        min_foot_height = min(left_foot_height, right_foot_height)
        max_foot_height = max(left_foot_height, right_foot_height)

        min_foot_down = rewards.tolerance(min_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        max_foot_up = rewards.tolerance(max_foot_height,
                                bounds=(_YOGA_FEET_UP_HEIGHT, float('inf')),
                                margin=_YOGA_FEET_UP_HEIGHT/2)
        
        return standing * max_foot_up * min_foot_down

    def _lunge_pose_reward(self, physics):
        # torso up, standing, but lower
        # leg up higher than leg down
        # horiontal thigh and leg
        standing = rewards.tolerance(physics.torso_height(),
                                    bounds=(_YOGA_KNEESTAND_HEIGHT, float('inf')),
                                    margin=_YOGA_KNEESTAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        torso = (3*standing + upright) / 4

        left_leg_height = physics.named.data.xpos['left_leg', 'z']
        right_leg_height = physics.named.data.xpos['right_leg', 'z']
        
        min_leg_height = min(left_leg_height, right_leg_height)
        max_leg_height = max(left_leg_height, right_leg_height)

        min_leg_down = rewards.tolerance(min_leg_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        max_leg_up = rewards.tolerance(max_leg_height,
                                bounds=(_YOGA_KNEE_HEIGHT, float('inf')),
                                margin=_YOGA_KNEE_HEIGHT / 2)
        
        max_thigh = 'left_thigh' if max_leg_height == left_leg_height else 'right_thigh'
        min_leg = 'left_leg' if min_leg_height == left_leg_height else 'right_leg'

        max_thigh_horiz = 1 - abs(physics.named.data.xmat[max_thigh, 'zz'])
        min_leg_horiz = 1 - abs(physics.named.data.xmat[min_leg, 'zz'])
        
        legs = (min_leg_down + max_leg_up + max_thigh_horiz + min_leg_horiz) / 4

        return torso * legs

    def _sit_knees_reward(self, physics):
        # torso up, standing, but lower
        # foot up higher than foot down
        standing = rewards.tolerance(physics.torso_height(),
                                    bounds=(_YOGA_SITTING_HEIGHT, float('inf')),
                                    margin=_YOGA_SITTING_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        torso_up = (3*standing + upright) / 4

        legs_height = (physics.named.data.xpos['left_leg', 'z'] + physics.named.data.xpos['right_leg', 'z']) / 2
        legs_down = rewards.tolerance(legs_height,
                                bounds=(-float('inf'), _YOGA_SITTING_LEGS_HEIGHT),
                                margin=_YOGA_SITTING_LEGS_HEIGHT*1.5)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT*1.5)
        
        l_thigh_foot_distance = max(0.1, abs(physics.named.data.xpos['left_foot', 'x'] - physics.named.data.xpos['left_thigh', 'x'])) - 0.1
        r_thigh_foot_distance = max(0.1, abs(physics.named.data.xpos['right_foot', 'x'] - physics.named.data.xpos['right_thigh', 'x'])) - 0.1
        close = np.exp(-(l_thigh_foot_distance + r_thigh_foot_distance)/2)
        
        legs = (3 * legs_down + feet_down) / 4
        return torso_up * legs * close

    def _urlb_flip_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        move_reward = rewards.tolerance(self._forward *
                                        physics.named.data.subtree_angmom['torso'][1], # physics.angmomentum(),
                                        bounds=(_SPIN_SPEED, float('inf')),
                                        margin=_SPIN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
        return stand_reward * (5 * move_reward + 1) / 6

    def _flip_reward(self, physics):
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_up = rewards.tolerance(thigh_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upside_down_reward = (3*legs_up + 2*thigh_up) / 5
        if self._move_speed == 0:
            return upside_down_reward
        move_reward = rewards.tolerance(physics.named.data.subtree_angmom['torso'][1], # physics.angmomentum(),
                                    bounds=(self._move_speed, float('inf')) if self._move_speed > 0 else (-float('inf'), self._move_speed),
                                    margin=abs(self._move_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return upside_down_reward * (5*move_reward + 1) / 6

    
    def _stand_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        return (3 * standing + upright) / 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.

    Args:
      physics: An instance of `Physics`.

    """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if self._goal in ['stand', 'walk', 'run']:
            stand_reward = self._stand_reward(physics)
            move_reward = rewards.tolerance(
                self._forward * physics.horizontal_velocity(),
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6
        if self._goal == 'flipping':
            self._move_speed = abs(self._move_speed)
            pos_rew = self._flip_reward(physics)
            self._move_speed = -abs(self._move_speed)
            neg_rew = self._flip_reward(physics)
            return max(pos_rew, neg_rew)
        try:
            reward_fn = getattr(self, f'_{self._goal}_reward')
            return reward_fn(physics)
        except Exception as e:
            print(e)
            raise NotImplementedError(f'Goal {self._goal} or function "_{self._goal}_reward" not implemented.')

if __name__ == '__main__':
    from dm_control import viewer
    import numpy as np
    
    env = boxing()
    env.task.visualize_reward = True

    action_spec = env.action_spec()

    def zero_policy(time_step):
        print(time_step.reward)
        return np.zeros(action_spec.shape)
    
    ts = env.reset()
    while True:
       ts = env.step(zero_policy(ts))

    viewer.launch(env, policy=zero_policy)
    
    # obs = env.reset()
    # next_obs, reward, done, info = env.step(np.zeros(6))