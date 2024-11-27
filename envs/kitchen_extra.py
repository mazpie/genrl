"""Environments using kitchen and Franka robot."""
import logging
import sys
from pathlib import Path
sys.path.append((Path(__file__).parent.parent / 'third_party' / 'relay-policy-learning' / 'adept_envs').__str__())  
import adept_envs
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
import os
import numpy as np
from dm_control.mujoco import engine

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

BONUS_THRESH = {
    "bottom burner": 0.5,
    "top burner": 0.5,
    "light switch": 0.5,
    "slide cabinet": 0.2,
    "microwave": 0.25,
    "hinge cabinet": 0.3, # default
    "kettle": 0.3, # default
}

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)
logger = logging.getLogger()

XPOS_NAMES = {
    "light switch" : "lightswitchroot",
    "slide cabinet" : "slidelink",
    "microwave" : "microdoorroot",
    "kettle" : "kettle",
}

class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = (
        True  # This allows for the tasks to be completed in arbitrary order.
    )
    GRIPPER_DISTANCE_REW = False

    def __init__(
        self, dense=True, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs
    ):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.goal_masking = True
        self.dense = dense
        self.use_grasp_rewards = False

        super(KitchenBase, self).__init__(**kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        completions = []
        dense = 0
        if self.GRIPPER_DISTANCE_REW:
            assert len(self.tasks_to_complete) == 1
            element = next(iter(self.tasks_to_complete))
            gripper_pos = (self.sim.named.data.xpos['panda0_leftfinger'] + self.sim.named.data.xpos['panda0_rightfinger']) / 2
            object_pos = self.sim.named.data.xpos[XPOS_NAMES[element]]
            gripper_obj_dist = np.linalg.norm(object_pos - gripper_pos)
            if self.dense:
                reward_dict["bonus"] = -gripper_obj_dist
                reward_dict["r_total"] = -gripper_obj_dist
                score = -gripper_obj_dist
            else:
                reward_dict["bonus"] = gripper_obj_dist < 0.15 
                reward_dict["r_total"] = gripper_obj_dist < 0.15
                score = gripper_obj_dist < 0.15               
            return reward_dict, score
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - OBS_ELEMENT_GOALS[element]
            )
            dense += -1 * distance  # reward must be negative distance for RL
            is_grasped = True
            if not self.initializing and self.use_grasp_rewards:
                if element == "slide cabinet":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("schandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.07
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.07
                        right = right_pad[0] < obj_pos[0]
                        left = obj_pos[0] < left_pad[0]
                        if (
                            right
                            and left
                            and within_sphere_right
                            and within_sphere_left
                        ):
                            is_grasped = True
                if element == "top left burner":
                    is_grasped = False
                    obj_pos = self.get_site_xpos("tlbhandle")
                    left_pad = self.get_site_xpos("leftpad")
                    right_pad = self.get_site_xpos("rightpad")
                    within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.035
                    within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.04
                    right = right_pad[0] < obj_pos[0]
                    left = obj_pos[0] < left_pad[0]
                    if within_sphere_right and within_sphere_left and right and left:
                        is_grasped = True
                if element == "microwave":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("mchandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
                        if (
                            right_pad[0] < obj_pos[0]
                            and obj_pos[0] < left_pad[0]
                            and within_sphere_right
                            and within_sphere_left
                        ):
                            is_grasped = True
                if element == "hinge cabinet":
                    is_grasped = False
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("hchandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
                        if (
                            right_pad[0] < obj_pos[0]
                            and obj_pos[0] < left_pad[0]
                            and within_sphere_right
                        ):
                            is_grasped = True
                if element == "light switch":
                    is_grasped = False
                    for i in range(1, 4):
                        obj_pos = self.get_site_xpos("lshandle{}".format(i))
                        left_pad = self.get_site_xpos("leftpad")
                        right_pad = self.get_site_xpos("rightpad")
                        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.045
                        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.03
                        if within_sphere_right and within_sphere_left:
                            is_grasped = True
            complete = distance < BONUS_THRESH[element] #  and is_grasped
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        if self.dense:
            reward_dict["r_total"] = dense
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH[wrong_task]
                if complete:
                    done = True
                    break
        env_info["completed_tasks"] = set(self.TASK_ELEMENTS) - set(
            self.tasks_to_complete
        )
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                dict(
                    states=data["observations"][start : end_idx + 1],
                    actions=data["actions"][start : end_idx + 1],
                )
            )
            start = end_idx + 1
        return seqs
    
    def render(self, mode='rgb_array', resolution=(64,64)):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, *resolution)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()


class KitchenSlideV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenHingeV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenLightV0(KitchenBase):
    TASK_ELEMENTS = ["light switch",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom burner",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenTopBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top burner",]
    COMPLETE_IN_ANY_ORDER = False

class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenKettleMicrowaveLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "microwave", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenAllV0(KitchenBase):
    TASK_ELEMENTS = KitchenBase.ALL_TASKS