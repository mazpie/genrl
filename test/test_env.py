import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import logging
LOGGER = logging.getLogger(__name__)

import pytest

import envs.main as envs
from tools.task_scores import MAX as task_max_scores

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
# @pytest.mark.filterwarnings('ignore:The distutils package is deprecated and slated for removal in Python 3.12. Use setuptools or check PEP 632 for potential alternatives')
def test_envs():
    for task_name in task_max_scores.keys():
        LOGGER.info(task_name)
        env = envs.make(task_name, 'pixels', action_repeat=2, seed=0)
        env.reset()
        env.step(env.act_space['action'].sample())

