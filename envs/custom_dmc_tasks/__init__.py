from . import cheetah
from . import walker
from . import quadruped
from . import jaco
from . import stickman
from dm_control import suite

suite._DOMAINS['stickman'] = stickman
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
    
def make_jaco(task, obs_type, seed, img_size, ):
    return jaco.make(task, obs_type, seed, img_size, )