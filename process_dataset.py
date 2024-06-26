import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import io
import os
from tqdm import tqdm

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from pathlib import Path
from collections import OrderedDict

import hydra
import numpy as np
import torch

import tools.utils as utils
from tools.replay import load_episode

torch.backends.cudnn.benchmark = True

if os.name == "nt":
    import msvcrt

    def portable_lock(fp):
        fp.seek(0)
        msvcrt.locking(fp, msvcrt.LK_LOCK, 1)

    def portable_unlock(fp):
        fp.seek(0)
        msvcrt.locking(fp, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def portable_lock(fp):
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def portable_unlock(fp):
        fcntl.flock(fp, fcntl.LOCK_UN)


class Locker:
    def __init__(self, lock_name):
        # e.g. lock_name = "./lockfile.lck"
        self.lock_name = lock_name 

    def __enter__(self,):
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        self.fd = os.open(self.lock_name, open_mode)
        portable_lock(self.fd)

    def __exit__(self, _type, value, tb):
        portable_unlock(self.fd)
        os.close(self.fd)
        try:
            os.remove(self.lock_name)
        except:
            pass

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None,):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        assert int(cfg.viclip_encode) == 1, "encoding only one (video or img)"  

        if cfg.viclip_encode:
            self.key_to_add = 'clip_video'

        self.key_to_process = getattr(cfg, 'key_to_process', 'observation')

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # create envs
        task = cfg.task
        self.task = task
        img_size = cfg.img_size
        
        import envs.main as envs
        self.train_env = envs.make(task, cfg.obs_type, cfg.action_repeat, cfg.seed, img_size=img_size,  viclip_encode=cfg.viclip_encode, device='cuda')

        self.dataset_path = Path(cfg.dataset_dir)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def process(self):
        filenames = sorted(self.dataset_path.glob('**/*.npz'))
        print(f"Found {len(filenames)} files")
        episodes_to_process = {}

        for idx, fname in tqdm(enumerate(filenames)):
            lockname = str(fname.absolute()) + ".lck"
            try:
                with Locker(lockname):
                    episode = load_episode(fname)

                    # validate before continuing
                    if type(episode[self.key_to_add]) == np.ndarray and episode[self.key_to_add].size > 1 and episode[self.key_to_add].shape[0] == episode[self.key_to_process].shape[0]:
                        continue
                    else:
                        del episode[self.key_to_add]

                    add_data = self.train_env.process_episode(episode[self.key_to_process]) # .cpu().numpy()
                    if idx == 0:
                        print(add_data.shape)
                    episode[self.key_to_add] = add_data

                    # save episode
                    with io.BytesIO() as f1:
                        np.savez_compressed(f1, **episode)
                        f1.seek(0)
                        with fname.open('wb') as f2:
                            f2.write(f1.read())
            except BlockingIOError:
                print(f"File busy: {str(fname)}")
                continue


def start_processing(cfg, savedir, workdir):
    from process_dataset import Workspace as W
    root_dir = Path.cwd()
    cfg.workdir = str(root_dir)
    workspace = W(cfg, savedir, workdir)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(workspace.root_dir)
    workspace.process()

@hydra.main(config_path='.', config_name='process_dataset')
def main(cfg):
    start_processing(cfg, None, None)

if __name__ == '__main__':
    main()