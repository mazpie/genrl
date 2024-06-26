from pathlib import Path 
import os
import sys
sys.path.append(str(Path(os.path.abspath(''))))

import torch
import numpy as np
from tools.genrl_utils import ViCLIPGlobalInstance

import time
import torchvision
from huggingface_hub import hf_hub_download

def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), 0., 1.)
        video = video.permute(1, 0, 2, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

class Text2Video():
    def __init__(self,result_dir='./tmp/',gpu_num=1) -> None:
        model_folder = str(Path(os.path.abspath('')) / 'models')
        model_filename = 'genrl_stickman_500k_2.pt'
        
        if not os.path.isfile(os.path.join(model_folder, model_filename)):
            self.download_model(model_folder, model_filename)
        if not os.path.isfile(os.path.join(model_folder, 'InternVideo2-stage2_1b-224p-f4.pt')):
            self.download_internvideo2(model_folder)
        self.agent = torch.load(os.path.join(model_folder, model_filename))
        model_name = 'internvideo2'

        # Get ViCLIP
        viclip_global_instance = ViCLIPGlobalInstance(model_name)
        if not viclip_global_instance._instantiated:
            print("Instantiating InternVideo2")
            viclip_global_instance.instantiate()
        self.clip = viclip_global_instance.viclip
        self.tokenizer = viclip_global_instance.viclip_tokenizer

        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def get_prompt(self, prompt, duration):
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()

        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str

        labels_list = [prompt_str]
        with torch.no_grad():
            wm = world_model = self.agent.wm
            connector = self.agent.wm.connector
            decoder = world_model.heads['decoder']
            n_frames = connector.n_frames
            
            # Get text(video) embed
            text_feat = []
            for text in labels_list:
                with torch.no_grad():
                    text_feat.append(self.clip.get_txt_feat(text,))
            text_feat = torch.stack(text_feat, dim=0).to(self.clip.device)

            video_embed = text_feat

            B = video_embed.shape[0]
            T = 1

            # Get actions
            video_embed = video_embed.repeat(1, duration, 1)
            with torch.no_grad():
                # Imagine
                prior = wm.connector.video_imagine(video_embed, None, sample=False, reset_every_n_frames=False, denoise=True)
                # Decode
                prior_recon = decoder(wm.decoder_input_fn(prior))['observation'].mean + 0.5

        save_videos(prior_recon.unsqueeze(0), self.result_dir, filenames=[prompt_str], fps=15)
        print(f"Saved in {prompt_str}.mp4. Time used: {(time.time() - start):.2f} seconds")
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")
    
    def download_model(self, model_folder, model_filename):
        REPO_ID = 'mazpie/genrl_models'
        filename_list = [model_filename]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for filename in filename_list:
            local_file = os.path.join(model_folder, filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=model_folder, local_dir_use_symlinks=False)
    
    def download_internvideo2(self, model_folder):
        REPO_ID = 'OpenGVLab/InternVideo2-Stage2_1B-224p-f4'
        filename_list = ['InternVideo2-stage2_1b-224p-f4.pt']
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for filename in filename_list:
            local_file = os.path.join(model_folder, filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=model_folder, local_dir_use_symlinks=False)

if __name__ == '__main__':
    t2v = Text2Video()
    video_path = t2v.get_prompt('a black swan swims on the pond', 8)
    print('done', video_path)