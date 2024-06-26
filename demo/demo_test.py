import os
from pathlib import Path
VIDEO_PATH = Path(os.path.abspath('')) / 'assets' / 'video_samples'

class Text2Video():
    def __init__(self, result_dir='./tmp/') -> None:
        pass

    def get_prompt(self, input_text, steps=50, cfg_scale=15.0, eta=1.0, fps=16):

        return str(VIDEO_PATH / 'headstand.mp4')
    
class Video2Video:
    def __init__(self, result_dir='./tmp/') -> None:
        pass

    def get_image(self, input_image, input_prompt, i2v_steps=50, i2v_cfg_scale=15.0, i2v_eta=1.0, i2v_fps=16):

        return str(VIDEO_PATH / 'dancing.mp4')
        
if __name__ == '__main__':
    t2v = Text2Video()
    print(t2v.get_prompt('test'))