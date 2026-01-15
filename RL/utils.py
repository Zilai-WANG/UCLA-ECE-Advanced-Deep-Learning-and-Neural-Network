import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML
import os
import cv2

def animate(frames):
    # Create animation
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def animate(i):
        im.set_array(frames[i])
        return im,
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames))
    return anim

class exponential_decay:
    def __init__(self, epsilon:float, half_life:int, min_epsilon:float):
        self.epsilon = epsilon
        self.decay_rate = 0.5 ** (1 / half_life)
        self.epsilon = self.epsilon/self.decay_rate
        self.min_epsilon = min_epsilon
        
    def __call__(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)
        return self.epsilon
    

class linear_decay:
    def __init__(self, epsilon:float, decay_time:int, min_epsilon:float):
        self.epsilon = epsilon
        self.decay_rate = (epsilon - min_epsilon) / decay_time
        self.epsilon = self.epsilon + self.decay_rate
        self.min_epsilon = min_epsilon
        
    def __call__(self):
        self.epsilon = max(self.epsilon - self.decay_rate, self.min_epsilon)
        return self.epsilon
    
def get_save_path(suffix,directory):
    save_path = os.path.join(directory,suffix)
    #find the number of run directories in the directory
    try:
        runs = [d for d in os.listdir(save_path) if "run" in d]
        runs = sorted(runs,key = lambda x: int(x.split("run")[1]))
        last_run = runs[-1]
        last_run = int(last_run.split("run")[1])
        save_path = os.path.join(save_path,f"run{last_run+1}")
    except:
        save_path = os.path.join(save_path,"run0")
    print("saving to",save_path)
    return save_path

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img
    
