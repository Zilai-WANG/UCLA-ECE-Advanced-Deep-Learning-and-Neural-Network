import torch
import torch.nn as nn
import traceback
from termcolor import colored
import gymnasium as gym
import numpy as np
import sys 

#get the operating system
if sys.platform.startswith('darwin'):
    # Mac OS X
    suffix = "_mac"
else:
    suffix = ""

def test_model_DQN(model):
    try:
        model = model((4,84,84),5)

        model.load_state_dict(torch.load("test_weights.pt", map_location="cpu"))
        # Test the forward function
        test_outputs =  torch.load(f"test_outputs{suffix}.pt",map_location=torch.device('cpu'))
        test_inputs = test_outputs["S"]
        test_outputs = test_outputs["outputs"]
        model.eval()
        with torch.no_grad():
            for i in range(len(test_inputs)):
                # print(torch.tensor(test_inputs[i]).float().shape)
                assert torch.allclose(model(torch.tensor(test_inputs[i]).float().unsqueeze(0)),torch.tensor(test_outputs[i])), f"expected {test_outputs[i]} but got {model(torch.tensor(test_inputs[i]).float().unsqueeze(0))}"
            
        print(colored("Passed","green"))
    except Exception as e:
        print(e)
        print(colored("Failed","red"))
        traceback.print_exc()
        return

def test_model_DDPG(model):
    #TODO: Implement the test for the DDPG model
    pass

def test_wrapper(wrapper):
    try:
        env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
        wrapper = wrapper(env)

        # Test the reset function
        test_outputs = torch.load(f"test_outputs{suffix}.pt",map_location=torch.device('cpu'))
        test_inputs = test_outputs["outputs"]
        test_outputs = test_outputs["S"]
        s,_ = wrapper.reset(seed = 42)
        # print(s.shape)
        # print(test_outputs[0].shape)
        wrong_indexs = ~np.isclose(test_outputs[0],s)
        assert np.allclose(test_outputs[0],s), f"at {np.where(wrong_indexs)} expected {test_outputs[0][wrong_indexs]} but got {s[wrong_indexs]}"
        print(colored("Passed reset","green"))

        for i  in range(len(test_outputs)-1):
            # Test the step function
            s,r,terminated, truncated, info = wrapper.step(np.argmax(test_inputs[i]))
            assert np.allclose(test_outputs[i+1],s), f"expected {test_outputs[i+1]} but got {s}"

        print(colored("Passed step","green"))
    except Exception as e:
        print(e)
        print(colored("Failed","red"))
        traceback.print_exc()
        return
    
import pickle 

def check_same_torch(a,b):
    #first check shape 
    if a.shape != b.shape:
        return False
    #then check values
    return torch.allclose(a,b)

    



def test_DQN_replay_buffer(buffer_class):
    with open(f"test_replay_buffer_inputs{suffix}.pkl","rb") as f:
        buffer_inputs = pickle.load(f)
    buffer_samples = torch.load(f"test_replay_buffer_samples{suffix}.pth")
    try:
        buffer = buffer_class(40,seed = 42)
        j = 0
        for i in range(100):
            buffer.add(buffer_inputs["states"][i],buffer_inputs["actions"][i],buffer_inputs["rewards"][i],buffer_inputs["next_states"][i],buffer_inputs["dones"][i])
            if i % 30 == 29:
                # print(i)
                target_outputs = buffer_samples[j]
                actual_outputs = buffer.sample(5)
                for k in range(len(target_outputs)):
                    # print(target_outputs[k],actual_outputs[k])
                    assert check_same_torch(target_outputs[k],actual_outputs[k]), f"expected {target_outputs[k][0]} but got {buffer.sample(1)[k][0]}"
                # assert np.all(buffer_samples[j] == buffer.sample(40)), f"expected {buffer_samples[j]} but got {buffer.sample(40)}"
                j += 1
        
        print(colored("Passed","green"))
    except:
        print(colored("Failed","red"))
        traceback.print_exc()
        return
        
            
    
    
if __name__ == "__main__":
    from replay_buffer import ReplayBufferDQN
    
    test_DQN_replay_buffer(ReplayBufferDQN)
    
    
    from env_wrapper import EnvWrapper
    test_wrapper(EnvWrapper)


    from model import Nature_Paper_Conv
    test_model_DQN(Nature_Paper_Conv)