from utils import animate_policy, str2bool
from cart_pole_env import CartPoleEnv
import numpy as np
from td3 import TD3_agent
import torch
import argparse


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
# parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=90, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(1e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(3e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2500), help='Model evaluating interval, in steps.')
parser.add_argument('--animate_interval', type=int, default=int(5000), help='Animation interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=32, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.15, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


if __name__ == "__main__":
    
    # define cart-pole system constants
    cart_mass = 1
    bob_mass = 1
    rod_length = 1

#     env = CartPoleEnv(cart_mass, bob_mass, rod_length, 
#                       x0=5, theta0=np.radians(180))
#     env_name = "CartPole"
    
    env = CartPoleEnv(cart_mass, bob_mass, rod_length, theta0=np.radians(200), rand_start=False)
    env_name = "CartPoleUp"
    
    opt.state_dim = env.observation_dim
    opt.action_dim = env.action_space.dim
    opt.max_action = float(env.action_space.upper_bound)   #remark: action space【-max,max】
    opt.max_e_steps = env.max_episode_steps
    print(f'Env:{env_name}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.lower_bound}  max_e_steps:{opt.max_e_steps}')

    agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
    agent.load(env_name, opt.ModelIdex)

    animate_policy(env, agent)