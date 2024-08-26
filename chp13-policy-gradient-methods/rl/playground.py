import numpy as np

from utils.helpers import animate_policy, generate_settings
from envs.cart_pole_up import CartPoleUp
from agents.td3 import TD3_agent


if __name__ == "__main__":
    
    ############ generate the experiment settings ############
    opt = generate_settings(load_model=True, model_idx=90)
    print(opt)
    
    env = CartPoleUp(rand_start=False, theta0=np.radians(170))
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