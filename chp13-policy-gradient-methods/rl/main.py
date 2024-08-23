from cart_pole_env import CartPoleEnv
# from deep_q_agent import DQAgent


from utils import str2bool, evaluate_policy, animate_policy
from datetime import datetime
from td3 import TD3_agent
import numpy as np
import os, shutil
import argparse
import torch


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
# parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(1e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(3e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--animate_interval', type=int, default=int(5e3), help='Animation interval, in steps.')

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



##########################################
def main():
    
    # define cart-pole system constants
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    
    # build environment
    env = CartPoleEnv(cart_mass, bob_mass, rod_length, theta0=np.radians(195))
    eval_env = CartPoleEnv(cart_mass, bob_mass, rod_length, theta0=np.radians(195))
    env_name = "CartPole"
    
    opt.state_dim = env.observation_dim
    opt.action_dim = env.action_space.dim
    opt.max_action = float(env.action_space.upper_bound)   #remark: action space【-max,max】
    opt.max_e_steps = env.max_episode_steps
    print(f'Env:{env_name}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.lower_bound}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(env_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(env_name, opt.ModelIdex)


    # if opt.render:
    #     while True:
    #         score = evaluate_policy(env, agent, turns=1)
    #         print('EnvName:', env_name, 'score:', score)
    # else:
    total_steps = 0
    while total_steps < opt.Max_train_steps:
        
        print("*"*20, end='')
        print("  This is step %d  " % total_steps, end='')
        print("*"*20)
        
        env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
        s = env.get_state()
        done = False

        '''Interact & trian'''
        while not done:
            if total_steps < (10*opt.max_e_steps): a = env.action_space.sample() # warm up
            else: a = agent.select_action(s, deterministic=False)
            s_next, r, done = env.step(a) # dw: dead&win; tr: truncated

            agent.replay_buffer.add(s, a, r, s_next, done)
            s = s_next
            total_steps += 1

            '''train if its time'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                for j in range(opt.update_every):
                    agent.train()

            '''record & log'''
            if total_steps % opt.eval_interval == 0:
                agent.explore_noise *= opt.explore_noise_decay
                ep_r = evaluate_policy(eval_env, agent, turns=3)
                if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{env_name}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')
                
            '''animate policy'''
            if total_steps % opt.animate_interval == 0:
                animate_policy(eval_env, agent)

            '''save model'''
            if total_steps % opt.save_interval == 0:
                agent.save(env_name, int(total_steps/1000))
    
        # env.close()
        # eval_env.close()
    
    
    print("="*100)
    print("\n\n  Finished  \n\n")
    print("="*100)
    
    
    
    
    # agent = DQAgent()
    
    # # simulate
    # total_sim_time = 10    # [seconds]
    # while env.get_state().t < total_sim_time:
    #     state = env.get_state()
    #     # action = agent.get_action(state)
    #     action = agent.get_random_action(state)
    #     env.step(action)
    
    # # simulation finished, display animation
    # env.animate()
    
    
    

    





##########################################
if __name__ == "__main__":
    main()



# ##########################################
# if __name__ == "__main__":
    
#     # define cart-pole system constants
#     cart_mass = 1
#     bob_mass = 1
#     rod_length = 1
    
#     # instantiate cart-pole environment and deep-Q agent
#     env = CartPoleEnv(cart_mass, bob_mass, rod_length)
#     agent = DQAgent()
    
#     # simulate
#     total_sim_time = 10    # [seconds]
#     while env.get_state().t < total_sim_time:
#         state = env.get_state()
#         # action = agent.get_action(state)
#         action = agent.get_random_action(state)
#         env.step(action)
    
#     # simulation finished, display animation
#     env.animate()