from datetime import datetime
import numpy as np
import os, shutil
import torch

from envs.cart_pole_up import CartPoleUp
from agents.td3 import TD3_agent
from utils.helpers import evaluate_policy, animate_policy, generate_settings


##########################################
def main():
    
    # generate experiment settings
    opt = generate_settings()
    print(opt)

    # # define cart-pole system constants
    # cart_mass = 1
    # bob_mass = 1
    # rod_length = 1

    # build environment
    env = CartPoleUp()
    eval_env = CartPoleUp()
    env_name = "CartPoleUp"

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
    agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(env_name, opt.ModelIdex)
    
    
    total_steps = 0
    while total_steps < opt.Max_train_steps:

        # print("*"*20, end='')
        # print("  This is step %d  " % total_steps, end='')
        # print("*"*20)

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


    print("="*100)
    print("\n\n  Finished  \n\n")
    print("="*100)





##########################################
if __name__ == "__main__":
    main()


