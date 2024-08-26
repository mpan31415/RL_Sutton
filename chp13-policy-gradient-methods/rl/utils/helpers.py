import argparse
import torch


def evaluate_policy(env, agent, turns = 3):
    
    total_scores = 0
    for j in range(turns):
        env.reset()
        s = env.get_state()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, done = env.step(a)

            total_scores += r
            s = s_next
            
    return int(total_scores/turns)


def animate_policy(env, agent):
    
    env.reset(rand=False)
    while env.get_sim_time() < 10:
        # Take deterministic actions at test time
        a = agent.select_action(env.get_state(), deterministic=True)
        env.step(a)
        
    env.animate()
    

#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
def generate_settings(load_model=False, model_idx=0):
    
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--Loadmodel', type=str2bool, default=load_model, help='Load pretrained model or Not')
    if load_model:
        parser.add_argument('--ModelIdex', type=int, default=model_idx, help='which model to load')
    
    # parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')
    parser.add_argument('--Max_train_steps', type=int, default=int(1e5), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(3e4), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
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
    
    return opt
