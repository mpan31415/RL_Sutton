from random import choice
import numpy as np

class RandomWalk:
    
    def __init__(self, num_states) -> None:
        self.num_states = num_states
        self.curr_state = int((num_states+1)/2)
        self.steps = [-1, 1]
        self.terminal = False
        
    def step(self):
        self.curr_state += choice(self.steps)
        if self.curr_state == 1 or self.curr_state == self.num_states:
            self.terminal = True
        # assign reward
        reward = 0
        if self.curr_state == self.num_states:
            reward = 1
        return reward
        
        
class TDAgent:
    
    def __init__(self, num_states, alpha, gamma, num_episode, variable_alpha=False, verbose=False) -> None:
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.num_episode = num_episode
        self.state_values = np.ones(num_states, dtype=float)*0.5   # init state values
        # fix terminal state values to be zero
        self.state_values[0] = 0.0
        self.state_values[-1] = 0.0
        
        self.true_state_values = [1/6, 2/6, 3/6, 4/6, 5/6]
        
        # state values memory, omits terminal states
        self.state_values_history = np.zeros((num_episode+1, num_states-2))
        self.state_values_history[0] = self.state_values[1:-1]
        
        # rms errors history
        self.rms_errors_history = [self.get_state_values_rms()]
        
        # other flags
        self.variable_alpha = variable_alpha
        self.verbose = verbose
        
        
    def run(self):
        for episode_num in range(1, self.num_episode+1):
            if self.verbose:
                print("Starting episode #%d" % int(episode_num))
            walk = RandomWalk(self.num_states)
            
            while not walk.terminal:
                last_state = walk.curr_state
                reward = walk.step()
                this_state = walk.curr_state
                if self.verbose:
                    print("Walked from state %d to %d" % (last_state, this_state))
                # TD(0) update
                td_error = reward + self.gamma*self.state_values[this_state-1] - self.state_values[last_state-1]
                if self.variable_alpha:
                    self.state_values[last_state-1] += np.sqrt(1/episode_num)*td_error
                else:
                    self.state_values[last_state-1] += self.alpha*td_error
            
            if self.verbose:
                print("Finished episode #%d" % int(episode_num))
                
            # add state values to memory
            self.state_values_history[episode_num] = self.state_values[1:-1]
            
            # add rms error to memory
            self.rms_errors_history.append(self.get_state_values_rms())
            
            
    def get_state_values_rms(self):
        return np.sqrt(((self.state_values[1:-1] - self.true_state_values) ** 2).mean())


if __name__ == "__main__":
    
    num_states = 7    # including left & right terminal states, ODD NUMBER
    alpha = 0.1
    gamma = 1
    num_episode = 3
    
    td_agent = TDAgent(num_states, alpha, gamma, num_episode)
    td_agent.run()
    
    print(td_agent.state_values_history)
    print(td_agent.rms_errors_history)