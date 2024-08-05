import numpy as np
from scipy.stats import bernoulli

####################################
class MultiArmedBandit:

    def __init__(self, num_bandits):
        self.num_bandits = num_bandits
        self.rng = np.random.Generator(np.random.PCG64())
        self.reward_means = self.rng.normal(0, 1, size=self.num_bandits)
        self.optimal_bandit_id = np.argmax(self.reward_means)+1
    
    def get_bandit_reward(self, bandit_num):
        return self.rng.normal(self.reward_means[bandit_num-1], 1)
    


####################################
class BanditAgent:

    def __init__(self, num_bandits, steps, epsilon, alpha=0):
        self.rng = np.random.Generator(np.random.PCG64())
        
        self.bandit_ids = np.linspace(1, num_bandits, num_bandits, dtype=int)  # [1, num_bandits]
        self.num_bandits = num_bandits
        self.epsilon = epsilon
        self.alpha = alpha
        
        self.ave_bandit_rewards = np.zeros(self.num_bandits)
        self.bandit_counts = np.zeros(self.num_bandits)
        
        self.total_steps = steps
        
        
    def choose_bandit(self):
        if bernoulli.rvs(self.epsilon)==1:
            return self.rng.choice(self.bandit_ids)
        else:
            return np.argmax(self.ave_bandit_rewards)+1
        
    def incremental_update(self, bandit_id, reward):
        bandit_count = self.bandit_counts[bandit_id-1]
        self.bandit_counts[bandit_id-1] += 1
        if self.alpha>0:
            self.ave_bandit_rewards[bandit_id-1] += self.alpha*(reward-self.ave_bandit_rewards[bandit_id-1])
        else:
            scaling = 0
            if bandit_count>0:
                scaling = (1/bandit_count)
            self.ave_bandit_rewards[bandit_id-1] += scaling*(reward-self.ave_bandit_rewards[bandit_id-1])

    def play(self):
        
        mab = MultiArmedBandit(self.num_bandits)
        optimal_bandit_id = mab.optimal_bandit_id
        
        self.rewards = []
        self.ave_rewards = []
        
        self.choices = []
        self.percent_optimal_choice = []
        
        current_step = 1
        
        while current_step <= self.total_steps:
            
            bandit_id = self.choose_bandit()
            reward = mab.get_bandit_reward(bandit_id)
            self.incremental_update(bandit_id, reward)
            
            self.rewards.append(reward)
            self.ave_rewards.append(sum(self.rewards)/len(self.rewards))
            
            self.choices.append(bandit_id==optimal_bandit_id)
            self.percent_optimal_choice.append(sum(self.choices)/len(self.choices))
            
            # print("Finished step", current_step)
            current_step += 1


####################################
if __name__ == "__main__":
    
    num_bandits = 10
    epsilon = 0.01
    
    agent = BanditAgent(num_bandits, 1000, epsilon)
    agent.play()