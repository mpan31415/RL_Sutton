import numpy as np
from envs.cart_pole import CartPole


#################################################################################
############ implementation of cart-pole-up environment for learning ############
############ how to balance the inverted pendulum ###############################
#################################################################################
        

##########################################
class ActionSpace:
    def __init__(self, lower, upper, dim) -> None:
        self.lower_bound = lower
        self.upper_bound = upper
        self.dim = dim
        self.rng = np.random.default_rng()
        
    def sample(self):
        # return self.rng.uniform(self.lower_bound, self.upper_bound)
        return np.array([self.rng.uniform(self.lower_bound, self.upper_bound)])
    

##########################################
class CartPoleUp(CartPole):
    
    def __init__(self, cart_mass=1, bob_mass=1, rod_length=1,
                 dt=0.01, x0=0, v0=0, theta0=np.radians(0), omega0=0, loop_animation=False, rand_start=True):
        
        super().__init__(cart_mass, bob_mass, rod_length, dt, x0, v0, theta0, omega0, loop_animation, rand_start)
        
        # constants
        self._m1 = cart_mass
        self._m2 = bob_mass
        self._L = rod_length
        self._g = 9.81
        
        # cart and bob sizes
        self._cart_height, self._cart_width = 0.25, 0.5
        self._bob_radius = 0.08
        
        # simulation parameters
        self._dt = dt                  # time step for numerical integration of the equation of motion [s]
        
        # flags for animation
        self._repeat = loop_animation
        
        # specified initial conditions
        self._x0, self._v0 = x0, v0                        # cart position and velocity
        self._theta0, self._omega0 = theta0, omega0        # pendulum angular position and velocity
        
        # reset (re-initilialize) env
        self.reset(rand=rand_start)
        
        # constants, for reinforcement learning algorithms
        self.observation_dim = 4         # [x, v, theta, omega]
        self.action_space = ActionSpace(lower=-1, upper=1, dim=1)
        self.force = 5                   # [Newtons]
        self.max_episode_steps = 1000
        
        
    ##########################################
    def step(self, action):
        
        # get action, and convert to left (-1) / right (1) moves
        dir = 1 if action[0] > 0 else -1
        F = self.force * dir
        
        super().step(F)
        
        return self.get_state(), self.get_reward(), self.is_terminal()
    
    
    ##########################################
    def is_terminal(self):
        # terminal state reached in 3 cases:
        # 1. theta > 15 degrees
        # 2. cart position > 5
        # 3. time > 5 seconds
        s = self.get_state()
        x = abs(s[0])
        theta_err = abs(np.degrees(s[2]) - 180)
        time = self._t
        if theta_err > 15 or x > 5 or time > 5:
            return True
        return False
        
    ##########################################
    def get_reward(self):
        
        reward = 1
        
        # s = self.get_state()
        # theta_deg = np.degrees(s[2])
        # omega_deg = np.degrees(s[3])
        
        # costs = (theta_deg-180)**2 + 0.1*np.degrees(omega_deg)**2
        # reward = -costs
        
        # reward = 1
        # if abs(theta_deg) > 15:
        #     reward = -1
        
        return reward
    


##########################################
if __name__ == "__main__":
    
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    total_time = 10
    
    # # define the force function
    # def force_func(env:CartPoleEnv):
    #     t = env.get_sim_time()
    #     return 15*np.sin(5*t)
    
    env = CartPoleUp(cart_mass, bob_mass, rod_length)
    
    while env.get_sim_time() < total_time:
        cont_input = env.action_space.sample()
        env.step(cont_input)
    
    env.animate()