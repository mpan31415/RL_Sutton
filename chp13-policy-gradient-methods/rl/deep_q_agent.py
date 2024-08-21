import numpy as np
from cart_pole_env import State


class DQAgent:
    
    def __init__(self) -> None:
        self.rng = np.random.default_rng(69)
    
    def get_action(self, state:State):
        t = state.t
        return 5*np.sin(5*t)
    
    def get_random_action(self, state:State):
        return self.rng.normal(0.0, 15.0)
