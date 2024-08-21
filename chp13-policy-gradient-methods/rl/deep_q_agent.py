import numpy as np
from cart_pole_env import State


class DQAgent:
    
    def __init__(self) -> None:
        pass
    
    def get_action(self, state:State):
        t = state.t
        return 5*np.sin(5*t)
