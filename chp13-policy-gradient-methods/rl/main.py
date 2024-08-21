from cart_pole_env import CartPoleEnv
from deep_q_agent import DQAgent


##########################################
if __name__ == "__main__":
    
    # define cart-pole system constants
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    
    # instantiate cart-pole environment and deep-Q agent
    env = CartPoleEnv(cart_mass, bob_mass, rod_length)
    agent = DQAgent()
    
    # simulate
    total_sim_time = 10    # [seconds]
    while env.get_state().t < total_sim_time:
        state = env.get_state()
        action = agent.get_action(state)
        env.step(action)
    
    # simulation finished, display animation
    env.animate()