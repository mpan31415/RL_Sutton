from cart_pole_system import CartPole
import numpy as np
import control


##########################################
if __name__ == "__main__":
    
    cart_mass = 5
    bob_mass = 1
    rod_length = 1
    total_time = 20
    g = 9.81
    
    # define controller function
    # linearized system about top point (theta = 180 deg)
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -bob_mass*g/cart_mass, 0],
                  [0, 0, 0, 1],
                  [0, 0, -(cart_mass+bob_mass)*g/(rod_length*cart_mass), 0]])
    B = np.array([[0, 0, 0, 0],
                 [1/cart_mass, 0, 0, 0],
                 [0, 0, 0, 0],
                 [1/(rod_length*cart_mass), 0, 0, 0]])
    # LQR gain matrices
    Q = np.diag([0.01, 10, 10000, 1])
    R = np.diag([0.001, 1, 1, 1])
    
    # control gain matrix computation
    K, S, E = control.lqr(A, B, Q, R)
    # K, S, E = control.dlqr(A, B, Q, R)
    print("K = ", K)
    # print("S = ", S)
    print("E = ", E)
    
    # memory for storing the active controller across the trajectory
    control_history = []
    
    def force_func(constants:dict, state:dict):
        theta = state['theta']
        force = 0
        if abs(theta-np.radians(180)) > np.radians(20):
            # energy-shaping swingup to get closer quicker
            # print("energy shaping running")
            omega = state['omega']
            force = -1 * omega / np.cos(theta)
            # force = -0.25 * omega / np.cos(theta) / abs(theta)
            control_history.append(0)
        else:
            # maintain equilibrium using LQR
            # print("LQR running")
            x = np.array([state['x'],state['v'],state['theta']-np.radians(180),state['omega']])
            cont = -K@x
            # print(cont)
            force = cont[0]
            # print(force)
            control_history.append(1)
            
        # print("force = ",force)
        return force
    
    # system = CartPole(cart_mass, bob_mass, rod_length, total_time, force_func, theta0=np.radians(162))
    system = CartPole(cart_mass, bob_mass, rod_length, total_time, force_func, theta0=np.radians(20))
    
    system.simulate()
    
    print(control_history)
    
    system.animate()