from cart_pole_system import CartPole
import numpy as np
import control



##########################################
if __name__ == "__main__":
    
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    total_time = 20
    g = 9.81
    
    # define controller function
    # LQR computation
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -bob_mass*g/cart_mass, 0],
                  [0, 0, 0, 1],
                  [0, 0, (cart_mass+bob_mass)*g/(rod_length*cart_mass), 0]])
    B = np.diag([0, 1/cart_mass, 0, -1/(rod_length*cart_mass)])
    # B = np.diag([0, 1/cart_mass, 0, 0])
    # LQR gains
    Q = np.diag([0.01, 0.01, 10, 100])
    R = np.diag([1, 1, 1, 1])
    # # state space
    # ss = control.StateSpace(A, B, np.identity(4), np.identity(4))
    # lqr = control.LQR(ss, Q, R, 100)
    # # gain matrix
    # K = lqr.solve()
    K, S, E = control.lqr(A, B, Q, R)
    
    def force_func(constants:dict, state:dict):
        theta = state['theta']
        
        if abs(theta-np.radians(180)) > 0.15:
            # energy-shaping swingup to get closer quicker
            omega = state['omega']
            # return -1 * omega / np.cos(theta)
            return -0.5 * omega / np.cos(theta) / abs(theta)
        else:
            # maintain equilibrium using LQR
            print("LQR running")
            x = np.array([state['x'],state['v'],state['theta']-np.radians(180),state['omega']])
            cont = -K@x
            print(cont)
            return cont[1]*60
    
    # k=1
    # def force_func(theta, omega, l):
    #     # return -k * omega * l / np.cos(theta)
    #     return -k * omega / np.cos(theta)
    
    # system = CartPole(cart_mass, bob_mass, rod_length, total_time, force_func, theta0=np.radians(180))
    system = CartPole(cart_mass, bob_mass, rod_length, total_time, force_func, theta0=np.radians(10))
    
    system.simulate()
    
    system.animate()