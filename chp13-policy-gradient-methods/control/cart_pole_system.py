import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import solve
import matplotlib.patches as ptch
# import control


##########################################
class CartPole:
    
    def __init__(self, cart_mass, bob_mass, rod_length, total_time, control_input_func,
                 dt=0.01, x0=0, v0=0, theta0=np.radians(0), omega0=0, loop_animation=False):
        
        # constants
        self.m1 = cart_mass
        self.m2 = bob_mass
        self.L = rod_length
        self.g = 9.81
        
        # cart and bob sizes
        self.cart_height, self.cart_width = 0.25, 0.5
        self.bob_radius = 0.08
        
        # initial conditions
        self.x0, self.v0 = x0, v0                        # cart position and velocity
        self.theta0, self.omega0 = theta0, omega0        # pendulum angular position and velocity
        
        # simulation parameters
        self.dt = dt                  # time step for numerical integration of the equation of motion [s]
        self.total_time = total_time  # total simulation time
        
        # memory for all past states
        self.xs, self.vs, self.aas = [self.x0], [self.v0], [0]
        self.thetas, self.omegas, self.alphas = [self.theta0], [self.omega0], [0]
        
        # flags for animation
        self.repeat = loop_animation
        
        # control input callable function
        self.input_func = control_input_func
        
        # constants and current state (initialized) dictionary
        self.constants_dict = {
            'm1': self.m1,
            'm2': self.m2,
            'L': self.L
        }
        self.curr_state_dict = {
            't': 0,
            'x': self.x0,
            'v': self.v0,
            'a': 0,
            'theta': self.theta0,
            'omega': self.omega0,
            'alpha': 0
        }
        
        
    ##########################################
    def simulate(self):

        self.t = 0
        
        # temporary variables for the states during simulation (initalized with initial conditions)
        self.x, self.v, self.a = self.x0, self.v0, 0
        self.theta, self.omega, self.alpha = self.theta0, self.omega0, 0
        
        while self.t < self.total_time:
            
            # forward Euler method for numerical integration of the ODE
            self.t += self.dt
            
            # calculate user input
            # F = self.input_func(self.t)
            F = self.input_func(self.constants_dict, self.curr_state_dict)
            
            # compute accelerations
            A = np.array([[self.m1+self.m2, self.m2*self.L*np.cos(self.theta)], [np.cos(self.theta), self.L]])
            b = np.array([F+self.m2*self.L*(self.omega**2)*np.sin(self.theta), -self.g*np.sin(self.theta)])
            acc_vec = solve(A, b)
            self.a = acc_vec[0]
            self.alpha = acc_vec[1]
            
            # semi-implicit Euler update
            self.v += self.a * self.dt
            self.x += self.v * self.dt
            self.omega += self.alpha * self.dt
            self.theta += self.omega * self.dt

            # add to memory of states
            self.xs.append(self.x)
            self.vs.append(self.v)
            self.aas.append(self.a)
            self.thetas.append(self.theta)
            self.omegas.append(self.omega)
            self.alphas.append(self.alpha)
            
            # update current state dictionary
            self.curr_state_dict['t'] = self.t
            self.curr_state_dict['x'] = self.x
            self.curr_state_dict['v'] = self.v
            self.curr_state_dict['a'] = self.a
            self.curr_state_dict['theta'] = self.theta
            self.curr_state_dict['omega'] = self.omega
            self.curr_state_dict['alpha'] = self.alpha
        
        # simulation finished
        print("="*30, end='')
        print("  Simulation finished!  ", end='')
        print("="*30)
        
        # compute number of time steps simulated
        self.num_steps = len(self.xs)
    
            
    ##########################################
    def animate(self):
        
        # Initialize the animation plot. Make the aspect ratio equal so it looks right.
        fig = plt.figure()
        ax = fig.add_subplot(aspect='equal')

        # compute initial positions of cart center and pendulum mass
        cx0, cy0 = self.x0, 0
        px0, py0 = self.get_bob_pos(cx0, self.theta0)

        # define visualization of the cart, pendulum rod, and the bob
        cart = ax.add_patch(ptch.Rectangle(self.get_cart_pos(cx0), self.cart_width, self.cart_height, 
                                           fc='b', zorder=1))
        line, = ax.plot([cx0, px0], [cy0, py0], lw=3, c='k')
        circle = ax.add_patch(ptch.Circle(self.get_bob_pos(cx0, self.theta0), self.bob_radius,
                              fc='r', zorder=3))
        
        # set plot limits
        ax.set_xlim(-self.L*10, self.L*10)
        ax.set_ylim(-self.L*1.5, self.L*1.5)

        # define the callable animation function
        steps_per_second = int(1/self.dt)
        def animate_func(i):
            if i%steps_per_second==0 and i>0:
                print("simulation wall time = %d seconds" % int(i/steps_per_second))
            # get the recorded positions at timestep i
            cx, cy = self.get_cart_pos(self.xs[i])
            px, py = self.get_bob_pos(self.xs[i], self.thetas[i])
            # set the positions
            cart.set_xy((cx-self.cart_width/2, cy-self.cart_height/2))
            line.set_data([cx, px], [cy, py])
            circle.set_center((px, py))

        # set animation parameters
        nframes = self.num_steps
        interval = self.dt * 1000        # delay between consecutive frames [milliseconds]
        ani = animation.FuncAnimation(fig, animate_func, frames=nframes, repeat=self.repeat,
                                    interval=interval)
        plt.show()
        
    
    ##########################################
    def get_current_sim_states(self):
        return self.t, self.x, self.v, self.a, self.theta, self.omega, self.alpha
    
    ##########################################
    def get_bob_pos(self, x, theta):
        return x + self.L * np.sin(theta), -self.L * np.cos(theta)

    ##########################################
    def get_cart_pos(self, x):
        return x, 0
    



##########################################
if __name__ == "__main__":
    
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    total_time = 10
    
    # define the force function
    def force_func(constants:dict, curr_state:dict):
        t = curr_state['t']
        return 5*np.sin(5*t)
    
    system = CartPole(cart_mass, bob_mass, rod_length, total_time, force_func)
    
    system.simulate()
    
    system.animate()