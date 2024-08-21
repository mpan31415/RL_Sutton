import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import solve
import matplotlib.patches as ptch


#################################################################
############ implementation of cart-pole environment ############
#################################################################


class State:
    def __init__(self, t, x, v, a, theta, omega, alpha) -> None:
        self.t = t
        self.x = x
        self.v = v
        self.a = a
        self.theta = theta
        self.omega = omega
        self.alpha = alpha


##########################################
class CartPoleEnv:
    
    def __init__(self, cart_mass, bob_mass, rod_length,
                 dt=0.01, x0=0, v0=0, theta0=np.radians(0), omega0=0, loop_animation=False):
        
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
        self.reset()
        
        
    ##########################################
    def reset(self):
        
        # memory for all past states
        self._xs, self._vs, self._as = [self._x0], [self._v0], [0]
        self._thetas, self._omegas, self._alphas = [self._theta0], [self._omega0], [0]
        
        # constants and current state (initialized) dictionary
        self._const = {
            'm1': self._m1,
            'm2': self._m2,
            'L': self._L
        }
        # self._state = {
        #     't': 0,
        #     'x': self._x0,
        #     'v': self._v0,
        #     'a': 0,
        #     'theta': self._theta0,
        #     'omega': self._omega0,
        #     'alpha': 0
        # }
        self._state = State(0, self._x0, self._v0, 0, self._theta0, self._omega0, 0)
        
        # variables for the states during simulation (initalized with initial conditions)
        self._t = 0
        self._x, self._v, self._a = self._x0, self._v0, 0
        self._theta, self._omega, self._alpha = self._theta0, self._omega0, 0
        
        
    ##########################################
    def step(self, input):
            
        # forward Euler method for numerical integration of the ODE
        self._t += self._dt
        
        # get input
        F = input
        
        # compute accelerations
        A = np.array([[self._m1+self._m2, self._m2*self._L*np.cos(self._theta)], [np.cos(self._theta), self._L]])
        b = np.array([F+self._m2*self._L*(self._omega**2)*np.sin(self._theta), -self._g*np.sin(self._theta)])
        acc_vec = solve(A, b)
        self._a = acc_vec[0]
        self._alpha = acc_vec[1]
        
        # semi-implicit Euler update
        self._v += self._a * self._dt
        self._x += self._v * self._dt
        self._omega += self._alpha * self._dt
        self._theta += self._omega * self._dt

        # add to memory of states
        self._xs.append(self._x)
        self._vs.append(self._v)
        self._as.append(self._a)
        self._thetas.append(self._theta)
        self._omegas.append(self._omega)
        self._alphas.append(self._alpha)
        
        # update current state dictionary
        self._state = State(self._t, self._x, self._v, self._a, self._theta, self._omega, self._alpha)
        
        # self._state['t'] = self._t
        # self._state['x'] = self._x
        # self._state['v'] = self._v
        # self._state['a'] = self._a
        # self._state['theta'] = self._theta
        # self._state['omega'] = self._omega
        # self._state['alpha'] = self._alpha
        
    
    ##########################################
    def get_state(self):
        return self._state
    
    
    ##########################################
    def get_constants(self):
        return self._const
    
        
    ##########################################
    def animate(self):
        
        # simulation finished
        print("="*30, end='')
        print("  Simulation finished! Showing animation ...  ", end='')
        print("="*30)
        
        # compute number of time steps simulated
        self.num_steps = len(self._xs)
        
        # Initialize the animation plot. Make the aspect ratio equal so it looks right.
        fig = plt.figure()
        ax = fig.add_subplot(aspect='equal')

        # compute initial positions of cart center and pendulum mass
        cx0, cy0 = self._x0, 0
        px0, py0 = self._get_bob_pos(cx0, self._theta0)

        # define visualization of the cart, pendulum rod, and the bob
        cart = ax.add_patch(ptch.Rectangle(self._get_cart_pos(cx0), self._cart_width, self._cart_height, 
                                           fc='b', zorder=1))
        line, = ax.plot([cx0, px0], [cy0, py0], lw=3, c='k')
        circle = ax.add_patch(ptch.Circle(self._get_bob_pos(cx0, self._theta0), self._bob_radius,
                              fc='r', zorder=3))
        
        # set plot limits
        ax.set_xlim(-self._L*10, self._L*10)
        ax.set_ylim(-self._L*1.5, self._L*1.5)

        # define the callable animation function
        steps_per_second = int(1/self._dt)
        def animate_func(i):
            if i%steps_per_second==0 and i>0:
                print("animation wall time = %d seconds" % int(i/steps_per_second))
            # get the recorded positions at timestep i
            cx, cy = self._get_cart_pos(self._xs[i])
            px, py = self._get_bob_pos(self._xs[i], self._thetas[i])
            # set the positions
            cart.set_xy((cx-self._cart_width/2, cy-self._cart_height/2))
            line.set_data([cx, px], [cy, py])
            circle.set_center((px, py))

        # set animation parameters
        nframes = self.num_steps
        interval = self._dt * 1000        # delay between consecutive frames [milliseconds]
        ani = animation.FuncAnimation(fig, animate_func, frames=nframes, repeat=self._repeat,
                                    interval=interval)
        plt.show()
    
    ##########################################
    def _get_bob_pos(self, x, theta):
        return x + self._L * np.sin(theta), -self._L * np.cos(theta)

    ##########################################
    def _get_cart_pos(self, x):
        return x, 0
    



##########################################
if __name__ == "__main__":
    
    cart_mass = 1
    bob_mass = 1
    rod_length = 1
    total_time = 10
    
    # define the force function
    def force_func(sys:CartPoleEnv):
        t = sys.get_state().t
        return 5*np.sin(5*t)
    
    sys = CartPoleEnv(cart_mass, bob_mass, rod_length)
    
    while sys.get_state().t < total_time:
        cont_input = force_func(sys)
        sys.step(cont_input)
    
    sys.animate()