import os


import numpy as np
import sys
import matplotlib.pyplot as plt 
from tqdm import tqdm
from plt_quiver import quiver3D, load, quiver2D
import pandas as pd
from helper import *
from animate import render_anim

class LC(object):
    """
    field - tensor (n, m, l, 3) represent LC grid
    gamma - update rate
    dr - [dx, dy, dz] of the grid
    K - Energy term
    E - tensor (n, m, l, 3) represent E field
    q0 - chirality of the LC
    E0, dE - electric field interaction term
    """
    def __init__(self, field_init, gamma ,dr, K, E, q0, E0 = 8.9e-12, dE = 3.7, dt = None):
        self.field = self.boundary_condition(field_init)
        self.gamma = gamma
        self.dr = dr
        self.dV = self.dr**3
        [self.K11, self.K22, self.K33] = K
        self.E = E
        self.q0 = q0
        self.E0 = E0
        self.dE = dE
        self.normalize()
        self.t = 0
        self.dt = dt if dt else  self.gamma *  self.dr**2 / (2 * self.K33) /5
        print('dt:', self.dt)

        print("Initiate model")
    
    """
    Assign Boundary Condition to the system
    """
    def boundary_condition(self, field):
        
        field[0,:,:] = field[-2,:,:]
        field[-1,:,:] = field[1,:,:]

        field[:,0,:] = field[:,-2,:]
        field[:,-1,:] = field[:,1,:]

        field[:,:,0] = np.array([0,0,1])
        field[:,:,-1] = np.array([0,0,1])

        return field

    def boundary_derivative(self, field):
        
        field[0,:,:] = field[-2,:,:]
        field[-1,:,:] = field[1,:,:]

        field[:,0,:] = field[:,-2,:]
        field[:,-1,:] = field[:,1,:]

        field[:,:,0] = 0
        field[:,:,-1] = 0

        return field

    """
    Calculate jacobian fo the field 
    """
    def gradient(self, field):
        grads = np.zeros(field.shape + (3,))
        grads[1:-1,1:-1,1:-1, 0] = (field[2:,1:-1,1:-1] - field[:-2,1:-1,1:-1])/self.dr/2
        grads[1:-1,1:-1,1:-1, 1] = (field[1:-1,2:,1:-1] - field[1:-1,:-2,1:-1])/self.dr/2
        grads[1:-1,1:-1,1:-1, 2] = (field[1:-1,1:-1,2:] - field[1:-1,1:-1,:-2])/self.dr/2
        return grads


    def laplacian(self, field):
        laplace = np.zeros(field.shape)
        laplace[1:-1,1:-1,1:-1] = -6 * field[1:-1, 1:-1, 1:-1] \
            + field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] \
                + field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] \
                    + field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] 


        return laplace/dr**2

    def dot_del(self, field1, field2):
        result = np.zeros(field1.shape)
        result += mult(field1[1:-1,1:-1,1:-1, 0], (field2[2:,1:-1,1:-1] - field2[:-2,1:-1,1:-1])/self.dr/2)
        result += mult(field1[1:-1,1:-1,1:-1, 1], (field[1:-1,2:,1:-1] - field[1:-1,:-2,1:-1])/self.dr/2)
        result += mult(field1[1:-1,1:-1,1:-1, 2], (field[1:-1,1:-1,2:] - field[1:-1,1:-1,:-2])/self.dr/2)
        return result

    def W(self):
        gradient_mat = self.gradient(self.field)
        curl_term = curl(gradient_mat)

        return np.mean(1 * self.K11/2 * div(gradient_mat)**2 \
            + 1 * self.K22/2 * (dot(field, curl_term) + self.q0)**2 \
            + 1 * self.K33/2 * norm_square(cross(field, curl_term)) \
            - 1 * self.E0 * self.dE/2 * dot(self.E, field)**2)

    """
    Calculate the functional derivative
    """

    def variational(self, field):
        gradient_mat = self.gradient(field)
        gradient_mat = self.boundary_derivative(gradient_mat)
        """
        Divergence term
        - K11 * grad(div(n))
        """

        div_force = - self.gradient(div(gradient_mat))

        """
        Twisted Term calculation
        K22 [(2 n . curl (n)  + q0)(curl n) -  n x grad(n . curl(n)) ]
        """
        curl_term = curl(gradient_mat)
        dot_curl_term = dot(field, curl_term) 
        zeroth_twisted_term = mult(dot_curl_term + self.q0, curl_term)
        grad_dot_curl_term = cross(field, self.gradient(dot_curl_term))
        twist_force = 2 * zeroth_twisted_term - grad_dot_curl_term
        """
        Curve Calculation
        """
        # anti_sym_grad[x,y,z][j,i] = d f_j /dx_i - d f_i /dx_j
        anti_sym_grad = gradient_mat - np.transpose(gradient_mat, [0,1,2,4,3])
        anti_field = np.einsum('pi,xyzl->xyzlpi',DELTA, field)
        anti_field = anti_field - np.transpose(anti_field, [0,1,2,4,3,5])
        cross_curl = cross(field, curl_term)
        twist_zeroth_term = np.einsum('xyzi,xyzij->xyzj',cross_curl, anti_sym_grad)
        grad_curve_momentum_term = np.einsum('xyzi,xyzlpi->xyzpl', cross_curl, anti_field)
        grad_curve_momentum_term = self.gradient(grad_curve_momentum_term)
        grad_curve_momentum_term = np.einsum('xyziij->xyzj',grad_curve_momentum_term)
        cross_force = twist_zeroth_term - grad_curve_momentum_term

        # momentum_term = np.einsum('xyziij->xyzj', 1 * self.K22 * grad_dot_curl_term + 1 *self.K33 * grad_curve_momentum_term)
        div_term = self.K11 * div_force

        twist_term = self.K22 * twist_force
        cross_term = self.K33 * cross_force

        """
        Electric Field Term
        """
        E_field_term = -self.E0 * self.dE * mult(dot(self.E, field), self.E)
        
        return div_term + twist_term + cross_term + E_field_term 
    """
    apply the gradient
    """
    def update(self):
        grads = self.variational(self.field)
        self.field -= 1/self.gamma * grads * self.dt
        self.normalize()
        self.field = self.boundary_condition(self.field)
        self.t += self.dt
        return grads
    """
    keep the LC length
    """
    def normalize(self):
        length = norm_square(self.field)**0.5
        length = np.reshape(length, length.shape + (1,))
        self.field = self.field / length
    
    """
    plot the result
    """
    def render(self, save = None, plot = False):
        quiver3D(self.convert_data(), save, plot)

    """
    project the field
    """
    def slice(self, axis, scale = [3, 3]):
        mid = int(self.field.shape[axis]/2)
        ax1 = (axis + 1)%3
        ax2 = (axis + 2)%3
        ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
        x, y = np.meshgrid(np.arange(0, self.field.shape[ax1], scale[0]),np.arange(0, self.field.shape[ax2], scale[1]))
        x = x.T
        y = y.T
        if axis == 0:
            xy_prog = self.field[mid,::scale[0],::scale[1]]
        elif axis ==1:
            xy_prog = self.field[::scale[0],mid,::scale[1]]
        else:
            xy_prog = self.field[::scale[0],::scale[1],mid]
        u = xy_prog[:,:, ax1]
        v = xy_prog[:,:, ax2]
        x, y, u, v = x.flatten(), y.flatten(), u.flatten(), v.flatten()
        time = np.ones(x.shape) * self.t
        data = np.stack([time, x, y, u, v], -1)
        return data
    


    def animate_evolution(self, num_step, save = None):
        tracex = [self.slice(axis = 0, scale = [3,1])]
        tracey = [self.slice(axis = 1, scale = [3,1])]
        tracez = [self.slice(axis =2)]
        iterator = tqdm(range(num_step))
        step = num_step//50
        for t in iterator:
            if self.dt < 0.1 :
                for _ in range(0, int(0.1/self.dt)):
                    grads = self.update()
            else:
                grads = self.update()
            tracex.append(self.slice(axis = 0, scale = [3,1]))
            tracey.append(self.slice(axis = 1, scale = [3,1]))
            tracez.append(self.slice(axis =2))
            if t%step == 0:
                print('W:', self.W())
        self.i = 0
        def updatex():
            new_frame = tracex[self.i%num_step]
            self.i += 1
            # print(int(self.t/self.dt)%num_step)
            return new_frame
        def updatey():
            new_frame = tracey[self.i%num_step]
            self.i+= 1
            # print(int(self.t/self.dt)%num_step)
            return new_frame
        def updatez():
            new_frame = tracez[self.i%num_step]
            self.i += 1
            # print(int(self.t/self.dt)%num_step)
            return new_frame
        savex = None
        savey = None
        savez = None
        if save:
            savex = save + 'yz.mp4'
            savey = save + 'xz.mp4'
            savez = save + 'xy.mp4'
        render_anim(tracex[0], updatex, savex, show = False, num = num_step - 2,  dt = self.dt)
        self.i = 0
        render_anim(tracey[0], updatey, savey, show = False, num = num_step - 2,  dt = self.dt)
        self.i = 0
        render_anim(tracez[0], updatez, savez, show = False, num = num_step - 2,  dt = self.dt)


    
"""
time evolution
"""

def random_field(size):
    theta = np.random.uniform(size = size) * 2 * np.pi 
    phi = np.random.uniform(size = size) *  np.pi 
    field_z = np.cos(phi)
    field_x = np.sin(phi) * np.cos(theta) 
    field_y = np.sin(phi) * np.sin(theta) 
    field = np.stack([field_x, field_y, field_z], axis = -1)
    return field



if __name__ == '__main__':
    """
    define parameter here
    """
    pitch = 10e-6
    size = [112, 112, 32]
    q0 = 2*np.pi/ pitch * 0 
    dr = pitch / 32
    # K = [17.2e-12, 7.51e-12, 17.9e-12]
    K = [17.9e-12, 17.9e-12, 17.9e-12]
    gamma = 162
    field = random_field(size)
    E = np.zeros(field.shape)
    dt = None


    model = LC(field, gamma = gamma, E = E, dr = dr, K = K, q0 = q0, dt = dt)
    model.animate_evolution(500, save = './fig/noq0_')
    sys.stdout.write('\a')
    sys.stdout.flush()








