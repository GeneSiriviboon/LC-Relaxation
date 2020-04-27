import tensorflow as tf
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
        self.K = K
        self.E = E
        self.q0 = q0
        self.E0 = E0
        self.dE = dE
        self.normalize()
        self.t = 0
        self.dt = dt if dt else  self.gamma *  self.dr**2 / (2 * self.K) /5
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

    def W(self):
        gradient_mat = self.gradient(self.field)
        curl_term = curl(gradient_mat)

        return np.mean(self.K/2 * div(gradient_mat)**2 \
            + self.K/2 * (dot(field, curl_term) + self.q0)**2 \
            + self.K/2 * norm_square(cross(field, curl_term)) \
            - self.E0 * self.dE/2 * dot(self.E, field)**2)
    def W2(self):
        gradient_mat = self.gradient(self.field)
        curl_term = curl(gradient_mat)

        return np.mean(self.K/2 * div(gradient_mat)**2 \
            + self.K/2 * (dot(field, curl_term) + self.q0)**2 \
            + self.K/2 * norm_square(cross(field, curl_term)) \
            - self.E0 * self.dE/2 * dot(self.E, field)**2)

    """
    Calculate the functional derivative
    """



    def variational(self, field):
        gradient_mat = self.gradient(field)
        gradient_mat = self.boundary_derivative(gradient_mat)
       
        """
        div term
        """
        div_term =  - self.K * self.laplacian(field)

        """
        Twisted term
        """
        twisted = 2 * self.K * self.q0 * curl(gradient_mat)
        """
        Electric Field Term
        """
        E_field_term = -self.E0 * self.dE * mult(dot(self.E, field), self.E)
        
        return div_term + twisted + E_field_term 
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
    
    def project(self, save = None, plot = False):
        quiver2D([self.slice(2)], save, plot)

    def convert_data(self):
        step = [10, 10, 20]
        field = self.field[::step[0], ::step[1], ::step[2]]
        x, y, z = np.meshgrid(np.arange(field.shape[0]) ,
                            np.arange(field.shape[1]),
                            np.arange(field.shape[2]))

        
        u = field[:,:,:,0] 
        v = field[:,:,:,1]  
        w = field[:,:,:,2] 
        
        x, y, z, u, v, w = x.flatten(), y.flatten(), z.flatten(), u.flatten(), v.flatten(), w.flatten()
        time = self.t * np.ones(x.shape[0])

        return np.stack([time, x, y, z, u, v, w], axis = -1)
    def animate(self, save = None, axis = 2, scale = [1, 1], show = False):
        def update():
            self.update()
            return self.slice(axis, scale)
        render_anim(self.slice(axis, scale), update = update, save = save, dt = self.dt, show = show)

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
    q0 = 2*np.pi/ pitch * 0.5
    dr = pitch / 32
    K = 17.9e-12
    gamma = 162
    field = random_field(size)
    E = np.zeros(field.shape)
    dt = None


    model = LC(field, gamma = gamma, E = E, dr = dr, K = K, q0 = q0, dt = dt)
    # model.project('init', plot = True)
    # num_step = 100
    # evolve(model, 100)
    # model.project('final', plot = True)
    # model.render('./fig/final')
    # load('./fig/init')
    # load('./fig/final')
    # model.animate(None, axis = 0, scale=[3,3], show = True)
    model.animate_evolution(3000, save = './fig/simplify/20um_')
    sys.stdout.write('\a')
    sys.stdout.flush()








