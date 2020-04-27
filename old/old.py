# import tensorflow as tf
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
        self.dV = self.dr[0] * self.dr[1], self.dr[2]
        [self.K11, self.K22, self.K33] = K
        self.E = E
        self.q0 = q0
        self.E0 = E0
        self.dE = dE
        self.free_energies = []
        self.normalize()
        self.t = 0
        self.dt = dt if dt else  self.gamma *  min(self.dr[0], self.dr[1], self.dr[2])**2 / (2 * self.K33) 
        print('dt:', self.dt)

        print("Initiate model")
    
    """
    Assign Boundary Condition to the system
    """
    def boundary_condition(self, field, padding = False):
        if len(field.shape) == 4:
            if padding:
                field_pad = np.pad(field , [[0,1],[0,1],[0,1],[0,0]], "wrap")
            else:
                field_pad = field
            field_pad[:,:,0] = np.array([0,0,1])
            field_pad[:,:,-1] = np.array([0,0,1])
            field_pad[:,:,0] = np.array([0,0,1])
            field_pad[:,:,-1] = np.array([0,0,1])
            field_pad[:,:,0] = np.array([0,0,1])
            field_pad[:,:,-1] = np.array([0,0,1])

        elif len(field.shape) == 5:
            if padding:
                field_pad = np.pad(field , [[1,0],[1,0],[1,0],[0,0],[0,0]], "wrap")
                # field_pad[-1,:,:] = field_pad[0,:,:]
                # field_pad[:,-1,:] = field_pad[:,0,:]
            else:
                 field_pad = field
         
        return field_pad

    """
    Calculate jacobian fo the field 
    """
    def gradient(self, field):
        field_pad = self.boundary_condition(field, True)
        grad_x = (field_pad[1:,:-1,:-1] - field_pad[:-1,:-1,:-1])/self.dr[0]
        grad_y = (field_pad[:-1,1:,:-1] - field_pad[:-1,:-1,:-1])/self.dr[1]
        grad_z = (field_pad[:-1,:-1,1:] - field_pad[:-1,:-1,:-1])/self.dr[2]
        grad = np.stack([grad_x, grad_y, grad_z], axis = 3)
        return grad

    def laplacian(self, field):
        laplace = np.zeros(field.shape)
        laplace[1:-1, 1:-1, 1:-1] = - 6 * field[1:-1, 1:-1, 1:-1] \
            + field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] \
                + field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] \
                    + field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2]
        return laplace/dr[0]**2

    def W(self):
        gradient_mat = self.gradient(self.field)
        curl_term = curl(gradient_mat)
        return np.mean(self.K11/2 * div(gradient_mat)**2 )
            # + self.K22/2 * (dot(field, curl_term) + self.q0)**2 \
            # + self.K33/2 * norm_square(cross(field, curl_term)) \
            # - self.E0 * self.dE/2 * dot(self.E, field)**2)

    """
    Calculate the functional derivative
    """

    def variational(self, field):
        # gradient_mat = self.gradient(field)

        # # anti_sym_grad[x,y,z][i,j] = d f_j /dx_i - d f_i /dx_j
        # anti_sym_grad = gradient_mat - np.transpose(gradient_mat, [0,1,2,4,3])
        # curl_term = curl(gradient_mat)
        # dot_curl_term = dot(field, curl_term) + self.q0
        # # add1 to the end to make it (x, y, z, 3, 1) size
        # N_square = (field**2).reshape(field.shape + (1,))
        # # difference_matrix[x,y,z][i,j] = nj**2 - n_i**2
        # difference_matrix = np.transpose(N_square, [0,1,2,4,3]) - N_square

        # W_dn = self.K22 * mult(dot_curl_term,  curl_term) \
        #     + self.K33 * np.einsum('xyzi,xyzji->xyzi', field, anti_sym_grad**2)\
        #     - self.E0 * self.dE * mult(dot(self.E, field), self.E)
        # # print(np.mean(W_dn) * self.dt)
        
        # W_ddn = self.K11 * np.einsum('xyz,ij->xyzij',div(gradient_mat), DELTA)\
        #     + self.K22 * mult_mat(dot_curl_term, np.einsum('kij,xyzk->xyzij', EPSILON, field))\
        #     + self.K33 * difference_matrix * anti_sym_grad

        # dW_ddn = self.gradient(W_ddn)
        # print(np.mean(W_ddn) * self.dt)
        # print('\n\n\n')
        # deriv = W_dn - np.einsum('xyziij->xyzj', dW_ddn)
        laplace = self.K11 * self.laplacian(field)
        return -laplace
    """
    apply the gradient
    """
    def update(self):
        grads = self.variational(self.field)
        self.field -= 1/self.gamma * grads * self.dt
        self.field = self.boundary_condition(self.field)
        self.normalize()
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
        field = self.field[::step[0], ::step[1], 15::step[2]]
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
        for t in iterator:
            if self.dt < 0.1 :
                for _ in range(0, int(0.1/self.dt)):
                    grads = self.update()
            else:
                grads = self.update()
            if t%10==0:
                # id_ = np.argmax(grads)
                # idx = int(id_/(self.field.shape[0]))
                # idy = int((id_-idx * self.field.shape[0])/self.field.shape[1]) 
                # idz = idy - idy * self.field.shape[1]
                # print(idx, idy, idz)
                print(self.W())
            tracex.append(self.slice(0, scale = [3,1]))
            tracey.append(self.slice(1, scale = [3,1]))
            tracez.append(self.slice(2))
            # iterator.set_description("updating: {:.2f}".format(np.max(np.abs(grads * self.dt))))
        self.t = 0
        def updatex():
            new_frame = tracex[int(self.t/self.dt)%num_step]
            self.t += self.dt
            # print(int(self.t/self.dt)%num_step)
            return new_frame
        def updatey():
            new_frame = tracey[int(self.t/self.dt)%num_step]
            self.t += self.dt
            # print(int(self.t/self.dt)%num_step)
            return new_frame
        def updatez():
            new_frame = tracez[int(self.t/self.dt)%num_step]
            self.t += self.dt
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
        render_anim(tracey[0], updatey, savey, show = False, num = num_step - 2,  dt = self.dt)
        render_anim(tracez[0], updatez, savez, show = False, num = num_step - 2,  dt = self.dt)


    
"""
time evolution
"""

def random_field(size):
    theta = np.random.uniform(size = size) * 2 * np.pi 
    phi = np.random.uniform(size = size) *  np.pi * 0.1
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
    dr = [pitch / 32, pitch / 32, pitch / 32]
    K = [17.2e-12, 7.51e-12, 17.9e-12]
    gamma = 162
    field = random_field(size)
    E = np.zeros(field.shape)
    dt = 0.02


    model = LC(field, gamma = gamma, E = E, dr = dr, K = K, q0 = q0, dt = dt)
    # model.project('init', plot = True)
    # num_step = 100
    # evolve(model, 100)
    # model.project('final', plot = True)
    # model.render('./fig/final')
    # load('./fig/init')
    # load('./fig/final')
    # model.animate('~/Desktop/LC/skyrmion-replicate/python/sim.mp4', axis = 0, scale=[3,3], show = True)
    model.animate_evolution(100, save = './fig/laplace')
    sys.stdout.write('\a')
    sys.stdout.flush()








