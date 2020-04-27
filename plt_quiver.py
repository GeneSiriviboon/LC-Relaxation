import plotly
import plotly.io as pio
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import axes3d
import pickle
import copy
import pandas as pd
import numpy as np
from numpy import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

def quiver2D(projections, save = None, plot = True):
    num_proj = len(projections)
    fig, axs = plt.subplots(num_proj, 1, figsize = [6, 6])
    if num_proj==1:
        data = projections[0]
        x = data[:,1] 
        y = data[:,2] 
        u = data[:,3] 
        v = data[:,4]  
        c = 1 - u**2 + v**2
        axs.quiver(x, y, u, v, c, pivot ='mid') 
        axs.set(adjustable='box', aspect='equal')
    else:
        for i, ax in enumerate(axs):
            data = projections[i]
            x = data[:,1] 
            y = data[:,2] 
            u = data[:,3] 
            v = data[:,4]  
            c = 1 - u**2 + v**2
            ax.quiver(x, y, u, v, c, pivot ='mid') 
            ax.set(adjustable='box', aspect='equal')
    fig.tight_layout(pad = 1.0)
    if save:
        plt.savefig(save + '.png', dpi = 200)
    if plot:
        plt.show()

def quiver3D(data, save, plot):
    LINE_WIDTH = 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')


    x = data[:,1] 
    y = data[:,2] 
    z = data[:,3] 
    u = data[:,4] 
    v = data[:,5]  
    w = data[:,6] 

    
    r = np.abs(u)
    g = np.abs(v)
    b = np.abs(w)
    a = np.ones(u.shape)
    colors = np.stack([r, g, b, a], axis = -1)
    ax.quiver(x, y, z, u, v, w, linewidths=LINE_WIDTH, pivot = 'middle', colors=colors, arrow_length_ratio = 0.0, length = 0.5, normalize = True)
    if save:
        pickle.dump(fig, open(save + '-FigureObject.fig.pickle', 'wb'))
    if plot:
        plt.show()

def load(name):
    figx = pickle.load(open(name+'-FigureObject.fig.pickle', 'rb'))

    figx.show() # Show the figure, edit it, etc.!
    plt.show()

if __name__ == '__main__':
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    u = np.random.random(x.shape)
    v = np.random.random(y.shape)
    x, y, u, v = x.flatten(), y.flatten(), u.flatten(), v.flatten()
    t = np.ones(x.shape)
    data = np.stack([t, x, y, u, v], axis =  -1)
    # print(data.shape)
    quiver2D([data, data])