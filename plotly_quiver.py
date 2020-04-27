import plotly
import plotly.io as pio
import plotly.graph_objs as go
# import chart_studio.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import copy
import pandas as pd
import numpy as np
from numpy import pi, sin, cos, sqrt

def quiver3D(field):

     x, y, z = np.meshgrid(np.arange(field.shape[0]) ,
                         np.arange(field.shape[1]),
                         np.arange(field.shape[2]))


     u = field[:,:,:,0] 
     v = field[:,:,:,1]  
     w = field[:,:,:,2] 

     angle = np.arctan(v/(u+1e-5))/np.pi * 2

     # u = u*angle
     # v = v*angle
     # w = w*angle

     x, y, z, u, v, w = x.flatten(), y.flatten(), z.flatten(), u.flatten(), v.flatten(), w.flatten()

     # pl_deep = [[0.0, 'rgb(39, 26, 44)'],
     #           [0.1, 'rgb(53, 41, 74)'],
     #           [0.2, 'rgb(63, 57, 108)'],
     #           [0.3, 'rgb(64, 77, 139)'],
     #           [0.4, 'rgb(61, 99, 148)'],
     #           [0.5, 'rgb(65, 121, 153)'],
     #           [0.6, 'rgb(72, 142, 157)'],
     #           [0.7, 'rgb(80, 164, 162)'],
     #           [0.8, 'rgb(92, 185, 163)'],
     #           [0.9, 'rgb(121, 206, 162)'],
     #           [1.0, 'rgb(165, 222, 166)']]

     pl_deep = 'RdBu'

     trace2 = dict(type='cone',
               x=x,
               y=y,
               z=z,
               u=u,
               v=v,
               w=w,
               sizemode='absolute',
               sizeref=0.25, #this is the default value 
               showscale=True,
               color = angle.flatten(),
               colorscale=pl_deep,
               colorbar=dict(thickness=20, ticklen=4), 
               anchor='cm'
               )
     fig2 = dict(data=[trace2])
     title = 'Vertical levels of a numerical weather model'
     pio.show(fig2, filename='vertical_levels',validate=False)

if __name__=="__main__":
     # x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
     #                     np.arange(-0.8, 1, 0.2),
     #                     np.arange(-0.8, 1, 0.8))
     # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
     # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
     # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     #      np.sin(np.pi * z))
     # field = np.stack([u,v,w], axis = -1)

     u = np.array([0,0,1])
     v = np.array([0,1,1])

     field = np.array([[[u,v,v, -v], [u,v,v, -v], [u,v,v, -v]],
               [[v,v,u, -v], [v,u,v, -v], [v,v,v, -v]],
               [[v,v,-u, v], [v,v,u, v], [u, u, u, v]]])
     quiver3D(field)

