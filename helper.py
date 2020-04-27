import numpy as np
"""
Global variable
"""
# Cronecker Delta
DELTA = np.eye(3, dtype=np.float32)
# Levi-Cevita Symbol
EPSILON = np.zeros((3, 3, 3))
EPSILON[0, 1, 2] = EPSILON[1, 2, 0] = EPSILON[2, 0, 1] = 1.
EPSILON[0, 2, 1] = EPSILON[2, 1, 0] = EPSILON[1, 0, 2] = -1.
"""
HELPER function
"""
# helper function
def grad(gradient_mat):
    return np.einsum('xyzii->xyzi', gradient_mat)

def div(gradient_mat):
    return np.einsum('xyzii->xyz', gradient_mat)

def curl(gradient_mat):
    curl_x = gradient_mat[:,:,:,1,2] - gradient_mat[:,:,:,2,1]
    curl_y = gradient_mat[:,:,:,2,0] - gradient_mat[:,:,:,0,2]
    curl_z = gradient_mat[:,:,:,0,1] - gradient_mat[:,:,:,1,0]
    return np.stack([curl_x, curl_y, curl_z], axis = -1)

def dot(u, v):
    return np.einsum('xyzi,xyzi->xyz', u, v)

def cross(u, v):
    cross_x = u[:,:,:,1]*v[:,:,:,2]  - u[:,:,:,2]*v[:,:,:,1]
    cross_y = u[:,:,:,2]*v[:,:,:,0]  - u[:,:,:,0]*v[:,:,:,2]
    cross_z = u[:,:,:,0]*v[:,:,:,1]  - u[:,:,:,1]*v[:,:,:,0]
    return np.stack([cross_x, cross_y, cross_z], axis = -1)

def mult(scalar_field, field):
    return np.einsum('xyz, xyzi->xyzi', scalar_field, field)
# scalar * matrix
def mult_mat(scalar_field, mat_field):
    return np.einsum('xyz, xyzij->xyzij', scalar_field, mat_field)

def norm_square(field):
    return dot(field, field)




