import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import float64
# import pyvista as pv
from typing import Callable

from sys import exit



#######################
# Auxiliary functions #
#######################
# Double the center one half grid; used in adaptive mesh_refinement in 1D
def double_grid(arr):
    new_arr = np.zeros(len(arr) * 2 - 1)
    for i in range(len(new_arr)):
        if i % 2 == 0:
            new_arr[i] = arr[int(i/2)]
        else:
            new_arr[i] = (arr[int((i-1)/2)] + arr[int((i+1)/2)]) / 2
    return new_arr


# Halve the grid; used in adaptive_mesh_refinement in 2D for the coarsening problem
def half_grid(arr):
    print(arr.size)
    new_arr = np.zeros(arr.size // 2 + 1)
    for i in range(new_arr.size):
        new_arr[i] = arr[2 * i]
    return new_arr


def adaptive_mesh_coarsening(dt, mesh_grid, eps, s):
    '''
    Coarsen the mesh grid in the simulation of the Cahn-Hilliard coarsening
    We implement the grid coarsening by using the composite grid method.
    '''

    dx = ((12 * eps**1.5 * dt) / (s**2 + 3 * s - 4)) ** (1/4)
    dy = dx

    x_grid, y_grid = mesh_grid
    a_x = x_grid[0, 0]
    b_x = x_grid[-1, 0]
    a_y = y_grid[0, 0]
    b_y = y_grid[0, -1]

    Nx = int((b_x - a_x) / dx)
    Ny = int((b_y - a_y) / dy)

    x_grid_new = np.linspace(a_x, b_x, Nx+1)
    y_grid_new = np.linspace(a_y, b_y, Ny+1)

    mesh_grid_new = np.meshgrid(x_grid_new, y_grid_new)




    return mesh_grid_new
    
    





    # current_threshold = threshold

    # # Calculate the gradient norm square at the inner grid points
    # x_grid, y_grid = mesh_grid
    
    # # Store the partial derivatives
    # u_x = np.zeros(x_grid[1:-1, 1:-1].shape)
    # u_y = np.zeros(y_grid[1:-1, 1:-1].shape)        # u_x and u_y should have the same shape

    # # Calculate the partial derivatives and gradient norm square at the inner grid points
    # # Here we use the 3-point central difference scheme that has second order accuracy
    # h1_x = x_grid[1:-1, 1:-1] - x_grid[1:-1, :-2]
    # h2_x = x_grid[1:-1, 2:] - x_grid[1:-1, 1:-1]
    # u_x = (h2_x**2 * u[1:-1, :-2] + (h1_x**2 - h2_x**2) * u[1:-1, 1:-1] - h1_x**2 * u[1:-1, 2:]) / (h1_x * h2_x * (h1_x + h2_x))

    # h1_y = y_grid[1:-1, 1:-1] - y_grid[:-2, 1:-1]
    # h2_y = y_grid[2:, 1:-1] - y_grid[1:-1, 1:-1]
    # u_y = (h2_y**2 * u[:-2, 1:-1] + (h1_y**2 - h2_y**2) * u[1:-1, 1:-1] - h1_y**2 * u[2:, 1:-1]) / (h1_y * h2_y * (h1_y + h2_y))
    # u_gradient_magnitude = u_x**2 + u_y**2

    # gradient_energy = eps**2 * np.sum(u_gradient_magnitude) / 2
    # print("Summation of inner gradient energy is " + str(gradient_energy))



    # # If the threshold is not yet defined or the gradient energy is increased (at the beginning),
    # # update the threshold value
    # if gradient_energy / 4 >= threshold:
    #     current_threshold = gradient_energy / 4
    
    # # If the gradient energy is decreased below a threshold, coarsen the grid
    # if gradient_energy < threshold:
    #     # Update the mesh grid
    #     x_grid = half_grid(x_grid[0, :])
    #     y_grid = half_grid(y_grid[:, 0])
    #     mesh_grid = np.meshgrid(x_grid, y_grid)
        
    #     # Update the threshold
    #     current_threshold = current_threshold / 4

    #     # Update u
    #     u_new = u[::2, ::2]
    #     return u_new, mesh_grid, current_threshold

    # return u, mesh_grid, current_threshold



def b_coeff_RKL2(j):
    if j < 0:
        print("The b_j coefficient in the RKL method can't have j negative")
        return
    if j == 0 or j == 1 or j == 2:
        return 1 / 3

    return (j**2 + j - 2) / (2 * j * (j + 1))




def b_coeff_RKG2(j):
    if j < 0:
        print("The b_j coefficient in the RKG method can't have j negative")
        return
    if j == 0:
        return 1
    if j == 1:
        return 1 / 3
    

    return 4 * (j - 1) * (j + 4) / (3 * j * (j + 1) * (j + 2) * (j + 3))



######################
# Discrete operators #
######################
# Auxiliary function for the Engquist-Osher flux in the expanding flow case
def enquist_osher_flux_expanding(uL, uR, V0, face):
    vx = V0 * face
    return np.maximum(vx, 0) * uL + np.minimum(vx, 0) * uR


def flux_1D(u, x_grid, V0, vector_field, bc="Neumann"):
    '''
    Compute the flux for a general vector field \vec{V}
    '''
    if vector_field == "expanding":
        vec_field = V0 * x_grid
    elif type(vector_field) == Callable:
        vec_field = vector_field(x_grid)
    else:
        print("Wrong vector field; you need to require the vector field is expanding or a callable function.")

    x_faces = 0.5 * (x_grid[1:] + x_grid[:-1])
    flux = enquist_osher_flux_expanding(u[:-1], u[1:], V0, x_faces)

    return flux

    



# 1D approximation of the second derivative
def second_derivative(u, x_grid, bc="Periodic"):
    u_second_derivative = np.zeros(u.size, dtype=float64)

    if bc == "Dirichlet":
        # Homogeneous Dirichlet boundary condition
        u_second_derivative[0] = 0  
        u_second_derivative[-1] = 0
    elif bc == "Periodic":
        # Periodic boundary condition
        u_second_derivative[0] = (u[1] - 2 * u[0] + u[-2]) / (x_grid[1] - x_grid[0])**2
        u_second_derivative[-1] = (u[1] - 2 * u[0] + u[-2]) / (x_grid[1] - x_grid[0])**2
    elif bc == "Neumann":
        # Homogeneous Neumann boundary condition
        u_second_derivative[0] = 2 * (u[1] - u[0]) / (x_grid[1] - x_grid[0])**2
        u_second_derivative[-1] = 2 * (u[-2] - u[-1]) / (x_grid[-1] - x_grid[-2])**2
    else:
        print("Wrong boundary condition; check your spelling.")

    x_central = x_grid[1:-1]
    x_left = x_grid[:-2]
    x_right = x_grid[2:]
    u_second_derivative[1:-1] = (u[2:] * (x_central - x_left) + u[:-2] * (x_right - x_central) - u[1:-1] * (x_right - x_left)) / (0.5 * (x_right - x_left) * (x_right - x_central) * (x_central - x_left))

    return u_second_derivative

def laplacian_2D(u, mesh_grid, bc="Periodic"):
    '''
    Implement the 2D discrete Laplacian, under Dirichlet, Neumann, and periodic BC
    '''
    x_grid, y_grid = mesh_grid
    # Only consider the uniform grid case
    u_laplacian = np.zeros(u.shape, dtype=float64)
    h = x_grid[0, 1] - x_grid[0, 0]

    if bc == "Periodic":
        # Periodic boundary condition
        # Note: For periodic boundary condition, we assume that the initial condition is periodic, otherwise it fails.
        # If the initial condition is not periodic, consider using the Neumann boundary condition instead.
        # Edge case
        u_laplacian[0, 1:-1] = (u[1, 1:-1] - 2 * u[0, 1:-1] + u[-2, 1:-1]) / h**2 + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / h**2
        u_laplacian[-1, 1:-1] = (u[1, 1:-1] - 2 * u[-1, 1:-1] + u[-2, 1:-1]) / h**2 + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / h**2
        u_laplacian[1:-1, 0] = (u[1:-1, 1] - 2 * u[1:-1, 0] + u[1:-1, -2]) / h**2 + (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / h**2
        u_laplacian[1:-1, -1] = (u[1:-1, 1] - 2 * u[1:-1, -1] + u[1:-1, -2]) / h**2 + (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / h**2

        # Corner case
        u_laplacian[0, 0] = (u[1, 0] - 2 * u[0, 0] + u[-2, 0]) / h**2 + (u[0, 1] - 2 * u[0, 0] + u[0, -2]) / h**2
        u_laplacian[-1, 0] = (u[1, 0] - 2 * u[-1, 0] + u[-2, 0]) / h**2 + (u[-1, 1] - 2 * u[-1, 0] + u[-1, -2]) / h**2
        u_laplacian[0, -1] = (u[0, 1] - 2 * u[0, -1] + u[0, -2]) / h**2 + (u[1, -1] - 2 * u[0, -1] + u[-2, -1]) / h**2
        u_laplacian[-1, -1] = (u[-2, -1] - 2 * u[-1, -1] + u[1, -1]) / h**2 + (u[-1, -2] - 2 * u[-1, -1] + u[-1, 1]) / h**2
    elif bc == "Neumann":
        # (Homogeneous) Neumann boundary condition
        # Here we use the second order Neumann boundary condition approximation
        # Assume that near the boundary, the mesh grid is uniform
         u_laplacian[0, 1:-1] = (2 * u[1, 1:-1] + u[0, :-2] + u[0, 2:] - 4 * u[0, 1:-1]) / (y_grid[1, 1:-1] - y_grid[0, 1:-1])**2
         u_laplacian[-1, 1:-1] = (2 * u[-2, 1:-1] + u[-1, :-2] + u[-1, 2:] - 4 * u[-1, 1:-1]) / (y_grid[-1, 1:-1] - y_grid[-2, 1:-1])**2
         u_laplacian[1:-1, 0] =(2 * u[1:-1, 1] + u[:-2, 0] + u[2:, 0] - 4 * u[1:-1, 0]) / (x_grid[1:-1, 1] - x_grid[1:-1, 0])**2
         u_laplacian[1:-1, -1] = (2 * u[1:-1, -2] + u[:-2, -1] + u[2:, -1] - 4 * u[1:-1, -1]) / (x_grid[1:-1, -1] - x_grid[1:-1, -2])**2
 
         # Treat corner points for the Neumann boundary condition
         # We assume that the grid are uniform and the x&y direction spatial discretization are of the same size, for all corners
         u_laplacian[0, 0] = (2 * u[0, 1] + 2 * u[1, 0] - 4 * u[0, 0]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[0, -1] = (2 * u[0, -2] + 2 * u[1, -1] - 4 * u[0, -1]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[-1, 0] = (2 * u[-2, 0] + 2 * u[-1, 1] - 4 * u[-1, 0]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[-1, -1] = (2 * u[-2, -1] + 2 * u[-1, -2] - 4 * u[-1, -1]) / (y_grid[1, 0] - y_grid[0, 0])**2
    elif bc == "Dirichlet":
        # For Dirichlet boundary condition, all the boundary points are zero, so we don't need to do anything
        pass
    else:
        print("Wrong boundary condition; check your spelling.")

    x_central = x_grid[1:-1, 1:-1]
    x_left = x_grid[1:-1, :-2]
    x_right = x_grid[1:-1, 2:]

    y_central = y_grid[1:-1, 1:-1]
    y_left = y_grid[:-2, 1:-1]
    y_right = y_grid[2:, 1:-1]

    u_laplacian[1:-1, 1:-1] = 2 / ((x_right - x_central) * (x_central - x_left) * (x_right - x_left)) * (u[1:-1, 2:] * (x_central - x_left) - u[1:-1, 1:-1] * (x_right - x_left) + u[1:-1, :-2] * (x_right - x_central)) \
              + 2 / ((y_right - y_central) * (y_central - y_left) * (y_right - y_left)) * (u[2:, 1:-1] * (y_central - y_left) - u[1:-1, 1:-1] * (y_right - y_left) + u[:-2, 1:-1] * (y_right - y_central))

    return u_laplacian



def laplacian_3D(u, mesh_grid, bc="Periodic"):
    '''
    Implement the 3D discrete Laplacian using standard central differences
    '''
    x_grid, y_grid, z_grid = mesh_grid
    u_laplacian = np.zeros(u.shape, dtype=float64)
    h = x_grid[0, 1, 0] - x_grid[0, 0, 0]  # Assuming uniform grid spacing

    if bc == "Periodic":
        # Periodic boundary condition

        # Face points (excluding edges and corners)
        # x = 0 face
        u_laplacian[0, 1:-1, 1:-1] = (
            (u[1, 1:-1, 1:-1] - 2*u[0, 1:-1, 1:-1] + u[-2, 1:-1, 1:-1]) / h**2 +
            (u[0, 2:, 1:-1] - 2*u[0, 1:-1, 1:-1] + u[0, :-2, 1:-1]) / h**2 +
            (u[0, 1:-1, 2:] - 2*u[0, 1:-1, 1:-1] + u[0, 1:-1, :-2]) / h**2
        )
        # x = -1 face
        u_laplacian[-1, 1:-1, 1:-1] = (
            (u[1, 1:-1, 1:-1] - 2*u[-1, 1:-1, 1:-1] + u[-2, 1:-1, 1:-1]) / h**2 +
            (u[-1, 2:, 1:-1] - 2*u[-1, 1:-1, 1:-1] + u[-1, :-2, 1:-1]) / h**2 +
            (u[-1, 1:-1, 2:] - 2*u[-1, 1:-1, 1:-1] + u[-1, 1:-1, :-2]) / h**2
        )
        # y = 0 face
        u_laplacian[1:-1, 0, 1:-1] = (
            (u[2:, 0, 1:-1] - 2*u[1:-1, 0, 1:-1] + u[:-2, 0, 1:-1]) / h**2 +
            (u[1:-1, 1, 1:-1] - 2*u[1:-1, 0, 1:-1] + u[1:-1, -2, 1:-1]) / h**2 +
            (u[1:-1, 0, 2:] - 2*u[1:-1, 0, 1:-1] + u[1:-1, 0, :-2]) / h**2
        )
        # y = -1 face
        u_laplacian[1:-1, -1, 1:-1] = (
            (u[2:, -1, 1:-1] - 2*u[1:-1, -1, 1:-1] + u[:-2, -1, 1:-1]) / h**2 +
            (u[1:-1, 1, 1:-1] - 2*u[1:-1, -1, 1:-1] + u[1:-1, -2, 1:-1]) / h**2 +
            (u[1:-1, -1, 2:] - 2*u[1:-1, -1, 1:-1] + u[1:-1, -1, :-2]) / h**2
        )
        # z = 0 face
        u_laplacian[1:-1, 1:-1, 0] = (
            (u[2:, 1:-1, 0] - 2*u[1:-1, 1:-1, 0] + u[:-2, 1:-1, 0]) / h**2 +
            (u[1:-1, 2:, 0] - 2*u[1:-1, 1:-1, 0] + u[1:-1, :-2, 0]) / h**2 +
            (u[1:-1, 1:-1, 1] - 2*u[1:-1, 1:-1, 0] + u[1:-1, 1:-1, -2]) / h**2
        )
        # z = -1 face
        u_laplacian[1:-1, 1:-1, -1] = (
            (u[2:, 1:-1, -1] - 2*u[1:-1, 1:-1, -1] + u[:-2, 1:-1, -1]) / h**2 +
            (u[1:-1, 2:, -1] - 2*u[1:-1, 1:-1, -1] + u[1:-1, :-2, -1]) / h**2 +
            (u[1:-1, 1:-1, 1] - 2*u[1:-1, 1:-1, -1] + u[1:-1, 1:-1, -2]) / h**2
        )

        # Edge points (excluding corners)
        # x = 0, y = 0 edge
        u_laplacian[0, 0, 1:-1] = (
            (u[1, 0, 1:-1] - 2*u[0, 0, 1:-1] + u[-2, 0, 1:-1]) / h**2 +
            (u[0, 1, 1:-1] - 2*u[0, 0, 1:-1] + u[0, -2, 1:-1]) / h**2 +
            (u[0, 0, 2:] - 2*u[0, 0, 1:-1] + u[0, 0, :-2]) / h**2
        )
        # x = 0, y = -1 edge
        u_laplacian[0, -1, 1:-1] = (
            (u[1, -1, 1:-1] - 2*u[0, -1, 1:-1] + u[-2, -1, 1:-1]) / h**2 +
            (u[0, 1, 1:-1] - 2*u[0, -1, 1:-1] + u[0, -2, 1:-1]) / h**2 +
            (u[0, -1, 2:] - 2*u[0, -1, 1:-1] + u[0, -1, :-2]) / h**2
        )
        # x = -1, y = 0 edge
        u_laplacian[-1, 0, 1:-1] = (
            (u[1, 0, 1:-1] - 2*u[-1, 0, 1:-1] + u[-2, 0, 1:-1]) / h**2 +
            (u[-1, 1, 1:-1] - 2*u[-1, 0, 1:-1] + u[-1, -2, 1:-1]) / h**2 +
            (u[-1, 0, 2:] - 2*u[-1, 0, 1:-1] + u[-1, 0, :-2]) / h**2
        )
        # x = -1, y = -1 edge
        u_laplacian[-1, -1, 1:-1] = (
            (u[1, -1, 1:-1] - 2*u[-1, -1, 1:-1] + u[-2, -1, 1:-1]) / h**2 +
            (u[-1, 1, 1:-1] - 2*u[-1, -1, 1:-1] + u[-1, -2, 1:-1]) / h**2 +
            (u[-1, -1, 2:] - 2*u[-1, -1, 1:-1] + u[-1, -1, :-2]) / h**2
        )
        # x = 0, z = 0 edge
        u_laplacian[0, 1:-1, 0] = (
            (u[1, 1:-1, 0] - 2*u[0, 1:-1, 0] + u[-2, 1:-1, 0]) / h**2 +
            (u[0, 2:, 0] - 2*u[0, 1:-1, 0] + u[0, :-2, 0]) / h**2 +
            (u[0, 1:-1, 1] - 2*u[0, 1:-1, 0] + u[0, 1:-1, -2]) / h**2
        )
        # x = 0, z = -1 edge
        u_laplacian[0, 1:-1, -1] = (
            (u[1, 1:-1, -1] - 2*u[0, 1:-1, -1] + u[-2, 1:-1, -1]) / h**2 +
            (u[0, 2:, -1] - 2*u[0, 1:-1, -1] + u[0, :-2, -1]) / h**2 +
            (u[0, 1:-1, 1] - 2*u[0, 1:-1, -1] + u[0, 1:-1, -2]) / h**2
        )
        # x = -1, z = 0 edge
        u_laplacian[-1, 1:-1, 0] = (
            (u[1, 1:-1, 0] - 2*u[-1, 1:-1, 0] + u[-2, 1:-1, 0]) / h**2 +
            (u[-1, 2:, 0] - 2*u[-1, 1:-1, 0] + u[-1, :-2, 0]) / h**2 +
            (u[-1, 1:-1, 1] - 2*u[-1, 1:-1, 0] + u[-1, 1:-1, -2]) / h**2
        )
        # x = -1, z = -1 edge
        u_laplacian[-1, 1:-1, -1] = (
            (u[1, 1:-1, -1] - 2*u[-1, 1:-1, -1] + u[-2, 1:-1, -1]) / h**2 +
            (u[-1, 2:, -1] - 2*u[-1, 1:-1, -1] + u[-1, :-2, -1]) / h**2 +
            (u[-1, 1:-1, 1] - 2*u[-1, 1:-1, -1] + u[-1, 1:-1, -2]) / h**2
        )
        # y = 0, z = 0 edge
        u_laplacian[1:-1, 0, 0] = (
            (u[2:, 0, 0] - 2*u[1:-1, 0, 0] + u[:-2, 0, 0]) / h**2 +
            (u[1:-1, 1, 0] - 2*u[1:-1, 0, 0] + u[1:-1, -2, 0]) / h**2 +
            (u[1:-1, 0, 1] - 2*u[1:-1, 0, 0] + u[1:-1, 0, -2]) / h**2
        )
        # y = 0, z = -1 edge
        u_laplacian[1:-1, 0, -1] = (
            (u[2:, 0, -1] - 2*u[1:-1, 0, -1] + u[:-2, 0, -1]) / h**2 +
            (u[1:-1, 1, -1] - 2*u[1:-1, 0, -1] + u[1:-1, -2, -1]) / h**2 +
            (u[1:-1, 0, 1] - 2*u[1:-1, 0, -1] + u[1:-1, 0, -2]) / h**2
        )
        # y = -1, z = 0 edge
        u_laplacian[1:-1, -1, 0] = (
            (u[2:, -1, 0] - 2*u[1:-1, -1, 0] + u[:-2, -1, 0]) / h**2 +
            (u[1:-1, 1, 0] - 2*u[1:-1, -1, 0] + u[1:-1, -2, 0]) / h**2 +
            (u[1:-1, -1, 1] - 2*u[1:-1, -1, 0] + u[1:-1, -1, -2]) / h**2
        )
        # y = -1, z = -1 edge
        u_laplacian[1:-1, -1, -1] = (
            (u[2:, -1, -1] - 2*u[1:-1, -1, -1] + u[:-2, -1, -1]) / h**2 +
            (u[1:-1, 1, -1] - 2*u[1:-1, -1, -1] + u[1:-1, -2, -1]) / h**2 +
            (u[1:-1, -1, 1] - 2*u[1:-1, -1, -1] + u[1:-1, -1, -2]) / h**2
        )

        # Corner points
        # x = 0, y = 0, z = 0 corner
        u_laplacian[0, 0, 0] = (
            (u[1, 0, 0] - 2*u[0, 0, 0] + u[-2, 0, 0]) / h**2 +
            (u[0, 1, 0] - 2*u[0, 0, 0] + u[0, -2, 0]) / h**2 +
            (u[0, 0, 1] - 2*u[0, 0, 0] + u[0, 0, -2]) / h**2
        )
        # x = 0, y = 0, z = -1 corner
        u_laplacian[0, 0, -1] = (
            (u[1, 0, -1] - 2*u[0, 0, -1] + u[-2, 0, -1]) / h**2 +
            (u[0, 1, -1] - 2*u[0, 0, -1] + u[0, -2, -1]) / h**2 +
            (u[0, 0, 1] - 2*u[0, 0, -1] + u[0, 0, -2]) / h**2
        )
        # x = 0, y = -1, z = 0 corner
        u_laplacian[0, -1, 0] = (
            (u[1, -1, 0] - 2*u[0, -1, 0] + u[-2, -1, 0]) / h**2 +
            (u[0, 1, 0] - 2*u[0, -1, 0] + u[0, -2, 0]) / h**2 +
            (u[0, -1, 1] - 2*u[0, -1, 0] + u[0, -1, -2]) / h**2
        )
        # x = 0, y = -1, z = -1 corner
        u_laplacian[0, -1, -1] = (
            (u[1, -1, -1] - 2*u[0, -1, -1] + u[-2, -1, -1]) / h**2 +
            (u[0, 1, -1] - 2*u[0, -1, -1] + u[0, -2, -1]) / h**2 +
            (u[0, -1, 1] - 2*u[0, -1, -1] + u[0, -1, -2]) / h**2
        )
        # x = -1, y = 0, z = 0 corner
        u_laplacian[-1, 0, 0] = (
            (u[1, 0, 0] - 2*u[-1, 0, 0] + u[-2, 0, 0]) / h**2 +
            (u[-1, 1, 0] - 2*u[-1, 0, 0] + u[-1, -2, 0]) / h**2 +
            (u[-1, 0, 1] - 2*u[-1, 0, 0] + u[-1, 0, -2]) / h**2
        )
        # x = -1, y = 0, z = -1 corner
        u_laplacian[-1, 0, -1] = (
            (u[1, 0, -1] - 2*u[-1, 0, -1] + u[-2, 0, -1]) / h**2 +
            (u[-1, 1, -1] - 2*u[-1, 0, -1] + u[-1, -2, -1]) / h**2 +
            (u[-1, 0, 1] - 2*u[-1, 0, -1] + u[-1, 0, -2]) / h**2
        )
        # x = -1, y = -1, z = 0 corner
        u_laplacian[-1, -1, 0] = (
            (u[1, -1, 0] - 2*u[-1, -1, 0] + u[-2, -1, 0]) / h**2 +
            (u[-1, 1, 0] - 2*u[-1, -1, 0] + u[-1, -2, 0]) / h**2 +
            (u[-1, -1, 1] - 2*u[-1, -1, 0] + u[-1, -1, -2]) / h**2
        )
        # x = -1, y = -1, z = -1 corner
        u_laplacian[-1, -1, -1] = (
            (u[1, -1, -1] - 2*u[-1, -1, -1] + u[-2, -1, -1]) / h**2 +
            (u[-1, 1, -1] - 2*u[-1, -1, -1] + u[-1, -2, -1]) / h**2 +
            (u[-1, -1, 1] - 2*u[-1, -1, -1] + u[-1, -1, -2]) / h**2
        )

    elif bc == "Neumann":
        # For Neumann boundary conditions, we use ghost points
        # Ghost points have the same value as their corresponding interior points
        # This effectively enforces zero Neumann boundary conditions

        # Face points (excluding edges and corners)
        # x = 0 face (ghost point has same value as u[1,...])
        u_laplacian[0, 1:-1, 1:-1] = (
            (u[1, 1:-1, 1:-1] - 2*u[0, 1:-1, 1:-1] + u[1, 1:-1, 1:-1]) / h**2 +
            (u[0, 2:, 1:-1] - 2*u[0, 1:-1, 1:-1] + u[0, :-2, 1:-1]) / h**2 +
            (u[0, 1:-1, 2:] - 2*u[0, 1:-1, 1:-1] + u[0, 1:-1, :-2]) / h**2
        )
        # x = -1 face (ghost point has same value as u[-2,...])
        u_laplacian[-1, 1:-1, 1:-1] = (
            (u[-2, 1:-1, 1:-1] - 2*u[-1, 1:-1, 1:-1] + u[-2, 1:-1, 1:-1]) / h**2 +
            (u[-1, 2:, 1:-1] - 2*u[-1, 1:-1, 1:-1] + u[-1, :-2, 1:-1]) / h**2 +
            (u[-1, 1:-1, 2:] - 2*u[-1, 1:-1, 1:-1] + u[-1, 1:-1, :-2]) / h**2
        )
        # y = 0 face (ghost point has same value as u[...,1,...])
        u_laplacian[1:-1, 0, 1:-1] = (
            (u[2:, 0, 1:-1] - 2*u[1:-1, 0, 1:-1] + u[:-2, 0, 1:-1]) / h**2 +
            (u[1:-1, 1, 1:-1] - 2*u[1:-1, 0, 1:-1] + u[1:-1, 1, 1:-1]) / h**2 +
            (u[1:-1, 0, 2:] - 2*u[1:-1, 0, 1:-1] + u[1:-1, 0, :-2]) / h**2
        )
        # y = -1 face (ghost point has same value as u[...,-2,...])
        u_laplacian[1:-1, -1, 1:-1] = (
            (u[2:, -1, 1:-1] - 2*u[1:-1, -1, 1:-1] + u[:-2, -1, 1:-1]) / h**2 +
            (u[1:-1, -2, 1:-1] - 2*u[1:-1, -1, 1:-1] + u[1:-1, -2, 1:-1]) / h**2 +
            (u[1:-1, -1, 2:] - 2*u[1:-1, -1, 1:-1] + u[1:-1, -1, :-2]) / h**2
        )
        # z = 0 face (ghost point has same value as u[...,1])
        u_laplacian[1:-1, 1:-1, 0] = (
            (u[2:, 1:-1, 0] - 2*u[1:-1, 1:-1, 0] + u[:-2, 1:-1, 0]) / h**2 +
            (u[1:-1, 2:, 0] - 2*u[1:-1, 1:-1, 0] + u[1:-1, :-2, 0]) / h**2 +
            (u[1:-1, 1:-1, 1] - 2*u[1:-1, 1:-1, 0] + u[1:-1, 1:-1, 1]) / h**2
        )
        # z = -1 face (ghost point has same value as u[...,-2])
        u_laplacian[1:-1, 1:-1, -1] = (
            (u[2:, 1:-1, -1] - 2*u[1:-1, 1:-1, -1] + u[:-2, 1:-1, -1]) / h**2 +
            (u[1:-1, 2:, -1] - 2*u[1:-1, 1:-1, -1] + u[1:-1, :-2, -1]) / h**2 +
            (u[1:-1, 1:-1, -2] - 2*u[1:-1, 1:-1, -1] + u[1:-1, 1:-1, -2]) / h**2
        )

        # Edge points (excluding corners)
        # x = 0, y = 0 edge
        u_laplacian[0, 0, 1:-1] = (
            (u[1, 0, 1:-1] - 2*u[0, 0, 1:-1] + u[1, 0, 1:-1]) / h**2 +
            (u[0, 1, 1:-1] - 2*u[0, 0, 1:-1] + u[0, 1, 1:-1]) / h**2 +
            (u[0, 0, 2:] - 2*u[0, 0, 1:-1] + u[0, 0, :-2]) / h**2
        )
        # x = 0, y = -1 edge
        u_laplacian[0, -1, 1:-1] = (
            (u[1, -1, 1:-1] - 2*u[0, -1, 1:-1] + u[1, -1, 1:-1]) / h**2 +
            (u[0, -2, 1:-1] - 2*u[0, -1, 1:-1] + u[0, -2, 1:-1]) / h**2 +
            (u[0, -1, 2:] - 2*u[0, -1, 1:-1] + u[0, -1, :-2]) / h**2
        )
        # x = -1, y = 0 edge
        u_laplacian[-1, 0, 1:-1] = (
            (u[-2, 0, 1:-1] - 2*u[-1, 0, 1:-1] + u[-2, 0, 1:-1]) / h**2 +
            (u[-1, 1, 1:-1] - 2*u[-1, 0, 1:-1] + u[-1, 1, 1:-1]) / h**2 +
            (u[-1, 0, 2:] - 2*u[-1, 0, 1:-1] + u[-1, 0, :-2]) / h**2
        )
        # x = -1, y = -1 edge
        u_laplacian[-1, -1, 1:-1] = (
            (u[-2, -1, 1:-1] - 2*u[-1, -1, 1:-1] + u[-2, -1, 1:-1]) / h**2 +
            (u[-1, -2, 1:-1] - 2*u[-1, -1, 1:-1] + u[-1, -2, 1:-1]) / h**2 +
            (u[-1, -1, 2:] - 2*u[-1, -1, 1:-1] + u[-1, -1, :-2]) / h**2
        )
        # x = 0, z = 0 edge
        u_laplacian[0, 1:-1, 0] = (
            (u[1, 1:-1, 0] - 2*u[0, 1:-1, 0] + u[1, 1:-1, 0]) / h**2 +
            (u[0, 2:, 0] - 2*u[0, 1:-1, 0] + u[0, :-2, 0]) / h**2 +
            (u[0, 1:-1, 1] - 2*u[0, 1:-1, 0] + u[0, 1:-1, 1]) / h**2
        )
        # x = 0, z = -1 edge
        u_laplacian[0, 1:-1, -1] = (
            (u[1, 1:-1, -1] - 2*u[0, 1:-1, -1] + u[1, 1:-1, -1]) / h**2 +
            (u[0, 2:, -1] - 2*u[0, 1:-1, -1] + u[0, :-2, -1]) / h**2 +
            (u[0, 1:-1, -2] - 2*u[0, 1:-1, -1] + u[0, 1:-1, -2]) / h**2
        )
        # x = -1, z = 0 edge
        u_laplacian[-1, 1:-1, 0] = (
            (u[-2, 1:-1, 0] - 2*u[-1, 1:-1, 0] + u[-2, 1:-1, 0]) / h**2 +
            (u[-1, 2:, 0] - 2*u[-1, 1:-1, 0] + u[-1, :-2, 0]) / h**2 +
            (u[-1, 1:-1, 1] - 2*u[-1, 1:-1, 0] + u[-1, 1:-1, 1]) / h**2
        )
        # x = -1, z = -1 edge
        u_laplacian[-1, 1:-1, -1] = (
            (u[-2, 1:-1, -1] - 2*u[-1, 1:-1, -1] + u[-2, 1:-1, -1]) / h**2 +
            (u[-1, 2:, -1] - 2*u[-1, 1:-1, -1] + u[-1, :-2, -1]) / h**2 +
            (u[-1, 1:-1, -2] - 2*u[-1, 1:-1, -1] + u[-1, 1:-1, -2]) / h**2
        )
        # y = 0, z = 0 edge
        u_laplacian[1:-1, 0, 0] = (
            (u[2:, 0, 0] - 2*u[1:-1, 0, 0] + u[:-2, 0, 0]) / h**2 +
            (u[1:-1, 1, 0] - 2*u[1:-1, 0, 0] + u[1:-1, 1, 0]) / h**2 +
            (u[1:-1, 0, 1] - 2*u[1:-1, 0, 0] + u[1:-1, 0, 1]) / h**2
        )
        # y = 0, z = -1 edge
        u_laplacian[1:-1, 0, -1] = (
            (u[2:, 0, -1] - 2*u[1:-1, 0, -1] + u[:-2, 0, -1]) / h**2 +
            (u[1:-1, 1, -1] - 2*u[1:-1, 0, -1] + u[1:-1, 1, -1]) / h**2 +
            (u[1:-1, 0, -2] - 2*u[1:-1, 0, -1] + u[1:-1, 0, -2]) / h**2
        )
        # y = -1, z = 0 edge
        u_laplacian[1:-1, -1, 0] = (
            (u[2:, -1, 0] - 2*u[1:-1, -1, 0] + u[:-2, -1, 0]) / h**2 +
            (u[1:-1, -2, 0] - 2*u[1:-1, -1, 0] + u[1:-1, -2, 0]) / h**2 +
            (u[1:-1, -1, 1] - 2*u[1:-1, -1, 0] + u[1:-1, -1, 1]) / h**2
        )
        # y = -1, z = -1 edge
        u_laplacian[1:-1, -1, -1] = (
            (u[2:, -1, -1] - 2*u[1:-1, -1, -1] + u[:-2, -1, -1]) / h**2 +
            (u[1:-1, -2, -1] - 2*u[1:-1, -1, -1] + u[1:-1, -2, -1]) / h**2 +
            (u[1:-1, -1, -2] - 2*u[1:-1, -1, -1] + u[1:-1, -1, -2]) / h**2
        )

        # Corner points
        # x = 0, y = 0, z = 0 corner
        u_laplacian[0, 0, 0] = (
            (u[1, 0, 0] - 2*u[0, 0, 0] + u[1, 0, 0]) / h**2 +
            (u[0, 1, 0] - 2*u[0, 0, 0] + u[0, 1, 0]) / h**2 +
            (u[0, 0, 1] - 2*u[0, 0, 0] + u[0, 0, 1]) / h**2
        )
        # x = 0, y = 0, z = -1 corner
        u_laplacian[0, 0, -1] = (
            (u[1, 0, -1] - 2*u[0, 0, -1] + u[1, 0, -1]) / h**2 +
            (u[0, 1, -1] - 2*u[0, 0, -1] + u[0, 1, -1]) / h**2 +
            (u[0, 0, -2] - 2*u[0, 0, -1] + u[0, 0, -2]) / h**2
        )
        # x = 0, y = -1, z = 0 corner
        u_laplacian[0, -1, 0] = (
            (u[1, -1, 0] - 2*u[0, -1, 0] + u[1, -1, 0]) / h**2 +
            (u[0, -2, 0] - 2*u[0, -1, 0] + u[0, -2, 0]) / h**2 +
            (u[0, -1, 1] - 2*u[0, -1, 0] + u[0, -1, 1]) / h**2
        )
        # x = 0, y = -1, z = -1 corner
        u_laplacian[0, -1, -1] = (
            (u[1, -1, -1] - 2*u[0, -1, -1] + u[1, -1, -1]) / h**2 +
            (u[0, -2, -1] - 2*u[0, -1, -1] + u[0, -2, -1]) / h**2 +
            (u[0, -1, -2] - 2*u[0, -1, -1] + u[0, -1, -2]) / h**2
        )
        # x = -1, y = 0, z = 0 corner
        u_laplacian[-1, 0, 0] = (
            (u[-2, 0, 0] - 2*u[-1, 0, 0] + u[-2, 0, 0]) / h**2 +
            (u[-1, 1, 0] - 2*u[-1, 0, 0] + u[-1, 1, 0]) / h**2 +
            (u[-1, 0, 1] - 2*u[-1, 0, 0] + u[-1, 0, 1]) / h**2
        )
        # x = -1, y = 0, z = -1 corner
        u_laplacian[-1, 0, -1] = (
            (u[-2, 0, -1] - 2*u[-1, 0, -1] + u[-2, 0, -1]) / h**2 +
            (u[-1, 1, -1] - 2*u[-1, 0, -1] + u[-1, 1, -1]) / h**2 +
            (u[-1, 0, -2] - 2*u[-1, 0, -1] + u[-1, 0, -2]) / h**2
        )
        # x = -1, y = -1, z = 0 corner
        u_laplacian[-1, -1, 0] = (
            (u[-2, -1, 0] - 2*u[-1, -1, 0] + u[-2, -1, 0]) / h**2 +
            (u[-1, -2, 0] - 2*u[-1, -1, 0] + u[-1, -2, 0]) / h**2 +
            (u[-1, -1, 1] - 2*u[-1, -1, 0] + u[-1, -1, 1]) / h**2
        )
        # x = -1, y = -1, z = -1 corner
        u_laplacian[-1, -1, -1] = (
            (u[-2, -1, -1] - 2*u[-1, -1, -1] + u[-2, -1, -1]) / h**2 +
            (u[-1, -2, -1] - 2*u[-1, -1, -1] + u[-1, -2, -1]) / h**2 +
            (u[-1, -1, -2] - 2*u[-1, -1, -1] + u[-1, -1, -2]) / h**2
        )

    else:
        print("Wrong boundary condition; check your spelling.")

    # Interior points
    u_laplacian[1:-1, 1:-1, 1:-1] = (
        (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / h**2 +
        (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / h**2 +
        (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / h**2
    )


    return u_laplacian



'''
Potential functions for the Allen-Cahn and Cahn-Hilliard equation
'''
# Potential function for the usual Allen-Cahn and Cahn-Hilliard equation
def potential_non_advective(u):
    return u**3 - u

# Potential function for the advective Allen-Cahn and Cahn-Hilliard equation
def potential_advective(u):
    return 2 * u * (1 - u) * (1 - 2 * u)










'''
Calculate the exact Ginzburg-Landau energy for 2D simulations
'''
# This method has been checked and implemented correctly
def energy_exact(u, mesh_grid, eps, bc="Periodic", case=1):
    x_grid, y_grid = mesh_grid
    h = x_grid[0, 1] - x_grid[0, 0]

    # Store the partial derivatives
    u_x = np.zeros(x_grid.shape)
    u_y = np.zeros(y_grid.shape)        # u_x and u_y should have the same shape

    # Calculate the partial derivatives and gradient norm square at the inner grid points
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * h)
    u_y[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * h)

    # Calculate the partial derivatives and gradient norm square at the boundary non-corner grid points
    if bc == "Periodic":
        u_x[:, 0] = (u[:, 1] - u[:, -2]) / (2 * h)
        u_x[:, -1] = u_x[:, 0]

        u_y[0, :] = (u[1, :] - u[-2, :]) / (2 * h)
        u_y[-1, :] = u_y[0, :]

        # boundary_energy = np.sum(u_x[:, 0]**2) + np.sum(u_x[:, -1]**2) + np.sum(u_x[0, :]**2) + np.sum(u_x[-1, :]**2)\
        #             + np.sum(u_y[0, :]**2) + np.sum(u_y[-1, :]**2) + np.sum(u_y[:, 0]**2) + np.sum(u_y[:, -1]**2)
        # print("Boundary energy: " + str(boundary_energy * eps**2 / 2))
    elif bc == "Neumann":
        # We apply the ghost point treatment here, so that all the gradients are zero
        # Hence nothing needs to be done
        pass

    # Calculate the gradient square
    u_gradient_square = u_x**2 + u_y**2
    # Calculate the total energy
    if case == 1:
        energy_gradient_entrywise = 1 / 2 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2 / eps**2
    elif case == 2:
        energy_gradient_entrywise = eps / 2 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2 / eps
    elif case == 3:
        energy_gradient_entrywise = eps**2 / 2 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2
    else:
        print("Wrong energy case; check your spelling.")
    # Calculate the gradient and non-linear energy
    energy_gradient = np.sum(energy_gradient_entrywise)
    energy_non_linear = np.sum(energy_non_linear_entrywise)

    return energy_gradient, energy_non_linear

'''
Calculate the exact Ginzburg-Landau energy for 3D simulations
'''
def energy_exact_3D(u, mesh_grid, eps, bc="Periodic"):
    x, y, z = mesh_grid

    dx = x[0, 1, 0] - x[0, 0, 0]
    dy = y[1, 0, 0] - y[0, 0, 0]
    dz = z[0, 0, 1] - z[0, 0, 0]
    
    # Store the partial derivatives
    u_x = np.zeros(x.shape)
    u_y = np.zeros(y.shape)
    u_z = np.zeros(z.shape)

    # Calculate the partial derivatives and gradient norm square at the inner grid points
    u_x[:, 1:-1, 1:-1] = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    u_y[1:-1, :, 1:-1] = (u[2:, :, 1:-1] - u[:-2, :, 1:-1]) / (2 * dy)
    u_z[1:-1, 1:-1, :] = (u[2:, 1:-1, :] - u[:-2, 1:-1, :]) / (2 * dz)

    # Calculate the partial derivatives and gradient norm square at the boundary non-corner grid points and corners
    if bc == "Periodic":
        u_x[:, 0, 1:-1] = (u[:, 1, 1:-1] - u[:, -2, 1:-1]) / (2 * dx)
        u_x[:, -1, 1:-1] = u_x[:, 0, 1:-1]

        u_y[0, :, 1:-1] = (u[1, :, 1:-1] - u[-2, :, 1:-1]) / (2 * dy)
        u_y[-1, :, 1:-1] = u_y[0, :, 1:-1]

        u_z[0, 1:-1, :] = (u[1, 1:-1, :] - u[-2, 1:-1, :]) / (2 * dz)
        u_z[-1, 1:-1, :] = u_z[0, 1:-1, :]
    elif bc == "Neumann":
        # We apply the ghost point treatment here, so that all the gradients are zero
        # Hence nothing needs to be done
        pass
    
    # Calculate the gradient norm square
    u_gradient_square = u_x**2 + u_y**2 + u_z**2
    # Calculate the total energy
    energy_gradient_entrywise = 1 / 2 * u_gradient_square
    energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2 / eps**2
    # Calculate the gradient and non-linear energy
    energy_gradient = np.sum(energy_gradient_entrywise)
    energy_non_linear = np.sum(energy_non_linear_entrywise)

    return energy_gradient, energy_non_linear

# This method has beend checked and implemented correctly
# NOTE: The energy modified is only defined for the 2D case, energy case 3.
def energy_modified(SNu, SLSNu, dt):
    energy_modified_nonlinear_entrywise = 0.25 + 1 / (2 * dt) * SLSNu**2 - np.exp(dt) / (dt * np.expm1(2 * dt)) * (np.sqrt(1 + np.expm1(2 * dt) * SLSNu**2) - 1)
    energy_modified_nonlinear = np.sum(energy_modified_nonlinear_entrywise)

    SNu = SNu.reshape(-1)
    SLSNu = SLSNu.reshape(-1)
    energy_modified_gradient = 1 / (2 * dt) * np.dot(SNu-SLSNu, SLSNu)

    return energy_modified_gradient, energy_modified_nonlinear

# Anime for Cahn-Hilliard coarsening, etc.
def generate_anime(u_total, t_total, extent, save=False, name=None):
    fig, ax = plt.subplots()
    # Change the colormap upper and lower limits
    cax = ax.imshow(u_total[0], extent=extent, cmap='viridis', origin='lower', vmin=-1, vmax=1)
    fig.colorbar(cax)

    def update(frame):
        cax.set_data(u_total[frame])
        ax.set_title(f'Current time: {round(t_total[frame], 5)}')
        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=len(u_total), blit=True)

    # Save as GIF
    if save:
        if name is None:
            ani.save('Natural Timestepping Cahn-Hilliard Coarsening T=' + str(t_total[-1]) + '.gif', writer='pillow', fps=10)
        else:
            ani.save(name + '.gif', writer='pillow', fps=10)

def volume_render(u, mesh_grid, t=0, save=False):
    x, y, z = mesh_grid

    # Create a PyVista grid
    grid = pv.StructuredGrid(x, y, z)
    grid["values"] = u.flatten(order="F")  # PyVista expects Fortran order

    # Volume rendering
    plotter = pv.Plotter(title="3D Volume Rendering")
    plotter.add_text("Cahn-Hilliard Coarsening at time " + str(t), position="upper_edge", font_size=14, color="black")
    plotter.add_volume(grid, scalars="values", cmap="viridis", opacity="sigmoid")
    plotter.add_axes()



    # Save a snapshot as png
    if save:
        plotter.save_graphic('volume_render.pdf')


    plotter.show()


# L-shaped domain for Allen-Cahn and Cahn-Hilliard equations
def L_shaped_domain(resolution, a_x, b_x, a_y, b_y):
    x = np.linspace(a_x, b_x, resolution+1)
    y = np.linspace(a_y, b_y, resolution+1)
    X, Y = np.meshgrid(x, y)
    mask = ~((X > 0.5 * (b_x - a_x)) & (Y > 0.5 * (b_y - a_y)))
    return X, Y, mask


def laplacian_2D_L_shaped_domain(u, mesh_grid, bc, mask):
    """
    Discrete 2D Laplacian on an embedded L-shaped domain specified by `mask`.

    Parameters
    ----------
    u : (ny, nx) array
        Field on the full rectangular grid. Values outside mask are ignored.
    mesh_grid : (X, Y)
        Meshgrid arrays.
    mask : (ny, nx) bool array
        True for points inside L-shaped domain.
    bc : {"Neumann", "Dirichlet"}
        Boundary condition applied on the L-shaped boundary (mask boundary).
        - Neumann is homogeneous (du/dn = 0).
        - Dirichlet uses value g (default 0).
    g : float or callable
        Dirichlet boundary value. If callable: g(x, y, side) -> value
        side in {"E","W","N","S"} indicating which neighbor was outside.

    Returns
    -------
    Lu : (ny, nx) array
        Laplacian values on mask; 0 outside mask.
    """
    X, Y = mesh_grid
    u = np.asarray(u, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    ny, nx = u.shape
    hx = float(X[0, 1] - X[0, 0])
    hy = float(Y[1, 0] - Y[0, 0])

    bc = bc.lower()
    if bc not in ("neumann", "dirichlet"):
        raise ValueError("bc must be 'Neumann' or 'Dirichlet'")

    Lu = np.zeros_like(u, dtype=np.float64)

    # --- shift masks for neighbor existence inside the embedded domain ---
    # East neighbor exists for cells where mask[i,j] and mask[i,j+1]
    mC = mask
    mE = np.zeros_like(mask); mE[:, :-1] = mask[:, 1:]
    mW = np.zeros_like(mask); mW[:, 1:]  = mask[:, :-1]
    mN = np.zeros_like(mask); mN[:-1, :] = mask[1:, :]
    mS = np.zeros_like(mask); mS[1:,  :] = mask[:-1, :]

    # --- shift u for interior neighbor values (undefined where neighbor missing, but masked out later) ---
    uE_in = np.zeros_like(u); uE_in[:, :-1] = u[:, 1:]
    uW_in = np.zeros_like(u); uW_in[:, 1:]  = u[:, :-1]
    uN_in = np.zeros_like(u); uN_in[:-1, :] = u[1:, :]
    uS_in = np.zeros_like(u); uS_in[1:,  :] = u[:-1, :]

    # Start with something (will overwrite everywhere mask==True)
    uC = u

    if bc == "dirichlet":
        # ghost for missing neighbor: -uC
        uE = np.where(mE, uE_in, -uC)
        uW = np.where(mW, uW_in, -uC)
        uN = np.where(mN, uN_in, -uC)
        uS = np.where(mS, uS_in, -uC)

    else:  # neumann (homogeneous), match your mirror + fallback behavior
        # Opposite neighbor existence/value
        # If East missing, use West if exists else uC
        uE = np.where(mE, uE_in, np.where(mW, uW_in, uC))
        # If West missing, use East if exists else uC
        uW = np.where(mW, uW_in, np.where(mE, uE_in, uC))
        # If North missing, use South if exists else uC
        uN = np.where(mN, uN_in, np.where(mS, uS_in, uC))
        # If South missing, use North if exists else uC
        uS = np.where(mS, uS_in, np.where(mN, uN_in, uC))

    # Compute Laplacian everywhere, then zero outside mask
    Lu_full = (uE - 2.0 * uC + uW) / (hx * hx) + (uN - 2.0 * uC + uS) / (hy * hy)
    Lu[mC] = Lu_full[mC]
    # Lu already 0 outside mask

    return Lu


def energy_exact_irregular_domain(u, mesh_grid, eps, mask, bc="Neumann", case=1):
    """
    Compute Ginzburg-Landau energy on an L-shaped (irregular) domain.
    
    Parameters
    ----------
    u : (ny, nx) array
        Field on the full rectangular grid. Values outside mask are ignored.
    mesh_grid : (X, Y)
        Meshgrid arrays.
    eps : float
        Interface width parameter.
    mask : (ny, nx) bool array
        True for points inside the L-shaped domain.
    bc : {"Neumann", "Dirichlet"}
        Boundary condition on the irregular boundary.
    case : int
        Energy scaling case (1, 2, or 3).
    
    Returns
    -------
    energy_gradient : float
        Gradient energy contribution.
    energy_non_linear : float
        Double-well potential energy contribution.
    """
    X, Y = mesh_grid
    u = np.asarray(u, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    
    ny, nx = u.shape
    hx = X[0, 1] - X[0, 0]
    hy = Y[1, 0] - Y[0, 0]
    
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)
    
    def in_mask(ii, jj):
        return (0 <= ii < ny) and (0 <= jj < nx) and mask[ii, jj]
    
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            
            uC = u[i, j]
            
            # East neighbor (j+1)
            if in_mask(i, j + 1):
                uE = u[i, j + 1]
            else:
                if bc == "Dirichlet":
                    uE = -uC
                else:  # Neumann: ghost point reflects interior
                    if in_mask(i, j - 1):
                        uE = u[i, j - 1]
                    else:
                        uE = uC
            
            # West neighbor (j-1)
            if in_mask(i, j - 1):
                uW = u[i, j - 1]
            else:
                if bc == "Dirichlet":
                    uW = -uC
                else:
                    if in_mask(i, j + 1):
                        uW = u[i, j + 1]
                    else:
                        uW = uC
            
            # North neighbor (i+1)
            if in_mask(i + 1, j):
                uN = u[i + 1, j]
            else:
                if bc == "Dirichlet":
                    uN = -uC
                else:
                    if in_mask(i - 1, j):
                        uN = u[i - 1, j]
                    else:
                        uN = uC
            
            # South neighbor (i-1)
            if in_mask(i - 1, j):
                uS = u[i - 1, j]
            else:
                if bc == "Dirichlet":
                    uS = -uC
                else:
                    if in_mask(i + 1, j):
                        uS = u[i + 1, j]
                    else:
                        uS = uC
            
            # Central difference for gradients
            u_x[i, j] = (uE - uW) / (2 * hx)
            u_y[i, j] = (uN - uS) / (2 * hy)
    
    # Gradient squared (only inside mask)
    u_gradient_square = u_x**2 + u_y**2
    
    # Compute energy densities based on case
    if case == 1:
        energy_gradient_entrywise = 0.5 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2 / eps**2
    elif case == 2:
        energy_gradient_entrywise = eps / 2 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2 / eps
    elif case == 3:
        energy_gradient_entrywise = eps**2 / 2 * u_gradient_square
        energy_non_linear_entrywise = 0.25 * (u**2 - 1)**2
    else:
        raise ValueError("Invalid energy case; must be 1, 2, or 3.")
    
    # Sum only over points inside the mask
    energy_gradient = np.sum(energy_gradient_entrywise[mask])
    energy_non_linear = np.sum(energy_non_linear_entrywise[mask])

    return energy_gradient, energy_non_linear