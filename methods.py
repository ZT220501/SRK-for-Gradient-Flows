import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from numpy import float64
import time

import utils



def get_method(method):
    if method == "RKL2_2D":
        return RKL2_2D
    elif method == "RKG2_2D":
        return RKG2_2D
    elif method == "RKG2_2D_CH":
        return RKG2_2D_CH
    elif method == "RKL2_2D_CH":
        return RKL2_2D_CH
    elif method == "RKL2_3D":
        return RKL2_3D
    elif method == "RKG2_3D":
        return RKG2_3D
    elif method == "RKL2_2D_irregular":
        return RKL2_2D_irregular
    elif method == "RKG2_2D_irregular":
        return RKG2_2D_irregular
    elif method == "RKL2_2D_CH_irregular":
        return RKL2_2D_CH_irregular
    elif method == "RKG2_2D_CH_irregular":
        return RKG2_2D_CH_irregular
    else:
        print("Method not found; check your spelling.")

##########################
# Regular domain solvers #
##########################
# RKL2 method for solving the 2D heat equation
# With Strang splitting, it can be used for solving the Allen-Cahn equation
# The parameters have been printed and checked in the solver
def RKL2_2D(u_previous, mesh_grid, dt, s, bc="Periodic"):
    # Solve for the diffusion termusing RKG2, with dt_used = dt
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(Y_j_1, mesh_grid, bc)
        else:
            mu = (2 * j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 1))
            nu = -(j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 2))
            mu_tilde = mu * 4 / (s**2 + s - 2)
            gamma_tilde = -mu_tilde * (1 - utils.b_coeff_RKL2(j-1))

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(Y_j_1, mesh_grid, bc) + gamma_tilde * dt * utils.laplacian_2D(u_previous, mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKG2 method for solving the 2D heat equation
# With Strang splitting, it can be used for solving the Allen-Cahn equation
# The parameters have been printed and checked in the solver
def RKG2_2D(u_previous, mesh_grid, dt, s, bc="Periodic"):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(Y_j_1, mesh_grid, bc)
        else:
            mu = (2 * j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 1))
            nu = -(j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * utils.b_coeff_RKG2(j-1)/ 2)

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(Y_j_1, mesh_grid, bc) + gamma_tilde * dt * utils.laplacian_2D(u_previous, mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

def RKL2_3D(u_previous, mesh_grid, dt, s, bc="Periodic"):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_3D(Y_j_1, mesh_grid, bc)
        else:
            mu = (2 * j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 1))
            nu = -(j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 2))
            mu_tilde = mu * 4 / (s**2 + s - 2)
            gamma_tilde = -mu_tilde * (1 - utils.b_coeff_RKL2(j-1))

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_3D(Y_j_1, mesh_grid, bc) + gamma_tilde * dt * utils.laplacian_3D(u_previous, mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

def RKG2_3D(u_previous, mesh_grid, dt, s, bc="Periodic"):
    # Solve for the diffusion termusing RKG2, with dt_used = dt
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_3D(Y_j_1, mesh_grid, bc)
        else:
            mu = (2 * j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 1))
            nu = -(j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * utils.b_coeff_RKG2(j-1)/ 2)

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_3D(Y_j_1, mesh_grid, bc) + gamma_tilde * dt * utils.laplacian_3D(u_previous, mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKL2 method for solving the 2D Cahn-Hilliard equation
# Three energy cases are implemented, depending on where to put the eps
# The parameters have been printed and checked in the solver
def RKL2_2D_CH(u_previous, mesh_grid, eps, dt, s, advective=False, bc="Periodic", energy_case=2):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    # Avoid mistakes when inputting the parameters
    if not isinstance(advective, bool):
        raise ValueError("advective parameter must be a boolean.")
    if not isinstance(bc, str):
        raise ValueError("boundary condition must be a string.")
    if not isinstance(energy_case, int):
        raise ValueError("energy_case must be an integer.")
    if energy_case not in [1, 2, 3]:
        raise ValueError("energy_case must be 1, 2, or 3.")

    # Choose the potential function
    if advective:
        potential = utils.potential_advective
    else:
        potential = utils.potential_non_advective
    
    # 2D Cahn-Hilliard
    # Three energy cases are implemented, depending on where to put the eps
    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            if energy_case == 1:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps**2 - utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 2:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps - eps * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 3:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) - eps**2 * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
        else:
            mu = (2 * j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 1))
            nu = -(j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 2))
            mu_tilde = mu * 4 / (s**2 + s - 2)
            gamma_tilde = -mu_tilde * (1 - utils.b_coeff_RKL2(j-1))

            if energy_case == 1:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps**2 - utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                    + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) / eps**2 - utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 2:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps - eps * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                    + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) / eps - eps * utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 3:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) - eps**2 * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                        + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) - eps**2 * utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKG2 method for solving the 2D Cahn-Hilliard equation
# Three energy cases are implemented, depending on where to put the eps
# The parameters have been printed and checked in the solver
def RKG2_2D_CH(u_previous, mesh_grid, eps, dt, s, advective=False, bc="Periodic", energy_case=2):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    # Avoid mistakes when inputting the parameters
    if not isinstance(advective, bool):
        raise ValueError("advective parameter must be a boolean.")
    if not isinstance(bc, str):
        raise ValueError("boundary condition must be a string.")
    if not isinstance(energy_case, int):
        raise ValueError("energy_case must be an integer.")
    if energy_case not in [1, 2, 3]:
        raise ValueError("energy_case must be 1, 2, or 3.")

    # Choose the potential function
    if advective:
        potential = utils.potential_advective
    else:
        potential = utils.potential_non_advective

    # 2D Cahn-Hilliard
    # Three energy cases are implemented, depending on where to put the eps
    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            if energy_case == 1:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps**2 - utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 2:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps - eps * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 3:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) - eps**2 * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)
        else:
            mu = (2 * j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 1))
            nu = -(j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * utils.b_coeff_RKG2(j-1)/ 2)

            if energy_case == 1:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps**2 - utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                    + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) / eps**2 - utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 2:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) / eps - eps * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                    + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) / eps - eps * utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)
            elif energy_case == 3:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D(potential(Y_j_1) - eps**2 * utils.laplacian_2D(Y_j_1, mesh_grid, bc), mesh_grid, bc)\
                        + gamma_tilde * dt * utils.laplacian_2D(potential(u_previous) - eps**2 * utils.laplacian_2D(u_previous, mesh_grid, bc), mesh_grid, bc)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

############################
# Irregular domain solvers #
############################
# RKL2 method for solving the 2D heat equation, in L-shaped domain
# With Strang splitting, it can be used for solving the Allen-Cahn equation
# The parameters have been printed and checked in the solver
def RKL2_2D_irregular(u_previous, mesh_grid, mask, dt, s, bc="Neumann"):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask)
        else:
            mu = (2 * j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 1))
            nu = -(j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 2))
            mu_tilde = mu * 4 / (s**2 + s - 2)
            gamma_tilde = -mu_tilde * (1 - utils.b_coeff_RKL2(j-1))

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask) + gamma_tilde * dt * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKG2 method for solving the 2D heat equation, in L-shaped domain
# With Strang splitting, it can be used for solving the Allen-Cahn equation
# The parameters have been printed and checked in the solver
def RKG2_2D_irregular(u_previous, mesh_grid, mask, dt, s, bc="Neumann"):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask)
        else:
            mu = (2 * j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 1))
            nu = -(j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * utils.b_coeff_RKG2(j-1)/ 2)

            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask) + gamma_tilde * dt * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKL2 method for solving the 2D Cahn-Hilliard equation, in L-shaped domain
# Three energy cases are implemented, depending on where to put the eps
# The parameters have been printed and checked in the solver
def RKL2_2D_CH_irregular(u_previous, mesh_grid, mask, eps, dt, s, bc="Periodic", energy_case=2):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    potential = utils.potential_non_advective

    if energy_case == 1:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) / eps**2 - utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)
    elif energy_case == 2:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) / eps - eps * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)
    elif energy_case == 3:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) - eps**2 * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)
    
    # 2D Cahn-Hilliard
    # Three energy cases are implemented, depending on where to put the eps
    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            if energy_case == 1:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps**2 - utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
            elif energy_case == 2:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps - eps * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
            elif energy_case == 3:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) - eps**2 * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
        else:
            mu = (2 * j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 1))
            nu = -(j - 1) * utils.b_coeff_RKL2(j) / (j * utils.b_coeff_RKL2(j - 2))
            mu_tilde = mu * 4 / (s**2 + s - 2)
            gamma_tilde = -mu_tilde * (1 - utils.b_coeff_RKL2(j-1))

            if energy_case == 1:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps**2 - utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                    + gamma_tilde * dt * Y_0_RHS
            elif energy_case == 2:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps - eps * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                    + gamma_tilde * dt * Y_0_RHS
            elif energy_case == 3:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) - eps**2 * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                        + gamma_tilde * dt * Y_0_RHS

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new

# RKG2 method for solving the 2D Cahn-Hilliard equation, in L-shaped domain
# Three energy cases are implemented, depending on where to put the eps
# The parameters have been printed and checked in the solver
def RKG2_2D_CH_irregular(u_previous, mesh_grid, mask, eps, dt, s, bc="Periodic", energy_case=1):
    Y_j_2 = u_previous
    Y_j_1 = u_previous
    Y_j = u_previous

    potential = utils.potential_non_advective

    if energy_case == 1:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) / eps**2 - utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)
    elif energy_case == 2:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) / eps - eps * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)
    elif energy_case == 3:
        Y_0_RHS = utils.laplacian_2D_L_shaped_domain(potential(u_previous) - eps**2 * utils.laplacian_2D_L_shaped_domain(u_previous, mesh_grid, bc, mask), mesh_grid, bc, mask)

    # 2D Cahn-Hilliard
    # Three energy cases are implemented, depending on where to put the eps
    for j in range(1, s + 1):
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            if energy_case == 1:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps**2 - utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
            elif energy_case == 2:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps - eps * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
            elif energy_case == 3:
                Y_j = Y_j_1 + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) - eps**2 * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)
        else:
            mu = (2 * j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 1))
            nu = -(j + 1) * utils.b_coeff_RKG2(j) / (j * utils.b_coeff_RKG2(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * utils.b_coeff_RKG2(j-1)/ 2)

            if energy_case == 1:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps**2 - utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                    + gamma_tilde * dt * Y_0_RHS
            elif energy_case == 2:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) / eps - eps * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                    + gamma_tilde * dt * Y_0_RHS
            elif energy_case == 3:
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * u_previous + mu_tilde * dt * utils.laplacian_2D_L_shaped_domain(potential(Y_j_1) - eps**2 * utils.laplacian_2D_L_shaped_domain(Y_j_1, mesh_grid, bc, mask), mesh_grid, bc, mask)\
                        + gamma_tilde * dt * Y_0_RHS

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    u_new = Y_j
    return u_new