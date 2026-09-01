import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from numpy import float64
import time
from tqdm.auto import tqdm
import logging
import os
from datetime import datetime

import utils
import methods
from methods import get_method
from typing import Callable
import IPython.display as display
from skimage.measure import marching_cubes


'''
The base class for all the solvers.
TODO: Make other classes inherit from this class.
'''
class Solver():
    def __init__(self, method_name:str, s:int, N:int, dt:float, extents:list, initial_condition:Callable, T:float, eps:float):
        self.method_name = method_name
        self.method = get_method(self.method_name)

        self.s = s
        self.N = N                                      # For now, we assume that Nx=Ny=Nz
        self.dt = dt
        self.extents = extents
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps

        self.running_time = 0
        self.iter_num = 0
        

        # Check for the validity of the domain specification
        if len(extents) % 2 != 0:
            raise ValueError("The domain must be defined with left and right ends.")
        


    def initialize(self, display=False):
        if len(self.extents) == 2:
            # Here although the name is mesh_grid, it is actually a 1D grid, i.e. a np array
            self.mesh_grid = np.linspace(self.extents[0], self.extents[1], self.N+1)
        elif len(self.extents) == 4:
            x_grid = np.linspace(self.extents[0], self.extents[1], self.N+1)
            y_grid = np.linspace(self.extents[2], self.extents[3], self.N+1)
            self.mesh_grid = np.meshgrid(x_grid, y_grid)
        elif len(self.extents) == 6:
            x_grid = np.linspace(self.extents[0], self.extents[1], self.N+1)
            y_grid = np.linspace(self.extents[2], self.extents[3], self.N+1)
            z_grid = np.linspace(self.extents[4], self.extents[5], self.N+1)
            self.mesh_grid = np.meshgrid(x_grid, y_grid, z_grid)
        else:
            print("Simulations with more than 3 dimensions are not supported yet.")
            return
        
        self.t_total = np.linspace(0, self.T, int(self.T / self.dt) + 1)
        self.u_total = [self.initial_condition(self.mesh_grid)]

        if display:
            self.display(self.u_total[0], self.mesh_grid, 0)

    def solve(self):
        start_time = time.time()
        t_current = 0

        while t_current <= self.T:
            # Update the current time
            t_current += self.dt

            self.iter_num += 1
            print("This is iteration " + str(self.iter_num))

            u_previous = self.u_total[-1]
            # Solve the equation for one step
            u_new = self.step(u_previous)
            self.u_total.append(u_new)

        end_time = time.time()
        self.running_time = end_time - start_time

    def step(self, u_previous):
        print("The step function is not implemented for this solver. Need to write a solver function!")

    def display(self, u, x, t):
        print("The display function is not implemented for this solver. Need to write a visualization function!")
    
    def get_results(self):
        return self.u_total, self.t_total, self.x_grid

#####################################
# Allen-Cahn equation related solvers
#####################################
class Allen_Cahn_Solver_2D_Constant_Timestep():

    def __init__(self, method, s, Nx, Ny, dt, dx, dy, a_x, b_x, a_y, b_y, initial_condition, T, eps, bc, energy_case=1, plot_frequency=100, splitting_order="NLN", save=False, save_path=None):
        self.method_name = method                    # Name of the method; is supposed to be int
        self.method = get_method(self.method_name)

        self.s = s
        self.Nx = Nx
        self.Ny = Ny

        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps

        self.bc = bc                # "Neumann" or "Periodic"
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.running_time = 0
        self.iter_num = 0

        self.splitting_order = splitting_order
        self.save = save
        self.save_path = save_path

    def initialize(self, display=False):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)

        self.mesh_grid = np.meshgrid(x_grid, y_grid)

        self.t_total = [0]
        self.u_total = [self.initial_condition(self.mesh_grid)]
        energy_gradient_new, energy_non_linear_new = utils.energy_exact(self.u_total[0], self.mesh_grid, self.eps, self.bc, self.energy_case)
        self.energy_total = [[energy_gradient_new, energy_non_linear_new]]
        self.modified_energy_total = [[energy_gradient_new, energy_non_linear_new]]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=0)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        # Calculate total number of iterations for progress bar
        total_iterations = round((self.T - self.t_current) / self.dt)

        for i in tqdm(range(total_iterations+1), desc="Solving", unit="iter", ncols=100, leave=True):

            if self.iter_num % self.plot_frequency == 0 or self.iter_num == total_iterations:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=self.save, save_path=self.save_path)
                print("Max value of u_total[-1] is " + str(np.max(self.u_total[-1])))
                print("Min value of u_total[-1] is " + str(np.min(self.u_total[-1])))

            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact(u_new, self.mesh_grid, self.eps, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                # break
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])

        end_time = time.time()
        self.running_time = end_time - start_time

    def step(self, u_previous):
        '''
        One step of solving using the Strang splitting
        '''
        # Splitting order: NLN (Nonlinear-Linear-Nonlinear) or LNL (Linear-Nonlinear-Linear)
        if self.splitting_order == "NLN":
            u_intermediate1 = self.nonlinear_update(u_previous, self.dt / 2, energy_case=self.energy_case)
            # Diffusion part update
            if self.energy_case == 1:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.dt, self.s, self.bc)
            elif self.energy_case == 2:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.eps * self.dt, self.s, self.bc)
            elif self.energy_case == 3:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.eps**2 * self.dt, self.s, self.bc)
            else:
                raise ValueError("Energy case must be 1, 2, or 3.")
            # Nonlinear part update
            u_new = self.nonlinear_update(u_intermediate2, self.dt / 2, energy_case=self.energy_case)
        # TODO: Change the splitting of LNL to the correct corresponding energy cases
        elif self.splitting_order == "LNL":
            u_intermediate1 = self.method(u_previous, self.mesh_grid, self.eps**2 * self.dt / 2, self.s, self.bc)
            # Diffusion part update
            u_intermediate2 = self.nonlinear_update(u_intermediate1, self.dt)
            # Nonlinear part update
            u_new = self.method(u_intermediate2, self.mesh_grid, self.eps**2 * self.dt / 2, self.s, self.bc)
        else:
            raise ValueError("Splitting order must be either 'NLN' or 'LNL'.")
        
        # Calculate the modified energy
        energy_modified_gradient_new, energy_modified_nonlinear_new = utils.energy_modified(u_intermediate1, u_intermediate2, self.dt)
        energy_modified_new = energy_modified_gradient_new + energy_modified_nonlinear_new
        # Check for monotonicity of the energy
        if self.modified_energy_total != [] and energy_modified_new > sum(self.modified_energy_total[-1]):
            print("Modified energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
            print("Modified energy is " + str(energy_modified_new) + " and the previous modified energy is " + str(sum(self.modified_energy_total[-1])))
            # break
        if np.isnan(energy_modified_new):
            print("Modified energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
        self.modified_energy_total.append([energy_modified_gradient_new, energy_modified_nonlinear_new])
        return u_new

    # Solve the non-linear part analytically
    def nonlinear_update(self, u_previous, dt, energy_case=1):
        if energy_case == 1:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps**2))
        elif energy_case == 2:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps))
        elif energy_case == 3:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt))
        else:
            raise ValueError("Energy case must be 1, 2, or 3.")
        return u_new
    
    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), np.array(self.modified_energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None):
        if not save:
            plt.figure(figsize=(5, 4))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Allen-Cahn coarsening at time ' + str(round(t, 5)))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Allen_Cahn_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

class Allen_Cahn_Solver_2D_Adaptive_Timestep():

    def __init__(self, method, s, Nx, Ny, dt, dx, dy, 
    a_x, b_x, a_y, b_y, initial_condition, T, eps, bc, 
    energy_case=1, plot_frequency=100, splitting_order="NLN", 
    save=False, save_path=None, adaptive_timestep_tolerance=1e-1):
        self.method_name = method                    # Name of the method; is supposed to be string
        self.method = get_method(self.method_name)

        self.s = s
        self.Nx = Nx
        self.Ny = Ny

        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps

        self.bc = bc                # "Neumann" or "Periodic"
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.running_time = 0
        self.iter_num = 0

        self.splitting_order = splitting_order
        self.save = save
        self.save_path = save_path

        self.adaptive_timestep_tolerance = adaptive_timestep_tolerance
        self.adaptive_timestep_count = 0            # Count until 3 to update the timestep

    def initialize(self, display=False):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)

        self.mesh_grid = np.meshgrid(x_grid, y_grid)

        self.t_total = [0]
        self.u_total = [self.initial_condition(self.mesh_grid)]
        energy_gradient_new, energy_non_linear_new = utils.energy_exact(self.u_total[0], self.mesh_grid, self.eps, self.bc, self.energy_case)
        self.energy_total = [[energy_gradient_new, energy_non_linear_new]]
        self.modified_energy_total = [[energy_gradient_new, energy_non_linear_new]]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=0)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        while self.t_current <= self.T:

            # print("Current dt is " + str(self.dt) + ", current s is " + str(self.s))
            if self.iter_num % self.plot_frequency == 0:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=self.save, save_path=self.save_path)
                print("Max value of u_total[-1] is " + str(np.max(self.u_total[-1])))
                print("Min value of u_total[-1] is " + str(np.min(self.u_total[-1])))

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact(u_new, self.mesh_grid, self.eps, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                self.dt = self.dt * 0.8
                self.substep_s_update(case="min")
                continue
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])
            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)
            print("Length of u_total is " + str(len(self.u_total)))
            print("Length of t_total is " + str(len(self.t_total)))

            # Adaptive timestep procedure
            if self.iter_num >= 2:
                # Define the necessary quantities for the adaptive timestep procedure
                # Mask to avoid division by zero where u_total[-2] or u_total[-3] are (near) zero
                e_k_plus_1 = np.max(np.abs(self.u_total[-1] - self.u_total[-2])) / np.max(np.abs(self.u_total[-2]))
                e_k = np.max(np.abs(self.u_total[-2] - self.u_total[-3])) / np.max(np.abs(self.u_total[-3]))
                dt_current = self.t_total[-1] - self.t_total[-2]
                dt_previous = self.t_total[-2] - self.t_total[-3]
                # Calculate the dimensionless local truncation error
                LTE = np.abs(e_k_plus_1 - (dt_current / dt_previous) * e_k)
                print("LTE is " + str(LTE))
                # Check if the LTE is within the tolerance; if yes, increment the count; 
                # if the count is greater than 3, update the timestep and reset the count.
                if LTE < self.adaptive_timestep_tolerance:
                    self.adaptive_timestep_count += 1
                    if self.adaptive_timestep_count >= 3:
                        self.dt = self.dt * 1.2
                        self.adaptive_timestep_count = 0
                        # Update the number of substeps in the super-time-stepping method accordingly
                        self.substep_s_update(case="max")
        end_time = time.time()
        self.running_time = end_time - start_time
    
    def substep_s_update(self, case="max"):
        if self.method_name == "RKL2_2D":
            if self.energy_case == 1:
                self.s_new = (-1 + np.sqrt(9 + 64 * self.dt / self.dx**2)) / 2
            elif self.energy_case == 2:
                self.s_new = (-1 + np.sqrt(9 + 64 * self.dt * self.eps / self.dx**2)) / 2
            elif self.energy_case == 3:
                self.s_new = (-1 + np.sqrt(9 + 64 * self.dt * self.eps**2 / self.dx**2)) / 2
        elif self.method_name == "RKG2_2D":
            if self.energy_case == 1:
                self.s_new = (-3 + np.sqrt(25 + 96 * self.dt / self.dx**2)) / 2
            elif self.energy_case == 2:
                self.s_new = (-3 + np.sqrt(25 + 96 * self.dt * self.eps / self.dx**2)) / 2
            elif self.energy_case == 3:
                self.s_new = (-3 + np.sqrt(25 + 96 * self.dt * self.eps**2 / self.dx**2)) / 2
        else:
            raise ValueError("Method name must be 'RKL2_2D' or 'RKG2_2D'. Check the spelling and the class implementation!")
        
        self.s_new = math.ceil(self.s_new)
        if case == "max":
            self.s = max(self.s, self.s_new)
        elif case == "min":
            self.s = min(self.s, self.s_new)
        else:
            raise ValueError("Case must be 'max' or 'min'. Check the spelling and the class implementation!")

    def step(self, u_previous):
        '''
        One step of solving using the Strang splitting
        '''
        # Splitting order: NLN (Nonlinear-Linear-Nonlinear) or LNL (Linear-Nonlinear-Linear)
        if self.splitting_order == "NLN":
            u_intermediate1 = self.nonlinear_update(u_previous, self.dt / 2, energy_case=self.energy_case)
            # Diffusion part update
            if self.energy_case == 1:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.dt, self.s, self.bc)
            elif self.energy_case == 2:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.eps * self.dt, self.s, self.bc)
            elif self.energy_case == 3:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.eps**2 * self.dt, self.s, self.bc)
            else:
                raise ValueError("Energy case must be 1, 2, or 3.")
            # Nonlinear part update
            u_new = self.nonlinear_update(u_intermediate2, self.dt / 2, energy_case=self.energy_case)
        # TODO: Change the splitting of LNL to the correct corresponding energy cases
        elif self.splitting_order == "LNL":
            u_intermediate1 = self.method(u_previous, self.mesh_grid, self.eps**2 * self.dt / 2, self.s, self.bc)
            # Diffusion part update
            u_intermediate2 = self.nonlinear_update(u_intermediate1, self.dt)
            # Nonlinear part update
            u_new = self.method(u_intermediate2, self.mesh_grid, self.eps**2 * self.dt / 2, self.s, self.bc)
        else:
            raise ValueError("Splitting order must be either 'NLN' or 'LNL'.")
        
        # Calculate the modified energy
        energy_modified_gradient_new, energy_modified_nonlinear_new = utils.energy_modified(u_intermediate1, u_intermediate2, self.dt)
        energy_modified_new = energy_modified_gradient_new + energy_modified_nonlinear_new
        # Check for monotonicity of the energy
        if self.modified_energy_total != [] and energy_modified_new > sum(self.modified_energy_total[-1]):
            print("Modified energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
            print("Modified energy is " + str(energy_modified_new) + " and the previous modified energy is " + str(sum(self.modified_energy_total[-1])))
            # break
        if np.isnan(energy_modified_new):
            print("Modified energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
        self.modified_energy_total.append([energy_modified_gradient_new, energy_modified_nonlinear_new])
        return u_new

    # Solve the non-linear part analytically
    def nonlinear_update(self, u_previous, dt, energy_case=1):
        if energy_case == 1:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps**2))
        elif energy_case == 2:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps))
        elif energy_case == 3:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt))
        else:
            raise ValueError("Energy case must be 1, 2, or 3.")
        return u_new
    
    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), np.array(self.modified_energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None):
        if not save:
            plt.figure(figsize=(5, 4))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Allen-Cahn coarsening at time ' + str(round(t, 5)))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Allen_Cahn_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

class Allen_Cahn_Solver_3D_Constant_Timestep():

    def __init__(self, method, s, Nx, Ny, Nz, dt, dx, dy, dz, a_x, b_x, a_y, b_y, a_z, b_z, initial_condition, T, eps, bc, plot_frequency=1, splitting_order="NLN", save=False, save_path=None):
        self.method_name = method                    # Name of the method; is supposed to be string
        self.method = get_method(self.method_name)

        self.s = s
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.a_z = a_z
        self.b_z = b_z
        
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps
        self.bc = bc

        self.running_time = 0
        self.iter_num = 0

        self.plot_frequency = plot_frequency
        self.splitting_order = splitting_order

        self.save = save
        self.save_path = save_path

    def initialize(self, display):
        self.x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        self.y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)
        self.z_grid = np.linspace(self.a_z, self.b_z, self.Nz+1)

        self.mesh_grid = np.meshgrid(self.x_grid, self.y_grid, self.z_grid)

        self.t_total = [0]
        self.u_total = [self.initial_condition(self.mesh_grid, self.eps)]
        self.energy_total = [utils.energy_exact_3D(self.u_total[0], self.mesh_grid, self.eps, bc=self.bc)]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], t=self.t_current)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        # Calculate total number of iterations for progress bar
        total_iterations = round((self.T - self.t_current) / self.dt)

        for i in tqdm(range(total_iterations + 1), desc="Solving", unit="iter", ncols=100, leave=True):

            if self.iter_num % self.plot_frequency == 0 or self.iter_num == total_iterations:
                self.display(self.u_total[-1], t=self.t_current)

            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact_3D(u_new, self.mesh_grid, self.eps, bc=self.bc)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                # break
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])

        end_time = time.time()
        self.running_time = end_time - start_time

    def step(self, u_previous):
        '''
        One step of solving using the Strang splitting
        '''
        # Splitting order: NLN (Nonlinear-Linear-Nonlinear) or LNL (Linear-Nonlinear-Linear)
        if self.splitting_order == "NLN":
            u_intermediate1 = self.nonlinear_update(u_previous, self.dt / 2)
            # Diffusion part update
            u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.dt, self.s, bc=self.bc)
            # Nonlinear part update
            u_new = self.nonlinear_update(u_intermediate2, self.dt / 2)
        elif self.splitting_order == "LNL":
            u_intermediate1 = self.method(u_previous, self.mesh_grid, self.dt / 2, self.s, bc=self.bc)
            # Diffusion part update
            u_intermediate2 = self.nonlinear_update(u_intermediate1, self.dt)
            # Nonlinear part update
            u_new = self.method(u_intermediate2, self.mesh_grid, self.dt / 2, self.s, bc=self.bc)
        else:
            raise ValueError("Splitting order must be either 'NLN' or 'LNL'.")
        return u_new
    
    # Solve the non-linear part analytically
    def nonlinear_update(self, u_previous, dt):
        u_new = u_previous / np.sqrt(u_previous**2 - (u_previous**2 - 1) * np.exp(-2 * dt / self.eps**2))
        return u_new
    
    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), self.mesh_grid
        # return self.u_total, self.t_total, self.mesh_grid
    
    def display(self, u, t):
        level = 0.0
        verts, faces, _, _ = marching_cubes(u, level=level)

        # Convert voxel coords -> physical coords
        sx = (self.x_grid.max() - self.x_grid.min()) / (self.Nx - 1)
        sy = (self.y_grid.max() - self.y_grid.min()) / (self.Ny - 1)
        sz = (self.z_grid.max() - self.z_grid.min()) / (self.Nz - 1)

        # Notice that here u is arranged as (y, x, z), so that the column stack should be the following
        verts_phys = np.column_stack([
            self.x_grid.min() + verts[:, 1] * sx,
            self.y_grid.min() + verts[:, 0] * sy,
            self.z_grid.min() + verts[:, 2] * sz
        ])

        # --- Plot ---
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(
            verts_phys[:, 0], verts_phys[:, 1], faces, verts_phys[:, 2],
            linewidth=0.0, antialiased=True, shade=True
        )

        ax.set_xlim(self.a_x, self.b_x)
        ax.set_ylim(self.a_y, self.b_y)
        ax.set_zlim(self.a_z, self.b_z)
        ax.set_box_aspect((self.b_x - self.a_x, self.b_y - self.a_y, self.b_z - self.a_z))
        ax.view_init(elev=15, azim=45)  # Angle for the plot
        ax.set_axis_off()

        plt.tight_layout()
        if not self.save:
            plt.title("3D Allen-Cahn Dumbbell RKG isosurface at time " + str(round(t, 5)))
        else:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(self.save_path + f"/3D_Allen_Cahn_Dumbbell_isosurface_time_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches="tight")
        plt.show()

########################################
# Cahn-Hilliard equation related solvers
########################################
class Cahn_Hilliard_Solver_2D_Constant_Timestep():

    def __init__(self, method, s, Nx, Ny, dt, dx, dy, a_x, b_x, a_y, b_y, initial_condition, T, eps, bc, energy_case=3, plot_frequency=100, **kwargs):
        self.method_name = method                    # Name of the method; is supposed to be string
        self.method = get_method(self.method_name)

        self.s = s
        self.Nx = Nx
        self.Ny = Ny

        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps
        self.bc = bc
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.save = kwargs.get("save", False)
        self.save_path = kwargs.get("save_path", "Results\Cahn-Hilliard Coarsening\Constent_timestep")
        self.test_case = kwargs.get("test_case", None)

        self.running_time = 0
        self.iter_num = 0

    def initialize(self, display):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)

        self.mesh_grid = np.meshgrid(x_grid, y_grid)

        self.t_total = [0]
        self.u_total = [self.initial_condition(self.mesh_grid)]
        self.energy_total = [utils.energy_exact(self.u_total[0], self.mesh_grid, self.eps, self.bc, self.energy_case)]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=False)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        # Calculate total number of iterations for progress bar
        total_iterations = round((self.T - self.t_current) / self.dt)

        for i in tqdm(range(total_iterations + 1), desc="Solving", unit="iter", ncols=100, leave=True):

            if self.iter_num % self.plot_frequency == 0 or self.iter_num == total_iterations:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=self.save, save_path=self.save_path, test_case=self.test_case)
                print("The max of u_total[-1] is " + str(np.max(self.u_total[-1])))
                print("The min of u_total[-1] is " + str(np.min(self.u_total[-1])))
                
            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact(u_new, self.mesh_grid, self.eps, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                # break
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])


        end_time = time.time()
        self.running_time = end_time - start_time


    def step(self, u_previous):
        '''
        One step of solving using the RKG method
        '''
        u_new = self.method(u_previous, self.mesh_grid, self.eps, self.dt, self.s, advective=False, bc=self.bc, energy_case=self.energy_case)
        return u_new

    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None, test_case=None):
        if not save:
            plt.figure(figsize=(5, 4))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Cahn-Hilliard coarsening at time ' + str(round(t, 5)))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Cahn_Hilliard_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

class Cahn_Hilliard_Solver_2D_Adaptive_Timestep():

    def __init__(self, method, s, Nx, Ny, dt, dx, dy, a_x, b_x, a_y, b_y, 
        initial_condition, T, eps, bc, energy_case=3, plot_frequency=100, 
        adaptive_timestep_tolerance=1e-1, save=False, save_path=None):
        self.method_name = method                    # Name of the method; is supposed to be string
        self.method = get_method(self.method_name)

        self.s = s
        self.Nx = Nx
        self.Ny = Ny

        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps
        self.bc = bc
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.adaptive_timestep_tolerance = adaptive_timestep_tolerance
        self.adaptive_timestep_count = 0            # Count until 3 to update the timestep
        
        self.save = save
        self.save_path = save_path

        self.running_time = 0
        self.iter_num = 0

    def initialize(self, display):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)

        self.mesh_grid = np.meshgrid(x_grid, y_grid)

        self.t_total = [0]
        self.u_total = [self.initial_condition(self.mesh_grid)]
        self.energy_total = [utils.energy_exact(self.u_total[0], self.mesh_grid, self.eps, self.bc, self.energy_case)]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()

        while self.t_current <= self.T:

            # print("Current dt is " + str(self.dt) + ", current s is " + str(self.s) + ", current t is " + str(self.t_current))

            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact(u_new, self.mesh_grid, self.eps, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                self.dt = self.dt * 0.8
                self.substep_s_update(case="min")
                continue
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            # Plot when we cross t=2: previous t < 2 and current t >= 2 (using recorded times)
            condition1 = len(self.t_total) >= 2 and self.t_total[-2] < 1 and self.t_total[-1] >= 1
            condition2 = len(self.t_total) >= 4 and self.t_total[-2] < 2 and self.t_total[-1] >= 2
            condition3 = len(self.t_total) >= 6 and self.t_total[-2] < 5 and self.t_total[-1] >= 5
            condition4 = len(self.t_total) >= 8 and self.t_total[-2] < 8 and self.t_total[-1] >= 8
            condition = condition1 or condition2 or condition3 or condition4
            if condition:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_total[-1], save=self.save, save_path=self.save_path)

            # Adaptive timestep procedure
            if self.iter_num >= 2:
                # Define the necessary quantities for the adaptive timestep procedure
                # Mask to avoid division by zero where u_total[-2] or u_total[-3] are (near) zero
                e_k_plus_1 = np.max(np.abs(self.u_total[-1] - self.u_total[-2])) / np.max(np.abs(self.u_total[-2]))
                e_k = np.max(np.abs(self.u_total[-2] - self.u_total[-3])) / np.max(np.abs(self.u_total[-3]))
                dt_current = self.t_total[-1] - self.t_total[-2]
                dt_previous = self.t_total[-2] - self.t_total[-3]
                # Calculate the dimensionless local truncation error
                LTE = np.abs(e_k_plus_1 - (dt_current / dt_previous) * e_k)
                # print("LTE is " + str(LTE))
                # Check if the LTE is within the tolerance; if yes, increment the count; 
                # if the count is greater than 3, update the timestep and reset the count.
                if LTE < self.adaptive_timestep_tolerance:
                    self.adaptive_timestep_count += 1
                    if self.adaptive_timestep_count >= 3:
                        self.dt = self.dt * 1.2
                        self.adaptive_timestep_count = 0
                        # Update the number of substeps in the super-time-stepping method accordingly
                        self.substep_s_update(case="max")
        end_time = time.time()
        self.running_time = end_time - start_time
    
    def substep_s_update(self, case="max"):
        if self.method_name == "RKL2_2D_CH":
            if self.energy_case == 1:
                self.s_new = (-1 + np.sqrt(9 + 512 * self.dt / self.dx**4)) / 2
            elif self.energy_case == 2:
                self.s_new = (-1 + np.sqrt(9 + 512 * self.dt * self.eps / self.dx**4)) / 2
            elif self.energy_case == 3:
                self.s_new = (-1 + np.sqrt(9 + 512 * self.dt * self.eps**2 / self.dx**4)) / 2
        elif self.method_name == "RKG2_2D_CH":
            if self.energy_case == 1:
                self.s_new = (-3 + np.sqrt(25 + 768 * self.dt / self.dx**4)) / 2
            elif self.energy_case == 2:
                self.s_new = (-3 + np.sqrt(25 + 768 * self.dt * self.eps / self.dx**4)) / 2
            elif self.energy_case == 3:
                self.s_new = (-3 + np.sqrt(25 + 768 * self.dt * self.eps**2 / self.dx**4)) / 2
        else:
            raise ValueError("Method name must be 'RKL2_2D_CH' or 'RKG2_2D_CH'. Check the spelling and the class implementation!")
        # For the Cahn-Hilliard equation, we give s some free space so that s_new is multiplied by a factor of 1.02
        self.s_new = math.ceil(self.s_new * 1.02)
        # Compare s_new with s and update s accordingly
        if case == "max":
            self.s = max(self.s, self.s_new)
        elif case == "min":
            self.s = min(self.s, self.s_new)
        else:
            raise ValueError("Case must be 'max' or 'min'. Check the spelling and the class implementation!")

    def step(self, u_previous):
        '''
        One step of solving using the RKG method
        '''
        u_new = self.method(u_previous, self.mesh_grid, self.eps, self.dt, self.s, advective=False, bc=self.bc, energy_case=self.energy_case)
        return u_new

    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None, test_case=None):
        if not save:
            plt.figure(figsize=(5, 4))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Cahn-Hilliard coarsening at time ' + str(round(t, 5)))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(u, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Cahn_Hilliard_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

#########################################################
# Irregular domain Allen-Cahn and Cahn-Hilliard solvers #
#########################################################
class Allen_Cahn_Solver_2D_Constant_Timestep_Irregular_Domain():

    def __init__(self, method, s, dt, X, Y, mask, initial_condition, T, eps, bc, energy_case=1, plot_frequency=100, splitting_order="NLN", save=False, save_path=None):
        self.method_name = method                    # Name of the method; is supposed to be int
        self.method = get_method(self.method_name)

        self.s = s
        self.mask = mask
        self.dt = dt

        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]

        self.a_x = X[0, 0]
        self.b_x = X[0, -1]
        self.a_y = Y[0, 0]
        self.b_y = Y[-1, 0]

        self.Nx = round((self.b_x - self.a_x) / self.dx)
        self.Ny = round((self.b_y - self.a_y) / self.dy)

        print("Nx is " + str(self.Nx) + " and Ny is " + str(self.Ny))

        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps

        self.bc = bc                # "Neumann" or "Periodic"
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.running_time = 0
        self.iter_num = 0

        self.splitting_order = splitting_order
        self.save = save
        self.save_path = save_path

    def initialize(self, display=False):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)
        self.mesh_grid = np.meshgrid(x_grid, y_grid)
            
        self.t_total = [0]
        # Set up the initial condition and mask out the outside region to be 0
        u_initial_condition = self.initial_condition(self.mesh_grid)
        u_initial_condition[~self.mask] = 0
        self.u_total = [u_initial_condition]
        energy_gradient_new, energy_non_linear_new = utils.energy_exact_irregular_domain(self.u_total[0], self.mesh_grid, self.eps, self.mask, bc=self.bc, case=self.energy_case)
        self.energy_total = [[energy_gradient_new, energy_non_linear_new]]
        self.modified_energy_total = [[energy_gradient_new, energy_non_linear_new]]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=0)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        # Calculate total number of iterations for progress bar
        total_iterations = round((self.T - self.t_current) / self.dt)

        for i in tqdm(range(total_iterations+1), desc="Solving", unit="iter", ncols=100, leave=True):

            if self.iter_num % self.plot_frequency == 0 or self.iter_num == total_iterations:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=self.save, save_path=self.save_path)
                print("Max value of u_total[-1] is " + str(np.max(self.u_total[-1])))
                print("Min value of u_total[-1] is " + str(np.min(self.u_total[-1])))

            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact_irregular_domain(u_new, self.mesh_grid, self.eps, self.mask, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                # break
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])

        end_time = time.time()
        self.running_time = end_time - start_time

    def step(self, u_previous):
        '''
        One step of solving using the Strang splitting
        '''
        # Splitting order: NLN (Nonlinear-Linear-Nonlinear) or LNL (Linear-Nonlinear-Linear)
        if self.splitting_order == "NLN":
            u_intermediate1 = self.nonlinear_update(u_previous, self.dt / 2, energy_case=self.energy_case)
            # Diffusion part update
            if self.energy_case == 1:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.mask, self.dt, self.s, bc=self.bc)
            elif self.energy_case == 2:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.mask, self.eps * self.dt, self.s, bc=self.bc)
            elif self.energy_case == 3:
                u_intermediate2 = self.method(u_intermediate1, self.mesh_grid, self.mask, self.eps**2 * self.dt, self.s, bc=self.bc)
            else:
                raise ValueError("Energy case must be 1, 2, or 3.")
            # Nonlinear part update
            u_new = self.nonlinear_update(u_intermediate2, self.dt / 2, energy_case=self.energy_case)
        # TODO: Change the splitting of LNL to the correct corresponding energy cases
        elif self.splitting_order == "LNL":
            u_intermediate1 = self.method(u_previous, self.mesh_grid, self.mask, self.eps**2 * self.dt / 2, self.s, bc=self.bc)
            # Diffusion part update
            u_intermediate2 = self.nonlinear_update(u_intermediate1, self.dt)
            # Nonlinear part update
            u_new = self.method(u_intermediate2, self.mesh_grid, self.mask, self.eps**2 * self.dt / 2, self.s, bc=self.bc)
        else:
            raise ValueError("Splitting order must be either 'NLN' or 'LNL'.")

        # Calculate the modified energy
        energy_modified_gradient_new, energy_modified_nonlinear_new = utils.energy_modified(u_intermediate1[self.mask], u_intermediate2[self.mask], self.dt)
        energy_modified_new = energy_modified_gradient_new + energy_modified_nonlinear_new
        # Check for monotonicity of the energy
        if self.modified_energy_total != [] and energy_modified_new > sum(self.modified_energy_total[-1]):
            print("Modified energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
            print("Modified energy is " + str(energy_modified_new) + " and the previous modified energy is " + str(sum(self.modified_energy_total[-1])))
            # break
        if np.isnan(energy_modified_new):
            print("Modified energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
        self.modified_energy_total.append([energy_modified_gradient_new, energy_modified_nonlinear_new])
        return u_new

    # Solve the non-linear part analytically
    def nonlinear_update(self, u_previous, dt, energy_case=1):
        if energy_case == 1:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps**2))
        elif energy_case == 2:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt / self.eps))
        elif energy_case == 3:
            u_new = u_previous / np.sqrt(u_previous**2 + (1 - u_previous**2) * np.exp(-2 * dt))
        else:
            raise ValueError("Energy case must be 1, 2, or 3.")
        return u_new
    
    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), np.array(self.modified_energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None):
        # Outside-mask region: show white (set to NaN and use colormap bad color)
        # Create masked array (mask the OUTSIDE region)
        plot = np.ma.array(u, mask=~self.mask)

        # Copy colormap so we don't modify global one
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')   # masked values -> white


        if not save:
            plt.figure(figsize=(5, 5))
            plt.imshow(plot, origin='lower', cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Allen-Cahn coarsening at time ' + str(round(t, 5)))
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(plot, origin='lower', cmap=cmap, vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Allen_Cahn_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

class Cahn_Hilliard_Solver_2D_Constant_Timestep_Irregular_Domain():

    def __init__(self, method, s, dt, X, Y, mask, initial_condition, T, eps, bc, energy_case=1, plot_frequency=100, splitting_order="NLN", save=False, save_path=None):
        self.method_name = method                    # Name of the method; is supposed to be string
        self.method = get_method(self.method_name)

        self.s = s
        self.mask = mask

        self.dt = dt

        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]

        self.a_x = X[0, 0]
        self.b_x = X[0, -1]
        self.a_y = Y[0, 0]
        self.b_y = Y[-1, 0]

        self.Nx = round((self.b_x - self.a_x) / self.dx)
        self.Ny = round((self.b_y - self.a_y) / self.dy)

        self.initial_condition = initial_condition
        self.T = T
        self.eps = eps
        self.bc = bc
        self.energy_case = energy_case
        self.plot_frequency = plot_frequency

        self.save = save
        self.save_path = save_path

        self.running_time = 0
        self.iter_num = 0

    def initialize(self, display):
        x_grid = np.linspace(self.a_x, self.b_x, self.Nx+1)
        y_grid = np.linspace(self.a_y, self.b_y, self.Ny+1)
        self.mesh_grid = np.meshgrid(x_grid, y_grid)
        # Set up the initial condition and mask out the outside region to be 0
        u_initial_condition = self.initial_condition(self.mesh_grid)
        u_initial_condition[~self.mask] = 0
        self.u_total = [u_initial_condition]

        self.t_total = [0]
        energy_gradient_new, energy_non_linear_new = utils.energy_exact_irregular_domain(self.u_total[0], self.mesh_grid, self.eps, self.mask, bc=self.bc, case=self.energy_case)
        self.energy_total = [[energy_gradient_new, energy_non_linear_new]]

        self.t_current = 0

        if display:
            self.display(self.u_total[0], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=False)

    def solve(self):
        '''
        Using Strang's splitting, we solve for u_t=u_{xx} using RKG2 and u_t=-1/eps^2f(u) analytically
        The entire scheme is second order.
        '''
        start_time = time.time()
        
        # Calculate total number of iterations for progress bar
        total_iterations = round((self.T - self.t_current) / self.dt)

        for i in tqdm(range(total_iterations + 1), desc="Solving", unit="iter", ncols=100, leave=True):

            if self.iter_num % self.plot_frequency == 0 or self.iter_num == total_iterations:
                self.display(self.u_total[-1], extent=[self.a_x, self.b_x, self.a_y, self.b_y], t=self.t_current, save=self.save, save_path=self.save_path)
                print("The max of u_total[-1] is " + str(np.max(self.u_total[-1])))
                print("The min of u_total[-1] is " + str(np.min(self.u_total[-1])))
                
            # Update the current time and iteration number
            self.t_current += self.dt
            self.iter_num += 1

            u_previous = self.u_total[-1]
            # Run RKL or RKG Strang splitting for one step
            u_new = self.step(u_previous)
            # Update the time, energy, and the solution u
            self.u_total.append(u_new)
            self.t_total.append(self.t_current)

            energy_gradient_new, energy_non_linear_new = utils.energy_exact_irregular_domain(u_new, self.mesh_grid, self.eps, self.mask, bc=self.bc, case=self.energy_case)
            energy_new = energy_gradient_new + energy_non_linear_new
        
            # Check for monotonicity of the energy
            if energy_new > sum(self.energy_total[-1]):
                print("Energy is not monotone, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                print("Energy is " + str(energy_new) + " and the previous energy is " + str(sum(self.energy_total[-1])))
                # break
            if np.isnan(energy_new):
                print("Energy is NaN, at iteration " + str(self.iter_num) + "!!!!!!!!!!!!!!!!")
                break
            self.energy_total.append([energy_gradient_new, energy_non_linear_new])


        end_time = time.time()
        self.running_time = end_time - start_time


    def step(self, u_previous):
        '''
        One step of solving using the RKG method
        '''
        u_new = self.method(u_previous, self.mesh_grid, self.mask, self.eps, self.dt, self.s, bc=self.bc, energy_case=self.energy_case)
        return u_new

    def get_results(self):
        return self.u_total, self.t_total, np.array(self.energy_total), self.mesh_grid
    
    def display(self, u, extent, t, save=False, save_path=None):
        # Outside-mask region: show white (set to NaN and use colormap bad color)
        # Create masked array (mask the OUTSIDE region)
        plot = np.ma.array(u, mask=~self.mask)

        # Copy colormap so we don't modify global one
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')   # masked values -> white

        if not save:
            plt.figure(figsize=(5, 5))
            plt.imshow(plot, origin='lower', cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Heatmap of Cahn-Hilliard coarsening at time ' + str(round(t, 5)))
            plt.show()
        else:
            plt.figure(figsize=(5, 5))
            plt.imshow(plot, origin='lower', cmap=cmap, vmin=-1, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_path, self.method_name)
            os.makedirs(save_path, exist_ok=True) 
            plt.savefig(save_path + f"/Cahn_Hilliard_{self.method_name}_at_{round(t, 5)}_eps_{self.eps}_dt_{self.dt}_s_{self.s}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()