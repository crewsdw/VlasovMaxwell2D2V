import cupy as cp
import numpy as np
import variables as var
import scipy.optimize as opt
import dispersion


class Dynamic:
    """ Class for dynamic fields described by Ampere/Faraday laws, here E_y and B_z """
    def __init__(self, resolutions, vt_c):
        self.electric_x = var.SpaceScalar(resolutions=resolutions)
        self.electric_y = var.SpaceScalar(resolutions=resolutions)
        self.magnetic_z = var.SpaceScalar(resolutions=resolutions)
        self.vt_c = vt_c

    def initialize(self, grid, phase_shift1, phase_shift2):
        # space indicators
        ix, iy = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.y.device_arr)
        x, y = cp.tensordot(grid.x.device_arr, iy, axes=0), cp.tensordot(ix, grid.y.device_arr, axes=0)

        # initialize to zero
        self.electric_x.arr_nodal, self.electric_y.arr_nodal, self.magnetic_z.arr_nodal = 0, 0, 0

        # Get eigenvalue
        guess_r, guess_i = 0, 1
        amplitude = 2.0e-3
        for i in [1, 2]:  # k_parallel modes
            for j in [0, 1, 2]:  # k_perp modes
                # First perp mode
                # get eigenvalue
                k_x, k_y = grid.x.wavenumbers[grid.x.zero_idx + i], grid.y.wavenumbers[grid.y.zero_idx + j]
                k, phi = np.sqrt(k_x ** 2.0 + k_y ** 2.0), np.arctan2(k_y, k_x)
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(k, phi), jac=dispersion.jacobian_fsolve, tol=1.0e-12)
                guess_r, guess_i = solution.x
                z = solution.x[0] + 1j * solution.x[1]

                # get eigenvector
                eigs = dispersion.eigenvalue_matrix(z, k, phi)
                antirotated_ev = np.dot(dispersion.rotation_matrix(angle=-phi), eigs)

                # field mode
                wave = cp.exp(1j * (k_x * x + k_y * y + phase_shift1[i-1, j]))
                self.electric_x.arr_nodal += amplitude * cp.real(antirotated_ev[0] * wave)
                self.electric_y.arr_nodal += amplitude * cp.real(antirotated_ev[1] * wave)
                self.magnetic_z.arr_nodal += amplitude * cp.real(eigs[1] / z * wave)

                # Second perp mode
                if j > 0:
                    # get eigenvalue
                    k_x, k_y = grid.x.wavenumbers[grid.x.zero_idx + i], -grid.y.wavenumbers[grid.y.zero_idx + j]
                    k, phi = np.sqrt(k_x ** 2.0 + k_y ** 2.0), np.arctan2(k_y, k_x)
                    solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                        args=(k, phi), jac=dispersion.jacobian_fsolve, tol=1.0e-12)
                    guess_r, guess_i = solution.x
                    z = solution.x[0] + 1j * solution.x[1]

                    # get eigenvector
                    eigs = dispersion.eigenvalue_matrix(z, k, phi)
                    antirotated_ev = np.dot(dispersion.rotation_matrix(angle=-phi), eigs)

                    # field mode
                    wave = cp.exp(1j * (k_x * x + k_y * y + phase_shift2[i-1, j]))
                    self.electric_x.arr_nodal += amplitude * cp.real(antirotated_ev[0] * wave)
                    self.electric_y.arr_nodal += amplitude * cp.real(antirotated_ev[1] * wave)
                    self.magnetic_z.arr_nodal += amplitude * cp.real(eigs[1] / z * wave)

        # Fourier-transform
        self.electric_x.fourier_transform()
        self.electric_y.fourier_transform()
        self.magnetic_z.fourier_transform()

    # def eigenmode(self, grid, amplitude, wavenumber, eigenvalue):
    #     # Nodal values
    #     self.magnetic_z.arr_nodal = cp.real(amplitude * cp.exp(1j * wavenumber * grid.x.device_arr))
    #     self.electric_y.arr_nodal = cp.real(eigenvalue * amplitude * cp.exp(1j * wavenumber * grid.x.device_arr))

    def compute_magnetic_energy(self, grid):
        self.magnetic_z.inverse_fourier_transform()
        return self.magnetic_z.integrate_energy(grid=grid) / (self.vt_c ** 2)

    def compute_electric_energy_x(self, grid):
        self.electric_x.inverse_fourier_transform()
        return self.electric_x.integrate_energy(grid=grid)

    def compute_electric_energy_y(self, grid):
        self.electric_y.inverse_fourier_transform()
        return self.electric_y.integrate_energy(grid=grid)

# class Static:
#     """ Class for static fields governed by Gauss's law, here E_x """
#     def __init__(self, resolution):
#         self.electric_x = var.SpaceScalar(resolution=resolution)
#
#     def gauss(self, distribution, grid, invert=True):
#         # Compute zeroth moment, integrate(c_n(v)dv)
#         distribution.compute_zero_moment(grid=grid)
#
#         # Adjust for charge neutrality
#         distribution.moment0.arr_spectral[grid.x.zero_idx] -= 1.0
#
#         # Compute field spectrum
#         self.electric_x.arr_spectral = (-1j * grid.charge_sign *
#                                         cp.nan_to_num(cp.divide(distribution.moment0.arr_spectral,
#                                                                 grid.x.device_wavenumbers)))
#
#         if invert:
#             self.electric_x.inverse_fourier_transform()
#
#     def compute_field_energy(self, grid):
#         self.electric_x.inverse_fourier_transform()
#         return self.electric_x.integrate_energy(grid=grid)
