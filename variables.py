import cupy as cp
import numpy as np
import dispersion
import scipy.optimize as opt


class SpaceScalar:
    """ Class for configuration-space scalars """

    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), norm='forward')

    def integrate(self, grid, array):
        """ Integrate an array, possibly self """
        # arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        arr_add = cp.zeros((self.res_x + 1, self.res_y + 1))
        arr_add[:-1, :-1] = array
        arr_add[-1, :-1] = array[0, :]
        arr_add[:-1, -1] = array[:, 0]
        arr_add[-1, -1] = array[0, 0]
        return trapz(arr_add, grid.x.dx, grid.y.dx)

    def integrate_energy(self, grid):
        return self.integrate(grid=grid, array=0.5 * self.arr_nodal ** 2.0)


class SpaceVector:
    """ Class for configuration-space vectors """

    def __init__(self, resolutions):
        self.resolutions = resolutions
        self.arr_nodal, self.arr_spectral = None, None
        self.arr_nodal = cp.zeros((2, resolutions[0], resolutions[1]))
        self.init_spectral_array()

        self.energy = SpaceScalar(resolutions=resolutions)

    def init_spectral_array(self):
        if self.arr_spectral is not None:
            return
        else:
            x_spec = cp.fft.rfft2(self.arr_nodal[0, :, :])
            y_spec = cp.fft.rfft2(self.arr_nodal[1, :, :])
            self.arr_spectral = cp.array([x_spec, y_spec])

    def integrate_energy(self, grid):
        self.inverse_fourier_transform()
        self.energy.arr_nodal = self.arr_nodal[0, :, :] ** 2.0 + self.arr_nodal[1, :, :] ** 2.0
        return 0.5 * self.energy.integrate(grid=grid, array=self.energy.arr_nodal)

    def fourier_transform(self):
        self.arr_spectral[0, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[0, :, :], norm='forward'), axes=0)
        self.arr_spectral[1, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[1, :, :], norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal[0, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[0, :, :], axes=0), norm='forward')
        self.arr_nodal[1, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[1, :, :], axes=0), norm='forward')


class VelocityScalar:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order
        # arrays
        self.arr_nodal = None


class Distribution:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order

        # arrays
        self.arr_spectral, self.arr_nodal = None, None
        self.moment0, self.moment2 = (SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]]),
                                      SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]]))
        self.moment1 = SpaceVector(resolutions=[self.resolutions[0], self.resolutions[1]])
        self.moment1_magnitude = SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]])

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, axes=(0, 1), norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), axes=(0, 1), norm='forward')

    def total_density(self, grid):
        self.compute_moment0(grid=grid)
        return self.moment0.integrate(grid=grid, array=self.moment0.arr_nodal)

    def total_thermal_energy(self, grid):
        self.compute_moment2(grid=grid)
        return 0.5 * self.moment2.integrate(grid=grid, array=self.moment2.arr_nodal)

    def compute_moment0(self, grid):
        self.moment0.arr_spectral = grid.moment0(variable=self.arr_spectral)
        self.moment0.inverse_fourier_transform()

    def compute_moment1(self, grid):
        self.moment1.arr_spectral[0, :, :] = grid.moment1_u(variable=self.arr_spectral)
        self.moment1.arr_spectral[1, :, :] = grid.moment1_v(variable=self.arr_spectral)
        self.moment1.inverse_fourier_transform()

    def compute_moment1_magnitude(self):
        self.moment1_magnitude.arr_nodal = cp.sqrt(cp.square(self.moment1.arr_nodal[0, :, :]) +
                                                   cp.square(self.moment1.arr_nodal[1, :, :]))

    def compute_moment2(self, grid):
        self.moment2.arr_spectral = grid.moment2(variable=self.arr_spectral)
        self.moment2.inverse_fourier_transform()

    def set_modal_distribution(self, idx, velocity_scalar):
        velocity_scalar.arr_nodal = self.arr_spectral[idx[0], idx[1], :, :, :, :]

    def nodal_flatten(self):
        return self.arr_nodal.reshape(self.resolutions[0], self.resolutions[1],
                                      self.resolutions[2] * self.order, self.resolutions[3] * self.order)

    def initialize_aniso_max(self, grid, vx, vy, phase_shift1, phase_shift2):
        # space indicators
        ix, iy = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.y.device_arr)
        x, y = cp.tensordot(grid.x.device_arr, iy, axes=0), cp.tensordot(ix, grid.y.device_arr, axes=0)

        # velocity indicators
        iu, iv = cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        u, v = cp.tensordot(grid.u.device_arr, iv, axes=0), cp.tensordot(iu, grid.v.device_arr, axes=0)

        # set basic-ass maxwellian
        factor = 1 / (2 * cp.pi * vx * vy)
        maxwell = factor * cp.exp(-0.5 * ((u/vx) ** 2.0 + (v/vy) ** 2.0))
        self.arr_nodal = cp.tensordot(ix, cp.tensordot(iy, maxwell, axes=0), axes=0)
        # cartesian gradient
        grad_u, grad_v = -(u/vx**2) * maxwell, -(v/vy**2) * maxwell

        # perturbation
        guess_r, guess_i = 0, 1
        amplitude = 2.0e-3
        for i in [1, 2]:  # k_parallel modes
            for j in [0, 1, 2]:  # k_perp modes
                # get eigenvalue, first perp mode
                k_x, k_y = grid.x.wavenumbers[grid.x.zero_idx + i], grid.y.wavenumbers[grid.y.zero_idx + j]
                k = np.sqrt(k_x ** 2.0 + k_y ** 2.0)
                phi = np.arctan2(k_y, k_x)
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(k, phi), jac=dispersion.jacobian_fsolve, tol=1.0e-12)
                guess_r, guess_i = solution.x

                z = solution.x[0] + 1j * solution.x[1]
                # print(z)
                # print(z**2)
                # get eigenvector
                eigs = dispersion.eigenvalue_matrix(z, k, phi)
                # print('\neigs are ')
                # print(eigs)
                antirotated_ev = np.dot(dispersion.rotation_matrix(angle=-phi), eigs)
                # antirotated_ev = cp.array([[np.cos(phi) * eigs[0] - np.sin(-phi) * eigs[1]],
                #                            [np.sin(-phi) * eigs[0] + np.cos(phi) * eigs[1]]])
                ex, ey = antirotated_ev[0], antirotated_ev[1]

                self.arr_nodal += amplitude * self.eigenfunction(x, y, u, v, k_x, k_y, z,
                                                                 ex, ey, grad_u, grad_v, phase_shift1[i-1, j])

                if j > 0:
                    # Second perp mode
                    # get eigenvalue, first perp mode
                    k_x, k_y = grid.x.wavenumbers[grid.x.zero_idx + i], -grid.y.wavenumbers[grid.y.zero_idx + j]
                    k = np.sqrt(k_x ** 2.0 + k_y ** 2.0)
                    phi = np.arctan2(k_y, k_x)
                    solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                        args=(k, phi), jac=dispersion.jacobian_fsolve, tol=1.0e-12)
                    guess_r, guess_i = solution.x

                    z = solution.x[0] + 1j * solution.x[1]
                    # get eigenvector
                    eigs = dispersion.eigenvalue_matrix(z, k, phi)
                    # print('\neigs are ')
                    # print(eigs)
                    antirotated_ev = np.dot(dispersion.rotation_matrix(angle=-phi), eigs)
                    # antirotated_ev = cp.array([[np.cos(phi) * eigs[0] - np.sin(-phi) * eigs[1]],
                    #                            [np.sin(-phi) * eigs[0] + np.cos(phi) * eigs[1]]])
                    ex, ey = antirotated_ev[0], antirotated_ev[1]

                    self.arr_nodal += amplitude * self.eigenfunction(x, y, u, v, k_x, k_y, z,
                                                                     ex, ey, grad_u, grad_v, phase_shift2[i-1, j])

        self.fourier_transform()
        self.compute_moment0(grid=grid)
        self.compute_moment1(grid=grid)

    def eigenfunction(self, x, y, u, v, kx, ky, z, Ex, Ey, grad_u, grad_v, phase_shift):
        k = np.sqrt(kx**2 + ky**2)  # wavenumber
        o = z*k  # frequency
        wave = cp.exp(1j * (kx * x + ky * y + phase_shift))
        Ex_prop = grad_u + u * (kx * grad_u + ky * grad_v) / (o - kx * u - ky * v)
        Ey_prop = grad_v + v * (kx * grad_u + ky * grad_v) / (o - kx * u - ky * v)
        return -1.0 * cp.real(cp.tensordot(wave, Ex * Ex_prop + Ey * Ey_prop, axes=0) / (1j * o))
        # -1, charge sign
        # return -1 * (cp.real(cp.tensordot(wave, Ex*Ex_prop/(1j*o), axes=0))
        #              + cp.real(cp.tensordot(wave, Ey*Ey_prop/(1j*o), axes=0))
        #              )

    def initialize_maxwellian(self, grid):
        # space indicators
        ix, iy = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.y.device_arr)

        # velocity indicators
        iu, iv = cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        u, v = cp.tensordot(grid.u.device_arr, iv, axes=0), cp.tensordot(iu, grid.v.device_arr, axes=0)

        # set basic-ass maxwellian
        factor = 1 / (2 * cp.pi)
        maxwell = factor * cp.exp(-0.5 * (u ** 2.0 + v ** 2.0))
        self.arr_nodal = cp.tensordot(ix, cp.tensordot(iy, maxwell, axes=0), axes=0)

        # perturb
        # self.arr_nodal += 0.01 * cp.tensordot(cp.sin(grid.x.fundamental * grid.x.device_arr),
        #                                       cp.tensordot(cp.sin(grid.y.fundamental * grid.y.device_arr),
        #                                                    maxwell, axes=0), axes=0)
        self.arr_nodal += 0.01 * cp.tensordot(cp.sin(grid.x.fundamental * grid.x.device_arr),
                                              cp.tensordot(iy, maxwell, axes=0), axes=0)
        self.fourier_transform()


def trapz(f, dx, dy):
    """ Custom trapz routine using cupy """
    sum_y = cp.sum(f[:, :-1] + f[:, 1:], axis=1) * dy / 2.0
    return cp.sum(sum_y[:-1] + sum_y[1:]) * dx / 2.0

