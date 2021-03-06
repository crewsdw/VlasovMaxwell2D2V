import cupy as cp
import variables as var


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, order, grid, nu):
        self.x_ele, self.y_ele, self.u_res, self.v_res = resolutions
        self.x_res = grid.x.wavenumbers.shape[0]
        self.y_res = grid.y.wavenumbers.shape[0]
        self.order = order

        # permutations after tensor-dot with basis array
        self.permutations = [(0, 1, 2, 5, 3, 4),  # for contraction with u nodes
                             (0, 1, 2, 3, 4, 5)]  # for contraction with v nodes

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [[(slice(self.x_res), slice(self.y_res),
                                  slice(self.u_res), 0,
                                  slice(self.v_res), slice(self.order)),
                                 (slice(self.x_res), slice(self.y_res),
                                  slice(self.u_res), -1,
                                  slice(self.v_res), slice(self.order))],
                                [(slice(self.x_res), slice(self.y_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), 0),
                                 (slice(self.x_res), slice(self.y_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), -1)]]
        self.boundary_slices_pad = [[(slice(self.x_res), slice(self.y_res),
                                      slice(self.u_res + 2), 0,
                                      slice(self.v_res), slice(self.order)),
                                     (slice(self.x_res), slice(self.y_res),
                                      slice(self.u_res + 2), -1,
                                      slice(self.v_res), slice(self.order))],
                                    [(slice(self.x_res), slice(self.y_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), 0),
                                     (slice(self.x_res), slice(self.y_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), -1)]]
        self.flux_input_slices = [(slice(self.x_res), slice(self.y_res),
                                   slice(1, self.u_res + 1), slice(self.order),
                                   slice(self.v_res), slice(self.order)),
                                  (slice(self.x_res), slice(self.y_res),
                                   slice(self.u_res), slice(self.order),
                                   slice(1, self.v_res + 1), slice(self.order)),
                                  (slice(self.x_res), slice(self.y_res),
                                   slice(self.u_res), slice(self.order),
                                   slice(self.v_res), slice(self.order))]
        self.pad_slices = [(slice(self.x_res), slice(self.y_res),
                            slice(1, self.u_res + 1),
                            slice(self.v_res), slice(self.order)),
                           (slice(self.x_res), slice(self.y_res),
                            slice(self.u_res), slice(self.order),
                            slice(1, self.v_res + 1))]
        self.num_flux_sizes = [(self.x_res, self.y_res, self.u_res, 2, self.v_res, self.order),
                               (self.x_res, self.y_res, self.u_res, self.order, self.v_res, 2)]
        self.padded_flux_sizes = [(self.x_res, self.y_res, self.u_res + 2, self.order, self.v_res, self.order),
                                  (self.x_res, self.y_res, self.u_res, self.order, self.v_res + 2, self.order)]

        self.directions = [2, 4]
        self.sub_elements = [3, 5]

        # arrays
        self.flux_ex = var.Distribution(resolutions=resolutions, order=order)
        self.flux_ey = var.Distribution(resolutions=resolutions, order=order)
        self.flux_bz = var.Distribution(resolutions=resolutions, order=order)

        # particle charge
        self.charge = -1.0
        self.nu = nu  # hyper-viscosity

        self.pad_field, self.pad_spectrum = None, None

    def semi_discrete_fully_explicit(self, distribution, field, grid):
        """ Computes the semi-discrete equation with a full advection step """
        # Compute the flux (pseudospectral method)
        self.flux_ex.arr_spectral = self.compute_flux(distribution=distribution, field=field.electric_x, grid=grid)
        self.flux_ey.arr_spectral = self.compute_flux(distribution=distribution, field=field.electric_y, grid=grid)
        self.flux_bz.arr_spectral = self.compute_flux(distribution=distribution, field=field.magnetic_z, grid=grid)
        # Compute semi-discrete RHS
        return (grid.u.J[None, None, :, None, None, None] * self.u_flux(distribution=distribution, grid=grid) +
                grid.v.J[None, None, None, None, :, None] * self.v_flux(distribution=distribution, grid=grid) +
                self.source_term(distribution=distribution, grid=grid))

    def semi_discrete_semi_implicit(self, distribution, field, grid):
        """ Computes the semi-discrete equation with a half advection step"""
        # Compute the flux (pseudospectral method)
        self.flux_ex.arr_spectral = self.compute_flux(distribution=distribution, field=field.electric_x, grid=grid)
        self.flux_ey.arr_spectral = self.compute_flux(distribution=distribution, field=field.electric_y, grid=grid)
        self.flux_bz.arr_spectral = self.compute_flux(distribution=distribution, field=field.magnetic_z, grid=grid)
        # Compute semi-discrete RHS
        return (grid.u.J[None, None, :, None, None, None] * self.u_flux(distribution=distribution, grid=grid) +
                grid.v.J[None, None, None, None, :, None] * self.v_flux(distribution=distribution, grid=grid) -
                self.nu * (grid.x.device_wavenumbers_fourth[:, None, None, None, None, None] +
                           grid.y.device_wavenumbers_fourth[None, :, None, None, None, None]) *
                distribution.arr_spectral)

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros((grid.x.modes + 2 * grid.x.pad_width, grid.y.modes + grid.y.pad_width)) + 0j
        self.pad_spectrum = cp.zeros((grid.x.modes + 2 * grid.x.pad_width, grid.y.modes + grid.y.pad_width,
                                      grid.u.elements, grid.u.order,
                                      grid.v.elements, grid.v.order)) + 0j

    def compute_flux(self, distribution, field, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # Dealias with two-thirds rule
        self.pad_field[grid.x.pad_width:-grid.x.pad_width, :-grid.y.pad_width] = field.arr_spectral
        self.pad_spectrum[grid.x.pad_width:-grid.x.pad_width, :-grid.y.pad_width, :, :, :, :] = (
            distribution.arr_spectral
        )

        # inverse transform zero-padded array
        return forward_distribution_transform(cp.multiply(
            inverse_field_transform(self.pad_field)[:, :, None, None, None, None],
            inverse_distribution_transform(self.pad_spectrum))
        )[grid.x.pad_width:-grid.x.pad_width, :-grid.y.pad_width, :, :, :, :]

    def u_flux(self, distribution, grid):
        # # flux = self.charge * (self.flux_ex.arr_spectral + cp.einsum('rps,nmijrs->nmijrp',
        # #                                                             grid.v.translation_matrix,
        # #                                                             self.flux_bz.arr_spectral))
        # flux = self.charge * (self.flux_ex.arr_spectral + (grid.v.device_arr[None, None, None, None, :, :] *
        #                                                    self.flux_bz.arr_spectral))
        # return (basis_product(flux=flux, basis_arr=grid.u.local_basis.internal,
        #                       axis=3, permutation=self.permutations[0]) -
        #         self.numerical_flux(distribution=distribution,
        #                             flux=flux, grid=grid, dim=0))
        # flux = self.charge * (self.flux_ex.arr_spectral + cp.einsum('rps,nmijrs->nmijrp',
        #                                                             grid.v.translation_matrix,
        #                                                             self.flux_bz.arr_spectral))
        return (basis_product(flux=self.charge * (self.flux_ex.arr_spectral + cp.einsum('rps,nmijrs->nmijrp',
                                                                                        grid.v.translation_matrix,
                                                                                        self.flux_bz.arr_spectral)),
                              basis_arr=grid.u.local_basis.internal,
                              axis=3, permutation=self.permutations[0]) -
                self.numerical_flux(distribution=distribution,
                                    flux=self.charge * (self.flux_ex.arr_spectral +
                                                        (grid.v.device_arr[None, None, None, None, :, :] *
                                                         self.flux_bz.arr_spectral)),
                                    grid=grid, dim=0))

    def v_flux(self, distribution, grid):
        # flux = self.charge * (self.flux_ey.arr_spectral - cp.einsum('ijk,nmikrs->nmijrs',
        #                                                             grid.u.translation_matrix,
        #                                                             self.flux_bz.arr_spectral))
        # flux = self.charge * (self.flux_ey.arr_spectral - (grid.u.device_arr[None, None, :, :, None, None] *
        #                                                    self.flux_bz.arr_spectral))
        # return (basis_product(flux=flux, basis_arr=grid.v.local_basis.internal,
        #                       axis=5, permutation=self.permutations[1]) -
        #         self.numerical_flux(distribution=distribution,
        #                             flux=flux, grid=grid, dim=1))
        return (basis_product(flux=self.charge * (self.flux_ey.arr_spectral - cp.einsum('ijk,nmikrs->nmijrs',
                                                                                        grid.u.translation_matrix,
                                                                                        self.flux_bz.arr_spectral)),
                              basis_arr=grid.v.local_basis.internal,
                              axis=5, permutation=self.permutations[1]) -
                self.numerical_flux(distribution=distribution,
                                    flux=self.charge * (self.flux_ey.arr_spectral -
                                                        (grid.u.device_arr[None, None, :, :, None, None] *
                                                         self.flux_bz.arr_spectral)),
                                    grid=grid, dim=1))

    def source_term(self, distribution, grid):
        return -1j * (cp.multiply(grid.x.device_wavenumbers[:, None, None, None, None, None],
                                  cp.einsum('axb,ijabcd->ijaxcd',
                                            grid.u.translation_matrix, distribution.arr_spectral)) +
                      cp.multiply(grid.y.device_wavenumbers[None, :, None, None, None, None],
                                  cp.einsum('cxd,ijabcd->ijabcx',
                                            grid.v.translation_matrix, distribution.arr_spectral)))

    def numerical_flux(self, distribution, flux, grid, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim]) + 0j

        # set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim]) + 0j
        padded_flux[self.flux_input_slices[dim]] = flux

        # Compute a central flux
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                                                 shift=+1,
                                                                 axis=self.directions[dim])[self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][0]]) / 2.0
        num_flux[self.boundary_slices[dim][1]] = (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                                          shift=-1,
                                                          axis=self.directions[dim])[self.pad_slices[dim]] +
                                                  flux[self.boundary_slices[dim][1]]) / 2.0

        # re-use padded_flux array for padded_distribution
        padded_flux[self.flux_input_slices[dim]] = distribution.arr_spectral
        constant = cp.amax(cp.absolute(flux), axis=self.sub_elements[dim])

        # Lax-Friedrichs flux
        num_flux[self.boundary_slices[dim][0]] += -1.0 * cp.multiply(constant,
                                                                     (cp.roll(
                                                                         padded_flux[self.boundary_slices_pad[dim][1]],
                                                                         shift=+1,
                                                                         axis=self.directions[dim])[
                                                                          self.pad_slices[dim]] -
                                                                      distribution.arr_spectral[
                                                                          self.boundary_slices[dim][0]]) / 2.0)
        num_flux[self.boundary_slices[dim][1]] += -1.0 * cp.multiply(constant,
                                                                     (cp.roll(
                                                                         padded_flux[self.boundary_slices_pad[dim][0]],
                                                                         shift=-1,
                                                                         axis=self.directions[dim])[
                                                                          self.pad_slices[dim]] -
                                                                      distribution.arr_spectral[
                                                                          self.boundary_slices[dim][1]]) / 2.0)

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical,
                             axis=self.sub_elements[dim], permutation=self.permutations[dim])


class SpaceFlux:
    def __init__(self, c):
        # self.resolutions = resolutions
        self.c = c

    def faraday(self, dynamic_field, grid):
        return -1j * (grid.x.device_wavenumbers[:, None] * dynamic_field.electric_y.arr_spectral -
                      grid.y.device_wavenumbers[None, :] * dynamic_field.electric_x.arr_spectral)

    def ampere_y(self, distribution, dynamic_field, grid):
        return ((self.c ** 2.0) * (-1j * grid.x.device_wavenumbers[:, None] * dynamic_field.magnetic_z.arr_spectral) -
                grid.charge_sign * distribution.moment1.arr_spectral[1, :, :])

    def ampere_x(self, distribution, dynamic_field, grid):
        return ((self.c ** 2.0) * (1j * grid.y.device_wavenumbers[None, :] * dynamic_field.magnetic_z.arr_spectral) -
                grid.charge_sign * distribution.moment1.arr_spectral[0, :, :])


# def inverse_field_transform(field, dim):
#     return cp.fft.irfft2(cp.fft.fftshift(field[dim, :, :], axes=0), norm='forward')
def inverse_field_transform(field):
    return cp.fft.irfft2(cp.fft.fftshift(field, axes=0), norm='forward')


def inverse_distribution_transform(distribution):
    return cp.fft.irfft2(cp.fft.fftshift(distribution, axes=0), axes=(0, 1), norm='forward')


def forward_distribution_transform(nodal_array):
    return cp.fft.fftshift(cp.fft.rfft2(nodal_array, axes=(0, 1), norm='forward'), axes=0)
