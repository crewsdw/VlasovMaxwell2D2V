import numpy as np
import time as timer
import variables as var
import fields
import cupy as cp

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, resolutions, order, steps, grid, phase_space_flux, space_flux):
        self.x_res, self.y_res, self.u_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.steps = steps
        self.phase_space_flux = phase_space_flux
        self.space_flux = space_flux

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.ex_energy = np.array([])
        self.ey_energy = np.array([])
        self.bz_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])

        # semi-implicit advection matrix
        self.implicit_x_advection_matrix = None
        self.implicit_y_advection_matrix = None
        self.build_advection_matrices(grid=grid, dt=0.5 * self.dt)  # applied twice, once on each side of nonlinear term

        # save-times
        self.save_times = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
                                    6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
                                    12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5,
                                    18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5,
                                    24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
                                    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5,
                                    36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40, 0])

    def main_loop(self, distribution, dynamic_field, grid, datafile):
        print('Beginning main loop')
        # Compute first two steps with ssp-rk3 and save fluxes
        # zero stage
        distribution.compute_moment1(grid=grid)
        ps_flux0 = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                     field=dynamic_field, grid=grid)
        ex_flux0 = self.space_flux.ampere_x(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        ey_flux0 = self.space_flux.ampere_y(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        bz_flux0 = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # first step
        self.ssp_rk3(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        self.time += self.dt

        # first stage
        distribution.compute_moment1(grid=grid)
        ps_flux1 = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                     field=dynamic_field, grid=grid)
        ex_flux1 = self.space_flux.ampere_x(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        ey_flux1 = self.space_flux.ampere_y(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        bz_flux1 = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # second step
        self.ssp_rk3(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        self.time += self.dt

        # store first two fluxes
        previous_phase_space_fluxes = [ps_flux1, ps_flux0]
        previous_dynamic_fluxes = [[ex_flux1, ex_flux0],
                                   [ey_flux1, ey_flux0],
                                   [bz_flux1, bz_flux0]]

        # Main loop
        t0, save_counter = timer.time(), 0
        for i in range(self.steps):
            previous_phase_space_fluxes, previous_dynamic_fluxes = self.strang_split_adams_bashforth(
                distribution=distribution, dynamic_field=dynamic_field, grid=grid,
                previous_phase_space_fluxes=previous_phase_space_fluxes, previous_dynamic_fluxes=previous_dynamic_fluxes
            )
            self.time += self.dt

            if i % 10 == 0:
                self.time_array = np.append(self.time_array, self.time)
                self.ex_energy = np.append(self.ex_energy, dynamic_field.compute_electric_energy_x(grid=grid))
                self.ey_energy = np.append(self.ey_energy, dynamic_field.compute_electric_energy_y(grid=grid))
                self.bz_energy = np.append(self.bz_energy, dynamic_field.compute_magnetic_energy(grid=grid))
                self.thermal_energy = np.append(self.thermal_energy, distribution.total_thermal_energy(grid=grid))
                self.density_array = np.append(self.density_array, distribution.total_density(grid=grid))
                print('\nTook 10 steps, time is {:0.3e}'.format(self.time))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')

            if np.abs(self.time - self.save_times[save_counter]) < 6.0e-3:
                print('Reached save time at {:0.3e}'.format(self.time) + ', saving data...')
                distribution.inverse_fourier_transform()
                distribution.moment0.inverse_fourier_transform()
                distribution.moment1.inverse_fourier_transform()
                datafile.save_data(distribution=distribution.arr_nodal.get(),
                                   density=distribution.moment0.arr_nodal.get(),
                                   current_x=distribution.moment1.arr_nodal[0, :, :].get(),
                                   current_y=distribution.moment1.arr_nodal[1, :, :].get(),
                                   electric_x=dynamic_field.electric_x.arr_nodal.get(),
                                   electric_y=dynamic_field.electric_y.arr_nodal.get(),
                                   magnetic_z=dynamic_field.magnetic_z.arr_nodal.get(), time=self.time)
                save_counter += 1

        print('\nAll done at time is {:0.3e}'.format(self.time))
        print('Total steps were ' + str(self.steps))
        print('Time since start is {:0.3e}'.format((timer.time() - t0)))

    def build_advection_matrices(self, grid, dt):
        """ Construct the global backward advection matrix """
        # x-directed velocity
        backward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] -
                                       0.5 * dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                       grid.u.translation_matrix[None, :, :, :])
        forward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] +
                                      0.5 * dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                      grid.u.translation_matrix[None, :, :, :])
        inv_backward_advection = cp.linalg.inv(backward_advection_operator)
        self.implicit_x_advection_matrix = cp.matmul(inv_backward_advection, forward_advection_operator)

        # y-directed advection
        backward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] -
                                       0.5 * dt * -1j * grid.y.device_wavenumbers[:, None, None, None] *
                                       grid.v.translation_matrix[None, :, :, :])
        forward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] +
                                      0.5 * dt * -1j * grid.y.device_wavenumbers[:, None, None, None] *
                                      grid.v.translation_matrix[None, :, :, :])
        inv_backward_advection = cp.linalg.inv(backward_advection_operator)
        self.implicit_y_advection_matrix = cp.matmul(inv_backward_advection, forward_advection_operator)

    def ssp_rk3(self, distribution, dynamic_field, grid):
        # Cut-off (avoid CFL advection instability as this is fully explicit)
        # cutoff = 50

        # Stage 0 set-up
        phase_space_stage0 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)
        dynamic_field_stage0 = fields.Dynamic(resolutions=[self.resolutions[0], self.resolutions[1]],
                                              vt_c=1 / self.space_flux.c)
        # Stage 1 set-up
        phase_space_stage1 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)
        dynamic_field_stage1 = fields.Dynamic(resolutions=[self.resolutions[0], self.resolutions[1]],
                                              vt_c=1 / self.space_flux.c)

        # zero stage
        distribution.compute_moment1(grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=distribution,
                                                                             field=dynamic_field,
                                                                             grid=grid)
        electric_x_rhs = self.space_flux.ampere_x(distribution=distribution,
                                                  dynamic_field=dynamic_field, grid=grid)
        electric_y_rhs = self.space_flux.ampere_y(distribution=distribution,
                                                  dynamic_field=dynamic_field, grid=grid)
        magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # update
        phase_space_stage0.arr_spectral = (distribution.arr_spectral + self.dt * phase_space_rhs)
        dynamic_field_stage0.electric_x.arr_spectral = (dynamic_field.electric_x.arr_spectral +
                                                        self.dt * electric_x_rhs)
        dynamic_field_stage0.electric_y.arr_spectral = (dynamic_field.electric_y.arr_spectral +
                                                        self.dt * electric_y_rhs)
        dynamic_field_stage0.magnetic_z.arr_spectral = (dynamic_field.magnetic_z.arr_spectral +
                                                        self.dt * magnetic_z_rhs)

        # first stage
        phase_space_stage0.compute_moment1(grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=phase_space_stage0,
                                                                             field=dynamic_field_stage0,
                                                                             grid=grid)
        electric_x_rhs = self.space_flux.ampere_x(distribution=phase_space_stage0,
                                                  dynamic_field=dynamic_field_stage0, grid=grid)
        electric_y_rhs = self.space_flux.ampere_y(distribution=phase_space_stage0,
                                                  dynamic_field=dynamic_field_stage0, grid=grid)
        magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field_stage0, grid=grid)

        phase_space_stage1.arr_spectral = self.rk_stage_update(stage=0, arr0=distribution.arr_spectral,
                                                               arr1=phase_space_stage0.arr_spectral,
                                                               rhs=phase_space_rhs)
        dynamic_field_stage1.electric_x.arr_spectral = self.rk_stage_update(
            stage=0,
            arr0=dynamic_field.electric_x.arr_spectral,
            arr1=dynamic_field_stage0.electric_x.arr_spectral,
            rhs=electric_x_rhs)
        dynamic_field_stage1.electric_y.arr_spectral = self.rk_stage_update(
            stage=0,
            arr0=dynamic_field.electric_y.arr_spectral,
            arr1=dynamic_field_stage0.electric_y.arr_spectral,
            rhs=electric_y_rhs)
        dynamic_field_stage1.magnetic_z.arr_spectral = self.rk_stage_update(
            stage=0,
            arr0=dynamic_field.magnetic_z.arr_spectral,
            arr1=dynamic_field_stage0.magnetic_z.arr_spectral,
            rhs=magnetic_z_rhs)

        # second stage
        phase_space_stage1.compute_moment1(grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=phase_space_stage1,
                                                                             field=dynamic_field_stage1,
                                                                             grid=grid)
        electric_x_rhs = self.space_flux.ampere_x(distribution=phase_space_stage1,
                                                  dynamic_field=dynamic_field_stage1, grid=grid)
        electric_y_rhs = self.space_flux.ampere_y(distribution=phase_space_stage1,
                                                  dynamic_field=dynamic_field_stage1, grid=grid)
        magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field_stage1, grid=grid)

        # Full update
        distribution.arr_spectral = self.rk_stage_update(stage=0, arr0=distribution.arr_spectral,
                                                         arr1=phase_space_stage1.arr_spectral,
                                                         rhs=phase_space_rhs)
        dynamic_field.electric_x.arr_spectral = self.rk_stage_update(
            stage=1,
            arr0=dynamic_field.electric_x.arr_spectral,
            arr1=dynamic_field_stage1.electric_x.arr_spectral,
            rhs=electric_x_rhs)
        dynamic_field.electric_y.arr_spectral = self.rk_stage_update(
            stage=1,
            arr0=dynamic_field.electric_y.arr_spectral,
            arr1=dynamic_field_stage1.electric_y.arr_spectral,
            rhs=electric_y_rhs)
        dynamic_field.magnetic_z.arr_spectral = self.rk_stage_update(
            stage=1,
            arr0=dynamic_field.magnetic_z.arr_spectral,
            arr1=dynamic_field_stage1.magnetic_z.arr_spectral,
            rhs=magnetic_z_rhs)

    def strang_split_adams_bashforth(self, distribution, dynamic_field, grid, previous_phase_space_fluxes,
                                     previous_dynamic_fluxes):
        """ strang-split phase space advance """
        # half crank-nicholson fractional advance advection step
        next_arr = cp.einsum('njkl,nmjlpq->nmjkpq',
                             self.implicit_x_advection_matrix, distribution.arr_spectral)
        next_arr = cp.einsum('mpqr,nmjkpr->nmjkpq',
                             self.implicit_y_advection_matrix, next_arr)

        # full explicit adams-bashforth nonlinear momentum flux step
        distribution.compute_moment1(grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                            field=dynamic_field, grid=grid)
        electric_x_rhs = self.space_flux.ampere_x(distribution=distribution,
                                                  dynamic_field=dynamic_field, grid=grid)
        electric_y_rhs = self.space_flux.ampere_y(distribution=distribution,
                                                  dynamic_field=dynamic_field, grid=grid)
        magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        next_arr += self.dt * (
            (23 / 12 * phase_space_rhs -
             4 / 3 * previous_phase_space_fluxes[0] +
             5 / 12 * previous_phase_space_fluxes[1]))
        dynamic_field.electric_x.arr_spectral += self.dt * (
            (23 / 12 * electric_x_rhs -
             4 / 3 * previous_dynamic_fluxes[0][0] +
             5 / 12 * previous_dynamic_fluxes[0][1])
        )
        dynamic_field.electric_y.arr_spectral += self.dt * (
            (23 / 12 * electric_y_rhs -
             4 / 3 * previous_dynamic_fluxes[1][0] +
             5 / 12 * previous_dynamic_fluxes[1][1])
        )
        dynamic_field.magnetic_z.arr_spectral += self.dt * (
            (23 / 12 * magnetic_z_rhs -
             4 / 3 * previous_dynamic_fluxes[2][0] +
             5 / 12 * previous_dynamic_fluxes[2][1])
        )

        # further half crank-nicholson fractional advection step
        next_arr = cp.einsum('njkl,nmjlpq->nmjkpq',
                             self.implicit_x_advection_matrix, next_arr)
        distribution.arr_spectral = cp.einsum('mpqr,nmjkpr->nmjkpq',
                                              self.implicit_y_advection_matrix, next_arr)

        # save fluxes
        previous_phase_space_fluxes = [phase_space_rhs, previous_phase_space_fluxes[0]]
        previous_dynamic_fluxes = [[electric_x_rhs, previous_dynamic_fluxes[0][0]],
                                   [electric_y_rhs, previous_dynamic_fluxes[1][0]],
                                   [magnetic_z_rhs, previous_dynamic_fluxes[2][0]]]
        return previous_phase_space_fluxes, previous_dynamic_fluxes

    def rk_stage_update(self, stage, arr0, arr1, rhs):
        return (
                self.rk_coefficients[stage, 0] * arr0 +
                self.rk_coefficients[stage, 1] * arr1 +
                self.rk_coefficients[stage, 2] * self.dt * rhs
        )
