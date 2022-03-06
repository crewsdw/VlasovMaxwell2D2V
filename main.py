import numpy as np
import cupy as cp
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes
import timestep as ts
import data

# parameters
vt_c = 0.3

# Geometry and grid parameters
elements, order = [32, 128, 11, 11], 8

# Grid
wavenumber_x = 0.125
wavenumber_y = 0.01
length_x, length_y = 2.0 * np.pi / wavenumber_x, 2.0 * np.pi / wavenumber_y
lows = np.array([-length_x/2, -length_y/2, -11, -11])
highs = np.array([length_x/2, length_y/2, 11, 11])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# phase shifts
x_modes, y_modes = 3, 4
phase_shift1 = 2*cp.pi*cp.random.randn(x_modes, y_modes)
phase_shift2 = 2*cp.pi*cp.random.randn(x_modes, y_modes)

# Variables: distribution
distribution = var.Distribution(resolutions=elements, order=order)
# distribution.initialize_maxwellian(grid=grid)
distribution.initialize_aniso_max(grid=grid, vx=1, vy=2, phase_shift1=phase_shift1, phase_shift2=phase_shift2)
# static and dynamic fields
dynamic_fields = fields.Dynamic(resolutions=[elements[0], elements[1]], vt_c=vt_c)
dynamic_fields.initialize(grid=grid, phase_shift1=phase_shift1, phase_shift2=phase_shift2)

# Plotter: check out IC
plotter = my_plt.Plotter2D(grid=grid)
# plotter.spatial_scalar_plot(scalar=distribution.moment0, title='Density', spectrum=False)
# plotter.spatial_vector_plot(vector=distribution.moment1)
plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_x, title='Electric x')
plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_y, title='Electric y')
plotter.scalar_plot_interpolated(scalar=dynamic_fields.magnetic_z, title='Magnetic z')
plotter.show()

# Set up fluxes
flux = fluxes.DGFlux(resolutions=elements, order=order, grid=grid, nu=1)
flux.initialize_zero_pad(grid=grid)
space_flux = fluxes.SpaceFlux(c=1/vt_c)

# Time information
dt, stop_time = 0.8e-2, 40
steps = int(stop_time // dt) + 1

# Save data
datafile = data.Data(folder='data\\', filename='weibel_222')
datafile.create_file(distribution=distribution.arr_nodal.get(),
                     density=distribution.moment0.arr_nodal.get(),
                     current_x=distribution.moment1.arr_nodal[0, :, :].get(),
                     current_y=distribution.moment1.arr_nodal[1, :, :].get(),
                     electric_x=dynamic_fields.electric_x.arr_nodal.get(),
                     electric_y=dynamic_fields.electric_y.arr_nodal.get(),
                     magnetic_z=dynamic_fields.magnetic_z.arr_nodal.get())

# Set up time-stepper
stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps, grid=grid,
                     phase_space_flux=flux, space_flux=space_flux)
stepper.main_loop(distribution=distribution, dynamic_field=dynamic_fields, grid=grid, datafile=datafile)


plotter.scalar_plot_interpolated(scalar=distribution.moment0, title='Density')
plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_x, title='Electric x')
plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_y, title='Electric y')
plotter.scalar_plot_interpolated(scalar=dynamic_fields.magnetic_z, title='Magnetic z')
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ex_energy,
                         y_axis='Electric x energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ey_energy,
                         y_axis='Electric y energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.bz_energy,
                         y_axis='Magnetic z energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=(stepper.ex_energy + stepper.ey_energy +
                                                                stepper.bz_energy + stepper.thermal_energy),
                         y_axis='Total energy', log=False)

plotter.show()
