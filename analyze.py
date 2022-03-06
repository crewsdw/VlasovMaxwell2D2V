import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import data
import cupy as cp
import matplotlib.pyplot as plt

# parameters
vt_c = 0.3

# Geometry and grid parameters
# elements, order = [20, 42, 11, 11], 8
# elements, order = [32, 64, 11, 11], 8
elements, order = [32, 128, 11, 11], 8

# Grid
wavenumber_x = 0.125
wavenumber_y = 0.01
length_x, length_y = 2.0 * np.pi / wavenumber_x, 2.0 * np.pi / wavenumber_y
lows = np.array([-length_x / 2, -length_y / 2, -11, -11])
highs = np.array([length_x / 2, length_y / 2, 11, 11])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Read data
DataFile = data.Data(folder='data\\', filename='weibel_222')
(time_data, distribution_data, density_data, current_x_data, current_y_data,
 electric_x_data, electric_y_data, magnetic_z_data, total_eng, total_den) = DataFile.read_file()

min_f, max_f = np.amin(distribution_data), np.amax(distribution_data[0, :, :, :, :, :, :])
min_n, max_n = np.amin(density_data), np.amax(density_data)
min_jx, max_jx = np.amin(current_x_data), np.amax(current_x_data)
min_jy, max_jy = np.amin(current_y_data), np.amax(current_y_data)
min_ex, max_ex = np.amin(electric_x_data), np.amax(electric_x_data)
min_ey, max_ey = np.amin(electric_y_data), np.amax(electric_y_data)
min_bz, max_bz = np.amin(magnetic_z_data), np.amax(magnetic_z_data)

if min_f < 0:
    min_f = 0
print(min_f), print(max_f)

# Set up plotter
plotter = my_plt.Plotter2D(grid=grid)
v_plotter = my_plt.VelocityPlotter(grid=grid)

jump = 0
for idx, time in enumerate(time_data[jump:]):
    print('Data at time {:0.3e}'.format(time))
    # Unpack data, distribution, density
    distribution = var.Distribution(resolutions=elements, order=order)
    distribution.arr_nodal = cp.asarray(distribution_data[idx])
    distribution.moment0.arr_nodal = cp.asarray(density_data[idx])
    # distribution.moment1.arr_nodal[0, :, :] = cp.asarray(current_x_data[idx])
    # distribution.moment1.arr_nodal[1, :, :] = cp.asarray(current_y_data[idx])
    distribution.fourier_transform(), distribution.moment0.fourier_transform()

    jx = var.SpaceScalar(resolutions=[elements[0], elements[1]])
    jy = var.SpaceScalar(resolutions=[elements[0], elements[1]])
    jx.arr_nodal = cp.asarray(current_x_data[idx])
    jy.arr_nodal = cp.asarray(current_y_data[idx])

    dynamic_fields = fields.Dynamic(resolutions=[elements[0], elements[1]], vt_c=vt_c)
    dynamic_fields.electric_x.arr_nodal = cp.asarray(electric_x_data[idx])
    dynamic_fields.electric_y.arr_nodal = cp.asarray(electric_y_data[idx])
    dynamic_fields.magnetic_z.arr_nodal = cp.asarray(magnetic_z_data[idx])

    plotter.scalar_plot_interpolated(scalar=distribution.moment0, title='Density', cb_lim=[min_n, max_n],
                                     save='figs\\n\\' + str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=jx, title='Current x', save='figs\\jx\\' +
    #                                                                     str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=jy, title='Current y', save='figs\\jy\\' +
    #                                                                     str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_x, title='Electric x', save='figs\\ex\\' +
    #                                  str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=dynamic_fields.electric_y, title='Electric y', save='figs\\ey\\' +
    #                                  str(idx) + '.png')
    plotter.stream_plot_interpolated(vector_x=jx, vector_y=jy, title='Current', save='figs\\j\\' +
                                                                                     str(idx) + '.png')
    plotter.stream_plot_interpolated(vector_x=dynamic_fields.electric_x,
                                     vector_y=dynamic_fields.electric_y, title='Electric Field', save='figs\\E\\' +
                                                                                                      str(idx) + '.png')
    plotter.scalar_plot_interpolated(scalar=dynamic_fields.magnetic_z, title='Magnetic z', cb_lim=[min_bz, max_bz],
                                     save='figs\\bz\\' + str(idx) + '.png')

    plt.close('all')
