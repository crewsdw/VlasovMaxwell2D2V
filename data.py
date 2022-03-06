import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self, distribution, density, current_x, current_y, electric_x, electric_y, magnetic_z):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1],
                                       distribution.shape[2], distribution.shape[3],
                                       distribution.shape[4], distribution.shape[5]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density]),
                             chunks=True,
                             maxshape=(None, density.shape[0], density.shape[1]),
                             dtype='f')
            f.create_dataset('current_x', data=np.array([current_x]),
                             chunks=True,
                             maxshape=(None, current_x.shape[0], current_x.shape[1]),
                             dtype='f')
            f.create_dataset('current_y', data=np.array([current_y]),
                             chunks=True,
                             maxshape=(None, current_y.shape[0], current_y.shape[1]),
                             dtype='f')
            f.create_dataset('electric_x', data=np.array([electric_x]),
                             chunks=True,
                             maxshape=(None, electric_x.shape[0], electric_x.shape[1]),
                             dtype='f')
            f.create_dataset('electric_y', data=np.array([electric_y]),
                             chunks=True,
                             maxshape=(None, electric_y.shape[0], electric_y.shape[1]),
                             dtype='f')
            f.create_dataset('magnetic_z', data=np.array([magnetic_z]),
                             chunks=True,
                             maxshape=(None, magnetic_z.shape[0], magnetic_z.shape[1]),
                             dtype='f')
            f.create_dataset('time', data=[0.0], chunks=True, maxshape=(None,))
            f.create_dataset('total_energy', data=[], chunks=True, maxshape=(None,))
            f.create_dataset('total_density', data=[], chunks=True, maxshape=(None,))

    def save_data(self, distribution, density, current_x, current_y, electric_x, electric_y, magnetic_z, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new timeline
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['current_x'].resize((f['current_x'].shape[0] + 1), axis=0)
            f['current_y'].resize((f['current_y'].shape[0] + 1), axis=0)
            f['electric_x'].resize((f['electric_x'].shape[0] + 1), axis=0)
            f['electric_y'].resize((f['electric_y'].shape[0] + 1), axis=0)
            f['magnetic_z'].resize((f['magnetic_z'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['density'][-1] = density
            f['current_x'][-1] = current_x
            f['current_y'][-1] = current_y
            f['electric_x'][-1] = electric_x
            f['electric_y'][-1] = electric_y
            f['magnetic_z'][-1] = magnetic_z
            f['time'][-1] = time

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            pdf = f['pdf'][()]
            den = f['density'][()]
            jx = f['current_x'][()]
            jy = f['current_y'][()]
            ex = f['electric_x'][()]
            ey = f['electric_y'][()]
            bz = f['magnetic_z'][()]
            total_eng = f['total_energy'][()]
            total_den = f['total_density'][()]
        return time, pdf, den, jx, jy, ex, ey, bz, total_eng, total_den
