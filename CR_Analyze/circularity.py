# script charged with calculating the circularities of an 
# imported hdf5 file with illustris_python

import illustris_python as il
import numpy as np
import h5py

# import of methods
from .rotator import table_rotated_n_angularmomenta


def frame_of_reference_change(r, v, central_pos, central_vel, L_box):
    """
    A function that changes the frame of reference
    in a periodic box.

    Parameters
    ----------
    r : np.array(N,3)
        coordinates of particles
    v : np.array(N,3)
        velocities of particles
    central_pos : np.array(1,3)
        coordinate of center
    central_vel : np.array(1,3)
        velocity of center
    L_box : float
        the length of the periodic box

    Returns
    -------
    tuple(r, v)
        The r and v arrays with their coordinates changed
        to fit the frame of reference provided
        through central_pos and central_vel.
    """
    r -= central_pos
    v -= central_vel
    # now we need to check if any coordinate from the particles
    # goes beyond L_box/2
    for i in range(3):
        periodicity_fix = np.where(r[:, i] > L_box/2, L_box, 0)
        r[:, i] -= periodicity_fix
        continue
    # now the whole system should be centered around central_pos
    return (r, v)


def comoving_to_physical(r, v,
                         snapNum=None, basePath=None,
                         scalefactor_a=None, h=None,
                         omega_0=None, omega_lambda=None):
    """
    Changes the comoving system to a physical one
    It requires either the snapshot number (assuming IllustrisTNG)
    or to be given the cosmological parameters explicitly

    Parameters
    ----------
    r : np.array(N, 3)
        coordinates of particles
    v : np.array(N, 3)
        velocities of particles
    snapNum : int
        The snapshot to load from the simulation(illustristng)
        used to obtain cosmological parameters,
        otherwise they can be provided individually
    scalefactor_a : float
        The scale factor a
    h : float
        Adimensional hubble constant
    omega_0 : float
        Cosmological parameter
    omega_lambda : float
        Cosmological parameter
    Returns
    -------
    tuple
        The r an v arrays changed to physical magnitudes.
    """
    if ((snapNum is not None) and (basePath is not None)):
        # Load the snapshot header and obtain the cosmological parameters
        snap_header = il.groupcat.loadHeader(basePath, snapNum)
        scalefactor_a = snap_header["Time"]
        h = snap_header["HubbleParam"]
        omega_0 = snap_header["Omega0"]
        omega_lambda = snap_header["OmegaLambda"]
    else:
        if ((scalefactor_a is None) or (h is None) or
                (omega_0 is None) or (omega_lambda is None)):
            # Then not all cosmological parameters were defined
            # and a snapshot number was not given.
            raise Exception(
                """ snapshot number and basepath was not given,
                or not all cosmological parameters were provided""")
    H_0 = h/10
    w_a = omega_0/(omega_lambda*(scalefactor_a**3))
    H_a = H_0*np.sqrt(omega_lambda)*np.sqrt(1+w_a)
    # cosmological parameters needed for conversion, obtained
    r *= scalefactor_a
    v = H_a*r+scalefactor_a*v
    # With that, physicial velocities have been calculated
    return (r, v)


def calculate_the_circularities(r, v, U,
                                npm=50):
    """
    Method in charge of calculating the circularities and angular momentum
    of a set of particles, the coordinates and velocities
    should have been rotated and changed in the frame of reference

    Parameters
    ----------
    r : array(3,N)
        Coordinates of particles,
        already rotated and centered on 0,0
    v : array(3,N)
        Velocities of particles,
        already rotated and changed the frame of reference
    U : array(N)
        Potential energies of the particles
    npm : int
        Max interval in discrete energies to search for
        maximum angular momentum
        (lower values tend to be more accurate,
        but are risky depending on the amount of particles)
        50 by default
    Returns
    -------
    tuple
        The arrays for the circularity of the particles and angular momentum
        respectively in the same order as the coordinates and velocities.
    """
    angular_momentums = np.cross(r, v)

    norm_velocities = np.linalg.norm(v)

    E = (0.5*(norm_velocities**2)) + U

    sorted_indexes = np.argsort(E)
    revert_sorted_indexes = np.argsort(sorted_indexes)

    angular_momentums_sorted = angular_momentums[sorted_indexes]

    jz = angular_momentums_sorted[:, 2]
    jz_abs = abs(jz)

    max_jz = [np.max(jz_abs[:i+npm]) if i < npm
              else np.max(jz_abs[i-npm:]) if i > len(jz_abs)-npm
              else np.max(jz_abs[i-npm:i+npm])
              for i in range(len(jz_abs))]

    circularities_sorted = jz/max_jz

    circularities = circularities_sorted[revert_sorted_indexes]
    return (circularities, angular_momentums)


def generate_data_hdf5(subhaloID, basepath,
                       redundant_data=True, use_CRindex=True, savepath='',
                       debug=False):
    """
    Integrates all methods to generate a single file
    containing the circularities and angular momentums of all particles,
    as well as additional data used in the calculations(if enabled).
    It saves it inside savepath, on a file named subhaloID.hdf5
    The hdf5 file contains datasets for each snapshot,
    this function updates the file accordingly, inside each dataset:
    Metadata:
        R200, CR_Rhalf, disk_mass, CR_mass, CR_index(if enabled)
    Particle Dataset:
    Fields:
        ParticleIDs , Coordinates, Velocities, Potential_energy,
        GFM_StellarFormationTime, GFM_metallicity_solar, R, J, circularity
    """
    try:
        file = h5py.File(f'{savepath}{subhaloID}', 'r')
    except FileNotFoundError:
        if debug is True:
            print("File has not been found, so creation is possible")
        # Startup of loading particle data from illustris
        fields_to_load = ['Masses', 'Coordinates', 'Velocities',
                          'GFM_Metallicity', 'Potential',
                          'GFM_StellarFormationTime', 'ParticleIDs']
        subhalo_star_data = il.loadSubhalo(basePath=basepath, snapNum=snapNum,
                                           id=subhaloID, partType='star',
                                           fields=fields_to_load)
        # Data calculated, saving to file
        with h5py.File(f'Subhalo_{subhaloID}', 'w') as savefile:
            for key in subhalo_star_data.keys():
                    savefile[f'{snapNum}']
    else:
        if debug is True:
            print("File found. unable to create data file")
        file.close()
    return (0)
