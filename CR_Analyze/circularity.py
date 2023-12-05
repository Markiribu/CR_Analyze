# script charged with calculating the circularities of an 
# imported hdf5 file with illustris_python
# 

import illustris_python as il
import numpy as np

# importar funcion propia
from .rotator import table_rotated_n_angularmomenta


def frame_of_reference_change(r, v, central_pos, central_vel, L_box):
    """
    A function that changes the frame of reference
    in a periodic box.
    Parameters
    - r np.array(N,3) coordinates of particles
    - v np.array(N,3) velocities of particles
    - central_pos np.array(1,3) coordinate of center
    - central_vel np.array(1,3) velocity of center
    - L_box float the length of the periodic box
    Returns
    - r np.array(N, 3) the coordinates of the particles
    - v np.array(N, 3) the velocities of the particles
    """
    r -= central_pos
    v -= central_vel
    # now we need to check if any coordinate from the particles
    # goes beyond L_box/2
    for i in range(3):
        periodicity_fix = np.where(r[:,i]>L_box/2,L_box,0)
        r[:,i] -= periodicity_fix
        continue
    # now the whole system should be centered around central_pos
    return (r, v)


def comoving_to_physical(r, v, snapNum=None, scalefactor_a=None, h=None, omega_0=None, omega_lambda=None):
    """
    Changes the comoving system to a physical one
    It requires either the snapshot number (assuming IllustrisTNG)
    or to be given the cosmological parameters explicitly
    """
    if snapNum != None:
        # We load the snapshot header and obtain the cosmological parameters needed
        continue
    else:
        if ((scalefactor_a == None) | (h == None) | (omega_0 == None) | (omeg == None)):
            # Then no cosmological parameters were defined
            continue
        continue
    H_0 = h/10
    w_a = omega_0/(omega_lambda*(scalefactor_a**3))
    H_a = H_0*np.sqrt(omega_lambda)*np.sqrt(1+w_a)
    # Now we know for sure that we have the cosmological parameters needed for conversion
    r *= scalefactor_a
    v = H_a*r+scalefactor_a*v
    # With that, physicial velocities have been calculated
    return (r, v)


def calculate_the_circularities():
    return (0)


def generate_cr_hdf5(subhaloID, snapNum, basepath):
    return (0)
