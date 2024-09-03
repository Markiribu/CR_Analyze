# Methods used for executing on illustrisTNG data
# This can probably be generalized to other simulations
from .circularity import *

def optional_dependencies():
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required in order to save computed data!")
    try:
        import illustris_python as il
    except ImportError:
        raise SystemExit("illustris_python is required in order to load data!")
    return 0

def save_snap_data_hdf5(subhaloID, snapnum, basepath, savefile):
    """
    Computes the circularities of the given subhaloID at snapnum snapshot.
    Meaning that both should be given for a galaxy(consider the merger trees).
    Parameters:
    - subhaloID (int) The subhaloID of the galaxy at snapshot snapnum
    - snapnum (int) The snapshot to lookup
    - basepath (str) The main simulation data folder\
    - savefile (HDF5 file object) file to save or append data to
    """
    optional_dependencies()
    index = 99 - snapnum
    logging.info(f"selected {subhaloID} with snap {snapnum}, index {index}")
    # Load header data, used afterwards when considering the cosmology
    snap_data = il.groupcat.loadHeader(basePath=basepath,
                                       snapNum=snapnum)
    # Startup of loading particle data from illustris
    fields_to_load = ['Masses', 'Coordinates', 'Velocities',
                      'GFM_Metallicity', 'Potential',
                      'GFM_StellarFormationTime', 'ParticleIDs']
    subhalo_star_data = il.snapshot.loadSubhalo(basePath=basepath,
                                                snapNum=snapnum,
                                                id=subhaloID,
                                                partType='star',
                                                fields=fields_to_load)
    # Data for snapshot snapnum loaded, begin calculation process
    logging.info(f"snapshot {snapnum} loaded")
    logging.info("Loading subhalo data")
    subhalo_data = il.groupcat.loadSingle(basePath=basepath,
                                          snapNum=snapnum,
                                          subhaloID=subhaloID)
    # Subhalo data loaded, obtain central_pos and central_vel
    central_pos = subhalo_data["SubhaloPos"]
    central_vel = subhalo_data["SubhaloVel"]
    L_box = snap_data["BoxSize"]
    # Now changing the frame of reference
    r = subhalo_star_data["Coordinates"]
    v = subhalo_star_data["Velocities"]
    r, v = frame_of_reference_change(r, v,
                                     central_pos, central_vel, L_box)
    # Now changing the coordinates to physical
    r, v = comoving_to_physical(r, v,
                                basePath=basepath, snapNum=snapnum)
    subhalo_star_data["Coordinates"] = r
    subhalo_star_data["Velocities"] = v
    # Before rotating first some measure of radius must be obtained
    # if central is true then search for R200 in the group catalog
    # otherwise halfmassrad is used as radius
    group_catalog = il.groupcat.loadHalos(basePath=basepath,
                                          snapNum=snapnum)
    index_where = np.where(group_catalog["GroupFirstSub"] == subhaloID)[0]
    if len(index_where) == 1:
        centralsubhalo = True
        R_gal = group_catalog["Group_R_Crit200"][index_where]
        logging.info('This subhalo is central!, R200 = ', R_gal)
    else:
        centralsubhalo = False
        R_gal = subhalo_data["SubhaloHalfmassRad"]
        logging.info('The subhalo is not central!, using HalfMassRadius', R_gal)
    # Now begin rotation process
    subhalo_star_data, M_rot = table_rotated_n_angularmomenta(
        subhalo_star_data, R_gal)
    # Now calculate the circularities, this requires the potential
    U = subhalo_star_data["Potential"]
    epsilon, J = calculate_the_circularities(r, v, U)
    # Data calculated, saving to file
    subhalo_star_data["Potential"] = U
    subhalo_star_data["J"] = J
    subhalo_star_data["Circularity"] = epsilon
    # Now particle data is rotated and circularities are calculated
    # Save particle data
    logging.info("Saving the following dictionary as hdf5\n",subhalo_star_data)
    for key in subhalo_star_data.keys():
        savefile[f'{snapnum}/{key}'] = subhalo_star_data[key]
    # Save metadata
    if centralsubhalo is True:
        savefile[f"{snapnum}/"].attrs["R200"] = R_gal
    else:
        savefile[f"{snapnum}/"].attrs["Rgal"] = R_gal
    savefile[f"{snapnum}/"].attrs["Mrot"] = M_rot
    logging.info(f'Data created, circularities computed, {snapnum}')
    return 0


def generate_data_hdf5(subhaloID, snapnum, basepath,
                       savepath='',
                       append=True):
    """
    Integrates all methods to generate a single file
    containing the circularities and angular momentums of all particles,
    as well as additional data used in the calculations(if enabled).
    It saves it inside savepath, on a file named subhalo_ID.hdf5
    The hdf5 file contains datasets for each snapshot,
    this function updates the file accordingly, inside each dataset:
    Metadata:
        R200, CR_Rhalf, disk_mass, CR_mass, CR_index(if enabled)
    Particle Dataset:
    Fields:
        ParticleIDs , Coordinates, Velocities, Potential_energy,
        GFM_StellarFormationTime, GFM_metallicity_solar, R, J, circularity
    Parameters:
    - subhaloID (int) The subhalo id at snap 99 of the galaxy of interest
    - snapnum (int) The snapshot to lookup for data generation
    - basepath (str) The main simulation data folder
    - savepath (str)(optional) folder where computed data is saved
    - append (bool)(optional) if the function should append new data.
    """
    optional_dependencies()
    filename = f'{savepath}subhalo_{subhaloID}.hdf5'
    index = 99 - snapnum
    # now estimate the subhaloid at the requested snap
    subfindid_arr = il.sublink.loadTree(basepath, 99, subhaloID,
                                        fields=['SubfindID'])
    subhaloIDatsnapnum = subfindid_arr[index]
    try:
        with h5py.File(filename, 'r+') as file:
            if append is False:
                print('FILE FOUND, Interrupting...')
                return (1)
            logging.info('File found, computing data and appending in ',filename)
            save_snap_data_hdf5(subhaloIDatsnapnum,
                                snapnum, basepath, savefile=file)
    except OSError:
        with h5py.File(filename, 'w') as file:
            logging.info("File has not been found, creating file with name ",filename)
            save_snap_data_hdf5(subhaloIDatsnapnum,
                                snapnum, basepath, savefile=file)
    return (0)

