# Methods used for executing on illustrisTNG data
# This can probably be generalized to other simulations
from .circularity import *
import logging

def optional_dependencies(hdf5=True,illustris=True,progressbar=True):
    if hdf5:
        try:
            import h5py
        except ImportError:
            raise SystemExit("h5py is required in order to save computed data!")
    if illustris:
        try:
            import illustris_python as il
        except ImportError:
            raise SystemExit("illustris_python is required in order to load data!")
    if progressbar:
        try:
            from tqdm import tqdm
        except ImportError:
            raise SystemExit("tqdm is required for showing progress bars in long tasks")
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
    optional_dependencies(progressbar=False)
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
    optional_dependencies(progressbar=False)
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


def append_haloID(subfindid, snap, snaps_arr, basepath, savepath=''):
    """
    Obtains all of the HaloIDs for the given subhalo.
    Esentially generating a merger tree that tracks the HaloID of the subhalo considered.
    As it is being estimated it adds it as an attribute of the snapshot
    under filename 'subhalo_subfindid.hdf5'.
    Parameters:
    subfindid (int) the SubhaloID, only central subhalos are recommended.
    snap (int) Snapshot at which the subhalo is given.
    snaps_arr (1-D np.array) array of snapshots  where to obtain the HaloID.
    Returns:
    0 in case there where no problems.
    """
    # Check required libraries
    optional_dependencies()
    # load the tree
    print(f'loading tree of {subfindid}')
    firstsubfind_tree = il.sublink.loadTree(basepath,snap,subfindid,fields=["GroupFirstSub"])
    # Start snapshot cycle
    with tqdm(total=len(snaps_arr)) as pbar:
        for snapnum in snaps_arr:
            # Begin cycle
            snapid = 99 - snapnum
            # Load groupcat of all halos and their respective firstsubfind
            pbar.set_description(f'loading halocat of snap {snapnum}')
            halocat = il.groupcat.loadHalos('/virgotng/universe/IllustrisTNG/TNG50-1/output',snapnum,fields=["GroupFirstSub"])
            # Now we strip the unnecesary values like -1
            # since we know that the halo we search for has at least 1 subhalo
            halocat = halocat[halocat != -1]
            # Now lets do and intersection of the 2 arrays
            # its an intersection in order to catch problems where the halo appears twice
            pbar.set_description(f'processing: snap {snapnum}')
            xy, xindx, yindx = np.intersect1d(halocat,firstsubfind_tree[snapid],return_indices=True)
            haloID = xindx
            if len(haloID) > 1:
                logging.warning("I seem to have found more than 1 halo, please revise")
            # update the progressbar
            pbar.set_description(f'saving: snap {snapnum}')
            # now that haloID has been calculated, register as attribute
            with h5py.File(f'{savepath}subhalo_{subfindid}.hdf5', 'r+') as data:
                data[f'{snapnum}'].attrs['HaloID'] = haloID[0]
            pbar.update(1)
            pass
        pass
    return 0


def CR_track(subfindid):
    """
    Tracks the counterrotating particles back in time and assigns them an origin.
    Specifically only considers exsitu particles.
    """
    return 0
