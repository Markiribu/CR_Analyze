# Methods used for executing on illustrisTNG data
# This can probably be generalized to other simulations
from .circularity import *
import logging
import numpy as np

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
    # import them
    import illustris_python as il
    import h5py
    from tqdm import tqdm
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
            if len(haloID) == 0:
                logging.warning("The subhalo does not seem to belong to a halo, please revise, skipping snap")
                continue
            # update the progressbar
            pbar.set_description(f'saving: snap {snapnum}')
            # now that haloID has been calculated, register as attribute
            with h5py.File(f'{savepath}subhalo_{subfindid}.hdf5', 'r+') as data:
                data[f'{snapnum}'].attrs['HaloID'] = haloID[0]
            pbar.update(1)
            pass
        pass
    return 0


def obtain_all_nextprogenitors(tree,rootid,snapid):
    """
    Gives the index in the tree of all nextprogenitors for the given snapshotid.
    Esentially all other parents besides the main branch.
    """
    nextprogenitors_dict = {}
    nextprogenitorid = tree['NextProgenitorID'][snapid+1]
    nextprogenitorsids_list = []
    while nextprogenitorid != -1:
        indexintree = nextprogenitorid - rootid
        nextprogenitorsids_list.append(int(indexintree))
        nextprogenitorid = tree['NextProgenitorID'][indexintree]
    nextprogenitorsids_arr = np.array(nextprogenitorsids_list, dtype='int64')
    nextprogenitors_dict['indexesintree'] = nextprogenitorsids_arr
    nextprogenitors_dict['count'] = len(nextprogenitorsids_arr)
    if len(nextprogenitorsids_list) == 0:
        return nextprogenitors_dict
    for field in tree.keys():
        if field is 'count':
            continue
        nextprogenitors_dict[field] = tree[field][nextprogenitors_dict['indexesintree']]
    return nextprogenitors_dict


def exsitu_tracker(subfindid, snapnum, particleIDs, maxsnapdepth=10,
                   basepath='/virgotng/universe/IllustrisTNG/TNG50-1/output',
                   deep_tracking=False, halotracker=[]):
    """
    Tracks the counterrotating particles back in time and assigns them an origin. Must be Exsitu particles.
    It does this in a 2 step process. first it looks at the merger trees, and finds to which nextprogenitor
    The particles come from, then it takes any particles that a progenitor wasn't determined and searches for the last
    subhalo it was from before getting to the main branch by loading the main halo.
    The 2nd step is optional and is available as a parameter.
    Parameters:
    subfindid (int) the SubfindID
    snap (int) SnapNum
    maxsnapdepth (int) Maximum snap that the tracker searches through
    deep_tracking (bool) Whether to run the 2nd search step by loading all particles in the FoF Halo.
    halo_tracker (np.array(int)) Array with the HaloIDs starting with snap 99
    Returns:
    origins (dict) a dictionmary with fields (subfindid, snap) with the corresponding particleids
    in case no origin was determined it will be given under 'undefined'
    """
    origins = {}
    # required dependencies
    optional_dependencies(hdf5=False)
    import illustris_python as il
    # Check parameters for disallowed values
    if (maxsnapdepth >= 99) or (maxsnapdepth < 0):
        raise Exception('maxsnapdepth must be between 0-98')
    if (deep_tracking is True) and (len(halotracker) == 0):
        raise Exception('A list of HaloIDs must be given under halotracker')
    #####
    # Load initial tree
    fieldstoload = ['SubfindID','NextProgenitorID','SubhaloID','SubhaloIDRaw','SnapNum','FirstProgenitorID']
    merger_tree = il.sublink.loadTree(basepath, snapnum, subfindid, fields=fieldstoload)
    # Cycle trough the given snapshots
    maxsnapid = 99 - maxsnapdepth
    rootid = merger_tree['SubhaloID'][0]
    particleidsnotfound = particleIDs.copy() # we assign a new variable to the particleIDs that we'll be searching for
    for snapid in range(0,maxsnapid):
        # first obtain all nextprogenitors for the snapshot within the tree
        nextprogenitor_dict = obtain_all_nextprogenitors(merger_tree,rootid,snapid)
        # Now lets check one by one
        for i in range(nextprogenitor_dict['count']):
            nextsubfindid = nextprogenitor_dict['SubfindID'][i]
            nextsnapnum = nextprogenitor_dict['SnapNum'][i]
            # Load stellar particles and their IDs
            nextprogenitordata = il.snapshot.loadSubhalo(basepath, nextsnapnum, nextsubfindid, 'star', fields=['ParticleIDs'])
            # if count is zero, then skip
            if nextprogenitordata['count'] == 0:
                continue
            # Now we check for any intersection between the particleids in the nextprogenitor and
            # The particle ids we are still searching for.
            indexfound = np.isin(particleidsnotfound,nextprogenitordata['ParticleIDs'])
            indexnotfound = np.invert(indexfound)
            particleidsfound = particleidsnotfound[indexfound]
            if len(particleidsfound) == 0:
                continue
            else:
                particleidsnotfound = particleidsnotfound[indexnotfound]
                origin_name = 'SubfindID:' + str() + '|Snap:' + str()
                origins[origin_name] = particleidsfound
                if len(particleidsnotfound) == 0:
                    break
                pass
            continue
        # before going with next snapshot lets check there are still particleids to find
        if len(particleidsnotfound) == 0:
            break
        # Otherwise just continue
        continue
    # All snapshots checked for nextprogenitors
    # If deep_search is on and there are still particleids to find then continue searching
    if (deep_tracking is True) and (len(particleidsnotfound) != 0):
        print('This is not Finished!')
        pass
    # If even then there are still particleids to find then assign them undefined
    if len(particleidsnotfound) != 0:
        origins['Undefined'] = particleidsnotfound
        pass
    # Now return all of the estimated origins.
    return origins
