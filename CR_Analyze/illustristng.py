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
                   deep_tracking=False):
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
    deep_tracking (bool) Whether to run the 2nd search step, using the latest subhalo before getting to the main branch.
    Returns:
    origins (dict) a dictionmary with fields (subfindid, snap) with the corresponding particleids
    in case no origin was determined it will be given under 'undefined'.
    """
    origins = {}
    # required dependencies
    optional_dependencies(hdf5=False)
    import illustris_python as il
    # Check parameters for disallowed values
    if (maxsnapdepth >= 99) or (maxsnapdepth < 0):
        raise Exception('maxsnapdepth must be between 0-98')
    #####
    # Load initial tree
    fieldstoload = ['SubfindID','NextProgenitorID','SubhaloID','SubhaloIDRaw','SnapNum','FirstProgenitorID','SubhaloGrNr']
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
            nextprogenitordata = il.snapshot.loadSubhalo(basepath, nextsnapnum, nextsubfindid, 'star', fields=['ParticleIDs','Masses'])
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
                origin_name = 'SubfindID:' + str(nextsubfindid) + '|Snap:' + str(nextsnapnum)
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
    # If deep_tracking is on and there are still particleids to find then continue searching
    # deep_tracking will first estimate the snapshot that the particle was accreted into the main branch
    # then it will estimate the snapshots at which the particle was incorporated into the FoF Halo
    # Having these 2 data points then, it searches for the last subhalo outside of the main branch that the particle was a part of.
    if (deep_tracking is True) and (len(particleidsnotfound) != 0):
        # Lets setup the SH_origins, saving the snap when the particles were incorporated into the main branch.
        SH_origins = {} # format {'55':np.array(particleids),...}
        particleidsSHnotfound = particleidsnotfound.copy()
        # Lets setup FoF_origins, saving snap at which the particles appear in the Halo
        FoF_origins = {}
        particleidsFoFnotfound = particleidsnotfound.copy()
        # Begin first passtrough trough all snapshots
        for snapid in range(0,maxsnapid):
            #inverse snapid, to start from the oldest snapshot
            snapid_inv = maxsnapid - snapid - 1
            # Lets start checking the Subhalo
            snapnum = merger_tree['SnapNum'][snapid_inv]
            haloid = merger_tree['SubhaloGrNr'][snapid_inv]
            subfindid = merger_tree['SubfindID'][snapid_inv]
            if particleidsSHnotfound.size != 0:
                subhalodata = il.snapshot.loadSubhalo(basepath, snapnum, subfindid,
                                                      'star', fields=['ParticleIDs','Masses'])
                # check is not empty
                if subhalodata['count'] != 0:
                    indexSHfound = np.isin(particleidsSHnotfound,subhalodata['ParticleIDs'])
                    indexSHnotfound = np.invert(indexSHfound)
                    particleidsSHfound = particleidsSHnotfound[indexSHfound]
                    particleidsSHnotfound = particleidsSHnotfound[indexSHnotfound]
                    if particleidsSHfound.size != 0:
                        origin_name = str(snapnum - 1)
                        SH_origins[origin_name] = particleidsSHfound
                    pass
                # cleanupdata
                del subhalodata
                pass
            # Now check the FoF
            if particleidsFoFnotfound.size != 0:
                fofdata = il.snapshot.loadHalo(basepath, snapnum, haloid,
                                               'star', fields=['ParticleIDs','Masses'])
                # check is not empty
                if fofdata['count'] != 0:
                    indexFoFfound = np.isin(particleidsFoFnotfound,fofdata['ParticleIDs'])
                    indexFoFnotfound = np.invert(indexFoFfound)
                    particleidsFoFfound = particleidsFoFnotfound[indexFoFfound]
                    particleidsFoFnotfound = particleidsFoFnotfound[indexFoFnotfound]
                    if particleidsFoFfound.size != 0:
                        origin_name = str(snapnum - 1)
                        FoF_origins[origin_name] = particleidsFoFfound
                    pass
                # cleanupdata
                del fofdata
                pass
            pass
        # With that we have SH_origins and FoF origins, we need to make them into batches now for revision.
        batches = batch_maker(SH_origins,FoF_origins)
        # Now the batches are in format {"minsnap|maxsnap":[particleids]}
        # So checking should be direct.
        for batch_name in batches.keys():
            particleidbatch = batches[batch_name]
            minsnap = int(batch_name[:2])
            maxsnap = int(batch_name[3:])
            for snapnum_front in range(minsnap,maxsnap):
                snapnum = maxsnap - snapnum_front + minsnap
                snapid = 99 - snapnum
                # Now what is the subfindid of the main branch at this snap?
                mainsubfindid = merger_tree['SubfindID'][snapid]
                # And the HaloID
                FoFID = merger_tree['SubhaloGrNr'][snapid]
                # For this snap lets see the list of subfindids for the given halo.
                halodata = il.groupcat.loadSingle(basepath, snapnum, haloID=FoFID)
                centralsubfindid = halodata['GroupFirstSub']
                subhalosinFoF = halodata['GroupNsubs']
                furthestsubfindid = centralsubfindid + subhalosinFoF
                subfindids_arr = np.arange(centralsubfindid, furthestsubfindid)
                # delete the halodata, no longer being used
                del halodata
                # We will not considerate the main branch in the search obviously.
                for subfindid in subfindids_arr:
                    if subfindid == mainsubfindid:
                        continue
                    # Lets load the particle data
                    subfind_data = il.snapshot.loadSubhalo(basepath, snapnum, subfindid, 'stars', fields=['ParticleIDs','Masses'])
                    # check is not empty
                    if subfind_data['count'] != 0:
                        # Now lets check if there are common particleids
                        indexbatchfound = np.isin(particleidbatch,subfind_data['ParticleIDs'])
                        indexbatchnotfound = np.invert(indexbatchfound)
                        particleidsbatchfound = particleidbatch[indexbatchfound]
                        if particleidsbatchfound.size != 0:
                            particleidbatch = particleidbatch[indexbatchnotfound]
                            # lets take a moment to delete the ids from the global not found list
                            indexfound = np.isin(particleidsnotfound,particleidsbatchfound)
                            particleidsnotfound = particleidsnotfound[indexfound]
                            origin_name = 'SubfindID:' + str(subfindid) + '|Snap:' + str(snapnum)
                            if origin_name in origins.keys():
                                logging.info(f'This origin was already registered, appending...')
                                origins[origin_name] = np.append(origins[origin_name],particleidsbatchfound)
                                pass
                            else:
                                origins[origin_name] = particleidsbatchfound
                                pass
                            pass
                        pass
                    # lets clear the subfind_data
                    del subfind_data
        pass
    # If even then there are still particleids to find then assign them undefined
    if len(particleidsnotfound) != 0:
        origins['Undefined'] = particleidsnotfound
        pass
    # Now return all of the estimated origins.
    return origins


def batch_maker(SH_origins,FoF_origins):
    """
    Makes a dictionary with the snapshot ranges to search the particleids for.
    Parameters:
    SH_origins (dict) format {'snap':[particleids]}
    FoF_origins (dict) format {'snap':[particleids]}
    Returns:
    batches (dict) format {'FoFsnap|SHsnap':[particleids]}
    """
    batches = {}
    # For making the batches we want to go trough all combinations of FoF origins and SH origins, making pairs.
    logging.info('Making pairs of snaps')
    for FoFsnap in FoF_origins.keys():
        particleidsinFoF = FoF_origins[FoFsnap]
        # The FoFsnap always should be less than the SHsnap
        for SHsnap in SH_origins.keys():
            particleidsinSH = SH_origins[SHsnap]
            if int(SHsnap) < int(FoFsnap):
                logging.info(f'Beware SHsnap lower than FoFsnap! SHsnap:{SHsnap}|FoFsnap:{FoFsnap}')
                continue
            elif particleidsinSH.size == 0:
                logging.info('This SH_origins is already empty, not checking...')
                continue
            # Obtain particleids in common
            indexincommon = np.isin(particleidsinSH,particleidsinFoF)
            indexnotincommon = np.invert(indexincommon)
            particleidsincommon = particleidsinSH[indexincommon]
            # what particleids are in common, if empty, do not save.
            if particleidsincommon.size != 0:
                batches_name = str(FoFsnap) + '|' + str(SHsnap)
                batches[batches_name] = particleidsincommon
                # And lets not consider the already checked particleids anymore.
                SH_origins[SHsnap] = particleidsinSH[indexnotincommon]
                pass
            continue
        # All possible SH_origins checked, next FoF!
        continue
    # All pairs made!
    return batches


def sort_keys(origins):
    """
    Given a dictionary of origins it gives the sorted key list according to the snap number.
    Parameters:
    origins (dict) dictionary such that {'SubfindID:N|Snap:N':[particleids],...}
    """
    keys = list(origins.keys())
    keys_sorted = keys.copy()
    sorting = True
    while sorting:
        sorting = False
        for i in range(len(keys)):
            name = keys_sorted[i]
            nextname = keys_sorted[i+1]
            # obtain the snap
            snap = int([element.split(sep=':') for element in name.split(sep='|')][1][1])
            nextsnap = int([element.split(sep=':') for element in nextname.split(sep='|')][1][1])
            if nextsnap > snap:
                keys_sorted[i] = nextname
                keys_sorted[i+1] = name
                sorting = True
    return keys_sorted


def tree_merger(origins, basepath = '/virgotng/universe/IllustrisTNG/TNG50-1/output'):
    """
    """
    optional_dependencies(hdf5=False)
    import illustris_python as il
    fixed_origins = {}
    # before using the origins for checking the satellites, we need to sort the keys in descending snap order.
    sorted_keys = sort_keys(origins)
    i = 0
    while len(sorted_keys) > 1:
        main_origin = sorted_keys[0]
        subfindid = int([s.split(sep=':') for s in main_origin.split(sep='|')][0][1])
        snap = int([s.split(sep=':') for s in main_origin.split(sep='|')][1][1])
        tree = il.sublink.loadTree(basepath, snap, subfindid, fields=['SubfindID'])
        satellite_origins = {}
        satellite_origins[main_origin] = origins[main_origin]
        sorted_keys.remove(main_origin)
        sorted_Keys_copy = sorted_keys.copy()
        for key_tocheck in sorted_keys_copy:
            subfindid_tocheck = int([s.split(sep=':') for s in main_origin.split(sep='|')][0][1])
            snap_tocheck = int([s.split(sep=':') for s in main_origin.split(sep='|')][1][1])
            subfindid_intree = tree[snap - snap_tocheck]
            if (subfindid_intree == subfindid_tocheck):
                satellite_origins[key_tocheck] = origins[key_tocheck]
                sorted_keys.remove(key_tocheck)
                pass
            pass
        # this satellite is fully considered now.
        satellitename = 'satellite_' + str(i)
        satellite_origins['All'] = np.array([],dtype=int64)
        # Now change to next satellite to consider or finish
        for key in satellite_origins:
            if key == 'All':
                continue
            satellite_origins['All'] = np.append(satellite_origins['All'],satellite_origins[key])
            pass
        satellitename = 'satellite_' + str(i)
        fixed_origins[satellitename] = satellite_origins
        i += 1
        pass
    # Now we have taken every main branch into account
    return(fixed_origins)
