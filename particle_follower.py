"""
These globalish scripts will be deleted and fragmented in order to make a kind of importable package
"""


#making functions that follow a given set of particleIDs trough time, then generates a folder with tables in the following structure
#time_tables
#|-'subhalo_%s'%subhaloid
# |-snap_99.csv
# |-snap_98.csv
# |-(...)
# |-snap_3.csv
# |-snap_2.csv
# |-snap_1.csv
#|-'subhalo_%s'%subhaloid2
# |-snap_99.csv
# |-snap_98.csv
# |-(...)
# |-snap_3.csv
# |-snap_2.csv
# |-snap_1.csv
#AND THE STRUCTURE FOR EVERY CSV file:
#x,y,z,ParticleIDs
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import illustris_python as il
from tqdm import tqdm



def load_counterrotating_particles(subhaloid):
    #loads the counterrotating particles of the given subhalo to a dataframe with the following keys
    #Mass coordinates Velocities Metallicity Potentialenergy formation_time Distance_center Angular_momentum Circularity ParticleIDs
    filename = 'tables/subhalo'
    filename += str(subhaloid)
    filename += '.0_counterrotating.csv'
    subhalo_df_counterrotating = pd.read_csv(filename)
    return(subhalo_df_counterrotating)

def load_stellar_particles(snapshot):
    #loads all stellar particles for a given snapshot and gives a dictionary with the keys
    #Coordinates ParticleIDs
    fields = ['Coordinates','ParticleIDs']
    basePath = '/virgotng/universe/IllustrisTNG/L35n2160TNG/output'
    star_data = il.snapshot.loadSubset(basePath, snapshot, 'star', fields=fields)
    return(star_data)

def filter_particles_byID(subhalo_df, star_data):
    #given a dataframe for a set of particles with ParticleIDs as a key, this function filters the full star data by the aformentioned ID
    #at the end a dataframe with the coordinates and ParticleIDs for the star_data is obtained.
    filtered_star_data_df = pd.DataFrame()
    filtro_particulas = np.isin(star_data['ParticleIDs'],subhalo_df['ParticleIDs'])
    
    filtered_star_data_df['x'] = star_data['Coordinates'][filtro_particulas,0]
    filtered_star_data_df['y'] = star_data['Coordinates'][filtro_particulas,1]
    filtered_star_data_df['z'] = star_data['Coordinates'][filtro_particulas,2]
    filtered_star_data_df['ParticleIDs'] = star_data['ParticleIDs'][filtro_particulas]
    
    return(filtered_star_data_df)

def generate_time_table(subhalo_df, subhalospos_arr, snapshot):
    #generates a single time table dataframe
    print('Cargando particulas en snapshot %s'%snapshot)
    star_data = load_stellar_particles(snapshot)
    print('Comenzando filtrado')
    star_data_df = filter_particles_byID(subhalo_df, star_data)
    print('cambio del marco de referencia')
    snapid = 99 - snapshot
    x = subhalospos_arr[snapid][0]
    y = subhalospos_arr[snapid][1]
    z = subhalospos_arr[snapid][2]
    star_data_df['x'] = star_data_df['x'] - x
    star_data_df['y'] = star_data_df['y'] - y
    star_data_df['z'] = star_data_df['z'] - z
    return(star_data_df)

def generate_time_tables(subhaloid, start_snap, end_snap):
    #generates a set of time tables and saves them
    print('Cargando particulas del subhalo')
    subhalo_df = load_counterrotating_particles(subhaloid)
    subhalospos_arr = obtain_subhalo_pos(subhaloid)
    for snapshot in range(start_snap,end_snap+1):
        star_data_df = generate_time_table(subhalo_df, subhalospos_arr, snapshot)
        star_data_df.to_csv(f'time_tables/{subhaloid}/snap_{snapshot}.csv')
        print(f'Tabla para particulas CR {subhaloid} en la snapshot {snapshot} CONSTRUIDA')
    return('Tablas generadas')

def obtain_subhalo_pos(subhaloid):
    #generates an array that contains the position across time of the subhalo.
    #'x:',test[snapid][0],'y:',test[snapid][1],'z:',test[snapid][2]
    basePath = '/virgotng/universe/IllustrisTNG/L35n2160TNG/output'
    fields = ['SubhaloPos']
    subhalopos_arr = il.sublink.loadTree(basePath,99,subhaloid,fields=fields,onlyMPB=True)
    return(subhalopos_arr)

def load_subhalo_tree(subhaloid):
    #generates a dictionary with all the particles of a given subhalo for a specific snapshot.
    basePath = '/virgotng/universe/IllustrisTNG/L35n2160TNG/output'
    Fields = ['SubfindID','SnapNum','SubhaloPos']
    arbolito = il.sublink.loadTree(basePath, 99, subhaloid, fields=Fields,onlyMPB=True)
    return(arbolito)

def generate_time_table_full(subhaloid, snapid, subhalopos_arr):
    basePath = '/virgotng/universe/IllustrisTNG/L35n2160TNG/output'
    Fields = ['Coordinates','ParticleIDs']
    snapshot = 99 - snapid
    print('Cargando particulas del subhalo')
    subhalo_dict = il.snapshot.loadSubhalo(basePath, snapshot, subhaloid, 'stellar', fields=Fields)
    x = subhalopos_arr[snapid][0]
    y = subhalopos_arr[snapid][1]
    z = subhalopos_arr[snapid][2]
    print('Transformando Dict a DataFrame')
    subhalo_df = pd.DataFrame()
    subhalo_df['ParticleIDs'] = subhalo_dict['ParticleIDs'][:]
    subhalo_df['x'] = subhalo_dict['Coordinates'][:,0]
    subhalo_df['y'] = subhalo_dict['Coordinates'][:,1]
    subhalo_df['z'] = subhalo_dict['Coordinates'][:,2]
    print('Cambiando marco de referencia')
    subhalo_df['x'] = subhalo_df['x'] - x
    subhalo_df['y'] = subhalo_df['y'] - y
    subhalo_df['z'] = subhalo_df['z'] - z
    return(subhalo_df)

def generate_time_tables_full(subhaloid, start_snap, end_snap):
    print('Cargando tree del subhalo')
    arbolito = load_subhalo_tree(subhaloid)
    subhalopos_arr = arbolito['SubhaloPos']
    for snapshot in range(start_snap,end_snap+1):
        snapid = 99 - snapshot
        subhaloid_i = arbolito['SubfindID'][snapid]
        subhalo_df = generate_time_table_full(subhaloid_i, snapid, subhalopos_arr)
        subhalo_df.to_csv(f'time_tables_full/{subhaloid}/snap_{snapshot}.csv')
        print(f'Tabla para particulas {subhaloid} en la snapshot {snapshot} CONSTRUIDA')
    return('Tablas generadas')
