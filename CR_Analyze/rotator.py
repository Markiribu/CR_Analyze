#script charged with rotating particles in a dataframe that are distributed in a disk-like form.
import illustris_python as il
import numpy as np
import pandas as pd

def spherical_coords_from_vector(vector):
    """
    A function that takes a cartesian vector, and gives its spherical coordinates
    Parameters:
    - vector (array/list of floats): must be in a [x,y,z] format in cartesian coordinates
    Returns:
    - r (float) spherical coordinate
    - theta (float) spherical coordinate
    - phi (float) spherical coordinate
    """
    x = vector[0] ; y = vector[1] ; z=vector[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    if x > 0:
        phi = np.arctan(y/x)
    elif ((x < 0) and (y >= 0)):
        phi = np.arctan(y/x) + np.pi
    elif ((x < 0) and (y < 0)):
        phi = np.arctan(y/x) - np.pi
    elif ((x == 0) and (y > 0)):
        phi = np.pi / 2
    elif ((x == 0) and (y < 0)):
        phi = (-1 * (np.pi / 2))
    elif ((x == 0) and (y == 0)):
        phi = "INVALIDO"
    return( r, theta, phi)

def matrix_from_spherical( r, theta, phi):
    """
    Calculates the rotation matrix needed to rotate a given vector to make it parallel to the z axis in a cartesian coordinate system
    Parameters:
    - r (float) spherical coordinate
    - theta (float) spherical coordinate
    - phi (float) spherical coordinate
    Returns:
    - M (numpy.Matrix) the rotation matrix
    """
    Rz = np.matrix([[np.cos(-phi),-np.sin(-phi),0],
                    [np.sin(-phi), np.cos(-phi),0],
                    [           0,            0,1]
                   ])
    
    Ry = np.matrix([[ np.cos(-theta),            0,np.sin(-theta)],
                    [            0,              1,             0],
                    [-np.sin(-theta),            0,np.cos(-theta)]
                   ])
    
    M = np.dot(Ry,Rz)
    return(M)

def star_particles_rotated_once_eulermethod(subhaloID,snapNum,basepath,radius_limit,minmetal,maxmetal,snap_header):
    """
    Loads and rotates the particles of a given subhalo, it requires an object named snap_header in a global context, that corrects comoving systems to physical
    Parameters:
    - subhaloID (int) The identication of the subhalo
    - snapNum (int) The snapshot number
    - basepath (str) Directory in which the simulation data is allocated
    - radius_limit (float) maximum distance to the center in kpc of the rotation reference particles
    - minmetal (float) minimum metallicity in solar metallicity units for the reference particles
    - maxmetal (float) maximum metallicity in solar metallicity units for the reference particles
    - snap_header (pd.DataFrame) DataFrame containing information about the snapshots like scalefactor, redshifts, h, etc.
    Returns:
    - stars (pd.DataFrame)
    - M (np.Matrix)
    """
    #ROTACION MEDIANTE METODO DE MOMENTO ANGULAR
    #constantes
    redshift_z = np.round(snap_header['redshift'][snapNum-1],3)
    scalefactor_a = snap_header['scalefactor'][snapNum-1]
    
    #esto no es necesario, ya que h y los omegas son valores fijos en verdad, (._.)
    h = snap_header['h'][snapNum-1]
    H_0 = h/10
    omega_0 = snap_header['omega_0'][snapNum-1]
    omega_lambda = snap_header['omega_lambda'][snapNum-1]
    
    w_a = omega_0/(omega_lambda*(scalefactor_a**3))
    H_a = H_0 * np.sqrt(omega_lambda) * np.sqrt(1 + w_a)
    
    campos = ['Masses','Coordinates','Velocities','GFM_Metallicity','Potential','GFM_StellarFormationTime','ParticleIDs']
    
    #carga de datos
    stars = il.snapshot.loadSubhalo(basepath, snapNum, subhaloID, 'star', fields=campos)
    subhalo = il.groupcat.loadSingle(basepath, snapNum, subhaloID=subhaloID)
    
    shPos = subhalo['SubhaloPos']
    rHalf = subhalo['SubhaloHalfmassRadType'][4]
    shVel = subhalo['SubhaloVel']
    
    #cambio unidades de las particulas a terminos fisicos
    stars['Masses'] *= (1e10/h)
    stars['Coordinates'] = stars['Coordinates']*(scalefactor_a/h)
    stars['GFM_Metallicity'] /= 0.0127
    stars['Velocities'] *= np.sqrt(scalefactor_a)
    stars['Velocities'] = H_a * stars['Coordinates'] + stars['Velocities']
    
    
    rHalf *= (scalefactor_a/h)
    shPos *= (scalefactor_a/h)
    
    #velocidad peculiar a fisica
    shVel =  H_a * shPos + shVel
    
    
    
    print(rHalf)
    
    #distancias al centro
    vector_desde_centro = stars['Coordinates'] - shPos
    stars['Distance_to_center'] = np.array([np.linalg.norm(v) for v in vector_desde_centro])
    
    #filtrando por distancia al centro y por metalicidad similar a la solar
    wStars = np.where( (stars['GFM_Metallicity'] >= minmetal) & (stars['GFM_Metallicity'] <= maxmetal) & (stars['Distance_to_center'] <= radius_limit) & (stars['GFM_StellarFormationTime'] > 0) )
    stars['Masses'] = stars['Masses'][wStars]
    stars['Coordinates'] = stars['Coordinates'][wStars]
    stars['GFM_Metallicity'] = stars['GFM_Metallicity'][wStars]
    stars['Distance_to_center'] = stars['Distance_to_center'][wStars]
    stars['Velocities'] = stars['Velocities'][wStars]
    stars['Potential'] = stars['Potential'][wStars]
    stars['ParticleIDs'] = stars['ParticleIDs'][wStars]
    stars['count'] = len(stars['Masses'])
    
    #cambio de marco de referencia
    stars['Coordinates'] -= shPos
    stars['Velocities'] -= shVel
    
    
    #calculo del momento angular
    stars['Angular_Momentum'] = np.cross(stars['Coordinates'],stars['Velocities'])
    total_angular_momentum = sum(stars['Angular_Momentum'])
    
    r, theta, phi = spherical_coords_from_vector(total_angular_momentum)
    M = matrix_from_spherical(r, theta, phi)
    print(np.dot(M, total_angular_momentum))
    
    
    stars['Coordinates'] = np.array([np.array(np.dot(M, Coord))[0] for Coord in stars['Coordinates']])
    stars['Velocities'] = np.array([np.array(np.dot(M, Velocidad))[0] for Velocidad in stars['Velocities']])
    
    return(stars, M)

