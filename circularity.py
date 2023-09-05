import illustris_python as il
import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.linalg import eig


#funciones de rotacion mediante momentum angular
def spherical_coords_from_vector(vector):
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

#funciones de rotacion mediante tensor de inercia
def inertia_tensor(Masas, Coordenadas, CentralPos  ):
    #calcula el tensor de inercia mediante las masas, coordenadas, y posicion central de un set de particulas
    #en caso de querer acotar por radio, formacion, etc, debe hacerse antes de correr el comando
    
    #reordenando el array
    xyz = np.squeeze(Coordenadas)
    if xyz.ndim == 1:
        xyz = np.reshape( xyz, (1,3) )
    
    #moviendo marco de referencia al de la posicion central
    for i in range(3):
        xyz[:,i] -= CentralPos[i]
        
    #Construccion del momento de inercia
    I = np.zeros( (3,3), dtype='float32' )

    I[0,0] = np.sum( Masas * (xyz[:,1]*xyz[:,1] + xyz[:,2]*xyz[:,2]) )
    I[1,1] = np.sum( Masas * (xyz[:,0]*xyz[:,0] + xyz[:,2]*xyz[:,2]) )
    I[2,2] = np.sum( Masas * (xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) )
    I[0,1] = -1 * np.sum( Masas * (xyz[:,0]*xyz[:,1]) )
    I[0,2] = -1 * np.sum( Masas * (xyz[:,0]*xyz[:,2]) )
    I[1,2] = -1 * np.sum( Masas * (xyz[:,1]*xyz[:,2]) )
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    return(I)

def diagonalization_of_inertia(I):
    #entrega la matriz que diagonaliza el tensor de inercia
    autovalores,autovectores = eig(I)
    #reordenando autovectores y autovalores de manera ascendente
    ind_sort = np.argsort(autovalores)
    autovalores = autovalores[ind_sort]
    autovectores = autovectores[ind_sort]
    
    print("Autovalores \n",autovalores,"\n Autovectores \n",autovectores)
    #utiliza los autovectores para formar la matriz de rotacion
    rotation_matrix = inv(autovectores)
    
    return(rotation_matrix)

##Funciones de rotacion

def star_particles_rotated_once_eulermethod(subhaloID,snapNum,basepath,radius_limit,minmetal,maxmetal,snap_header):
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

def star_particles_rotated_once(subhaloID,snapNum,basepath,radius_limit,minmetal,maxmetal,snap_header):
    #ROTACION MEDIANTE EL TENSOR DE INERCIA
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
    stars['Coordinates'] /= h
    stars['GFM_Metallicity'] /= 0.0127
    rHalf /= h
    shPos /= h
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
    
    
    #obtencion del tensor de inercia
    I = np.matrix(inertia_tensor(stars['Masses'], stars['Coordinates'], shPos ))
    #obtencion de la matriz de rotacion
    S = np.matrix(diagonalization_of_inertia(I))
    S_1 = np.matrix(inv(S))
    print(S_1 * I * S,"|<-----Tensor de inercia Diagonalizado")
    stars['Coordinates'] = np.dot(stars['Coordinates'], S)
    stars['Velocities'] -= shVel
    stars['Velocities'] = np.dot(stars['Velocities'], S)
    
    return(stars, S, S_1)


## Funciones de calculo de circularidades individuales, las deja registradas en el diccionario de las particulas

def circularities_eulermethod(subhaloID,snapNum,basepath,radius_limit,minmetal,maxmetal,radius_limit_rotation,minmetal_rotation,maxmetal_rotation):
    #calcula las circularidades individuales dentro de un radio, y por sobre un valor de metalicidad, rotando con el metodo de momento angular y angulos
    
    #Extraer constantes de hubble, factor de escala, y otras cosas para conversion comoving--->peculiar--->Fisicas
    snap_header = pd.read_csv('snap_header.csv')
    redshift_z = np.round(snap_header['redshift'][snapNum-1],3)
    scalefactor_a = snap_header['scalefactor'][snapNum-1]
    
    #esto no es necesario, ya que h y los omegas son valores fijos en verdad, (._.)
    h = snap_header['h'][snapNum-1]
    H_0 = h/10
    omega_0 = snap_header['omega_0'][snapNum-1]
    omega_lambda = snap_header['omega_lambda'][snapNum-1]
    
    w_a = omega_0/(omega_lambda*(scalefactor_a**3))
    H_a = H_0 * np.sqrt(omega_lambda) * np.sqrt(1 + w_a)
    
    radius_limit = (radius_limit/h)*scalefactor_a
    radius_limit_rotation = (radius_limit_rotation/h)*scalefactor_a
    stars, M = star_particles_rotated_once_eulermethod(subhaloID,snapNum,basepath,radius_limit_rotation,minmetal_rotation,maxmetal_rotation,snap_header)
    
    campos = ['Masses','Coordinates','Velocities','GFM_Metallicity','Potential','GFM_StellarFormationTime','ParticleIDs']
    
    
    #rotando todas las particulas
    stars = il.snapshot.loadSubhalo(basepath,snapNum,subhaloID, 'star', fields=campos)

    subhalo = il.groupcat.loadSingle(basepath, snapNum, subhaloID=subhaloID)
    shPos = subhalo['SubhaloPos']
    shVel = subhalo['SubhaloVel']
    rHalf = subhalo['SubhaloHalfmassRadType'][4]


    stars['Coordinates'] -= shPos
    stars['Velocities']  -= shVel
    
    #pasando datos a valores fisicos
    stars['Masses'] *= (1e10/h)
    stars['Coordinates'] /= h
    stars['GFM_Metallicity'] /= 0.0127
    stars['Velocities'] *= np.sqrt(scalefactor_a)
    stars['Velocities'] = H_a * stars['Coordinates'] + stars['Velocities']
    
    
    rHalf *= (scalefactor_a/h)
    shPos *= (scalefactor_a/h)
    
    #velocidad peculiar a fisica
    shVel =  H_a * shPos + shVel
    
    
    
    #distancia al centro
    stars['Distance_to_center'] = np.array([np.linalg.norm(v) for v in stars['Coordinates']])
    
    #filtrando por radio y metalicidad
    wStars = np.where((stars['Distance_to_center'] <= radius_limit) & (stars['GFM_Metallicity'] >= minmetal) & (stars['GFM_Metallicity'] <= maxmetal) & (stars['GFM_StellarFormationTime'] > 0))
        
    stars['Masses'] = stars['Masses'][wStars]
    stars['Velocities'] = stars['Velocities'][wStars]
    stars['Coordinates'] = stars['Coordinates'][wStars]
    stars['GFM_Metallicity'] = stars['GFM_Metallicity'][wStars]
    stars['Potential'] = stars['Potential'][wStars]
    stars['ParticleIDs'] = stars['ParticleIDs'][wStars]
    stars['GFM_StellarFormationTime'] = stars['GFM_StellarFormationTime'][wStars]
    stars['Distance_to_center'] = stars['Distance_to_center'][wStars]
    
    #filtrado
    
    
    

    stars['Coordinates'] = np.array([np.array(np.dot(M, coords))[0] for coords in stars['Coordinates']])
    stars['Velocities'] = np.array([np.array(np.dot(M, vels))[0] for vels in stars['Velocities']])


    stars['Angular_Momentum'] = np.cross(stars['Coordinates'],stars['Velocities'])
    print("-----------------------\n",sum(stars['Angular_Momentum']))

    norm_velocities = [np.linalg.norm(v) for v in stars['Velocities']]
    norm_velocities = np.array(norm_velocities)
    norm_velocities

    stars['Specific_Energy'] = (0.5*norm_velocities**2) + stars['Potential']

    indices_ordenados = np.argsort(stars['Specific_Energy'])

    stars['Masses'] = stars['Masses'][indices_ordenados]
    stars['Coordinates'] = stars['Coordinates'][indices_ordenados]
    stars['Velocities'] = stars['Velocities'][indices_ordenados]
    stars['GFM_Metallicity'] = stars['GFM_Metallicity'][indices_ordenados]
    stars['Potential'] = stars['Potential'][indices_ordenados]
    stars['GFM_StellarFormationTime'] = stars['GFM_StellarFormationTime'][indices_ordenados]
    stars['Distance_to_center'] = stars['Distance_to_center'][indices_ordenados]
    stars['Angular_Momentum'] = stars['Angular_Momentum'][indices_ordenados]
    stars['Specific_Energy'] = stars['Specific_Energy'][indices_ordenados]
    stars['ParticleIDs'] = stars['ParticleIDs'][indices_ordenados]

    jz = stars['Angular_Momentum'][:,2]
    npm = 50
    max_jz = [np.max(jz[:i+npm]) if i < npm else np.max(jz[i-npm:]) if i > len(jz)-npm else np.max(jz[i-npm:i+npm]) for i in range(len(jz))]
    max_jz = np.array(max_jz)


    circularity = jz/max_jz
    stars['Circularity'] = circularity

    return(stars)

def circularities_diagmethod(subhaloID,snapNum,basepath,radius_limit,metallicity_cutoff,radius_limit_rotation,minmetal_rotation,maxmetal_rotation):
    #calcula las circularidades individuales, mediante el metodo de rotacion por diagonalizacion del tensor de inercia
    
    #Extraer constantes de hubble, factor de escala, y otras cosas para conversion comoving--->peculiar--->Fisicas
    snap_header = pd.read_csv('snap_header.csv')
    redshift_z = np.round(snap_header['redshift'][snapNum-1],3)
    scalefactor_a = snap_header['scalefactor'][snapNum-1]
    
    #esto no es necesario, ya que h y los omegas son valores fijos en verdad, (._.)
    h = snap_header['h'][snapNum-1]
    H_0 = h/10
    omega_0 = snap_header['omega_0'][snapNum-1]
    omega_lambda = snap_header['omega_lambda'][snapNum-1]
    
    w_a = omega_0/(omega_lambda*(scalefactor_a**3))
    H_a = H_0 * np.sqrt(omega_lambda) * np.sqrt(1 + w_a)
    
    stars, S, S_1 = star_particles_rotated_once(subhaloID,snapNum,basepath,radius_limit_rotation,minmetal_rotation,maxmetal_rotation,snap_header)
    
    campos = ['Masses','Coordinates','Velocities','GFM_Metallicity','Potential','GFM_StellarFormationTime','ParticleIDs','ParticleIDs']
    
    
    #rotando todas las particulas
    stars = il.snapshot.loadSubhalo(basepath,snapNum,subhaloID, 'star', fields=campos)

    subhalo = il.groupcat.loadSingle(basepath, snapNum, subhaloID=subhaloID)
    shPos = subhalo['SubhaloPos']
    shVel = subhalo['SubhaloVel']
    rHalf = subhalo['SubhaloHalfmassRadType'][4]


    stars['Coordinates'] -= shPos
    stars['Velocities']  -= shVel
    
    #pasando datos a valores fisicos
    stars['Masses'] *= (1e10/h)
    stars['Coordinates'] /= h
    stars['GFM_Metallicity'] /= 0.0127
    rHalf /= h
    shPos /= h
    
    #distancia al centro
    stars['Distance_to_center'] = np.array([np.linalg.norm(v) for v in stars['Coordinates']])
    
    #filtrando por radio y metalicidad
    if metallicity_cutoff != False:
        wStars = np.where((stars['Distance_to_center'] <= radius_limit) & (stars['GFM_Metallicity'] >= metallicity_cutoff) & (stars['GFM_StellarFormationTime'] > 0))
    else:
        wStars = np.where( (stars['Distance_to_center'] <= radius_limit) & (stars['GFM_StellarFormationTime'] > 0))
        
    stars['Masses'] = stars['Masses'][wStars]
    stars['Velocities'] = stars['Velocities'][wStars]
    stars['Coordinates'] = stars['Coordinates'][wStars]
    stars['GFM_Metallicity'] = stars['GFM_Metallicity'][wStars]
    stars['Potential'] = stars['Potential'][wStars]
    stars['Distance_to_center'] = stars['Distance_to_center'][wStars]
    stars['ParticleIDs'] = stars['ParticleIDs'][wStars]
    
    
    #filtrado

    stars['Coordinates'] = np.array([np.array(np.dot(coords,S))[0] for coords in stars['Coordinates']])
    stars['Velocities'] = np.array([np.array(np.dot(vels,S))[0] for vels in stars['Velocities']])


    stars['Angular_Momentum'] = np.cross(stars['Coordinates'],stars['Velocities'])
    print("-----------------------\n",sum(stars['Angular_Momentum']))

    norm_velocities = [np.linalg.norm(v) for v in stars['Velocities']]
    norm_velocities = np.array(norm_velocities)
    norm_velocities

    stars['Specific_Energy'] = (0.5*norm_velocities**2) + stars['Potential']

    indices_ordenados = np.argsort(stars['Specific_Energy'])

    stars['Masses'] = stars['Masses'][indices_ordenados]
    stars['Coordinates'] = stars['Coordinates'][indices_ordenados]
    stars['Velocities'] = stars['Velocities'][indices_ordenados]
    stars['GFM_Metallicity'] = stars['GFM_Metallicity'][indices_ordenados]
    stars['Potential'] = stars['Potential'][indices_ordenados]
    stars['Distance_to_center'] = stars['Distance_to_center'][indices_ordenados]
    stars['Angular_Momentum'] = stars['Angular_Momentum'][indices_ordenados]
    stars['Specific_Energy'] = stars['Specific_Energy'][indices_ordenados]
    stars['ParticleIDs'] = stars['ParticleIDs'][indices_ordenados]

    jz = stars['Angular_Momentum'][:,2]
    npm = 50
    max_jz = [np.max(jz[:i+npm]) if i < npm else np.max(jz[i-npm:]) if i > len(jz)-npm else np.max(jz[i-npm:i+npm]) for i in range(len(jz))]
    max_jz = np.array(max_jz)


    circularity = jz/max_jz
    stars['Circularity'] = circularity

    return(stars)