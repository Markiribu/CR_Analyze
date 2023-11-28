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

def table_rotate(table,rotation_matrix):
    """
    Rotates the table of xyz coordinates by the given matrix.
    Parameters:
    - table (array) an array of shape (N,3)
    - rotation_matrix (numpy.Matrix) the rotation matrix
    Returns:
    - table (array) the array of shape (N,3) multiplied by the rotation matrix
    """
    for i in range(len(table)):
        xyz = table[i]
        xyz = np.dot(rotation_matrix, xyz)
        table[i] = xyz
    return(table)

def table_rotated_once_angularmomenta(tabla, reference_tabla, debug=False):
    """
    Rotates a table of particles by using a given reference table. Both must be in dict form with values of "Velocities" and "Coordinates"
    Parameters:
    - tabla (dict) a dictionary with the form {"Coordinates":numpy.array(N,3),"Velocities":numpy.array(3,N)}
    - reference_tabla (dict) a dictionary with the form {"Coordinates":numpy.array(N,3),"Velocities":numpy.array(3,N)}
    Returns:
    - tabla (dict), the same dictionary but with its coordinates and velocities rotated
    - M (np.array(3,3)) the rotation matrix calculated and used. ideally for later saving.
    """
    # We assume that the conversion to physical coordinates has been done already, as such we calculate the angular momenta from the reference table that is going to be used to rotate
    # We assume as well that (0,0,0) would be the center of the subhalo or galaxy
    reference_tabla["Angular_Momentum"] = np.cross(reference_tabla["Coordinates"],reference_tabla["Velocities"])
    reference_J = sum(reference_tabla["Angular_Momentum"]) #Referential angular momentum
    r, theta, phi = spherical_coords_from_vector(reference_J) #Angles of rotation
    M = matrix_from_spherical(r, theta, phi)
    if debug: print(np.dot(M, reference_J)) #In case of debugging, check rotated angular momentum.

    #Rotate the table "tabla" by the reference matrix
    tabla["Coordinates"] = table_rotate(tabla["Coordinates"], M)
    tabla["Velocities"] = table_rotate(tabla["Velocities"],M)

    return(tabla,M)

def table_rotated_n_angularmomenta(tabla,debug=False):
    """
    Rotates the table multiple times trough an algorithm of consecutive smaller spheres, it assumes that tabla contains the LMAO sigo escribiendo de ahi
    """
    return(tabla)
