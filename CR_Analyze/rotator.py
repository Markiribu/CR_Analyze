# script charged with rotating particles in a dataframe that are distributed in
# a disk-like form.
import numpy as np


def spherical_coords_from_vector(vector):
    """
    A function that takes a cartesian vector
    and gives its spherical coordinates
    Parameters:
    - vector (array/list of floats)
    must be in a [x,y,z] format in cartesian coordinates
    Returns:
    - r (float) spherical coordinate
    - theta (float) spherical coordinate
    - phi (float) spherical coordinate
    """
    x = vector[0]
    y = vector[1]
    z = vector[2]
    if ((x == 0) and (y == 0)):
        phi, theta = 0, 0
        r = z
        return (r, theta, phi)
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
    return (r, theta, phi)


def matrix_from_spherical(r, theta, phi):
    """
    Calculates the rotation matrix needed to rotate a given vector to
    make it parallel to the z axis in a cartesian coordinate system
    Parameters:
    - r (float) spherical coordinate
    - theta (float) spherical coordinate
    - phi (float) spherical coordinate
    Returns:
    - M (numpy.Matrix) the rotation matrix
    """
    Rz = np.matrix([[np.cos(-phi), -np.sin(-phi), 0],
                    [np.sin(-phi), np.cos(-phi), 0],
                    [0,            0, 1]])
    Ry = np.matrix([[np.cos(-theta),            0, np.sin(-theta)],
                    [0,              1,             0],
                    [-np.sin(-theta),            0, np.cos(-theta)]])
    M = np.dot(Ry, Rz)
    return (M)


def table_rotate(table, rotation_matrix):
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
    return (table)


def table_rotated_once_angularmomenta(tabla, reference_tabla, debug=False):
    """
    Rotates a table of particles by using a given reference table.
    Both must be in dict form with values of "Velocities" and "Coordinates"
    Parameters:
    - tabla (dict) a dictionary with the form
        {"Coordinates":numpy.array(N,3),"Velocities":numpy.array(3,N)}
    - reference_tabla (dict) a dictionary with the form
        {"Coordinates":numpy.array(N,3),"Velocities":numpy.array(3,N)}
    Returns:
    - tabla (dict), the resulting dictionary of coordinates
    - M (np.array(3,3)) the rotation matrix calculated and used.
    """
    # We assume that the conversion to physical coordinates has been done,
    # as such we calculate the angular momenta from the reference table
    # that is going to be used to rotate
    # We assume as well that (0,0,0) is the center of the subhalo or galaxy
    reference_tabla["Angular_Momentum"] = np.cross(
        reference_tabla["Coordinates"], reference_tabla["Velocities"])
    # Referential angular momentum
    reference_J = np.sum(reference_tabla["Angular_Momentum"], axis=0)
    # Angles of rotation
    r, theta, phi = spherical_coords_from_vector(reference_J)
    M = matrix_from_spherical(r, theta, phi)
    if debug:
        # In case of debugging, check rotated angular momentum.
        print('New J', np.dot(M, reference_J))
        # And rotation matrix angles, and matrix
        print('Matrix', M)
        # Angle
        print('spherical', r, theta, phi)

    # Rotate the table "tabla" by the reference matrix
    tabla["Coordinates"] = table_rotate(tabla["Coordinates"], M)
    tabla["Velocities"] = table_rotate(tabla["Velocities"], M)

    return (tabla, M)


def table_rotated_n_angularmomenta(tabla, Rgal,
                                   N_rotation=3, Rmin=0, Zmin=0.5, Zmax=1,
                                   debug=False):
    """
    Rotates the table multiple times using consecutive smaller spheres,
    it's assumed that the table "tabla" contains star particles
    , their metallicities and the coordinate system is centered on the
    lowest bound particle of the galaxy.

    "tabla" must have physical values for the coordinates and velocities,
    and the metallicity is given in terms of solar metallicity
    (1 being solar metallicity).
    Parameters:
    - tabla (dict) a dictionary with
    "Coordinates": array(3,N)
    "Velocities": array(3,N)
    "GFM_Metallicity": array(N)
    - Rgal (float) some definition of radius(in kpc) for the galaxy, example:
    R200 Rvir, Ropt, etc.
    - N_rotation (int) number of rotations, 3 by default.
    - Rmin (float) some definition of the radius of the bulge of the galaxy
    by default = 0.
    - Zmin (float) minimum metallicity to consider for reference particles.
    - Zmax (float) maximum metallicity to consider for reference particles.
    - debug (bool) whether to print debugging messages.
    angular momentum or rotation matrix obtained etc.
    Returns:
    - tabla (dict) the dictionary with particles rotated
    - M_rot (matrix) the final rotation matrix
    """
    # We define a list of rotation matrixes,
    # used to obtain the final rotation matrix after the multiple rotations.
    M_list = np.zeros((3, 3, 3))
    # We obtain the list of radius to use
    R_length = Rgal - Rmin
    step = R_length/N_rotation
    # array with the radiuses to use
    R_list = np.arange(Rmin, Rgal, step)
    # Obtain distance to center
    tabla["Distance_to_center"] = np.array(
        [np.linalg.norm(r) for r in tabla["Coordinates"]])
    tabla["GFM_Metallicity_solar"] = tabla["GFM_Metallicity"] / 0.0127
    if debug is True:
        print("Rotation iteration begun")
    for n_index in range(N_rotation):
        if debug is True:
            print(f"Rotation number {n_index + 1}/{N_rotation}")
        # Doing a rotation consists of taking the new max radius,
        # and using as reference only particles inside this radius,
        # and are in the range of given solar metallicity.
        Rmax = R_list[n_index]

        # Now we filter using the new maximum radius
        filtered_index = np.where((tabla["Distance_to_center"] >= Rmin) &
                                  (tabla["Distance_to_center"] <= Rmax) &
                                  (tabla["GFM_Metallicity_solar"] >= Zmin)
                                  & (tabla["GFM_Metallicity_solar"] <= Zmax))

        reference_tabla = {}
        reference_tabla["Coordinates"] = tabla["Coordinates"][filtered_index]
        reference_tabla["Velocities"] = tabla["Velocities"][filtered_index]
        # We have obtained a reference table
        # as such we use to rotate the main table,
        # and redefine the definition of tabla.
        tabla, M = table_rotated_once_angularmomenta(
            tabla, reference_tabla, debug=debug)
        # tabla has been redefined,
        # we need to append M to the list of rotation matrixes.
        M_list[n_index] = M
        # Finished this cycle, so continue
        continue
    # After all the cycles are done tabla should be rotated,
    # and we have M_list to obtain the final rotation matrix.
    # I prefer to make two cycles,
    # one for rotating tabla, and one for obtaining the final matrix,
    # for clarity
    M_rot = M_list[0]
    for n in range(N_rotation-1):
        M_rot = np.dot(M_list[n+1], M_rot)
    # We have tabla and M_rot, so just return them
    return (tabla, M_rot)
