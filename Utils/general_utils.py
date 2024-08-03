import numpy as np
import scipy
from scipy.spatial.transform import Rotation


# Converts an angle in degree to radians

def to_radians(angle):
    return angle*np.pi/180

# Converts an angle in radians to degrees

def to_degrees(angle):
    return angle*180/np.pi

# Computes the trace of a given matrix

def trace(matrix):
    trace = 0.0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            trace = trace + matrix[r][c]
    return trace

# Checks if a given matrix is a rotation matrix

def is_rotation_matrix(matrix):
    is_orthogonal = np.allclose(np.transpose(matrix).dot(matrix), np.identity(matrix.shape[0]))
    is_determinant_one = np.isclose(abs(np.linalg.det(matrix)), 1)
    return is_orthogonal and is_determinant_one

# Returns the inverse of a rotation matrix (it's transpose)

def inverse_rotation(matrix):
    inverted_rotation = np.transpose(matrix)
    return inverted_rotation

# Invert a homogeneous transformation matrix
@ staticmethod
def inverse_homogeneous_matrix(matrix):
    rotated_matrix = matrix[:3, :3]
    position = matrix[:3, 3].reshape(-1, 1)
    inverse_rotation_matrix = inverse_rotation(rotated_matrix)
    new_position = -1 * apply_rotation(inverse_rotation_matrix, position)
    inverted_homogeneous_matrix = np.block([[inverse_rotation_matrix, new_position], [0, 0, 0, 1]])
    return inverted_homogeneous_matrix

# Transforms a position by a homogeneous transformation matrix

def transform_position(transformation,position):
    return (transformation @ np.block([[position], [1]]))[0:3]

# Transforms a vector by a homogeneous transformation matrix

def transform_vector(transformation, vector):
    return (transformation @ np.block([[vector], [0]]))[0:3]

# Converts a rotation matrix and a translation to a homogeneous matrix (both rotation and translation happens based on old axis)

def homogeneous_matrix(rotation_matrix, translation):
    if rotation_matrix is None:
        rotation_matrix = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    if translation is None:
        translation = np.array([[0.0],[0.0],[0.0]])
    return np.block([
        [rotation_matrix, translation],
        [np.zeros((1, 3)), 1]
    ])

# Applies rotation on a position

def apply_rotation (rotation, position):
    return rotation @ position

# Returns a 2D rotation matrix given an angle

def rotation_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

# Returns a 3D rotation matrix in the x dimension given an angle

def rotation_matrix_x(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

# Returns a 3D rotation matrix in the y dimension given an angle

def rotation_matrix_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

# Returns a 3D rotation matrix in the z dimension given an angle

def rotation_matrix_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

# Returns a 3D rotation matrix in the z dimension given an angle

def translation(x,y,z):
    return np.array([[x],[y],[z]])


# Returns a rotation matrix using the euler angles (alpha, beta, gamma)

def rotation_matrix_rpy(roll, pitch, yaw):
    rotation_matrix_x = rotation_matrix_x(roll)
    rotation_matrix_y = rotation_matrix_y(pitch)
    rotation_matrix_z = rotation_matrix_z(yaw)
    euler_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    return euler_rotation_matrix

# Returns Roll (x, gamma), Pitch (y, beta) and Yaw (z, aplha) (ZYX Euler Rotations) given a rotation matrix

def find_rpy(rotation_matrix):
    pitch = np.arctan2(-rotation_matrix[2][0], (rotation_matrix[0][0]*rotation_matrix[0][0] + rotation_matrix[1][0]*rotation_matrix[1][0])**0.5)
    yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])
    roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
    return roll, pitch, yaw

# Finds skew symmetric matrix given a 3D vector [x, y, z]

def to_skew_3d(vector):
    return np.array([
        [0.0, -vector[2][0], vector[1][0]],
        [vector[2][0], 0.0, -vector[0][0]],
        [-vector[1][0], vector[0][0], 0.0]])

# Finds skew symmetric matrix given a 6D vector [x, y, z]

def to_skew_6d(vector):
    top = vector[0:3,:]
    bottom = vector[3:6,:]
    return np.block([
        [to_skew_3d(top), bottom],
        [0.0,0.0,0.0,0.0]
    ])

# Finds the 3D vector [x, y, z] from a skew symmetric matrix

def from_skew_3d(matrix):
    return np.array([matrix[2][1], matrix[0][2], matrix[1][0]])

# Finds the 4D vector [x, y, z] from a skew symmetric matrix

def from_skew_6d(matrix):
    skew_w = matrix[0:3,0:3]
    w = from_skew_3d(skew_w)
    v = matrix[0:3,3]
    return to_twist_vector(v,w)

# Checks if a given matrix is a skew symmetric matrix

def is_skew_symmetric(matrix):
    if matrix[0][1] != -matrix[1][0]:
        return False
    if matrix[0][2] != -matrix[2][0]:
        return False
    if matrix[1][2] != -matrix[2][2]:
        return False
    return True

# Returns the ross product two 3x3 matrices

def cross_product_3d(vector1, vector2):
    return to_skew_3d(vector1) @ vector2

# Finds spatial angular velocity and returns [w]

def find_spatial_angular_velocity(rot, d_rot):
    return d_rot @ rot.T

# Finds body angular velocity and returns [w]

def find_body_angular_velocity(rot, d_rot):
    return rot.T @ d_rot

# Finds linear velocity given an angular velocity and position in a simular frame

def find_linear_velocity(angular_velocity, position):
    return cross_product_3d(angular_velocity, position)


# Returns a twist given an angular velocity and linear velocity

def to_twist_vector(linear_velocity, angular_velocity):
    return np.block([[angular_velocity,linear_velocity]]).T

# Returns a twist given an angular velocity and linear velocity

def to_twist_matrix(linear_velocity, angular_velocity):
    twist_vector = to_twist_vector(linear_velocity, angular_velocity)
    return twist_vector_to_twist_matrix(twist_vector)

# Angular velocity to rotation matrix

def angular_to_rotation(w):
    return exp_3d_skew(w)

# Rotation matrix to angular vector

def rotation_to_angular(matrix):
    skew_w = scipy.linalg.logm(matrix)
    w = from_skew_3d(skew_w)
    return w

# Transformation matrix to twist vector

def transformation_to_twist(transformation,v_top=False):
    skew_twist = log_matrix(transformation)
    twist = from_skew_6d(skew_twist)
    if v_top:
        temp = np.copy(twist[0:3,:])
        twist[0:3,:] = twist[3:,:]
        twist[3:,:]=temp
    return twist

# Returns a homogeneous transformation given a twist [[w0],[w1],[w2],[v0],[v1],v[2]]

def twist_vector_to_twist_matrix(vector):
    return exp(to_skew_6d(vector))

# Exponential of a matrix

def exp(matrix):
    return scipy.linalg.expm(matrix)

# Takes the exponential of a 3d vector like the angular velocity

def exp_3d_skew(w):
    w_norm = norm(w)
    w_unit = unit_vector(w)
    return np.identity(3) + to_skew_3d(w_unit)*np.sin(w_norm) + (to_skew_3d(w_unit)@to_skew_3d(w_unit))*(1-np.cos(w_norm))

# Takes the log of a 3x3 matrix like a rotation

def log_matrix(matrix):
    return scipy.linalg.logm(matrix)
    if np.array_equal(matrix,np.identity(3)):
        return np.array([1.0,0.0,0.0])
    if np.trace(matrix) == -1:
        return (1/np.sqrt(2*(1+matrix[2][2]))) * np.array([matrix[0][2],matrix[1][2],1+matrix[2][2]])*np.pi
    magnitude = np.arccos(0.5*(np.trace(matrix)-1))
    skew = (matrix-matrix.T)/ (2*np.sin(magnitude))
    return magnitude*skew

# Gets rotation matrix from a transformation matrix

def get_rotation(transformation):
    return transformation[0:3,0:3]

# Gets translation matrix from a transformation matrix

def get_translation(transformation):
    return np.block([[transformation[0:3,3]]]).T


# Finds norm of a 3d vector

def norm(vector):
    return np.linalg.norm(vector)

# Finds unit vector of a given 3d vector

def unit_vector(vector):
    return vector/norm(vector)

# Finds the adjoint representation of a transformation matrix

def adjoint(transformation):
    R = get_rotation(transformation)
    p = get_translation(transformation)
    adjoint_matrix = np.block([
        [R,np.zeros((3,3))],
        [to_skew_3d(p)@R,R],
    ])
    return adjoint_matrix

# Changes frame of a twist

def transform_twist(transformation_ab, twist_b):
    twist_a = adjoint(transformation_ab)@twist_b
    return twist_a

# Changes frame of a wrench

def transform_wrench(transformation_ab, twist_b):
    twist_a = adjoint(transformation_ab.T) @ twist_b
    return twist_a

# Returns the wrench vector

def wrench(center_mass, force):
    return np.block([center_mass.T,force.T])

# Returns inverse (or pseduo inverse if not invertible)

def inv(x):
    return np.linalg.pinv(x)

# Twist given jacobian and joints velocity

def twist_from_jacobian(J,dq):
    return J@dq

# Joints velocity given jacobian and twist

def dq_from_jacobian(J,V):
    return inv(J)@V

# Torque given jacobian and wrench

def torque_from_jacobian(J,F):
    return J.T@F

# Wrench given jacobian and torque

def wrench_from_jacobian(J,tor):
    return inv(J.T)@tor

# Convert quaternion to a rotation matrix

def quaternion_to_rot(q):

    qw, qx, qy, qz = q[3], q[0], q[1], q[2]

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R

# Convert Transformation Matrix to translation and quaternion rotation

def to_position_quaternion(transformation):
    position = transformation[0:3,3]
    rot = transformation[0:3,0:3]
    rot = Rotation.from_matrix(rot)
    quaternion = rot.as_quat()
    return position, quaternion





