import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from Utils.general_utils import *

def set_geom_pose(model, name, pos=None, rot=None, debug_mode=True):
    if debug_mode:
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if pos is not None:
            model.geom_pos[target_geom_id] = pos.tolist()
        if rot is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat,rot.flatten())
            model.geom_quat[target_geom_id] = quat
        model.geom_rgba[target_geom_id][3] = 0.5

def set_site_pose(model, name, pos=None, rot=None, debug_mode=True):
    if debug_mode:
        target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if pos is not None:
            model.site_pos[target_site_id] = pos.tolist()
        if rot is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat,rot.flatten())
            model.site_quat[target_site_id] = quat
        model.site_rgba[target_site_id][3] = 0.0

def set_geom_size(model, name, size):
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_size[target_geom_id] = size

def set_body_pose(model, name, pos=None, rot=None, euler=None, quat=None, debug_mode=True):
    
    if debug_mode:
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if pos is not None:
            model.body_pos[target_body_id] = pos.tolist()
        if rot is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            model.body_quat[target_body_id] = quat
        if quat is not None:
             model.body_quat[target_body_id] = quat
        if quat is not None:
             model.body_quat[target_body_id] = quat
             

def zero_velocities(data):
    data.qvel = np.zeros(data.qvel.shape)

def get_latest_number(folder_path):
        files = os.listdir(folder_path)
        npz_files = [f for f in files if f.endswith('.npz')]
        if not npz_files:
            return -1
        numbers = [int(f.split('.')[0]) for f in npz_files]
        latest_number = max(numbers)
        return latest_number