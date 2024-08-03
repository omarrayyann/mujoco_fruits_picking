import mujoco
import mujoco.viewer
import numpy as np
import time
from Utils.utils import *
import threading
from controller import OpsaceController
from mujoco_ar import MujocoARConnector
import random
import rerun as rr
import cv2
import torch

class ImitationSimulation:
    
    def __init__(self):

        # Configs
        self.scene_path = 'Environment/pick_place_scene.xml'
        self.mjmodel = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.dt = 0.002
        self.grasp = 0
        self.button = 0
        self.placement_time = -1
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=120, width=160)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=120, width=160)
        self.depth_renderer.enable_depth_rendering()
        self.cameras = ["third_pov","gripper_camera"]

        # Override the simulation timestep
        self.pick_object = "orange_fruit"
        self.pick_joint = "orange_joint"
        self.place_object = "green_bowl"
        self.mjmodel.opt.timestep = self.dt
        self.frequency = 1000
        self.target_pos = np.array([0.5, 0.0, 0.4])
        self.target_rot = rotation_matrix_x(np.pi)@np.identity(3)
        self.pos_origin = self.target_pos.copy()
        self.rot_origin = self.target_rot.copy()
        self.target_quat = np.zeros(4)
        self.rerun = False
        self.eef_site_name = 'eef'
        self.site_id = self.mjmodel.site(self.eef_site_name).id
        self.camera_name = 'gripper_camera'
        self.last_recording_time = -1
        self.camera_data = None
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            
        ]

        # Recording
        self.recording_frequency = 10

        # Controller
        self.controller = OpsaceController(self.mjmodel,self.joint_names,self.eef_site_name)
        self.q0 = np.array([0, 0.2448, 0, -1.4204, 0, 1.4765, 0])
        self.dof_ids = np.array([self.mjmodel.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.mjmodel.actuator(name).id for name in self.joint_names])
        self.grasp_actuator_id = self.mjmodel.actuator("fingers_actuator").id 
        self.mjdata.qpos[self.actuator_ids] = self.q0

        # MujocoAR
        self.mujocoAR = MujocoARConnector(controls_frequency=10)

        # if self.rerun:
        #     rr.init("IIWA-KUKA", spawn=True)
    
    def is_valid_position(self, pos1, pos2, min_dist):
        if pos1 is None or pos2 is None:
            return False
        distance = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])) 
        return distance>=min_dist
    
    def send_rerun(self) -> dict:
        data = {}    
        for camera in self.cameras:
            self.rgb_renderer.update_scene(self.mjdata, camera)
            self.depth_renderer.update_scene(self.mjdata, camera)
            data[camera+"_rgb"] = self.rgb_renderer.render()
            data[camera+"_depth"] = self.depth_renderer.render()
        return data
    
    def get_pos_from_range(self,range):
        return np.array([random.uniform(range[0,0], range[0,1]), random.uniform(range[1,0], range[1,1]),range[2,0]])

    def was_placed(self):
        pick_pos = self.mjdata.body(self.pick_object).xpos.copy()
        place_pos = self.mjdata.body(self.place_object).xpos.copy()
        if np.linalg.norm(np.array(pick_pos)[0:2] - np.array(place_pos)[0:2]) <= 0.02 and pick_pos[2]<0.25:
            return True
        
    def fell(self):
        pick_pos = self.mjdata.body(self.pick_object).xpos.copy()
        if pick_pos[2]<0.1 or pick_pos[2]>2 or abs(pick_pos[0]) > 1.0 or abs(pick_pos[1])>1.0:
            return True

    def random_placement(self, min_seperation=0.3):

        place_range = np.array([[0.4,0.7],[-0.33,0.33],[0.21,0.21]])
        pick_range = np.array([[0.4,0.7],[-0.33,0.33],[0.25,0.25]])

        place_range = np.array([[0.61,0.61],[-0.2,-0.2],[0.21,0.21]])
        pick_range = np.array([[0.61,0.61],[0.2,0.2],[0.25,0.25]])

        place_pos, pick_pos = None, None
        while not self.is_valid_position(place_pos,pick_pos,min_seperation):
            place_pos = self.get_pos_from_range(place_range)
            pick_pos = self.get_pos_from_range(pick_range)
        
        self.mujocoAR.pause_updates()
        self.mujocoAR.reset_position()
        self.mjdata.qpos[self.actuator_ids] = self.q0
        mujoco.mj_step(self.mjmodel, self.mjdata)
        self.grasp = 0
        self.mjdata.joint(self.pick_joint).qvel = np.zeros(6)
        self.mjdata.joint(self.pick_joint).qpos = np.block([pick_pos,1,0,0,0])        
        set_body_pose(self.mjmodel,self.place_object,place_pos)
        mujoco.mj_step(self.mjmodel, self.mjdata)


        self.mujocoAR.resume_updates()

    def start(self):
        self.mujocoAR.start()
        self.mac_launch()

    def mac_launch(self):

        with mujoco.viewer.launch_passive(self.mjmodel,self.mjdata,show_left_ui=False,show_right_ui=False) as viewer:
            self.random_placement()

            while viewer.is_running():

                step_start = time.time()

                tau = self.controller.get_tau(self.mjmodel,self.mjdata,self.target_pos,self.target_rot)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]
                self.mjdata.ctrl[self.grasp_actuator_id] = (self.grasp)*255.0
                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

                if self.was_placed() or self.fell() or self.button:
                    if time.time()-self.placement_time > 2.0 or self.placement_time==-1:
                        self.random_placement()
                        self.placement_time = time.time()

                online_data = self.mujocoAR.get_latest_data()
                if online_data["position"] is not None:
                    online_pos = online_data["position"].copy()
                    online_rot = online_data["rotation"].copy()
                    self.button = online_data["button"]
                    self.grasp = online_data["toggle"]
                    self.target_pos = self.pos_origin + 2.0*online_pos
                    
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":

    sim = ImitationSimulation()

    sim.start()
