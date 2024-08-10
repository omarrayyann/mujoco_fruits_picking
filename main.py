import mujoco
import mujoco.viewer
import numpy as np
import time
from Utils.utils import *
from controller import OpsaceController
from mujoco_ar import MujocoARConnector
import random
import rerun as rr
import threading
from Model.model import *
from collections import deque

class ImitationSimulation:
    
    def __init__(self):

        # Configs
        self.scene_path = 'Environment/scene.xml'
        self.mjmodel = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.dt = 0.002
        self.grasp = 0
        self.button = 0
        self.placement_time = -1
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=120, width=160)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=120, width=160)
        self.depth_renderer.enable_depth_rendering()
        self.cameras = ["front_camera","side_camera","top_camera"]

        # Override the simulation timestep
        self.pick_objects = ["banana","orange_fruit"]
        self.pick_joints = ["banana_joint","orange_joint"]
        self.place_object = "plate"
        self.mjmodel.opt.timestep = self.dt
        self.frequency = 1000
        self.target_pos = np.array([0.5, 0.0, 0.4])
        self.target_rot = rotation_matrix_x(np.pi)@np.identity(3)
        self.pos_origin = self.target_pos.copy()
        self.rot_origin = self.target_rot.copy()
        self.target_quat = np.zeros(4)
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

        # Recording and Policy Related
        self.record = True
        self.run_policy = False
        self.recording_frequency = 10
        self.prev_datas = deque(maxlen=10)
        self.prev_times = deque(maxlen=10)

        # Controller
        self.controller = OpsaceController(self.mjmodel,self.joint_names,self.eef_site_name)
        self.q0 = np.array([0, 0.2686, 0, -1.5423, 0, 1.3307, 0])
        self.dof_ids = np.array([self.mjmodel.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.mjmodel.actuator(name).id for name in self.joint_names])
        self.grasp_actuator_id = self.mjmodel.actuator("fingers_actuator").id 
        self.mjdata.qpos[self.actuator_ids] = self.q0

        # MujocoAR Initialization
        self.mujocoAR = MujocoARConnector(mujoco_model=self.mjmodel,mujoco_data=self.mjdata)

        # Linking the target site with the AR position
        self.mujocoAR.link_site(
            name="eef_target",
            scale=3.0,
            translation=self.pos_origin,
            toggle_fn=lambda: setattr(self, 'grasp', not self.grasp),
            button_fn=lambda: (self.random_placement(), setattr(self, 'placement_time', time.time()), self.reset_data()) if time.time() - self.placement_time > 2.0 else None,
            disable_rot=True,
        )

    
    def is_valid_position(self, pos1, pos2_multiple, min_dist):
        if pos1 is None or pos2_multiple is None:
            return False
        for pos2 in pos2_multiple:
            distance = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])) 
            if distance<min_dist:
                return False
        for i, pos2 in enumerate(pos2_multiple):
            for j, pos2_2nd in enumerate(pos2_multiple):
                if i==j:
                    continue
                if pos2_2nd is None or pos2 is None:
                    return False
                distance = np.linalg.norm(np.array(pos2_2nd[:2]) - np.array(pos2[:2])) 
                if distance<min_dist:
                    return False
        return True
        
    def get_camera_data(self) -> dict:
        data = {}    
        for camera in self.cameras:
            self.rgb_renderer.update_scene(self.mjdata, camera)
            self.depth_renderer.update_scene(self.mjdata, camera)
            data[camera+"_rgb"] = self.rgb_renderer.render()
            data[camera+"_depth"] = self.depth_renderer.render()
        self.prev_datas.append(data)
        self.prev_times.append(time.time())
        return data
    
    def get_pos_from_range(self,range):
        return np.array([random.uniform(range[0,0], range[0,1]), random.uniform(range[1,0], range[1,1]),range[2,0]])

    def was_placed(self):
        for pick_object in self.pick_objects:
            pick_pos = self.mjdata.body(pick_object).xpos.copy()
            place_pos = self.mjdata.body(self.place_object).xpos.copy()
            if np.linalg.norm(np.array(pick_pos)[0:2] - np.array(place_pos)[0:2]) > 0.07 or pick_pos[2]>0.25:
                return False
        return True
        
    def fell(self):
        for pick_object in self.pick_objects:
            pick_pos = self.mjdata.body(pick_object).xpos.copy()
            if pick_pos[2]<0.1 or pick_pos[2]>2 :
                return True
        return False

    def random_placement(self, min_seperation=0.2):

        place_range = np.array([[0.4,0.7],[-0.33,0.33],[0.21,0.21]])
        place_range = np.array([[0.61,0.61],[0.0,0.0],[0.21,0.21]]) # static

        pick_range = np.array([[0.4,0.7],[-0.3,0.3],[0.24,0.24]])
        
        place_pos, pick_pos_multiple = None, None
        while not self.is_valid_position(place_pos,pick_pos_multiple,min_seperation):
            pick_pos_multiple = []
            place_pos = self.get_pos_from_range(place_range)
            for _ in self.pick_objects:
                pick_pos_multiple.append(self.get_pos_from_range(pick_range))
        
        self.mujocoAR.pause_updates()
        self.mujocoAR.reset_position()
        self.mjdata.qpos[self.actuator_ids] = self.q0
        mujoco.mj_step(self.mjmodel, self.mjdata)
        self.grasp = 0
        for i in range(len(pick_pos_multiple)):
            self.mjdata.joint(self.pick_joints[i]).qvel = np.zeros(6)
            self.mjdata.joint(self.pick_joints[i]).qpos = np.block([pick_pos_multiple[i],1,0,0,0])        
        set_body_pose(self.mjmodel,self.place_object,place_pos)
        mujoco.mj_step(self.mjmodel, self.mjdata)


        self.mujocoAR.resume_updates()

    def start(self):
        
        threading.Thread(target=self.mac_launch).start()
        # if self.run_policy:
        #     self.run_model()

    def mac_launch(self):

        if not self.run_policy:
            self.mujocoAR.start()

        with mujoco.viewer.launch_passive(self.mjmodel,self.mjdata,show_left_ui=False,show_right_ui=False) as viewer:
            
            self.random_placement()
            self.reset_data()
            
            while viewer.is_running():

                step_start = time.time()
                self.record_data()   

                tau = self.controller.get_tau(self.mjmodel,self.mjdata,self.target_pos,self.target_rot)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]
                self.mjdata.ctrl[self.grasp_actuator_id] = (self.grasp)*255.0
                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

                if self.was_placed() or self.fell():
                    self.record_data()   
                    if not self.fell() and time.time()-self.placement_time > 2.0:
                        self.save_data()
                    if time.time()-self.placement_time > 2.0 or self.placement_time==-1:
                        self.random_placement()
                        self.reset_data()
                        self.placement_time = time.time()
                
                self.target_pos = self.mjdata.site("eef_target").xpos.copy()

                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def record_data(self):

        if not self.record:
            return
     
        if self.camera_data is not None and self.last_recording_time != -1 and time.time()-self.last_recording_time < (1/self.recording_frequency):
            return
        
        if self.record_start_time == None:
            self.record_start_time = time.time()
            time_diff = 0.0
        else:
            time_diff = time.time() - self.record_start_time
        pose = np.identity(4)
        pose[0:3,3] = self.mjdata.site(self.site_id).xpos.copy()
        pose[0:3,0:3] = self.mjdata.site(self.site_id).xmat.copy().reshape((3,3))

        q = self.mjdata.qpos[self.dof_ids].copy()
        dq = self.mjdata.qvel[self.dof_ids].copy()
        
        camera1_rgb = self.camera_data[self.cameras[0]+"_rgb"]
        camera1_depth = self.camera_data[self.cameras[0]+"_depth"]
        camera2_rgb = self.camera_data[self.cameras[1]+"_rgb"]
        camera2_depth = self.camera_data[self.cameras[1]+"_depth"]
        camera3_rgb = self.camera_data[self.cameras[2]+"_rgb"]
        camera3_depth = self.camera_data[self.cameras[2]+"_depth"] 

        self.camera1_rgbs.append(camera1_rgb)
        self.camera1_depths.append(camera1_depth)
        self.camera2_rgbs.append(camera2_rgb)
        self.camera2_depths.append(camera2_depth)
        self.camera3_rgbs.append(camera3_rgb)
        self.camera3_depths.append(camera3_depth)

        self.poses.append(pose)
        self.grasps.append(self.grasp)
        self.times.append(time_diff)
        self.q.append(q)
        self.dq.append(dq)

        self.last_recording_time = time.time()

    def save_data(self):

        if not self.record:
            return

        new_file_name = "Data/" + str(get_latest_number("Data")+1)+".npz"
        camera1_rgbs = np.array(self.camera1_rgbs)
        camera1_depths = np.array(self.camera1_depths)
        camera2_rgbs = np.array(self.camera2_rgbs)
        camera2_depths = np.array(self.camera2_depths)
        camera3_rgbs = np.array(self.camera3_rgbs)
        camera3_depths = np.array(self.camera3_depths)

        poses = np.array(self.poses)
        times = np.array(self.times)
        grasps = np.array(self.grasps)
        q = np.array(self.q)
        dq = np.array(self.dq)

        np.savez(new_file_name, camera1_rgbs=camera1_rgbs, camera1_depths=camera1_depths, camera2_rgbs=camera2_rgbs, camera2_depths=camera2_depths, camera3_rgbs=camera3_rgbs, camera3_depths=camera3_depths, poses=poses, grasps=grasps, times=times, q=q, dq=q)

    def reset_data(self):
        self.camera1_rgbs = []
        self.camera1_depths = []
        self.camera2_rgbs = []
        self.camera2_depths = []
        self.camera3_rgbs = []
        self.camera3_depths = []
        self.poses = []
        self.times = []
        self.grasps = []
        self.q = []
        self.dq = []
        self.record_start_time = time.time()
    
    def run_poses_from_npz(self, npz_file_path):

        data = np.load(npz_file_path)
        poses = data['poses']
        times = data['times']
        grasps = data['grasps']

        while True:
          
            start_time = time.time()
            data_time = times[0]

            i = 1

            while i<len(poses)-1:
                
                # Set the pose
                if (time.time()-start_time) - (times[i]-data_time) >= 0:
                    set_site_pose(self.mjmodel,"eef_target",poses[i][0:3,3])
                    self.grasp = grasps[i]
                    i += 1


    def run_model(self):

        checkpoint_path = "Model/checkpoint_epoch.pth.tar"
        policy = load_model(checkpoint_path,"cpu")
        last_time = None
        
        while True:

            sim.camera_data = sim.get_camera_data()
        
            if (last_time is None or (time.time()-last_time) >= 1/5):
                
            
                last_data = self.prev_datas[-1]
                last_time = self.prev_times[-1]

                prev_data = self.prev_datas[0]
                j = 0
                for i, prev_time in enumerate(self.prev_times):
                    if last_time - prev_time >= 0.2:
                        prev_data = self.prev_datas[i]
                        j = i
                    else:
                        break
                
                print(self.prev_times[j]-self.prev_times[-1])
                            
                rgb1 = last_data["top_camera_rgb"]
                depth1 = last_data["top_camera_depth"]
                rgb2 = last_data["front_camera_rgb"]
                depth2 = last_data["front_camera_depth"]

                prev_rgb1 = prev_data["top_camera_rgb"]
                prev_depth1 = prev_data["top_camera_depth"]
                prev_rgb2 = prev_data["front_camera_rgb"]
                prev_depth2 = prev_data["front_camera_depth"]

                delta_pos = predict_action(policy, rgb1, depth1, rgb2, depth2, prev_rgb1, prev_depth1, prev_rgb2, prev_depth2, torch.tensor([sim.mjdata.site(sim.site_id).xpos.copy()]).float(), torch.tensor([[sim.grasp]]), "cpu")[0]
                
                new_pos = sim.mjdata.site("eef_target").xpos.copy()+delta_pos[:3]
                new_pos[2] =  max(new_pos[2],0.21)

                set_site_pose(sim.mjmodel,"eef_target",new_pos,None)

                sim.grasp = round(delta_pos[-1])
                last_time = time.time()


if __name__ == "__main__":

    sim = ImitationSimulation()
    sim.start()

    while True:
        sim.camera_data = sim.get_camera_data()