# MuJoCo Fruit Picking Task 
**Part of the [MujocoAR](https://github.com/omarrayyann/MujocoAR) package demos**

A MuJoCo simulation environment of a fruit picking task. The goal is to pick the fruits around the table and place them onto the plate at the center of the table. The simulation includes an operational space controller to handle the movement of the KUKA-iiwa14 arm with a 2f85 grasper at the end.


<table>
<!--   <tr>
    <th><code>camera_name="whole_view"</code></th>
    <th><code>camera_name="top_view"</code></th>
    <th><code>camera_name="side_view"</code></th>
    <th><code>camera_name="front_view"</code></th>
  </tr> -->
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1d74ddb7-1f43-4ce4-b994-327d4071eac5" width="500px" /></td>
    <td><img src="https://github.com/user-attachments/assets/8fd2b0ae-f90a-4df5-b114-3feac7c87e37" width="500px" /></td>
    <td><img src="https://github.com/user-attachments/assets/286428d9-93bf-456b-9ef1-322793c0bb3a" width="500px" /></td>
    <td><img src="https://github.com/user-attachments/assets/3d496ce1-0b5d-4a1f-a6d2-dc2e19d1e3d8" width="500px" /></td>
  </tr>
</table>


## MuJoCo AR Setup

```python
# Initializing MuJoCo AR
self.mujocoAR = MujocoARConnector(mujoco_model=self.mjmodel,mujoco_data=self.mjdata)

# Linking a Target Site with the AR Position
self.mujocoAR.link_site(
   name="eef_target",
   scale=3.0,
   translation=self.pos_origin,
   toggle_fn=lambda: setattr(self, 'grasp', not self.grasp),
   button_fn=lambda: (self.random_placement(), setattr(self, 'placement_time', time.time()), self.reset_data()) if time.time() - self.placement_time > 2.0 else None,
   disable_rot=True,
)

# Start!
self.mujocoAR.start()
```

## Usage Guide

1. **Clone the repository**:

   ```bash
   git clone https://github.com/omarrayyann/mujoco_fruit_picking.git
   cd mujoco_fruit_picking
   
3. **Install MujocoAR and othe Requirements**:
   ```bash
   
   pip install mujoco_ar
   pip install requirements.txt
   
4. **Download the [MuJoCo AR App](https://apps.apple.com/jo/app/past-code/id1551535957) from the App Store.**
   
5. **Run the application**:

   ```bash
   mjpython main.py
   
6. **Enter the IP and Port shown into the app's start screen to start. Make sure to be connected to the same Wi-Fi network as the device. Incase of a latency, I recommend connecting to your phone's hotspot.**

## Pre-Collected Data and Pre-Trained Checkpoint

120 Collected Episodes can be downloaded [here](https://huggingface.co/datasets/omarrayyann/mujoco_pick_place_fruits/blob/main/Data.zip). A pre-trained policy checkpoint can be found [here](https://huggingface.co/datasets/omarrayyann/mujoco_pick_place_fruits/blob/main/checkpoint_epoch.pth.tar).

**Policy Rollout (using an architeceture that can capture multimodality like DP or VQ-BeT might increase the success rate since multimodality lies in the order of picking up the fruits (banana or orange first)):**

<img src="https://github.com/user-attachments/assets/e84345f9-62eb-4d89-a08b-dcd92d1dd10b" width="500px" />


## Author

Omar Rayyan (olr7742@nyu.edu)
