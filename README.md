# Fruit Picking Task MuJoCo
**Part of the [MujocoAR](https://github.com/omarrayyann/MujocoAR) package demos**

A MuJoCo simulation environment of a fruit picking task. The goal is to pick the fruits around the table and place them onto the plate at the center of the table. The simulation includes an operational space controller to handle the movement of the KUKA-iiwa14 arm with a 2f85 grasper at the end.

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
   
6. **Enter the IP and Port shown into the app's start screen to start.**

## Pre-Collected Data

120 Collected Episodes can be downloaded [here](https://huggingface.co/datasets/omarrayyann/mujoco_pick_place_fruits/blob/main/Data.zip). A pre-trained policy checkpoint can be found [here](https://huggingface.co/datasets/omarrayyann/mujoco_pick_place_fruits/blob/main/checkpoint_epoch.pth.tar).

