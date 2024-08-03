# Pick and Place Task with MuJoCo
**Part of the [MujocoAR](https://github.com/omarrayyann/MujocoAR) package demos**

A MuJoCo environment (```Environment/scene.xml```) simulation of a simple pick and place task. The robot's goal is to pick an object and place it into another. The simulation includes an operational space controller to handle the movement of a KUKA-iiwa14 arm with a 2f85 grasper at the end.

Main methods and attributes:

- `__init__(self)`: Initializes the simulation environment, controller, and MujocoAR connector.
- `random_placement(self, min_seperation=0.1)`: Places the objects randomly in the scene with specified separation constraints.
- `start(self)`: Starts the simulation and control loop.
- `done(self)`: Checks if the task is completed.
- `fell(self)`: Checks if the object has fallen.
