import rerun as rr
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

def log_frames(file_name):

    data = np.load(file_name)

    camera1_rgbs = data['camera1_rgbs']
    camera1_depths = data['camera1_depths']
    camera2_rgbs = data['camera2_rgbs']
    camera2_depths = data['camera2_depths']
    camera3_rgbs = data['camera3_rgbs']
    camera3_depths = data['camera3_depths']
    poses = data['poses']
    times = data['times']

    rr.init("My_Rerun", spawn=True)

    start_time = time.time()
    initial_time = times[0]
    for i, (rgb1, depth1, rgb2, depth2, rgb3, depth3, pose, log_time) in enumerate(zip(camera1_rgbs, camera1_depths, camera2_rgbs, camera2_depths, camera3_rgbs, camera3_depths, poses, times)):

        if i > 0:
            wait_time = log_time - times[i-1]
            time.sleep(wait_time)

        rr.log("camera1/rgb".format(i), rr.Image(rgb1))
        rr.log("camera1/depth".format(i), rr.Image(depth1))
        rr.log("camera2/rgb".format(i), rr.Image(rgb2))
        rr.log("camera2/depth".format(i), rr.Image(depth2))
        rr.log("camera3/rgb".format(i), rr.Image(rgb3))
        rr.log("camera3/depth".format(i), rr.Image(depth3))
        rr.log("poses", rr.Transform3D(translation=pose[0:3,3].tolist(), mat3x3=pose[0:3,0:3].tolist()))

    # Finalize and open the rerun viewer
    # rr.finalize()

    print("Frames and poses have been logged to rerun. Use the rerun viewer to inspect them.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log frames from a .npz file to Rerun.io")
    parser.add_argument("file_name", type=str, help="The path to the .npz file containing the frames and poses")
    args = parser.parse_args()

    log_frames(args.file_name)
