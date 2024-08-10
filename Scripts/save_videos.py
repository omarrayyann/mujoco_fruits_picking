import rerun as rr
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import cv2

def save_numpy_as_video_with_times(frames: np.ndarray, times: np.ndarray, file_name: str) -> str:
    """
    Save a numpy array as a video file with correct colors and custom frame durations based on a times array.

    Parameters:
    - frames: numpy array of shape (num_frames, height, width, 3)
    - times: numpy array of shape (num_frames,) representing the time at which each frame occurs
    - file_name: the name of the output video file

    Returns:
    - path to the saved video file
    """
    # Get the height, width, and number of frames from the numpy array
    height, width = frames.shape[1], frames.shape[2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 20, (width, height))  # Default fps is set to 20; we'll modify it with times
    
    # Calculate durations between frames based on the times array
    durations = np.diff(times)
    
    # Normalize durations to frame counts assuming a base fps (we'll use 20 fps as base)
    base_fps = 20.0
    frame_counts = (durations * base_fps).astype(int)

    # Convert BGR to RGB for each frame and write the appropriate number of times
    for i in range(frames.shape[0] - 1):
        rgb_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        for _ in range(frame_counts[i]):
            out.write(rgb_frame)
    
    # Convert the last frame and write it (since it does not have a subsequent duration)
    rgb_last_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
    out.write(rgb_last_frame)

    # Release the VideoWriter object
    out.release()

    return file_name


def save_videos(file_name):

    data = np.load(file_name)  
    
    times = data['times']
    camera1_rgbs = data['camera1_rgbs']
    camera2_rgbs = data['camera2_rgbs']
    camera3_rgbs = data['camera3_rgbs']
    camera4_rgbs = data['camera4_rgbs']


    print(save_numpy_as_video_with_times(camera1_rgbs,times,"camera1.mp4"))
    print(save_numpy_as_video_with_times(camera2_rgbs,times,"camera2.mp4"))
    print(save_numpy_as_video_with_times(camera3_rgbs,times,"camera3.mp4"))
    print(save_numpy_as_video_with_times(camera4_rgbs,times,"camera4.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save frames as an .mp4 video")
    parser.add_argument("file_name", type=str, help="The path to the .npz file containing the frames")
    args = parser.parse_args()

    save_videos(args.file_name)
