import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R

class NPZDataset(Dataset):
    def __init__(self, directory, frequency=1):
        self.directory = directory
        self.frequency = frequency
        self.files = [f for f in os.listdir(directory) if f.endswith('.npz')]
        self.files.sort(key=lambda f: int(os.path.splitext(f)[0]))

        self.transform_frames = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        self.data = []
        self.load_data()

    def load_data(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            data = np.load(file_path)

            times = data['times']
            camera1_rgbs = self.apply_transform_rgb(data['camera1_rgbs'])
            camera1_depths = self.apply_transform_depth(data['camera1_depths'])
            camera2_rgbs = self.apply_transform_rgb(data['camera2_rgbs'])
            camera2_depths = self.apply_transform_depth(data['camera2_depths'])
            camera3_rgbs = self.apply_transform_rgb(data['camera3_rgbs'])
            camera3_depths = self.apply_transform_depth(data['camera3_depths'])
            poses = torch.tensor(data['poses']).float()
            grasps = torch.tensor(data['grasps']).float()

            for i in range(len(times) - 1):
                current_time = times[i]
                for j in range(i + 1, len(times)):
                    next_time = times[j]
                    if next_time - current_time >= (1/self.frequency):

                        # Current data
                        current_pose = poses[i]
                        current_grasp = torch.tensor([grasps[i]])
                        current_rot = R.from_matrix(current_pose[:3, :3].numpy()).as_euler('xyz', degrees=False)
                        current_pose_vector = torch.cat((current_pose[:3, 3], torch.tensor(current_rot)), dim=0).float()
                        current_pos = torch.tensor(current_pose[:3,3])

                        # Next data
                        next_pose = poses[j]
                        next_grasp = torch.tensor([grasps[j]])
                        next_rot = R.from_matrix(next_pose[:3, :3].numpy()).as_euler('xyz', degrees=False)
                        next_pose_vector = torch.cat((next_pose[:3, 3], torch.tensor(next_rot)), dim=0).float()
                        delta_next_position = torch.tensor(next_pose[:3, 3] - current_pose[:3, 3])

                        # Previous frame data (use the same frame if it's the first in the sequence)
                        prev_rgb1 = camera3_rgbs[i-2].float() if i-2 >= 0 else camera3_rgbs[i].float()
                        prev_depth1 = camera3_depths[i-2].float() if i-2 >= 0 else camera3_depths[i].float()
                        prev_rgb2 = camera1_rgbs[i-2].float() if i-2 >= 0 else camera1_rgbs[i].float()
                        prev_depth2 = camera1_depths[i-2].float() if i-2 >= 0 else camera1_depths[i].float()

                        self.data.append({
                            'prev_rgb1': prev_rgb1,
                            'prev_depth1': prev_depth1,
                            'prev_rgb2': prev_rgb2,
                            'prev_depth2': prev_depth2,
                            'rgb1': camera3_rgbs[i].float(),
                            'depth1': camera3_depths[i].float(),
                            'rgb2': camera1_rgbs[i].float(),
                            'depth2': camera1_depths[i].float(),
                            'current_pose': current_pose_vector,
                            'current_pos': current_pos,
                            'next_pose': next_pose_vector,
                            'delta_next_position': delta_next_position,
                            'current_grasps': current_grasp,
                            'next_grasps': next_grasp,
                        })
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def apply_transform_rgb(self, array):
        transformed_images = torch.stack([self.transform_frames(Image.fromarray(image.astype(np.uint8))) for image in array])
        return transformed_images

    def apply_transform_depth(self, array):
        transformed_images = torch.stack([self.transform_frames(Image.fromarray(image)) for image in array])
        return transformed_images

def create_dataloader(directory, batch_size=1, shuffle=False, num_workers=0, frequency=1, train_percentage=0.8):
    dataset = NPZDataset(directory, frequency)
    
    # Calculate lengths for train and test splits
    train_size = int(train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, test_loader

