import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from scipy.spatial.transform import Rotation as R

class ImitationLearningModel(nn.Module):
    def __init__(self):
        super(ImitationLearningModel, self).__init__()

        # Define a smaller CNN for concatenated RGB and Depth images
        self.conva1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conva2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conva3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conva4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Define a smaller CNN for concatenated RGB and Depth images
        self.convb1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.convb2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convb3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.convb4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fca1 = nn.Linear(16384, 256)  # Adjust size according to the input dimensions
        self.fcb1 = nn.Linear(16384, 256)  # Adjust size according to the input dimensions

        self.fc2 = nn.Linear(1028, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)  # Output: next pose (x, y, z) + grasp (0-1)

    def forward(self, frame1, frame2, prev_frame1, prev_frame2, current_grasp, current_pos):
        # Concatenate RGB and depth images from camera2

        # concatenated_input = torch.cat((camera2_rgb, camera2_depth), dim=1)  # Concatenate along the channel dimension

        # Process concatenated input through the CNN
        x = self.conva1(frame1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conva2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conva3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conva4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fca1(x)
        x = F.relu(x)

        # Process concatenated input through the CNN
        x_prev = self.conva1(prev_frame1)
        x_prev = F.relu(x_prev)
        x_prev = self.pool(x_prev)
        x_prev = self.conva2(x_prev)
        x_prev = F.relu(x_prev)
        x_prev = self.pool(x_prev)
        x_prev = self.conva3(x_prev)
        x_prev = F.relu(x_prev)
        x_prev = self.pool(x_prev)
        x_prev = self.conva4(x_prev)
        x_prev = F.relu(x_prev)
        x_prev = self.pool(x_prev)
        x_prev = x_prev.view(x_prev.size(0), -1)  # Flatten
        x_prev = self.fca1(x_prev)
        x_prev = F.relu(x_prev)

        # Process concatenated input through the CNN
        y = self.convb1(frame2)
        y = F.relu(y)
        y = self.pool(y)
        y = self.convb2(y)
        y = F.relu(y)
        y = self.pool(y)
        y = self.convb3(y)
        y = F.relu(y)
        y = self.pool(y)
        y = self.convb4(y)
        y = F.relu(y)
        y = self.pool(y)
        y = y.view(y.size(0), -1)  # Flatten
        y = self.fcb1(y)
        y = F.relu(y)

        # Process concatenated input through the CNN
        y_prev = self.convb1(prev_frame2)
        y_prev = F.relu(y_prev)
        y_prev = self.pool(y_prev)
        y_prev = self.convb2(y_prev)
        y_prev = F.relu(y_prev)
        y_prev = self.pool(y_prev)
        y_prev = self.convb3(y_prev)
        y_prev = F.relu(y_prev)
        y_prev = self.pool(y_prev)
        y_prev = self.convb4(y_prev)
        y_prev = F.relu(y_prev)
        y_prev = self.pool(y_prev)
        y_prev = y_prev.view(y_prev.size(0), -1)  # Flatten
        y_prev = self.fcb1(y_prev)
        y_prev = F.relu(y_prev)

        # x = x.to("cpu")
        # current_pose = current_pose.to("cpu")
        # current_grasp = current_grasp.to("cpu")
        
        # Concatenate the features with the current pose
        x = torch.cat((x, y, x_prev, y_prev, current_pos, current_grasp), dim=1)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        output = self.fc4(x)
        
        return output

def load_model(checkpoint_path, device):
    model = ImitationLearningModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  
    return model

def preprocess_input(rgb, depth, device):
    transform_frame = Resize((128, 128))

    # Convert and preprocess RGB image
    rgb = Image.fromarray(rgb.astype(np.uint8))
    rgb = transform_frame(rgb)
    rgb = ToTensor()(rgb).unsqueeze(0).to(device)

    # Convert and preprocess depth image
    depth = Image.fromarray(depth)
    depth = transform_frame(depth)
    depth = ToTensor()(depth).unsqueeze(0).to(device)

    return rgb, depth

def predict_action(model, rgb1, depth1, rgb2, depth2, prev_rgb1, prev_depth1, prev_rgb2, prev_depth2, current_pos, current_grasp, device):

    rgb1, depth1 = preprocess_input(rgb1, depth1, device)
    rgb2, depth2 = preprocess_input(rgb2, depth2, device)

    frame1 = torch.cat((rgb1, depth1), dim=1)
    frame2 = torch.cat((rgb2, depth2), dim=1)

    prev_rgb1, prev_depth1 = preprocess_input(prev_rgb1, prev_depth1, device)
    prev_rgb2, prev_depth2 = preprocess_input(prev_rgb2, prev_depth2, device)

    prev_frame1 = torch.cat((prev_rgb1, prev_depth1), dim=1)
    prev_frame2 = torch.cat((prev_rgb2, prev_depth2), dim=1)

    # Make predictions
    with torch.no_grad():
        delta_pos = model(frame1, frame2, prev_frame1, prev_frame2, current_grasp, current_pos)

    return delta_pos.cpu().numpy()


