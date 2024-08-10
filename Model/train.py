import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import ImitationLearningModel
from dataloader import create_dataloader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_model(checkpoint_path, device):
    model = ImitationLearningModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def train_model(directory, batch_size=16, num_epochs=10, learning_rate=0.001, frequency=1, checkpoint_dir="checkpoints", train_percentage=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = create_dataloader(directory, batch_size=batch_size, shuffle=True, num_workers=1, frequency=frequency, train_percentage=train_percentage)
    model = ImitationLearningModel().to(device)
    # model = load_model("checkpoints/checkpoint_epoch.pth.tar",device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        
        train_running_loss = 0.0
        test_running_loss = 0.0

        model.train()
        for i, data in enumerate(train_dataloader):

            rgb1 = data['rgb1'].to(device).float()
            depth1 = data['depth1'].to(device).float()
            rgb2 = data['rgb2'].to(device).float()
            depth2 = data['depth2'].to(device).float()

            prev_rgb1 = data['prev_rgb1'].to(device).float()
            prev_depth1 = data['prev_depth1'].to(device).float()
            prev_rgb2 = data['prev_rgb2'].to(device).float()
            prev_depth2 = data['prev_depth2'].to(device).float()

            current_grasps = data['current_grasps'].to(device).float()
            next_grasps = data['next_grasps'].to(device).float()
            current_pos = data['current_pos'].to(device).float()
            delta_next_position = data['delta_next_position'].to(device).float()

            optimizer.zero_grad()
                        
            frame1 = torch.cat((rgb1, depth1), dim=1)
            frame2 = torch.cat((rgb2, depth2), dim=1)

            frame1_prev = torch.cat((prev_rgb1, prev_depth1), dim=1)
            frame2_prev = torch.cat((prev_rgb2, prev_depth2), dim=1)

            outputs = model.forward(frame1, frame2, frame1_prev, frame2_prev, current_grasps, current_pos)

            actual_outputs = torch.cat((delta_next_position,next_grasps),dim=1)

            loss = criterion(outputs, actual_outputs) 
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        model.eval()
        for i, data in enumerate(test_dataloader):

            rgb1 = data['rgb1'].to(device).float()
            depth1 = data['depth1'].to(device).float()
            rgb2 = data['rgb2'].to(device).float()
            depth2 = data['depth2'].to(device).float()

            prev_rgb1 = data['prev_rgb1'].to(device).float()
            prev_depth1 = data['prev_depth1'].to(device).float()
            prev_rgb2 = data['prev_rgb2'].to(device).float()
            prev_depth2 = data['prev_depth2'].to(device).float()

            current_grasps = data['current_grasps'].to(device).float()
            next_grasps = data['next_grasps'].to(device).float()
            current_pos = data['current_pos'].to(device).float()
            delta_next_position = data['delta_next_position'].to(device).float()
                        
            frame1 = torch.cat((rgb1, depth1), dim=1)
            frame2 = torch.cat((rgb2, depth2), dim=1)

            frame1_prev = torch.cat((prev_rgb1, prev_depth1), dim=1)
            frame2_prev = torch.cat((prev_rgb2, prev_depth2), dim=1)

            outputs = model.forward(frame1, frame2, frame1_prev, frame2_prev, current_grasps, current_pos)

            actual_outputs = torch.cat((delta_next_position,next_grasps),dim=1)

            loss = criterion(outputs, actual_outputs) 

            test_running_loss += loss.item()

        print(f"[{epoch + 1}, train loss: {train_running_loss/len(train_dataloader) :.3f}, test loss: {train_running_loss/len(train_dataloader) :.3f}")
          

        # Save checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if epoch%10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch.pth.tar")
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}, checkpoint_path)

    print("Finished Training")
    torch.save(model.state_dict(), "imitation_learning_model.pth")

if __name__ == "__main__":
    train_model("Data", batch_size=128, num_epochs=1000, learning_rate=0.001, frequency=5)


