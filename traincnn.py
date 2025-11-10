"""CNN Training Script"""

from time import sleep

'''Imports'''
# Core Libraries
import os
import pickle
import numpy as np

# For Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# For Progress bar
from tqdm import tqdm

'''GPU VARIABLE'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Classes'''

'''Wrap dataset'''
class DrivingDataset(Dataset):
    # Stores loaded data from the .pkl to variable dataset
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frames, controls = self.dataset[index]

        # Convert to numpy array (5, 240, 320, 3)
        frames_np = np.array(frames, dtype=np.float32)

        # Normalizes RGB values to 0-1 floats COMMENT IF NOT NORMALIZED
        frames_np /= 255.0

        '''Stacks the channels (3*15)'''
        # Rearrange axes: (5, 3, H, W)
        frames_np = frames_np.transpose(0, 3, 1, 2)  # (5, 3, H, W)
        # Flatten frames along channel dimension: (5*3, H, W) = (15, 240, 320)
        frames_np = frames_np.reshape(-1, frames_np.shape[2], frames_np.shape[3])

        # Convert to torch tensors
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32)
        controls_tensor = torch.tensor(controls, dtype=torch.float32)

        print("Dataset created!")

        return frames_tensor, controls_tensor

'''DescribeCNN'''
class StandardDriveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers (DON'T CHANGE THE VALUE OF THE SMALLEST IN CHANNEL (5*3channel=15))
        self.conv1 = nn.Conv2d(15, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Fully connected layers (fc)
        '''Wanna place this elsewhere dont know where yet...'''
        # Calculate the flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 15, 240, 320)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            flatten_size = x.numel() // x.shape[0]

        #Continueing code
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, 5)  # steer, throttle, brake, reset, handbrake

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        features = torch.relu(self.conv1(features))
        features = torch.relu(self.conv2(features))
        features = torch.relu(self.conv3(features))
        features = torch.relu(self.conv4(features))
        features = features.view(features.size(0), -1)  # flatten
        features = self.dropout(torch.relu(self.fc1(features)))
        features = self.fc2(features)

        # Output activations
        steer = torch.tanh(features[:, 0:1])        # -1..1
        throttle = torch.sigmoid(features[:, 1:2])  # 0..1
        brake = torch.sigmoid(features[:, 2:3])
        reset = torch.sigmoid(features[:, 3:4])  # 0..1
        handbrake = torch.sigmoid(features[:, 4:5])
        # 0..1

        return torch.cat([steer, throttle, brake, reset, handbrake], dim=1)

#For windows processes
if __name__ == "__main__":
    '''Variables'''


    #List pkl files and set save dir name
    save_dir = "trainingData"

    #list files
    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
    print("Training files:", pkl_files)
    file_path = os.path.join(save_dir, pkl_files[0])

    '''Hyperparameters'''
    # Learning
    learning_rate = 0.001         # How fast the model learns (too high = unstable, too low = slow)
    optimizer_type = "adam"       # ["adam", "sgd", "rmsprop"] — Adam is usually the best starting point
    loss_function = "mse"         # ["mse", "crossentropy"] — MSE for regression, CrossEntropy for classification
    epochs = 250                  # How many passes through the entire dataset
    batch_size = 128              # Number of samples per gradient update '32-64'

    # CNN Architecture
    input_shape = (5, 240, 320, 3)# (num_frames, height, width, channels) — adjust for your setup
    num_classes = 5               # Example: steer, throttle, brake → 3 outputs
    conv_filters = [32, 64, 128]  # Number of filters in each conv layer (depth of features)
    kernel_size = (3, 3)          # Size of convolution window (3x3 is standard)
    stride = 1                    # How far the filter moves each step
    padding = 1                   # Keeps output size stable (1 for "same" padding)

    # Regularization & Stability
    dropout_rate = 0.3            # Randomly deactivate % of neurons to prevent overfitting
    weight_decay = 1e-5           # Small penalty on weights (L2 regularization)
    activation_function = "relu"  # ["relu", "leakyrelu", "tanh", "sigmoid"]
    pool_size = (2, 2)            # MaxPooling window — reduces spatial size
    pool_stride = 2               # How much to slide the pooling window

    # Data & Input Processing
    shuffle_data = True           # Shuffle training data for better generalization
    normalize_input = True        # Normalize pixel values (0–1 range)
    augmentation = False          # Enable image augmentation (rotation, flip, etc.)

    # Checkpointing & Logging
    save_every = 25               # Save model every N epochs
    model_save_dir = "models"     # Directory for saved model weights
    log_interval = 10             # Print loss every N batches

    # Hardware
    use_gpu = True                # Enable CUDA if available
    num_workers = 4               # Data loader threads
    pin_memory = True             # Speeds up GPU training

    print("Hyperparameters initialized!")

    '''Load data'''
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"Loaded {len(data)} samples from {pkl_files[0]}!")

    '''Check GPU Availbility'''
    sleep(2)
    print("GPU available?")
    print(torch.cuda.is_available())
    print("GPU NAME=" + torch.cuda.get_device_name(0))
    sleep(5)

    '''
    #Each frame is like
    frames, (steer, throttle, brake) = data[0]
    print("Frame count, vPixels, hPixels, RGB Channels")
    print(np.array(frames).shape)
    '''


    '''Model Description'''
    model = StandardDriveCNN().to(device)
    model.train()

    # Wrap dataset in DataLoader
    dataset_wrapped = DrivingDataset(data)

    val_size = int(len(dataset_wrapped) * 0.1)
    train_size = len(dataset_wrapped) - val_size
    train_dataset, val_dataset = random_split(dataset_wrapped, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    print("Described models")

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Training optimization complete")
    sleep(2)
    print("Starting training...")

    '''Training loop'''
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        #Progress bar
        for batchNumber, (inputs, targets) in enumerate(tqdm(train_loader), 1):
            # Move to GPU if available
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()
            if batchNumber % log_interval == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch [{batchNumber}/{len(train_loader)}], Loss: {running_loss / log_interval:.4f}")
                running_loss = 0.0

        '''Validation'''
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        #Max of 1 for division safety
        avg_val_loss = val_loss / max(1,len(val_loader))

        #Printing
        print(f"Epoch [{epoch}/{epochs}] Validation Loss: {avg_val_loss:.4f}")
        model.train()

        '''Save model'''
        #Check if done and save
        if epoch % save_every == 0:
            os.makedirs(model_save_dir, exist_ok=True)
            save_path = os.path.join(model_save_dir, f"epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training finished!")
