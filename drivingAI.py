"""Autonomous Driving Script Using Trained CNN"""
from time import sleep

'''Imports'''
import time
import torch
import numpy as np
from PIL import Image
import mss.tools
from pynput.keyboard import Controller, Key
from traincnn import StandardDriveCNN

'''Variables'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "models/epoch250.pth"  # path to your trained model
fps = 25
frame_time = 1.0 / fps
num_frames = 5  # number of frames your model expects
frame_shape = (320, 240)  # width x height

# Screen capture
sct = mss.mss()
monitor = sct.monitors[1]

# Keyboard controller
keyboard = Controller()

#Frame Buffer
frames_buffer = []


'''Helper functions'''

'''Convert the frames into tensor frames'''
def preprocess_frames(frames_list):

    frames_np = np.array(frames_list, dtype=np.float32)
    frames_np /= 255.0
    frames_np = frames_np.transpose(0, 3, 1, 2)  # (num_frames, 3, H, W)
    frames_np = frames_np.reshape(-1, frames_np.shape[2], frames_np.shape[3])  # (num_frames*3, H, W)
    frames_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    return frames_tensor.to(device)


'''Press keys based on model output'''
'''Send the controls'''
def send_controls(steer, throttle, brake, reset, handbrake):
    # Steering
    if steer < -0.1:
        keyboard.press('a')
        keyboard.release('d')
    elif steer > 0.1:
        keyboard.press('d')
        keyboard.release('a')
    else:
        keyboard.release('a')
        keyboard.release('d')

    # Throttle
    if throttle > 0.5:
        keyboard.press('w')
    else:
        keyboard.release('w')

    # Brake
    if brake > 0.1:
        keyboard.press('s')
    else:
        keyboard.release('s')

    # Reset
    if reset > 0.8:
        keyboard.press('r')
        keyboard.release('r')

    # Handbrake
    if handbrake > 0.1:
        keyboard.press(Key.space)
    else:
        keyboard.release(Key.space)


'''Load the trained cnn'''
print("Loading model, switch into BeamNG!")
sleep(2)
model = StandardDriveCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded and ready!")
print("Starting autonomous driving in 5 seconds...")
time.sleep(5)

'''Main loop'''
while True:
    start_time = time.time()

    # Capture frame
    frame = np.array(sct.grab(monitor))[:, :, :3]
    frame_small = np.array(Image.fromarray(frame).resize(frame_shape, Image.Resampling.NEAREST))
    frames_buffer.append(frame_small)

    # Keep only last `num_frames` frames
    if len(frames_buffer) > num_frames:
        frames_buffer.pop(0)

    # Only predict when buffer is full
    if len(frames_buffer) == num_frames:
        input_tensor = preprocess_frames(frames_buffer)
        with torch.no_grad():
            output = model(input_tensor).cpu().numpy()[0]

        steer, throttle, brake, reset, handbrake = output
        send_controls(steer, throttle, brake, reset, handbrake)

    # Maintain FPS
    elapsed = time.time() - start_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
