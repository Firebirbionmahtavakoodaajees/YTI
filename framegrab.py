"""Framegrab and conversion script"""

'''Imports'''
#For LLM Method
'''
from ollama import chat
from ollama import ChatResponse
'''

#For Data Saving
import os
import pickle

#For FPS timing
import time

#For sleep
from time import sleep

#For ScreenGrab and Conversion
import mss.tools
from PIL import Image
import numpy as np

#For input listening
from pynput import keyboard
from pynput.keyboard import Key


'''Variables'''
#From screengrab
sct = mss.mss()
monitor = sct.monitors[1]

#From data saving
dataset = []
save_dir = "trainingData"
os.makedirs(save_dir, exist_ok=True)

#From picture conversion
frames = []

#From FPS timing
fps = 25
frame_time = 1.0 / fps

#From input listening
pressed = set()

"""Code"""
#Sleep 5
print("Waiting for connected devices to respond...")
sleep(5)

'''Read Controls'''
def read_controls() -> tuple[int, int, int, int, int]:
    #Steering
    steer_val = 0
    if "a" in pressed:
        steer_val -= 1
    if "d" in pressed:
        steer_val += 1
    #Throttle
    throttle_val = 1 if "w" in pressed else 0

    #Brake
    brake_val = 1 if "s" in pressed else 0

    #Reset
    reset_val = 1 if "r" in pressed else 0

    #Handbrake
    handbrake_val = 1 if 'space' in pressed else 0

    return steer_val, throttle_val, brake_val, reset_val, handbrake_val

def on_press(key) -> None:
    try:
        if key.char is not None:
            pressed.add(key.char)
    except AttributeError:
        if key == Key.space:
            pressed.add('space')


def on_release(key) -> None:
    try:
        if key.char is not None:
            pressed.discard(key.char)
    except AttributeError:
        if key == Key.space:
            pressed.discard('space')

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

'''Main Loop'''
while True:
    for i in range(5):
        #Start timing 25fps
        start = time.time()

        #Screenshot and put into ram as an array.
        frame = np.array(sct.grab(monitor))[:, :, :3]
        frame_small = np.array(Image.fromarray(frame).resize((320, 240), Image.Resampling.NEAREST))
        frames.append(frame_small)

        # Keep the frame count at 5
        if len(frames) > 5:
            frames.pop(0)

        # Timing 25fps
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


    '''Out of forloop'''
    steer, throttle, brake, reset, handbrake = read_controls()
    dataset.append((frames.copy(), (steer, throttle, brake, reset, handbrake)))

    #Save every 100 samples
    if len(dataset) >= 100:
        filename = os.path.join(save_dir, f"{int(time.time())}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(dataset, f) # type: ignore[arg-type]
        dataset.clear()
        print("saved to", filename)


#LLM Method
'''
    response: ChatResponse = chat(model='moondream:v2', messages=[
        {
            'role': 'user',
            'content': 'Watch frames. Say G everytime you want me to press gas',
            'images': [i1,i2,i3,i4,i5],
        },
    ])
    print(response.message.content)

'''








