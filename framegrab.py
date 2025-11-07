"""Code Begins"""

#For LLM Method
'''
from ollama import chat
from ollama import ChatResponse
'''

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

'''Variables'''
#From screengrab
sct = mss.mss()
monitor = sct.monitors[1]

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
def read_controls():
    steer_val = 0
    if "a" in pressed:
        steer_val -= 1
    if "d" in pressed:
        steer_val += 1
    throttle_val = 1 if "w" in pressed else 0
    brake_val = 1 if "s" in pressed else 0
    return steer_val, throttle_val, brake_val

def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        pressed.add(key.char)

def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        pressed.discard(key.char)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

#InfLoop
while True:
    for i in range(5):
        #Start timing 25fps
        start = time.time()

        #Screenshot and put into ram as an array.
        frame = np.array(sct.grab(monitor))[:, :, :3]
        frame_small = np.array(Image.fromarray(frame).resize((128, 72), Image.Resampling.NEAREST))
        frames.append(frame_small)

        #Keep the frame count at 5
        if len(frames) > 5:
            frames.pop(0)

        #Timing 25fps
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        steer, throttle, brake = read_controls()


    #out of the loop
    i1, i2, i3, i4, i5 = frames

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








