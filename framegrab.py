#from ollama import chat
#from ollama import ChatResponse
import time

from time import sleep

import mss.tools
import base64
from PIL import Image
import numpy as np
from io import BytesIO

sct = mss.mss()
monitor = sct.monitors[1]

print("Waiting for connected devices to respond...")
sleep(5)

while True:
    frames_b64 = []


    fps = 25
    frame_time = 1.0 / fps
    for i in range(5):
        start = time.time()

        frame = np.array(sct.grab(monitor))[:, :, :3]

        img = Image.fromarray(frame).resize((128, 72), Image.Resampling.NEAREST)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=15)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        print("hello")

        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    #out of the loop
    i1, i2, i3, i4, i5 = frames_b64




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








