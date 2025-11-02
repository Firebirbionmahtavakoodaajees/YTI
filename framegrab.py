from time import sleep
import mss.tools
from ollama import chat
from ollama import ChatResponse
import base64
from resizeimage import resizeimage
from PIL import Image

x=0

sleep(20)
print("Starting script")

while x == 0:
    with mss.mss() as sct:
        sct.compression_level = 9

        for order in range(1, 6):
            sct.shot(output=f"{order}.png")

            with open(f"{order}.png", 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [256, 144])
                    cover.save(f"{order}.png", image.format)

    with open("1.png", "rb") as f:
        i1 = base64.b64encode(f.read()).decode("utf-8")

    with open("2.png", "rb") as f:
        i2 = base64.b64encode(f.read()).decode("utf-8")

    with open("3.png", "rb") as f:
        i3 = base64.b64encode(f.read()).decode("utf-8")

    with open("4.png", "rb") as f:
        i4 = base64.b64encode(f.read()).decode("utf-8")

    with open("5.png", "rb") as f:
        i5 = base64.b64encode(f.read()).decode("utf-8")

    print("done once")

'''

    response: ChatResponse = chat(model='', messages=[
        {
            'role': 'user',
            'content': 'Drive this car. Say F to go forwards, L to go left, R to go right and B to break. Write any combination of those to do both. If you see you fell off the edge say R', "images": [i1,i2,i3,i4,i5],
        },
    ])
    print(response.message.content)

'''










