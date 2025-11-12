# 🏎️ YTI Autonomous Driving AI — The Best YTI Project in the World 🌍

Welcome to **YTI Autonomous Driving AI**, the ultimate end-to-end self-driving simulation project.  
This system watches your gameplay, learns from your inputs, and then drives *just like you* — or better.

---

## 🚀 Project Overview

This repository contains a full pipeline for:
- Capturing your screen and driving inputs
- Building datasets automatically
- Training a deep Convolutional Neural Network (CNN)
- Deploying that trained model for autonomous control

Everything you need to create your own AI driver — neatly packaged and beautifully coded.

---

## 🧠 Features

- 🎥 **Live Frame Capture** — Records gameplay at 25 FPS and resizes to 320×240.  
- ⌨️ **Input Logging** — Captures `W`, `A`, `S`, `D`, `Space`, and `R` keys in real-time.  
- 🧩 **Automated Dataset Creation** — Saves frames + inputs into `.pkl` files every 100 samples.  
- 🧮 **Custom CNN Model** — Learns steering, throttle, braking, resetting, and handbrake.  
- ⚡ **GPU Acceleration** — Automatically detects and utilizes CUDA for training.  
- 💾 **Checkpoint System** — Saves model weights every 5 epochs for safety and recovery.  

---

## 📂 Folder Structure


---

## 🧱 Model Architecture

| Layer | Type        | Kernel | Stride | Padding | Channels | Notes |
|:------|:-------------|:-------:|:-------:|:--------:|:----------:|:------|
| 1 | Conv2d | 5×5 | 1 | 2 | 15 → 32 | Broad feature extraction |
| 2 | Conv2d | 3×3 | 2 | 1 | 32 → 64 | Local feature refinement |
| 3 | Conv2d | 3×3 | 2 | 1 | 64 → 128 | Object edge detection |
| 4 | Conv2d | 3×3 | 2 | 1 | 128 → 256 | Scene compression |
| 5 | Fully Connected | — | — | — | 256 → 5 | Output predictions |

**Outputs:** `[steer, throttle, brake, reset, handbrake]`

---

## ⚙️ How to Use

### 1️⃣ Record Data
Run:
```bash
python framegrab.py
Then drive using your normal controls.
```
Every 100 frames, the data will automatically save into 📂trainingData.

### 2️⃣ Train the Model

Run:

```bash
python traincnn.py
```

- Detects your GPU automatically
- Trains using Mean Squared Error (MSE) loss
- Saves checkpoints in /models/ every 5 epochs

### 3️⃣ Drive with AI

After training, load the model and connect it to your input system —
the AI will replicate your driving behavior frame-by-frame.
```bash
python drivingAI.py
```
---

## 📈 Recommended Settings
| Parameter          | Recommended Value | Description                               |
| :----------------- | :---------------: | :---------------------------------------- |
| `epochs`           |       50–100      | More = better learning (if data is large) |
| `batch_size`       |       32–256      | 64 recommended for strong GPUs            |
| `learning_rate`    |       0.001       | Stable for Adam optimizer                 |
| `input_resolution` |      320×240      | Great balance between speed and detail    |

---
## 🧰 Requirements
Install required libraries:
```Bash
pip install torch torchvision tqdm mss pillow pynput numpy
```
Optional (for advanced logging) *(Not implemented yet)*:
```Bash
pip install matplotlib seaborn tensorboard
```
---

## 🔮 Future Plans

- 🧭 Real-time inference with keyboard/mouse output
- 🎮 Integration with popular simulators (BeamNG, GTA, Assetto Corsa) (Technically already done)
- 📊 Training dashboard & visualizations
- 🤖 Adaptive reinforcement learning mode

---

## 💬 Notes

- The model uses 5 consecutive frames as temporal input to understand motion.
- Outputs are normalized between:
- Steering: -1 → 1
- Throttle/Brake/Reset/Handbrake: 0 → 1
- Designed to train even on mid-range GPUs efficiently.

---

## 🏁 Credits

Developed with passion, precision, and a little too much coffee ☕
by the YTI team — creators of The Best YTI Project in the World.

---

## 🪪 License

This project is licensed under the MIT License — free for all to use, modify, and build upon.

---


# 🌟 Star the repo

## Let’s make AI driving fun, fast, and free for everyone!

---
