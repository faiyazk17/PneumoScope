## 🫁 PneumoScope: AI-Based Lung Segmentation from Chest X-rays

**PneumoScope** is a deep learning-powered medical imaging project that segments lung regions from chest X-rays using a U-Net convolutional neural network (CNN). Built with **PyTorch**, this project supports clinical diagnosis by providing fast, accurate lung region isolation, useful for detecting conditions such as pneumonia or pneumothorax.

---

## 🧠 Project Overview

* **Goal**: Semantic segmentation of lungs in chest X-rays.
* **Architecture**: U-Net with skip connections.
* **Framework**: PyTorch.
* **Dataset**: 12,000+ real-world chest X-ray images and masks.
* **Image Resolution**: Input - 256x256 RGB; Output - 256x256 binary mask.

---

## 🚀 Key Features

* 🧾 **Custom Dataset Loader**: Dynamically loads image-mask pairs for training.
* 🔁 **U-Net Model**: Encoder-decoder CNN with skip connections for precise segmentation.
* 💾 **Checkpointing**: Saves model state and optimizer to resume interrupted training.
* 📊 **Training Visualization**: Live training metrics using TensorBoard.
* 📈 **Loss Tracking**: BCE + Dice loss function with plotted loss curves.

---

## 🖥️ How to Run

### 🧩 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/PneumoScope.git
cd PneumoScope

# Install Python dependencies
pip install -r requirements.txt
```

### 🚦 Start Training

```bash
python model/train.py
```

### 📉 Monitor Training Progress

```bash
tensorboard --logdir=runs
```

Then navigate to `http://localhost:6006/` in your browser.

---

## 🗂 Dataset Info

* **Source**: Kaggle Pneumothorax Chest X-ray Dataset
* **Format**: PNG/JPG images + RLE encoded masks
* **Preprocessing**:

  * Resized to 256x256
  * Masks converted to binary (lung = 1, background = 0)

---

## 📦 Project Structure

```
PneumoScope/
├── data/                # Contains images/ and masks/
├── model/
│   ├── dataset.py       # Custom PyTorch Dataset class
│   ├── unet_model.py    # U-Net architecture
│   ├── train.py         # Training script with TensorBoard support
│   └── test_dataset.py  # Dataset loader testing script
├── runs/                # TensorBoard logs
├── checkpoints/         # Saved model checkpoints
├── requirements.txt     # Python dependencies
└── README.md            # You're here!
```

---

## 🛠️ Tech Stack & Tools

* **Languages**: Python
* **Libraries**: PyTorch, OpenCV, NumPy, Matplotlib, TQDM
* **Visualization**: TensorBoard
* **Version Control**: Git

### 🔮 Planned Full Stack Deployment

| Tool       | Purpose                         |
| ---------- | ------------------------------- |
| Docker     | Containerize training/inference |
| Kubernetes | Scale model across clusters     |
| Jenkins    | Automate CI/CD pipeline         |
| AWS S3/EC2 | Host model + serve predictions  |
