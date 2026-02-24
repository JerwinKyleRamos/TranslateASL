# TranslateASL

This is a python based project that translates ASL Sign Language into Readable Texts 

## ✨ Features

- Fast and lightweight
- Realtime Hand Sign Translation
- Offline Translation
- Supports 24 ASL letters (A-Y) (Excluding Letters J and Z), plus special commands (delete, space)
- Multi-orientation support (palm and back of hand)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/JerwinKyleRamos/TranslateASL.git
cd TranslateASL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🛠 Usage

### Running the Translator
```bash
python asl_translator.py
```

### Collecting Your Own Dataset (Optional)
If you want to retrain the model with your own data:
```bash
python data_collection.py
```

### Training the Model (Optional)
```bash
python train_model.py
```

## 🧰 Tech Stack

- python 3.11
- mediapipe
- OpenCV (cv2)
- TensorFlow/keras
- Pandas

## Model Overview

A fully-connected neural network trained to classify ASL (American Sign Language) alphabet gestures from hand landmark coordinates.

### Dataset
- Size: 24,000 samples
- Features: 63 hand landmark coordinates (x, y, z for 21 landmarks)
- Classes: 24 (A-Y alphabet letters), plus special commands (delete, space)
- Split: 80% train / 10% validation / 10% test
- **Note**: Not all letters include samples from both hand orientations

**Model Architecture**: 

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │         8,192 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 128)            │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 64)             │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 26)             │         1,690 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 18,906 (73.85 KB)
 Trainable params: 18,522 (72.35 KB)
 Non-trainable params: 384 (1.50 KB)

### Performance
- **Training Accuracy**: 97.8%
- **Validation Accuracy**: 99.5%




