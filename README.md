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

***NOTE***
The Reference Material is under the 'ASL_HandSigns/' Folder

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

```
Input Layer (63 features)
    ↓
Dense Layer (128 units, ReLU) + Batch Normalization + Dropout (0.3)
    ↓
Dense Layer (64 units, ReLU) + Batch Normalization + Dropout (0.3)
    ↓
Output Layer (26 classes, Softmax)
```

### Performance
- **Training Accuracy**: 97.8%
- **Validation Accuracy**: 99.5%




