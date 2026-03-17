# DR-ResNet: Dynamic Routing Residual Network

This repository contains the implementation of **DR-ResNet** (Dynamic Routing Residual Network), a lightweight deep learning model for metabolic state classification based on physiological sensor data.

## Model Architecture

### DR-ResNet (Dynamic Routing Residual Network)

DR-ResNet introduces a dynamic routing mechanism into residual connections, allowing adaptive feature flow based on input importance. The key components are:

1. **Dynamic Routing Block**: 
   - Uses a gating mechanism to adaptively combine residual and transformed features
   - Applies 1D convolution with GELU activation
   - Employs channel-wise attention to compute routing gates

2. **Architecture**:
   - Input expansion layer: Projects 3D input (HB, Glu, pH) to higher dimensional space
   - 4 sequential dynamic routing blocks with group processing
   - Global average pooling and classification head

3. **Key Features**:
   - Parameter-efficient with ~10K parameters
   - Dynamic routing enhances feature selection
   - Residual connections alleviate gradient vanishing
   - Suitable for resource-constrained environments

## Dataset

### Physiological Sensor Data for Metabolic State Classification

The dataset contains physiological measurements for classifying different metabolic states:

### Data Format

Each CSV file contains the following columns:
- **Time (s)**: Time stamp in seconds
- **HB (mM)**: β-hydroxybutyrate concentration
- **Glu (mM)**: Glucose concentration
- **pH**: pH level
- **label**: Class label (0-3 representing different metabolic states)

### Data Files

The dataset is organized into 6 subjects with 6 conditions each:
- Subject 1-6: Different individuals
- Conditions: long-term ketogenic, non-ketogenic, short-term ketogenic
- Each condition has "Before" and "After" measurements

### Input Features

The model uses 3 physiological features as input:
1. **HB (β-hydroxybutyrate)**
2. **Glu (Glucose)**
3. **pH**

## Requirements

```
torch>=2.7.1
numpy>=1.26.1
pandas>=2.3.3
scikit-learn>=1.6.1
```

## Usage

### Training

```python
python train.py
```

The training script will:
1. Load all CSV files from the `data/` directory
2. Split data into training and validation sets
3. Standardize features using StandardScaler
4. Train DR-ResNet with Focal Loss and AdamW optimizer
5. Save the best model to `drresnet_model.pth`

### Model Configuration

Default configuration in `train.py`:
- Hidden dimension: 16
- Number of routing blocks: 4
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 500
- Weight decay: 1e-5

### Inference

```python
import torch
from model import DRResNet

# Load model
model = DRResNet(input_dim=3, hidden_dim=16, num_classes=4)
model.load_state_dict(torch.load('drresnet_model.pth'))
model.eval()

# Prepare input (HB, Glu, pH)
input_data = torch.tensor([[0.129, 2.083, 7.481]], dtype=torch.float32)

# Inference
with torch.no_grad():
    output = model(input_data)
    predicted_class = torch.argmax(output, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
```

## File Structure

```
DR-ResNet/
├── model.py              # DR-ResNet model definition
├── train.py              # Training script
├── data/                 # Dataset directory
│   ├── 1-long-term ketogenic-After.csv
│   ├── 1-non-ketogenic-Before.csv
│   └── ...
└── README.md             # This file
```
