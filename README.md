# pythondemo
this repository is just a space where I could save some demo for my python-learning

## LeNet-5 Demo

This repository includes an implementation of LeNet-5, a classic convolutional neural network architecture designed for handwritten digit recognition.

### Architecture

The LeNet-5 model follows this architecture:
- **Input**: 32x32 grayscale images
- **Conv1**: 6 filters of 5x5, followed by ReLU activation
- **Pool2**: 2x2 average pooling
- **Conv3**: 16 filters of 5x5, followed by ReLU activation
- **Pool4**: 2x2 average pooling
- **FC5**: Fully connected layer with 120 units, followed by ReLU activation
- **FC6**: Fully connected layer with 84 units, followed by ReLU activation
- **Output**: Fully connected layer with 10 units (for MNIST digits 0-9)

This implementation uses ReLU activation functions instead of the original tanh for improved performance.

### How to Run

1. Navigate to the `lenet5` directory:
   ```bash
   cd lenet5
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model on MNIST dataset:
   ```bash
   python train.py
   ```

The training script will:
- Automatically download the MNIST dataset
- Detect and use CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU
- Train the model for 10 epochs
- Display training progress and test accuracy after each epoch
- Save the trained model as `lenet5_mnist.pth`

### Requirements

- Python 3.6+
- PyTorch
- torchvision
