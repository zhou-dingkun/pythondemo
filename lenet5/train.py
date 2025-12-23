import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet5


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def get_data_loaders(batch_size=64):
    """
    Prepare MNIST dataset loaders with appropriate transformations.
    
    Args:
        batch_size (int): Batch size for training and testing
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # LeNet-5 expects 32x32 input, so we resize MNIST from 28x28
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to use for training
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        epoch: Current epoch number
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def test(model, device, test_loader, criterion):
    """
    Test the model on the test dataset.
    
    Args:
        model: The neural network model
        device: Device to use for testing
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(f'Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    return test_loss, test_acc


def main():
    """
    Main training function.
    """
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Setup
    device = get_device()
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Model, loss, and optimizer
    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'\nModel Architecture:')
    print(model)
    print(f'\nStarting training for {num_epochs} epochs...\n')
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{num_epochs} ---')
        train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
    
    # Save the trained model
    model_path = 'lenet5_mnist.pth'
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')


if __name__ == '__main__':
    main()
