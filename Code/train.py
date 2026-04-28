import torch
import torch.nn as nn
import torch.optim as optim
from models import MLPNet, SimpleCNN

def train_model(model_name, dataloaders, lr, optimizer_name, num_epochs=10, device='cuda'):
    trainloader, testloader, _ = dataloaders
    
    if model_name == 'MLP':
        model = MLPNet(num_classes=100).to(device)
    else:
        model = SimpleCNN(num_classes=100).to(device)
        
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Training {model_name} with {optimizer_name} (lr={lr})")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(trainloader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(testloader)
        val_acc = 100 * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    return model, history
