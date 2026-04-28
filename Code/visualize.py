import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import json
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import os
from models import SimpleCNN
from dataset import get_dataloaders

def plot_loss_curves():
    os.makedirs('results/plots', exist_ok=True)
    with open('results/history.json', 'r') as f:
        results = json.load(f)
        
    models = ['MLP', 'CNN']
    for m in models:
        plt.figure(figsize=(10, 5))
        for exp_name, history in results.items():
            if exp_name.startswith(m):
                plt.plot(history['train_loss'], label=f"Train {exp_name}")
                plt.plot(history['val_loss'], label=f"Val {exp_name}", linestyle='--')
        
        plt.title(f"{m} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'results/plots/{m}_loss_curves.png')
        plt.close()
        
    # Plot accuracy curves
    plt.figure(figsize=(10, 5))
    for exp_name, history in results.items():
        plt.plot(history['val_acc'], label=f"Val Acc {exp_name}")
            
    plt.title("Validation Accuracy Curves (All Models)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig('results/plots/accuracy_curves.png')
    plt.close()

def imshow(img):
    img = img * torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1) + torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1) # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualize_test_set():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, testloader, classes = get_dataloaders(batch_size=8)
    
    # Load best model (assuming CNN Adam is best)
    model = SimpleCNN(num_classes=100)
    model_path = 'results/models/CNN_Adam_lr0.001.pth'
    if not os.path.exists(model_path):
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    images_dev = images.to(device)
    outputs = model(images_dev)
    _, predicted = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(12, 6))
    for idx in range(8):
        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        pred_label = classes[predicted[idx]] if isinstance(classes[predicted[idx]], str) else str(classes[predicted[idx]])
        true_label = classes[labels[idx]] if isinstance(classes[labels[idx]], str) else str(classes[labels[idx]])
        ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                     color=("green" if predicted[idx]==labels[idx] else "red"))
        
    plt.tight_layout()
    plt.savefig('results/plots/test_predictions.png')
    plt.close()

if __name__ == '__main__':
    plot_loss_curves()
    visualize_test_set()
    print("Visualizations generated and saved in results/plots/")
