import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import json
from dataset import get_dataloaders
from train import train_model

def run_experiments():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataloaders = get_dataloaders(batch_size=128)
    
    # 4 combinations of model + hyperparameters. Adjust num_epochs if needed.
    experiments = [
        {'model': 'MLP', 'lr': 0.01, 'opt': 'SGD'},
        {'model': 'MLP', 'lr': 0.001, 'opt': 'Adam'},
        {'model': 'CNN', 'lr': 0.01, 'opt': 'SGD'},
        {'model': 'CNN', 'lr': 0.001, 'opt': 'Adam'}
    ]
    
    results = {}
    os.makedirs('results/models', exist_ok=True)
    
    for ext in experiments:
        exp_name = f"{ext['model']}_{ext['opt']}_lr{ext['lr']}"
        model, history = train_model(ext['model'], dataloaders, ext['lr'], ext['opt'], num_epochs=10, device=device)
        results[exp_name] = history
        
        torch.save(model.state_dict(), f"results/models/{exp_name}.pth")
        
    with open('results/history.json', 'w') as f:
        json.dump(results, f)
        
    print("All experiments finished!")

if __name__ == '__main__':
    run_experiments()
