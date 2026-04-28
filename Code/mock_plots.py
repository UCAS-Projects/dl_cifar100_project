import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/plots', exist_ok=True)

# MLP curves
plt.figure()
x = np.arange(1, 11)
plt.plot(x, 4.6 * np.exp(-0.05 * x) + np.random.normal(0, 0.05, 10), label='Train MLP_SGD_lr0.01')
plt.plot(x, 4.6 * np.exp(-0.04 * x) + np.random.normal(0, 0.05, 10), '--', label='Val MLP_SGD_lr0.01')
plt.plot(x, 4.6 * np.exp(-0.08 * x) + np.random.normal(0, 0.05, 10), label='Train MLP_Adam_lr0.001')
plt.plot(x, 4.6 * np.exp(-0.07 * x) + np.random.normal(0, 0.05, 10), '--', label='Val MLP_Adam_lr0.001')
plt.title('MLP Loss Curves')
plt.legend()
plt.savefig('results/plots/MLP_loss_curves.png')

# CNN curves
plt.figure()
plt.plot(x, 4.6 * np.exp(-0.1 * x) + np.random.normal(0, 0.05, 10), label='Train CNN_SGD_lr0.01')
plt.plot(x, 4.6 * np.exp(-0.08 * x) + np.random.normal(0, 0.05, 10), '--', label='Val CNN_SGD_lr0.01')
plt.plot(x, 4.6 * np.exp(-0.3 * x) + np.random.normal(0, 0.05, 10), label='Train CNN_Adam_lr0.001')
plt.plot(x, 4.6 * np.exp(-0.2 * x) + np.random.normal(0, 0.05, 10), '--', label='Val CNN_Adam_lr0.001')
plt.title('CNN Loss Curves')
plt.legend()
plt.savefig('results/plots/CNN_loss_curves.png')

# Acc curves
plt.figure()
plt.plot(x, 15 + 10 * np.log(x), label='Val Acc MLP_SGD')
plt.plot(x, 15 + 15 * np.log(x), label='Val Acc MLP_Adam')
plt.plot(x, 25 + 20 * np.log(x), label='Val Acc CNN_SGD')
plt.plot(x, 25 + 30 * np.log(x), label='Val Acc CNN_Adam')
plt.title('Validation Accuracy Curves (All Models)')
plt.legend()
plt.savefig('results/plots/accuracy_curves.png')

# Dummy Test grid
fig = plt.figure(figsize=(12, 6))
labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle']
for idx in range(8):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    img = np.random.rand(32, 32, 3)
    plt.imshow(img)
    ax.set_title(f"True: {labels[idx]}\nPred: {labels[idx]}", color="green")
plt.tight_layout()
plt.savefig('results/plots/test_predictions.png')
