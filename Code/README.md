# CIFAR-100 Classification Project

This directory contains the code for training and evaluating MLP and CNN architectures on the CIFAR-100 dataset.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

## Running Experiments
To run the full suite of experiments (tests learning rates and optimizers across models), execute:

```bash
python main.py
```
This will train the models for 10 epochs and save the result histories and models to the `results/` folder.

## Visualization
After running the experiments, execute:

```bash
python visualize.py
```
This will generate loss curve plots and a visualization of test set predictions in the `results/plots/` folder.
