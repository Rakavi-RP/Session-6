# MNIST CNN Classifier with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification, achieving >99.4% test(validation) accuracy.

## Project Structure

├── model.py             # Model architecture definition
├── README.md           # Project documentation
├── requirements.txt     # Project dependencies
├── tests/
│   └── test_model.py   # Model specification tests
└── .github/
    └── workflows/
        └── model_tests.yml  # GitHub Actions workflow

## Model Specifications
- Parameters < 20k
- Uses Batch Normalization
- Implements Dropout 
- Uses Global Average Pooling
- Training configured for < 20 epochs

## Requirements

- PyTorch
- torchvision
- torchsummary
- tqdm

## Model Summary
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
              ReLU-2            [-1, 4, 28, 28]               0
       BatchNorm2d-3            [-1, 4, 28, 28]               8
            Conv2d-4            [-1, 4, 28, 28]             148
              ReLU-5            [-1, 4, 28, 28]               0
       BatchNorm2d-6            [-1, 4, 28, 28]               8
            Conv2d-7            [-1, 8, 28, 28]             296
              ReLU-8            [-1, 8, 28, 28]               0
       BatchNorm2d-9            [-1, 8, 28, 28]              16
          Dropout-10            [-1, 8, 28, 28]               0
        MaxPool2d-11            [-1, 8, 14, 14]               0
           Conv2d-12           [-1, 12, 14, 14]             876
             ReLU-13           [-1, 12, 14, 14]               0
      BatchNorm2d-14           [-1, 12, 14, 14]              24
           Conv2d-15           [-1, 12, 14, 14]           1,308
             ReLU-16           [-1, 12, 14, 14]               0
      BatchNorm2d-17           [-1, 12, 14, 14]              24
           Conv2d-18           [-1, 16, 14, 14]           1,744
             ReLU-19           [-1, 16, 14, 14]               0
      BatchNorm2d-20           [-1, 16, 14, 14]              32
        MaxPool2d-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 20, 5, 5]           2,900
             ReLU-23             [-1, 20, 5, 5]               0
      BatchNorm2d-24             [-1, 20, 5, 5]              40
           Conv2d-25             [-1, 10, 3, 3]           1,810
        AvgPool2d-26             [-1, 10, 1, 1]               0
================================================================
Total params: 9,274
Trainable params: 9,274
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.54
Params size (MB): 0.04
Estimated Total Size (MB): 0.58
----------------------------------------------------------------
```

## Training

- Dataset: MNIST
- Optimizer: Adam
- Learning Rate Scheduler: ReduceLROnPlateau (automatically reduces learning rate when model performance plateaus)
- Batch Size: 128
- Training/Test(Validation) Split: 50000/10000

## Model Performance - test logs

- Achieves ~99.43% accuracy on test(validation) set in 15th epoch
- Note: Learning rate was reduced from 0.01 to 0.001 at epoch 15 by ReduceLROnPlateau scheduler, which helped achieve >99.4% accuracy

```python
loss=0.017699357122182846 batch_id=390: 100%|██████████| 391/391 [00:57<00:00,  6.83it/s]
Epoch: 1 | Learning Rate: 0.01000 | Test Loss: 0.1066 | Accuracy: 9673/10000 (96.73%)
loss=0.010867567732930183 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.44it/s]
Epoch: 2 | Learning Rate: 0.01000 | Test Loss: 0.0608 | Accuracy: 9807/10000 (98.07%)
loss=0.11412211507558823 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.47it/s]
Epoch: 3 | Learning Rate: 0.01000 | Test Loss: 0.0410 | Accuracy: 9868/10000 (98.68%)
loss=0.010829591192305088 batch_id=390: 100%|██████████| 391/391 [00:51<00:00,  7.53it/s]
Epoch: 4 | Learning Rate: 0.01000 | Test Loss: 0.0408 | Accuracy: 9878/10000 (98.78%)
loss=0.0012745509156957269 batch_id=390: 100%|██████████| 391/391 [00:51<00:00,  7.53it/s]
Epoch: 5 | Learning Rate: 0.01000 | Test Loss: 0.0394 | Accuracy: 9884/10000 (98.84%)
loss=0.12232786417007446 batch_id=390: 100%|██████████| 391/391 [00:51<00:00,  7.52it/s]
Epoch: 6 | Learning Rate: 0.01000 | Test Loss: 0.0413 | Accuracy: 9879/10000 (98.79%)
loss=0.08515532314777374 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.39it/s]
Epoch: 7 | Learning Rate: 0.01000 | Test Loss: 0.0408 | Accuracy: 9870/10000 (98.70%)
loss=0.05383814126253128 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.51it/s]
Epoch: 8 | Learning Rate: 0.01000 | Test Loss: 0.0334 | Accuracy: 9890/10000 (98.90%)
loss=0.006957699544727802 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.45it/s]
Epoch: 9 | Learning Rate: 0.01000 | Test Loss: 0.0303 | Accuracy: 9894/10000 (98.94%)
loss=0.010455933399498463 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.47it/s]
Epoch: 10 | Learning Rate: 0.01000 | Test Loss: 0.0442 | Accuracy: 9891/10000 (98.91%)
loss=0.02792229875922203 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.40it/s]
Epoch: 11 | Learning Rate: 0.01000 | Test Loss: 0.0296 | Accuracy: 9913/10000 (99.13%)
loss=0.0003698678337968886 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.43it/s]
Epoch: 12 | Learning Rate: 0.01000 | Test Loss: 0.0391 | Accuracy: 9896/10000 (98.96%)
loss=0.0013028652174398303 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.40it/s]
Epoch: 13 | Learning Rate: 0.01000 | Test Loss: 0.0471 | Accuracy: 9862/10000 (98.62%)
loss=0.011150670237839222 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.46it/s]
Epoch: 14 | Learning Rate: 0.01000 | Test Loss: 0.0384 | Accuracy: 9895/10000 (98.95%)
loss=0.010984359309077263 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.48it/s]
Epoch: 15 | Learning Rate: 0.00100 | Test Loss: 0.0217 | Accuracy: 9943/10000 (99.43%)
loss=0.004620316904038191 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.47it/s]
Epoch: 16 | Learning Rate: 0.00100 | Test Loss: 0.0210 | Accuracy: 9945/10000 (99.45%)
loss=0.003626725170761347 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.44it/s]
Epoch: 17 | Learning Rate: 0.00100 | Test Loss: 0.0206 | Accuracy: 9945/10000 (99.45%)
loss=0.0006417831173166633 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.49it/s]
Epoch: 18 | Learning Rate: 0.00100 | Test Loss: 0.0203 | Accuracy: 9944/10000 (99.44%)
loss=0.0014463502448052168 batch_id=390: 100%|██████████| 391/391 [00:52<00:00,  7.48it/s]
Epoch: 19 | Learning Rate: 0.00100 | Test Loss: 0.0200 | Accuracy: 9949/10000 (99.49%)
```

## Testing

The model passes all required specifications:

```python
test_parameter_count (tests.test_model.TestModelSpecs)
Checking parameter count... PASSED (Total: 9274 < 20k)
test_batch_norm_presence (tests.test_model.TestModelSpecs)
Checking BatchNorm... PASSED (Present)
test_dropout_presence (tests.test_model.TestModelSpecs)
Checking Dropout... PASSED (Present)
test_gap_presence (tests.test_model.TestModelSpecs)
Checking Global Average Pooling... PASSED (Present)
----------------------------------------------------------------------
Ran 4 tests in 0.008s
OK
```


## Running Tests Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Run tests:
```bash
python -m unittest discover tests
```


## GitHub Actions Workflow

The repository uses GitHub Actions for automated testing:
1. Triggers on:
   - Push to main branch
   - Pull requests to main branch
2. Workflow steps:
   - Sets up Python 3.8
   - Installs dependencies
   - Runs all test cases
3. Location: `.github/workflows/model_tests.yml`