Real Estate LSTM


This repository contains a Python implementation of a Long Short-Term Memory (LSTM) neural network designed for predicting real estate prices based on the Boston Housing dataset. The model leverages the LSTM architecture, which is particularly effective for time series data and sequences, to predict housing prices.

Overview

The script `RealEstateLSTM.py` performs the following key steps:

1. Data Loading: The Boston Housing dataset is loaded from the TensorFlow Keras dataset library.

2. Data Normalization: The input features are normalized to have a mean of 0 and a standard deviation of 1. This step is crucial to ensure that the model converges efficiently during training.

3. Model Architecture: 
   - An LSTM model is built using the Sequential API from TensorFlow Keras.
   - The model includes one LSTM layer with dropout regularization to prevent overfitting.
   - A Dense output layer with a single neuron is used for the final prediction.

4. Model Compilation: 
   - The model is compiled with the Adam optimizer and a mean squared error loss function. 
   - Mean squared error is suitable for regression tasks as it penalizes large errors.

5. Model Training:
   - The model is trained for 10 epochs with a batch size of 32.
   - The training process optimizes the model weights to minimize the loss.

6. Model Evaluation:
   - The model is evaluated on a test set, and the final loss value is recorded.

Results

- The model was trained for 10 epochs, and the final test loss was **135.0305**. This value represents the mean squared error on the test set, indicating the model's performance in predicting housing prices.

Requirements

- Python 3.x
- TensorFlow 2.x
- Numpy

To install the required dependencies, use:

```bash
pip install tensorflow numpy
```

## Usage

To run the script:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RealEstateLSTM.git
   cd RealEstateLSTM
   ```

2. Execute the Python script:
   ```bash
   python RealEstateLSTM.py
   ```

The script will train the model on the Boston Housing dataset and output the loss on the test set.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to adjust any section based on the specifics of your project or preferences.

