# PyTorch Linear Regression with ONNX Export

This project demonstrates how to implement a linear regression model using PyTorch, export it to the ONNX format, and run the exported model using the `ai.onnxruntime` library.

## Project Structure

```
pytorch-linear-regression-onnx
├── src
│   ├── train.py          # Code to define and train the linear regression model
│   ├── export_onnx.py    # Code to export the trained model to ONNX format
│   ├── run_onnx.py       # Code to run the exported ONNX model
│   └── model.py          # Definition of the linear regression model
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pytorch-linear-regression-onnx
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the linear regression model, run the following command:
```
python src/train.py
```
This will prepare the data, instantiate the model, set up the loss function and optimizer, and execute the training loop.

### Exporting the Model to ONNX

After training the model, you can export it to the ONNX format by running:
```
python src/export_onnx.py
```
This will save the trained model in the ONNX format, which can be used for inference in other environments.

### Running the ONNX Model

To run the exported ONNX model, use the following command:
```
python src/run_onnx.py
```
This script will load the ONNX model, prepare the input data, and execute inference, displaying the results.

## Code Explanation

- **src/train.py**: Contains the logic for training the linear regression model, including data preparation and the training loop.
- **src/export_onnx.py**: Handles the export of the trained model to ONNX format.
- **src/run_onnx.py**: Loads the ONNX model and performs inference using the `ai.onnxruntime` library.
- **src/model.py**: Defines the architecture of the linear regression model.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.