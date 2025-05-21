# This file provides documentation for the PyTorch model, including instructions on how to train the model and export it to ONNX.

## PyTorch Linear Regression Model

This project implements a linear regression model using PyTorch and provides functionality to export the trained model to the ONNX format.

### Requirements

- Python 3.x
- PyTorch
- ONNX

### Training the Model

1. **Implement the Model**: The linear regression model is defined in `linear_regression.py`. You can modify the input features and training parameters as needed.

2. **Train the Model**: Use the provided methods in `linear_regression.py` to train the model on your dataset. Ensure your dataset is properly formatted.

3. **Save the Model**: After training, save the model using PyTorch's `torch.save()` function.

### Exporting to ONNX

1. **Export the Model**: Use the `export_to_onnx.py` script to export the trained model to the ONNX format. Make sure to load your trained model in this script.

2. **Run the Export Script**: Execute the `export_to_onnx.py` script to generate the ONNX model file. The script uses `torch.onnx.export()` to perform the export.

### Example Usage

To train and export the model, you can run the following commands in your terminal:

```bash
python pytorch_model/linear_regression.py
python pytorch_model/export_to_onnx.py
```

### Notes

- Ensure that you have the necessary libraries installed before running the scripts.
- Check the output ONNX model file to verify that the export was successful.