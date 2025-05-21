# PyTorch ONNX Java Project

This project demonstrates how to implement a linear regression model using PyTorch, export it to the ONNX format, and then use the exported model in a Java application with the ai.onnxruntime library.

## Project Structure

```
pytorch-onnx-java-project
├── pytorch_model
│   ├── linear_regression.py       # Implementation of the linear regression model in PyTorch
│   ├── export_to_onnx.py         # Script to export the trained model to ONNX format
│   └── README.md                  # Documentation for the PyTorch model
├── java_onnx_runtime
│   ├── src
│   │   └── Main.java              # Main class to load and run inference with the ONNX model
│   └── README.md                  # Documentation for using the Java ONNX runtime
└── README.md                      # Main documentation for the entire project
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- ONNX
- Java Development Kit (JDK)
- ONNX Runtime for Java

### PyTorch Model

1. Navigate to the `pytorch_model` directory.
2. Train the linear regression model by running `linear_regression.py`.
3. Export the trained model to ONNX format by executing `export_to_onnx.py`.

### Java ONNX Runtime

1. Navigate to the `java_onnx_runtime` directory.
2. Compile the Java application using the provided `Main.java`.
3. Run the application to perform inference using the exported ONNX model.

## Additional Notes

- Ensure that all dependencies are installed before running the scripts.
- Refer to the individual README files in the `pytorch_model` and `java_onnx_runtime` directories for more detailed instructions.