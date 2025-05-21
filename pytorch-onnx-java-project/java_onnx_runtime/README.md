# This file provides documentation for the Java ONNX runtime usage, including setup instructions and how to run the Java application.

## Java ONNX Runtime

This directory contains the Java implementation for using the exported ONNX model. The main class is located in `src/Main.java`.

### Setup Instructions

1. Ensure you have Java Development Kit (JDK) installed on your machine.
2. Download the ONNX Runtime Java library from the official ONNX Runtime GitHub repository.
3. Add the ONNX Runtime library to your project's classpath.

### Running the Application

To run the Java application:

1. Compile the `Main.java` file using the following command:
   ```
   javac -cp <path_to_onnxruntime_jar> src/Main.java
   ```
2. Execute the compiled class:
   ```
   java -cp .:<path_to_onnxruntime_jar> src.Main
   ```

### Notes

- Ensure that the ONNX model file is accessible to the Java application.
- Modify the input data format in `Main.java` as needed to match the model's requirements.