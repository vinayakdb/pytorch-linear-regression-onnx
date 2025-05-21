import onnxruntime as ort
import numpy as np

def run_onnx_model(onnx_model_path, input_data):
    # Load the ONNX model
    session = ort.InferenceSession(onnx_model_path)

    # Prepare the input data as a dictionary
    input_name = session.get_inputs()[0].name
    input_data = np.array(input_data, dtype=np.float32)

    # Run inference
    result = session.run(None, {input_name: input_data})

    return result

if __name__ == "__main__":
    # Example usage
    onnx_model_path = "path/to/your/model.onnx"
    input_data = [[1.0, 2.0], [3.0, 4.0]]  # Replace with your actual input data

    output = run_onnx_model(onnx_model_path, input_data)
    print("Model output:", output)