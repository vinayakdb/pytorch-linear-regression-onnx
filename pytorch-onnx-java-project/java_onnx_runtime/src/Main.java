import ai.onnxruntime.*;

import java.util.Collections;
import java.util.Map;

public class Main {
    private OrtEnvironment env;
    private OrtSession session;

    public Main(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath);
    }

    public float[] runInference(float[] inputData) throws OrtException {
        // The model expects input shape [1, 3] (batch size 1, 3 features)
        long[] shape = new long[]{1, 3};
        java.nio.FloatBuffer buffer = java.nio.FloatBuffer.wrap(inputData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buffer, shape);

        // The input name is usually "input.1" for PyTorch ONNX exports, but you should check your model
        String inputName = session.getInputNames().iterator().next();

        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);
        OrtSession.Result outputs = session.run(inputs);

        float[][] outputArray = (float[][]) outputs.get(0).getValue();
        float[] outputData = outputArray[0];

        inputTensor.close();
        outputs.close();
        return outputData;
    }

    public static void main(String[] args) {
        try {
            String modelPath = "linear_regression_model.onnx"; // Update with your ONNX model path
            Main onnxRuntime = new Main(modelPath);

            // Example input data: 3 features as per the Python model
            float[] inputData = {1.0f, 2.0f, 3.0f};
            float[] outputData = onnxRuntime.runInference(inputData);

            System.out.println("Model output: " + java.util.Arrays.toString(outputData));
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}