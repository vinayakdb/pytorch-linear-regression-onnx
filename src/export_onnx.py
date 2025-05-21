import torch
import torch.onnx
from model import LinearRegressionModel

def export_model(model_path, onnx_path, input_size):
    model = LinearRegressionModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    export_model("model.pth", "model.onnx", input_size=3)