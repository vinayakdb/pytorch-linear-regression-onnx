import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def train_and_save_model(model_path, input_size=3, output_size=1, epochs=100):
    # Generate some dummy data for training
    X = torch.randn(100, input_size)
    true_weights = torch.randn(input_size, output_size)
    y = X @ true_weights + torch.randn(100, output_size) * 0.1

    model = LinearRegressionModel(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to {model_path}")
    return model

def export_model_to_onnx(model_path, onnx_path, input_size=3, output_size=1):
    model = LinearRegressionModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)
    print(f"Model exported to ONNX format at {onnx_path}")

if __name__ == "__main__":
    model_path = "trained_model.pth"
    onnx_path = "linear_regression_model.onnx"
    input_size = 3
    output_size = 1

    train_and_save_model(model_path, input_size, output_size)
    export_model_to_onnx(model_path, onnx_path, input_size, output_size)