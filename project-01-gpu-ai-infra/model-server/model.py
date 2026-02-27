import torch
import torch.nn as nn

# 1. Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
    
# 2. Choose GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Move the model to the device after creation
model = SimpleModel().to(device)

# 4. Set to evaluation mode
model.eval()

# 5. Inference function
def predict(input_data):
    with torch.no_grad():
        tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        output = model(tensor)
        return output.cpu().numpy().tolist()
    
print("Using device:", device)