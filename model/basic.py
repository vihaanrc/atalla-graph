import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.ones([32, 32], dtype=torch.bfloat16) * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias
  
