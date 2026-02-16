import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.ones([32, 32])
        self.b = torch.ones([32, 32]) * 4

    def forward(self) -> torch.Tensor:
        c = self.a + self.b
        return c
  
