import torch
import torch.nn as nn

def generateGraphModule(m : nn.Module) -> torch.fx.GraphModule:
    
    m.bfloat16() #convert all buffers and parameters to bfloat16

    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)
    return gm