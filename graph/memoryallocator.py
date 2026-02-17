from torch.fx import GraphModule


def allocate_memory(gm: GraphModule) -> GraphModule:
    
    # Modify gm.graph

    # <...>

    gm.graph.lint() #check if well-formed
    gm.recompile()

    return gm
