import torch
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.graph_drawer import FxGraphDrawer

from graph.memoryallocator import allocate_memory
from model.basic import BasicModule


def generate_gm(module: torch.nn.Module) -> GraphModule:
    module.bfloat16()
    gm = torch.fx.symbolic_trace(module)

    example_input = torch.ones((32, 32), dtype=torch.bfloat16)
    ShapeProp(gm).propagate(example_input)

    drawer = FxGraphDrawer(gm, "graph")
    drawer.get_dot_graph().write_svg("model/images/graph.svg")

    return gm


def main() -> None:
    basic_module = BasicModule()
    gm = generate_gm(basic_module)
    gm = allocate_memory(gm)

    #remove placeholder/get_attr ops from graph
    #map to kernels


if __name__ == "__main__":
    main()
