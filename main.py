import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graph.export_fx import render_graph
from graph.memoryallocator import allocate_memory
from model.basic import BasicModule


def main() -> None:
    module = BasicModule()
    module.bfloat16()

    gm = symbolic_trace(module)

    example_input = torch.ones((32, 32), dtype=torch.bfloat16)
    ShapeProp(gm).propagate(example_input)

    placeholder_data = {"x": example_input.clone()} #needed to populate dram.txt for input

    gm = allocate_memory(gm, "model/images/dram.txt", placeholder_data)

    render_graph(gm, "model/images/graph.svg", meta_keys=("dram_addr", "bytes"))


if __name__ == "__main__":
    main()
