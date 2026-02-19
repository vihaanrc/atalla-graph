import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node


TILE_BYTES = 32 * 32 * 2  # 2 KB tiles (32x32 bf16 values)
BF16_NUM_BYTES = 2
LINE_BYTES = 16 #number of bytes to display per line in dram.txt


def align(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def tensor_nbytes(node: Node) -> int:
    tensor_meta = node.meta.get("tensor_meta")

    if tensor_meta.dtype != torch.bfloat16:
        raise ValueError(
            f"dtype {tensor_meta.dtype} for node {node.name} is not torch.bfloat16")

    if len(tensor_meta.shape) != 2:
        raise ValueError(f"Shape {tensor_meta.shape} for node {node.name} is not 2D")

    numel = 1
    for dim in tensor_meta.shape:
        numel *= int(dim)

    return numel * BF16_NUM_BYTES


def tensor_for_node(
    node: Node, gm: GraphModule, placeholder_data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if node.op == "placeholder":
        if node.target not in placeholder_data:
            raise ValueError(f"Missing placeholder data for {node.target}")
        return placeholder_data[node.target]

    if node.op == "get_attr":
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part) #must be a tensor
        return attr

    return None

#TODO: write data in 32x32 tiles, not just row-wise
def tensor_bytes(tensor: torch.Tensor, allocation_size: int) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    data = tensor.view(torch.uint16).numpy().astype(np.uint16)
    raw = data.tobytes(order="C")
    return raw + bytes(allocation_size - len(raw)) #pad to allocation size if needed (tensor < 32x32)


def write_dram(file, start_addr: int, payload: bytes) -> None:
    for offset in range(0, len(payload), LINE_BYTES):
        chunk = payload[offset : offset + LINE_BYTES]
        hex_bytes = " ".join(f"{byte:02x}" for byte in chunk)
        file.write(f"0x{start_addr + offset:08x}: {hex_bytes}\n")


def assign_address(node: Node, next_addr: int) -> Tuple[int, int, int]:
    bytes_needed = tensor_nbytes(node)
    aligned_addr = align(next_addr, TILE_BYTES)
    allocation_size = align(bytes_needed, TILE_BYTES)

    node.meta["dram_addr"] = f"0x{aligned_addr:08x}"
    node.meta["bytes"] = allocation_size

    return aligned_addr, allocation_size, aligned_addr + allocation_size


def allocate_memory(gm: GraphModule, text_path: str, placeholder_data: Optional[Dict[str, torch.Tensor]] = None) -> GraphModule:
    placeholder_data = placeholder_data or {}
    next_addr = 0

    with open(text_path, "w") as dram_file:
        for node in gm.graph.nodes:
            if node.op != "output":
                start, size, next_addr = assign_address(node, next_addr)
                tensor_value = tensor_for_node(node, gm, placeholder_data)
                payload = tensor_bytes(tensor_value, size) if tensor_value is not None else bytes(size)
                write_dram(dram_file, start, payload)

    gm.graph.lint()
    gm.recompile()

    return gm
