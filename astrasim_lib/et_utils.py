"""Common Chakra ET helpers shared by AstraSim integration modules."""

from __future__ import annotations

import os
from typing import List

from .bootstrap import ensure_chakra_available

ensure_chakra_available()
import et_def_pb2 as pb  # type: ignore
from protolib import (  # type: ignore
    decodeMessage as chakra_decode,
    encodeMessage as chakra_encode,
    openFileRd as chakra_open,
)


__all__ = [
    "pb",
    "chakra_encode",
    "chakra_decode",
    "chakra_open",
    "write_et_node",
    "new_comm_node",
    "new_send_node",
    "new_recv_node",
    "new_comp_node",
    "ensure_dir",
    "size_label",
    "write_comm_microbenchmark",
    "write_point_to_point_microbenchmark",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_et_node(fh, node: pb.Node) -> None:
    chakra_encode(fh, node)


def new_comm_node(node_id: int, name: str, coll_type: int, size_bytes: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMM_COLL_NODE
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    node.attr.append(pb.AttributeProto(name="comm_type", int64_val=coll_type))
    node.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    return node


def new_send_node(node_id: int, name: str, size_bytes: int, dst_rank: int, tag: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMM_SEND_NODE
    node.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    node.attr.append(pb.AttributeProto(name="comm_dst", int32_val=int(dst_rank)))
    node.attr.append(pb.AttributeProto(name="comm_tag", int32_val=int(tag)))
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return node


def new_recv_node(node_id: int, name: str, size_bytes: int, src_rank: int, tag: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMM_RECV_NODE
    node.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    node.attr.append(pb.AttributeProto(name="comm_src", int32_val=int(src_rank)))
    node.attr.append(pb.AttributeProto(name="comm_tag", int32_val=int(tag)))
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return node


def new_comp_node(node_id: int, name: str, duration_micros: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMP_NODE
    node.duration_micros = duration_micros
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return node


def size_label(size_bytes: int) -> str:
    units: List[tuple[str, int]] = [
        ("TB", 1024 ** 4),
        ("GB", 1024 ** 3),
        ("MB", 1024 ** 2),
        ("KB", 1024 ** 1),
    ]
    for suffix, div in units:
        if size_bytes >= div:
            val = size_bytes / div
            return f"{val:.2f}{suffix}"
    val = size_bytes / 1024.0
    return f"{val:.4f}KB"


def write_comm_microbenchmark(prefix_path: str, npus_count: int, coll_type: int, size_bytes: int) -> str:
    ensure_dir(os.path.dirname(prefix_path))
    node_id = 0
    for rank in range(npus_count):
        et_path = f"{prefix_path}.{rank}.et"
        with open(et_path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            name = os.path.basename(prefix_path)
            node = new_comm_node(node_id, name, coll_type, size_bytes)
            write_et_node(fh, node)
            node_id += 1
    return prefix_path


def write_point_to_point_microbenchmark(prefix_path: str, size_bytes: int) -> str:
    ensure_dir(os.path.dirname(prefix_path))

    et_path = f"{prefix_path}.0.et"
    with open(et_path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        node = new_send_node(0, "pipeline_send", size_bytes, dst_rank=1, tag=0)
        write_et_node(fh, node)

    et_path = f"{prefix_path}.1.et"
    with open(et_path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        node = new_recv_node(0, "pipeline_recv", size_bytes, src_rank=0, tag=0)
        write_et_node(fh, node)

    return prefix_path
