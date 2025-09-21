#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

# import click
import math
import os
import pickle
import sys
import config
import shutil
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Set
# import numpy as np
import simulate_LLM
from parallelism import Parallelism
from topology import Topology
from simulate_LLM import Graph
import LLM_util
from hw_component import Core, MemoryHierarchy, Network
from model import Model_LSTM, Model_GEMM, Model_LLM
from tile import TiledGEMM, formatBytes
from astra_comparison import run_astra_simulation_only_onepath
from functools import lru_cache

from simulate_LLM import visualize_graph
from time_calculation import TimeCalculation
from util import disk_cache_method
# algByte = False  # algorithmic ops false
# proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True

debug = True
all_reduce = "every layer"  #"the end"  "every layer"


showing_ms = False # Show time in ms if True    show time in us if False
if showing_ms:
    m=1e3
    second = "ms"
else:
    m=1e6
    second = "us"


class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"


@dataclass
class ExecutionResult:
    total_time: float
    graph_root: Any
    mode: ExecutionMode


@dataclass
class TransformerTimings:
    forward: float
    backward: float


class PipelineGraphFlattener:
    """Expand pipeline transformer nodes into explicit tensor-parallel subgraphs."""

    def __init__(
        self,
        pipeline_graph: Graph,
        transformer_graph: Graph,
    ) -> None:
        if transformer_graph is None:
            raise ValueError("Transformer graph is required for flattening")

        transformer_cfg = getattr(transformer_graph, "transformer_cfg", None) or {}
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM template is missing")

        self.pipeline_graph = pipeline_graph
        self.transformer_graph = transformer_graph
        self._gemm_entries = list(gemm_entries)
        tp_degree = transformer_cfg.get("tp_degree")
        if tp_degree is None:
            tp_degree = max(1, getattr(pipeline_graph, "kp1", 1) * getattr(pipeline_graph, "kp2", 1))
        self._tp_degree = max(1, int(tp_degree))

        self._clone_cache: Dict[int, Any] = {}
        self._op_id_counter: int = 0

    @property
    def tp_degree(self) -> int:
        return self._tp_degree

    def build(self, root: Any) -> Any:
        """Return a flattened clone of the provided pipeline root."""

        if root is None:
            raise ValueError("Pipeline root is required for flattening")
        return self._clone(root)

    def _clone(self, obj: Any) -> Any:
        if obj is None:
            return None

        obj_id = id(obj)
        if obj_id in self._clone_cache:
            return self._clone_cache[obj_id]

        if isinstance(obj, simulate_LLM.Node):
            if obj.name in {"transformer", "transformer_b"}:
                expanded = self._expand_transformer_node(obj)
                self._clone_cache[obj_id] = expanded
                return expanded

            cloned = simulate_LLM.Node(
                obj.name,
                self._next_op_id(),
                obj.hw_id,
                obj.duration,
                fwd=obj.fwd,
            )
            self._clone_cache[obj_id] = cloned
            self._copy_metadata(obj, cloned)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned, child_clone)
            return cloned

        if isinstance(obj, simulate_LLM.Edge):
            cloned_edge = simulate_LLM.Edge(
                obj.name,
                self._next_op_id(),
                obj.duration,
                is_all_reduce=getattr(obj, "is_all_reduce", False),
                comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                comm_type=getattr(obj, "comm_type", None),
                participants=getattr(obj, "participants", 1),
                comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
            )
            self._clone_cache[obj_id] = cloned_edge
            self._copy_metadata(obj, cloned_edge)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_edge, child_clone)
            return cloned_edge

        if isinstance(obj, simulate_LLM.Data_batch):
            cloned_batch = simulate_LLM.Data_batch(obj.name, obj.batch_id, obj.duration)
            self._clone_cache[obj_id] = cloned_batch
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_batch, child_clone)
            return cloned_batch

        if isinstance(obj, simulate_LLM.Gradient):
            cloned_grad = simulate_LLM.Gradient(obj.name, self._next_op_id(), obj.hw_id, obj.duration)
            self._clone_cache[obj_id] = cloned_grad
            self._copy_metadata(obj, cloned_grad)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_grad, child_clone)
            return cloned_grad

        raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

    def _expand_transformer_node(self, node: simulate_LLM.Node) -> Tuple[Any, ...]:
        node_id = id(node)
        if node_id in self._clone_cache:
            cached_entry = self._clone_cache[node_id]
            if isinstance(cached_entry, (list, tuple)):
                return tuple(cached_entry)

        stage_id = getattr(node, "stage_id", node.hw_id)
        micro_batch = getattr(node, "micro_batch_index", None)
        layer_index = getattr(node, "layer_index", None)
        direction = getattr(node, "direction", "forward" if node.fwd else "backward")

        rank_heads: List[Any] = []
        rank_tails: List[Any] = []

        for tp_rank in range(self._tp_degree):
            previous: Optional[Any] = None
            head: Optional[Any] = None
            hw_id = self._hw_id_for_rank(stage_id, tp_rank)

            gemm_iterable = self._gemm_entries
            if direction == "backward":
                gemm_iterable = list(reversed(self._gemm_entries))

            for gemm_idx, entry in enumerate(gemm_iterable):
                entry_name = entry.get("name", f"g{gemm_idx}")
                cfg = entry.get(direction, {})
                duration = cfg.get("duration")
                if duration is None:
                    raise ValueError(
                        f"Missing duration for transformer entry '{entry_name}' in direction '{direction}'"
                    )

                gemm_node = simulate_LLM.Node(
                    name=self._format_gemm_name(entry_name, direction, micro_batch, layer_index, tp_rank),
                    op_id=self._next_op_id(),
                    hw_id=hw_id,
                    duration=duration,
                    fwd=(direction == "forward"),
                )
                gemm_node.stage_id = stage_id
                gemm_node.tp_rank = tp_rank
                gemm_node.micro_batch_index = micro_batch
                gemm_node.layer_index = layer_index
                gemm_node.direction = direction

                if previous is not None:
                    previous.add_child(gemm_node)
                previous = gemm_node
                if head is None:
                    head = gemm_node

                for comm_key in cfg.get("comm_keys", []):
                    comm_edge = self._create_transformer_comm_edge(
                        comm_key,
                        hw_id,
                        stage_id,
                        micro_batch,
                        layer_index,
                        direction,
                        tp_rank,
                    )
                    previous.add_child(comm_edge)
                    previous = comm_edge

            if head is None:
                raise ValueError("Transformer expansion produced no GEMM nodes")

            rank_heads.append(head)
            rank_tails.append(previous or head)

        dp_children: List[Any] = []
        other_children: List[Any] = []

        for child in getattr(node, "children", []):
            comm_type = getattr(child, "comm_interconnect_type", None)
            if comm_type == "dp":
                dp_children.append(child)
            else:
                other_children.append(child)

        # Keep the main trunk pointing to the per-rank compute tails.
        downstream_parents: List[Any] = list(rank_tails)

        # Attach DP collectives as side branches from the compute tails, without
        # reparenting the trunk. This preserves the true cross-layer pipeline
        # edge between compute nodes for ET conversion.
        for child in dp_children:
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(rank_tails[0], child_clone) # only attach to the first tail for DP collectives

        # Non-DP edges (e.g., cross_layer) stay on the trunk so (parent, target)
        # compute → compute pipeline edges remain visible.
        for child in other_children:
            # Special-case marked cross_layer edges (set in original graph):
            # create one per TP rank and wire tail[r] -> cross_layer_r -> next_head[r].
            is_pipeline_edge = False
            if isinstance(child, simulate_LLM.Edge):
                comm_type = getattr(child, "comm_type", None)
                if comm_type == "pipeline":
                    is_pipeline_edge = True
            if is_pipeline_edge:
                # Determine per-rank byte size (ceil split)
                try:
                    total_bytes = int(getattr(child, "comm_size_bytes", 0))
                except Exception:
                    total_bytes = 0
                per_rank_bytes = int(math.ceil(float(total_bytes) / float(max(1, self._tp_degree))))

                # Clone the original targets of this pipeline edge
                target_clones: List[Any] = []
                for tgt in getattr(child, "children", []):
                    tgt_clone = self._clone(tgt)
                    if tgt_clone is None:
                        continue
                    target_clones.append(tgt_clone)
                if not target_clones:
                    # No downstream target; skip safely
                    continue

                # For each TP rank, create its own pipeline edge and connect
                for r, tail in enumerate(rank_tails):
                    # Create rank-specific pipeline edge
                    edge_obj = simulate_LLM.Edge(
                        name=f"{getattr(child, 'name', '')}_rank{r}",
                        op_id=self._next_op_id(),
                        duration=0,
                        is_all_reduce=False,
                        comm_size_bytes=per_rank_bytes,
                        comm_type="pipeline",
                        participants=2,
                        comm_interconnect_type="lp",
                    )
                    edge_obj.is_cross_layer = True
                    tail.add_child(edge_obj)
                    # Also anchor to the compute node (two parents) for mapping clarity
                    last_compute = rank_heads[r]
                    # Find the nearest compute ancestor for this rank: walk back from tail if needed
                    compute_anchor = None
                    cur = tail
                    visited_ids = set()
                    while cur is not None and id(cur) not in visited_ids:
                        visited_ids.add(id(cur))
                        if isinstance(cur, simulate_LLM.Node):
                            compute_anchor = cur
                            break
                        parents = getattr(cur, "parents", [])
                        cur = parents[-1] if parents else None
                    if compute_anchor is not None and compute_anchor is not edge_obj:
                        compute_anchor.add_child(edge_obj)

                    # Connect to each cloned target, aligning ranks where possible
                    for tgt_clone in target_clones:
                        if isinstance(tgt_clone, (list, tuple)):
                            # Map by identity index when available
                            idx = r % len(tgt_clone)
                            edge_obj.add_child(tgt_clone[idx])
                        else:
                            edge_obj.add_child(tgt_clone)
                continue

            # Default path for non-pipeline children
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(downstream_parents, child_clone)

        heads_tuple = tuple(rank_heads)
        self._clone_cache[node_id] = heads_tuple
        return heads_tuple

    def _create_transformer_comm_edge(
        self,
        comm_key: str,
        hw_id: int,
        stage_id: int,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        direction: str,
        tp_rank: int,
    ) -> simulate_LLM.Edge:
        comm_info = self.transformer_graph.comm_metadata.get(comm_key, {})
        is_all_reduce = comm_info.get("type") == "all_reduce"

        comm_edge = self.transformer_graph.create_comm_edge(
            name=comm_key,
            op_id=self._next_op_id(),
            comm_key=comm_key,
            is_all_reduce=is_all_reduce,
            local_hw_id=hw_id,
        )
        comm_edge.stage_id = stage_id
        comm_edge.micro_batch_index = micro_batch
        comm_edge.layer_index = layer_index
        comm_edge.direction = direction
        comm_edge.tp_rank = tp_rank
        return comm_edge

    def _copy_metadata(self, source: Any, target: Any) -> None:
        for attr in (
            "micro_batch_index",
            "layer_index",
            "direction",
            "stage_id",
            "tp_rank",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _attach(self, parent: Any, child: Any) -> None:
        if parent is None or child is None:
            return

        if isinstance(parent, (list, tuple)):
            for item in parent:
                self._attach(item, child)
            return

        if isinstance(child, (list, tuple)):
            for item in child:
                self._attach(parent, item)
            return

        parent.add_child(child)

    def _format_gemm_name(
        self,
        base_name: str,
        direction: str,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        tp_rank: int,
    ) -> str:
        return f"{base_name}_{direction}_mb{micro_batch}_l{layer_index}_rank{tp_rank}"

    def _next_op_id(self) -> int:
        self._op_id_counter += 1
        return self._op_id_counter

    def _hw_id_for_rank(self, stage_id: int, tp_rank: int) -> int:
        stage_int = int(stage_id) if stage_id is not None else 0
        return stage_int * self._tp_degree + tp_rank

class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode):
# Mode parameter
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )
        self.execution_mode = execution_mode
        self._reduction_total_llm = 0.0
        self.all_reduce = all_reduce # when the reduce happens in data parallelism options: "the end"  "every layer"
        self.pipeline_graph: Optional[Graph] = None
        self.pipeline_root: Optional[Any] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None
        self.transformer_graph: Optional[Graph] = None
        self.transformer_forward_root: Optional[Any] = None
        self.transformer_backward_root: Optional[Any] = None
        self.transformer_analytical_time_forward: Optional[float] = None
        self.transformer_analytical_time_backward: Optional[float] = None
        self.transformer_astrasim_time_forward: Optional[float] = None
        self.transformer_astrasim_time_backward: Optional[float] = None
        self.transformer_astrasim_per_rank_forward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward: Optional[List[float]] = None
        self.pipeline_astrasim_time: Optional[float] = None
        self.pipeline_astrasim_per_rank: Optional[List[float]] = None

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astra":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astra", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode in (
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
            ExecutionMode.FULL_ASTRASIM_FLATTENED,
        ):
            return 'full'
        # Hybrid still uses the analytical pipeline, so keep the hybrid policy.
        return 'hybrid'

    def _distributed_gemm_forward(self, m: int, k: int, n: int, name: str) -> Tuple[float, float]:
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            gemm_time = self.getGEMMTime(m, k, n, name)[0]
            return gemm_time, 0.0
        if self.t == "CR":
            return self.getDistGEMM_f_kp1(m, k, n, self.kp1, name)
        if self.t == "RC":
            return self.getDistGEMM_f_kp2(m, k, n, self.kp1, self.kp2, name)
        raise ValueError(f"Invalid tensor parallel strategy: {self.t}")

    def _distributed_gemm_backward(self, m: int, k: int, n: int, name: str) -> Tuple[float, float]:
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            grad_act_time, _, _, _ = self.getGEMMTime(m, n, k, f"{name}_act")
            grad_wt_time, _, _, _ = self.getGEMMTime(k, m, n, f"{name}_wt")
            return grad_act_time + grad_wt_time, 0.0
        if self.t == "CR":
            return self.getDistGEMM_b_kp1(m, k, n, self.kp1, name)
        if self.t == "RC":
            return self.getDistGEMM_b_kp2(m, k, n, self.kp1, self.kp2, name)
        raise ValueError(f"Invalid tensor parallel strategy: {self.t}")

    def get_node_f(self, gemm, name):

        """Get node Time on Forward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_forward(m, k, n, name)

        if self.debug:
            print(f"{name} GEMM_time: {gemm_time:,}; reduction_time: {reduction_time:,}")

        total = gemm_time + reduction_time
        self._reduction_total_llm += reduction_time
        return total

    def get_node_b(self, gemm, name):
        """Get node Time on Backward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_backward(m, k, n, name)

        total = gemm_time + reduction_time
        self._reduction_total_llm += reduction_time
        return total

    def getEmbedding_f(self):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        return embedding_time + embedding_transfer_time

    def getLinearSoftmax_f(self, gemm):
        """
        Calculates the total computation time for a linear softmax operation using GEMM (General Matrix Multiply) and pointwise operations.
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        
        gemm_time, reduction_time = self._distributed_gemm_forward(m, k, n, "linear_softmax_f")
        point_flop = m * (3 * n - 1)
        point_mem = self.precision * m * (7 * n)
        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Linear Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("point_time: {:,}\n".format(point_time))

        total = gemm_time + reduction_time + point_time
        self._reduction_total_llm += reduction_time
        return total
    def getScaleSoftmax_f(self, gemm):

        m = gemm[0] # get the gemm shape of former layer which is attention_score layer here
        n = gemm[2]
        scale_flop = m * n
        scale_mem = self.precision * scale_flop * 2
        softmax_flop = m * n * 3
        softmax_mem = self.precision * m * n * 7
        scale_time = (
            self.roofline(scale_flop, scale_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        softmax_time = (
            self.roofline(softmax_flop, softmax_mem, name="pointwise-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Scale Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("scale point_time: {:,}\n".format(scale_time))
            print("softmax point_flop: {:,}, softmax point_mem: {:,}".format(
                int(softmax_flop / 1e9), int(softmax_mem / 1e9)
            ))
            print("softmax point_time: {:,}\n".format(softmax_time))

        return scale_time + softmax_time
    def getScaleSoftmax_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        scale_flop = m * n
        scale_mem = self.precision * scale_flop * 2
        softmax_flop = m * n * 5
        softmax_mem = self.precision * m * n * 11
        scale_time = (
            self.roofline(scale_flop, scale_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        softmax_time = (
            self.roofline(softmax_flop, softmax_mem, name="pointwise-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "(gr)Scale Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("(gr)scale point_time: {:,}\n".format(scale_time))
            print("(gr)softmax point_flop: {:,}, softmax point_mem: {:,}".format(
                int(softmax_flop / 1e9), int(softmax_mem / 1e9)
            ))
            print("(gr)softmax point_time: {:,}\n".format(softmax_time))

        return scale_time + softmax_time
    def getResidual_f(self, gemm):
        m = gemm[0] # get the gemm shape of former layer which is output_proj layer here
        n = gemm[2]
        residual_flop = m * n
        residual_mem = self.precision * residual_flop * 3

        residual_time = (
            self.roofline(residual_flop, residual_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
    

        if self.debug:
            print(
                "Residual point_flop: {:,}, point_mem: {:,}".format(
                    int(residual_flop / 1e9), int(residual_mem / 1e9)
                )
            )
            print("Residual point_time: {:,}\n".format(residual_time))
            
        return residual_time
    def getResidual_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        residual_flop = m * n
        residual_mem = self.precision * residual_flop * 3
        residual_time = (
            self.roofline(residual_flop, residual_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        if self.debug:
            print(
                "(gr)Residual point_flop: {:,}, point_mem: {:,}".format(
                    int(residual_flop / 1e9), int(residual_mem / 1e9)
                )
            )
            print("(gr)Residual point_time: {:,}\n".format(residual_time))
            
        return residual_time
    def getLayernorm_f(self, gemm):
        m = gemm[0] # get the gemm shape of former layer 
        n = gemm[2]
        flops = m * n * 5
        mem = self.precision * m * n * 9
        time = (
            self.roofline(flops, mem, name="pointwise-scale-f")
            + 3 * self.O
        )
        if self.debug:
            print(
                "Layernorm point_flop: {:,}, point_mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Layernorm point_time: {:,}\n".format(time))
            
        return time
    def getLayernorm_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        flops = m * n * 7
        mem = self.precision * m * n * 11
        time = (
            self.roofline(flops, mem, name="pointwise-scale-f")
            + 4 * self.O
        )
        if self.debug:
            print(
                "(gr)Layernorm point_flop: {:,}, point_mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("(gr)Layernorm point_time: {:,}\n".format(time))
            
        return time
    def getLinearSoftmax_b(self, gemm):
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_backward(m, k, n, "linear_softmax_b")
        point_flop = m * n * 5
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = self.precision * m * 11
        # 3: grad(A) in pointwise division
        # 3: grad(B) in pointwise division
        # 3: addition in copy backprop
        # 2: sigmoid

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-b")
            + 4 * self.O
        )

        if self.debug:
            print(
                "(gr) Linear Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("(gr) Linear Softmax point_time: {:,}\n".format(point_time))

        total = gemm_time + reduction_time + point_time
        self._reduction_total_llm += reduction_time
        return total
    
    def getEmbedding_b(self):
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name="embedding_b") + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    
    def check_memory(self, hw_config, model_config): #check whether memory usage exceeds capacity
        """Check memory usage."""
        total_mem_capacity = self.memory_capacity # in bytes
        print(f"Total Memory Capacity: {total_mem_capacity / 1e9} GB")
        total_mem = LLM_util.getTotMemReq(hw_config, model_config)[0]
        print(f"Total Memory Usage estimation: {total_mem / 1e9} GB")
        if total_mem > total_mem_capacity:
            print("Warning: Total memory usage exceeds memory capacity!")
            sys.exit("Program terminated due to memory capacity exceeded.")

        return
    def getDPOverhead(self, d, ffn_dim, n_layers): #calculate the reduction time and apply_grad_time for data parallelism 

        reduction_time = 0
        apply_grad_time = 0

        reduction_time += self.getR(Dim0=d, Dim1=3*d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="qkv_proj reduction") #get the reduction time between all nodes in one data parallel group
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj grad")

        reduction_time += self.getR(Dim0=d, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="output_proj reduction")
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj grad")

        reduction_time += 2*self.getR(Dim0=ffn_dim, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="ffn reduction")
        apply_grad_time += 2*self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn grad")
        print(f"reduction_time: {reduction_time}")
        print(f"apply_grad_time: {apply_grad_time}")
        
        if all_reduce == "the end":
            reduction_time *= n_layers
            apply_grad_time *= n_layers
        elif all_reduce == "every layer": #return the reduction time for each layer
            pass
        else:
            sys.exit("Invalid all_reduce option")



        return reduction_time , apply_grad_time
    
    
    def getInterLayerCommLatency_LLM(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.lp > 1:
            w_size = self.precision * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.IBL + self.LLL
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer")
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size
    def getDataParallelReductionSizes(self, d, ffn_dim):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0
            }

        # Calculate sizes only (no timing)
        qkv_size = math.ceil(self.precision * d * 3 * d)
        output_size = math.ceil(self.precision * d * d)
        ffn_size = math.ceil(self.precision * ffn_dim * d)
        total_size = qkv_size + output_size + 2 * ffn_size  # FFN appears twice

        return {
            'qkv_size': qkv_size,
            'output_size': output_size,
            'ffn_size': ffn_size,
            'total_size': total_size
        }

    def getDataParallelLocalComputation(self, d, ffn_dim):
        """Calculate local computation times for apply_grad operations."""
        qkv_local = self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        output_local = self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")
        ffn_local = 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        return {
            'qkv_local': qkv_local,
            'output_local': output_local,
            'ffn_local': ffn_local,
            'total_local': qkv_local + output_local + ffn_local
        }

    def getDataParallelReduction_LLM(self, d, ffn_dim):
        # If no data parallelism, still apply gradients locally but no cross-device reduction
        if not getattr(self, "dp", 1) or self.dp <= 1:
            apply_grad_time = 0.0
            apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
            apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")
            apply_grad_time += 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")
            if self.debug:
                print(f"(dp=1) apply_grad_time: {apply_grad_time}")
            return apply_grad_time

        # k = 2 * self.D
        # n = 4 * self.D
        # dim1 = self.kp_hidden_dim1
        # dim2 = self.kp_hidden_dim2
        w_data = 4*d*d + 2*ffn_dim*d # total parameters need to be reduced
        reduction_time = 0.0
        apply_grad_time = 0.0

        total_bytes = math.ceil(self.precision * d * 3 * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="qkv_proj reduction",
        )
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")

        total_bytes = math.ceil(self.precision * d * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="output_proj reduction",
        )
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")

        total_bytes = math.ceil(self.precision * ffn_dim * d)
        reduction_time += 2 * self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="ffn reduction",
        )
        apply_grad_time += 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        if self.debug:
            print(f"reduction_time: {reduction_time}")
            print(f"apply_grad_time: {apply_grad_time}")

        return reduction_time + apply_grad_time
    
    
    
    def getNodeLatency(self, batch_size, vocab_size, hidden_dim, seq_len, num_heads, ffn_dim):
        """Calculate latency for each node in the computation graph."""

        # Process GEMM shapes
        gemm_3d = LLM_util.process_gemm_shapes( #get gemm shape for each layer in transformer and linear softmax layer
            batch_size, seq_len, hidden_dim, num_heads, ffn_dim, vocab_size, option="multiply_batch_into_m"
        )
        gemm_qkv_proj, gemm_attention_score, gemm_attention_output, gemm_output_proj, gemm_ffn1, gemm_ffn2, gemm_linear = gemm_3d

        # Calculate time for each node
        embedding_f = self.getEmbedding_f()
        embedding_b = self.getEmbedding_b()

        qkv_proj_f, qkv_proj_b = (
            self.get_node_f(gemm=gemm_qkv_proj, name="qkv_projection_f"),
            self.get_node_b(gemm=gemm_qkv_proj, name="qkv_projection_b")
        )
        attention_score_f, attention_score_b = (
            self.get_node_f(gemm=gemm_attention_score, name="attention_score_f"),
            self.get_node_b(gemm=gemm_attention_score, name="attention_score_b")
        )
        attention_scale_softmax_f, attention_scale_softmax_b = (
            self.getScaleSoftmax_f(gemm=gemm_attention_score),
            self.getScaleSoftmax_b(gemm=gemm_attention_score)
        )
        attention_output_f, attention_output_b = (
            self.get_node_f(gemm=gemm_attention_output, name="attention_output_f"),
            self.get_node_b(gemm=gemm_attention_output, name="attention_output_b")
        )
        output_proj_f, output_proj_b = (
            self.get_node_f(gemm=gemm_output_proj, name="output_projection_f"),
            self.get_node_b(gemm=gemm_output_proj, name="output_projection_b")
        )
        residual1_f, residual1_b = (
            self.getResidual_f(gemm=gemm_output_proj),
            self.getResidual_b(gemm=gemm_output_proj)
        )
        layernorm1_f, layernorm1_b = (
            self.getLayernorm_f(gemm=gemm_output_proj),
            self.getLayernorm_b(gemm=gemm_output_proj)
        )
        ffn1_f, ffn1_b = (
            self.get_node_f(gemm=gemm_ffn1, name="ffn_f"),
            self.get_node_b(gemm=gemm_ffn1, name="ffn_b")
        )
        ffn2_f, ffn2_b = (
            self.get_node_f(gemm=gemm_ffn2, name="ffn2_f"),
            self.get_node_b(gemm=gemm_ffn2, name="ffn2_b")
        )
        residual2_f, residual2_b = (
            self.getResidual_f(gemm=gemm_ffn2),
            self.getResidual_b(gemm=gemm_ffn2)
        )
        layernorm2_f, layernorm2_b = (
            self.getLayernorm_f(gemm=gemm_ffn2),
            self.getLayernorm_b(gemm=gemm_ffn2)
        )
        linear_softmax_f, linear_softmax_b = (
            self.getLinearSoftmax_f(gemm=gemm_linear),
            self.getLinearSoftmax_b(gemm=gemm_linear)
        )

        # Calculate MHA and FFN times
        mha_time_f = (
            qkv_proj_f + attention_score_f + attention_output_f + output_proj_f + attention_scale_softmax_f
        )
        ffn_time_f = ffn1_f + ffn2_f
        mha_time_b = (
            qkv_proj_b + attention_score_b + attention_output_b + output_proj_b + attention_scale_softmax_b
        )
        ffn_time_b = ffn1_b + ffn2_b

        # Calculate transformer times
        transformer_time_f = (
            mha_time_f + ffn_time_f + residual1_f + residual2_f + layernorm1_f + layernorm2_f
        )
        transformer_time_b = (
            mha_time_b + ffn_time_b + residual1_b + residual2_b + layernorm1_b + layernorm2_b
        )

        # Write results to file
        output_file = "Transformer_breakdown_results.txt"
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            f.write(f"qkv_proj_f: {qkv_proj_f * m}{second}\n")
            f.write(f"attention_score_f: {attention_score_f * m}{second}\n")
            f.write(f"attention_scale_softmax_f: {attention_scale_softmax_f * m}{second}\n")
            f.write(f"attention_output_f: {attention_output_f * m}{second}\n")
            f.write(f"output_proj_f: {output_proj_f * m}{second}\n")
            f.write(f"residual1_f: {residual1_f * m}{second}\n")
            f.write(f"layernorm1_f: {layernorm1_f * m}{second}\n")
            f.write(f"ffn1_f: {ffn1_f * m}{second}\n")
            f.write(f"ffn2_f: {ffn2_f * m}{second}\n")
            f.write(f"residual2_f: {residual2_f * m}{second}\n")
            f.write(f"layernorm2_f: {layernorm2_f * m}{second}\n")
            f.write(f"linear_softmax_f: {linear_softmax_f * m}{second}\n")
            f.write(f"MHA Time: {mha_time_f * m}{second}\n")
            f.write(f"FFN Time: {ffn_time_f * m}{second}\n")
            f.write(f"Transformer Time (1 layer): {transformer_time_f * m}{second}\n")

        # Debugging output
        if debug:
            print(f"embedding_f: {embedding_f * m:.1f}{second}")
            print(f"qkv_proj_f: {qkv_proj_f * m:.1f}{second}")
            print(f"attention_score_f: {attention_score_f * m:.1f}{second}")
            print(f"attention_output_f: {attention_output_f * m:.1f}{second}")
            print(f"output_proj_f: {output_proj_f * m:.1f}{second}")
            print(f"residual1_f: {residual1_f * m:.1f}{second}")
            print(f"layernorm1_f: {layernorm1_f * m:.1f}{second}")
            print(f"ffn1_f: {ffn1_f * m:.1f}{second}")
            print(f"ffn2_f: {ffn2_f * m:.1f}{second}")
            print(f"residual2_f: {residual2_f * m:.1f}{second}")
            print(f"layernorm2_f: {layernorm2_f * m:.1f}{second}")
            print(f"linear_softmax_f: {linear_softmax_f * m:.1f}{second}")
            print(f"MHA Time: {mha_time_f * m:.1f}{second}")
            print(f"FFN Time: {ffn_time_f * m:.1f}{second}")
            print(f"Transformer Time (1 layer): {transformer_time_f * m:.1f}{second}")

        return (
            transformer_time_f, transformer_time_b, embedding_f, embedding_b,
            linear_softmax_f, linear_softmax_b
        )


    def _effective_transformer_batch(self) -> int:
        if self.lp > 1:
            return self.microB
        if self.dp > 1:
            return self.miniB
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes: Dict[str, int],
        local_comp: Dict[str, float],
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': 'pipeline',
                'participants': 2,
                'interconnect_type': 'lp',
                'local_comp_time': 0
            }
        }

    def _populate_transformer_comm_metadata(
        self,
        entry: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
        m: int,
        k: int,
        n: int,
    ) -> None:
        """Attach tensor-parallel collectives for a GEMM to metadata and entry."""

        tp_mode = self.t
        kp1 = int(self.kp1) if self.kp1 else 1
        kp2 = int(self.kp2) if self.kp2 else 1
        precision = self.precision

        if not tp_mode or (kp1 <= 1 and kp2 <= 1):
            return

        def add_comm(direction: str, suffix: str, kind: str, size_bytes: float, participants: int, interconnect: str) -> None:
            if participants <= 1:
                return
            bytes_int = int(math.ceil(size_bytes))
            if bytes_int <= 0:
                return
            key = f"{entry['name']}_{direction}_{suffix}"
            # ensure uniqueness when multiple collectives share the same suffix
            unique_key = key
            counter = 1
            while unique_key in metadata:
                counter += 1
                unique_key = f"{key}_{counter}"

            metadata[unique_key] = {
                'size': bytes_int,
                'type': kind,
                'participants': int(participants),
                'interconnect_type': interconnect,
            }
            entry[direction]['comm_keys'].append(unique_key)

        if tp_mode == "CR":
            if kp1 <= 1:
                return
            total_bytes = math.ceil(precision * m * n)
            add_comm('forward', 'reduce_scatter', 'reduce_scatter', total_bytes, kp1, 'kp1')

            # Backward all-gather mirrors forward reduce-scatter.
            total_bytes = math.ceil(precision * m * n)
            size_bytes = math.ceil(total_bytes / kp1)
            add_comm('backward', 'all_gather', 'all_gather', size_bytes, kp1, 'kp1')
            return

        if tp_mode == "RC":
            if kp2 > 1:
                total_bytes = math.ceil(precision * (m // max(1, kp1)) * n)
                size_bytes = math.ceil(total_bytes / kp2)
                add_comm('forward', 'all_gather', 'all_gather', size_bytes, kp2, 'kp2')

            if kp1 > 1:
                total_bytes_row = math.ceil(precision * k * m)
                size_row = math.ceil(total_bytes_row / kp1)

                total_bytes_col = math.ceil(precision * m * (n / max(1, kp2)))
                size_col = math.ceil(total_bytes_col / kp1)

                add_comm(
                    'backward',
                    'wt_all_gather',
                    'all_gather',
                    size_row + size_col,
                    kp1,
                    'kp1',
                )

            if kp2 > 1:
                total_bytes_row = math.ceil(precision * (m / max(1, kp1)) * n)
                size_row = math.ceil(total_bytes_row / kp2)

                total_bytes_col = math.ceil(precision * k * n)
                size_col = math.ceil(total_bytes_col / kp2)

                add_comm(
                    'backward',
                    'act_all_gather',
                    'all_gather',
                    size_row + size_col,
                    kp2,
                    'kp2',
                )

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'kp1': (self.IBK1, self.LLK1),
            'kp2': (self.IBK2, self.LLK2)
        }

    def _build_pipeline_graph(
        self,
        num_micro_batches: int,
        num_layers: int,
        transformer_time_f: float,
        transformer_time_b: float,
        embedding_f: float,
        embedding_b: float,
        linear_softmax_f: float,
        linear_softmax_b: float,
        comm_metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[Graph, Any, Dict[str, Tuple[float, float]]]:
        comp_times = {
            "embedding_f": embedding_f,
            "embedding_b": embedding_b,
            "linear_softmax_f": linear_softmax_f,
            "linear_softmax_b": linear_softmax_b,
            "transformer_f": transformer_time_f,
            "transformer_b": transformer_time_b,
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        misc_metadata = {
            "num_batch": num_micro_batches,
            "num_layer": num_layers,
            "all_reduce": self.all_reduce,
        }

        graph = simulate_LLM.Graph(
            mode="pipeline",
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1,
            kp2=self.kp2,
            tp_mode=self.t,
            comp_times=comp_times,
            comm_metadata=comm_metadata,
            misc_metadata=misc_metadata,
        )
        root = graph.construct_fwd_bwd_graph()
        interconnect_params = self._build_interconnect_params()
        return graph, root, interconnect_params

    def _tp_degree(self) -> int:
        kp1 = self.kp1 if self.kp1 else 1
        kp2 = self.kp2 if self.kp2 else 1
        return int(kp1 * kp2)
    
    def calcTime_LLM(self):
        """Calculate time for LLM model."""
        # Extract model parameters
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        num_layers = self.num_layers
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim
        num_micro_batches = self.mb

        # Adjust types and calculate node latencies
        self.readjust_type()
        transformer_time_f, transformer_time_b, embedding_f, embedding_b, linear_softmax_f, linear_softmax_b = self.getNodeLatency(batch_size, vocab_size, hidden_dim, seq_len, num_heads, ffn_dim)

        if self.debug:
            print(
                "dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}, kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding_dim1: {}, kp_embedding_dim2: {},  kp_hidden_type: {}, kp_softmax_type: {}, kp_embedding_type: {}\n".format(
                    self.dp, self.lp, self.kp_hidden_dim1, self.kp_hidden_dim2,
                    self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1,
                    self.kp_embedding_dim2, self.kp_hidden_type, self.kp_softmax_type,
                    self.kp_embedding_type,
                )
            )


            # print(
            #     "embedding_f: {}, embedding_b: {}, qkv_proj_f: {}, qkv_proj_b: {}, attention_score_f: {}, attention_score_b: {}, attention_scale_softmax_f: {}, attention_scale_softmax_b: {}, attention_output_f: {}, attention_output_b: {}, output_proj_f: {}, output_proj_b: {}, residual1_f: {}, residual1_b: {}, layernorm1_f: {}, layernorm1_b: {}, ffn1_f: {}, ffn1_b: {}, ffn2_f: {}, ffn2_b: {}".format(
            #     embedding_f, embedding_b,
                
            #     qkv_proj_f, qkv_proj_b, attention_score_f, attention_score_b,
            #     attention_scale_softmax_f, attention_scale_softmax_b,
            #     attention_output_f, attention_output_b, output_proj_f, output_proj_b,
            #     residual1_f, residual1_b, layernorm1_f, layernorm1_b,
            #     ffn1_f, ffn1_b, ffn2_f, ffn2_b
            # ))
        print("Calculating LLM time...")

        # Create graph and calculate times

        print("simulating parallelism with dp = {}, lp = {}, total data batch = {}, "
            "for each dp node, data batch = {}, for each pipeline stage, data batch = {}".format(self.dp, self.lp, self.batch_size, self.miniB, self.microB))
        print("total number of workers: {}".format(self.num_workers))
        print("number of workers for each data parallelism batch: {}".format(self.num_workers_dp))
        print("number of workers for each pipeline stage: {}".format(self.num_workers_lp))

        reduction_sizes = self.getDataParallelReductionSizes(hidden_dim, ffn_dim)
        local_comp = self.getDataParallelLocalComputation(hidden_dim, ffn_dim)
        embedding_size = math.ceil(self.precision * vocab_size * hidden_dim) + math.ceil(self.precision * seq_len * hidden_dim)
        softmax_size = math.ceil(self.precision * hidden_dim * vocab_size)
        cross_layer_bytes = self.getInterLayerCommLatency_LLM(batch_size, hidden_dim, seq_len)[1]

        comm_metadata = self._build_comm_metadata(
            reduction_sizes=reduction_sizes,
            local_comp=local_comp,
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
        )

        shapes = LLM_util.process_gemm_shapes(
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            ffn_dim,
            vocab_size,
            option="multiply_batch_into_m",
        )
        (
            gemm_qkv_proj,
            gemm_attention_score,
            gemm_attention_output,
            gemm_output_proj,
            gemm_ffn1,
            gemm_ffn2,
            _,
        ) = shapes

        gemm_specs = [
            ("gemm_qkv_proj", gemm_qkv_proj),
            ("gemm_attn_score", gemm_attention_score),
            ("gemm_attn_output", gemm_attention_output),
            ("gemm_output_proj", gemm_output_proj),
            ("gemm_ffn1", gemm_ffn1),
            ("gemm_ffn2", gemm_ffn2),
        ]

        transformer_gemm_entries = []
        transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}

        for base_name, (m, k, n) in gemm_specs:
            fwd_time, fwd_red = self._distributed_gemm_forward(m, k, n, base_name + "_fwd")
            bwd_time, bwd_red = self._distributed_gemm_backward(m, k, n, base_name + "_bwd")

            entry = {
                "name": base_name,
                "forward": {
                    "duration": fwd_time,
                    "reduction": fwd_red,
                    "comm_keys": [],
                },
                "backward": {
                    "duration": bwd_time,
                    "reduction": bwd_red,
                    "comm_keys": [],
                },
            }

            self._populate_transformer_comm_metadata(
                entry=entry,
                metadata=transformer_comm_metadata,
                m=m,
                k=k,
                n=n,
            )

            transformer_gemm_entries.append(entry)

        tp_degree = self._tp_degree()

        transformer_comp_times = {
            "transformer": {
                "gemms": transformer_gemm_entries,
                "tp_degree": tp_degree,
            }
        }
        self.transformer_graph = simulate_LLM.Graph(
            mode="transformer",
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1,
            kp2=self.kp2,
            tp_mode=self.t,
            comp_times=transformer_comp_times,
            comm_metadata=transformer_comm_metadata,
            misc_metadata={},
        )
        self.transformer_forward_root = self.transformer_graph.construct_transformer_graph(direction="forward")
        self.transformer_backward_root = self.transformer_graph.construct_transformer_graph(direction="backward")

        analytical_forward = 0.0
        analytical_backward = 0.0
        for entry in transformer_gemm_entries:
            forward_component = entry["forward"]["duration"] + entry["forward"].get("reduction", 0.0)
            backward_component = entry["backward"]["duration"] + entry["backward"].get("reduction", 0.0)
            analytical_forward += forward_component
            analytical_backward += backward_component
        self.transformer_analytical_time_forward = analytical_forward
        self.transformer_analytical_time_backward = analytical_backward

        pipeline_graph_obj, graph_root, interconnect_params = self._build_pipeline_graph(
            num_micro_batches=num_micro_batches,
            num_layers=num_layers,
            transformer_time_f=transformer_time_f,
            transformer_time_b=transformer_time_b,
            embedding_f=embedding_f,
            embedding_b=embedding_b,
            linear_softmax_f=linear_softmax_f,
            linear_softmax_b=linear_softmax_b,
            comm_metadata=comm_metadata,
        )

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params

        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
        )
        mode = self.execution_mode
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc

        time_fw_bw = result.total_time

        pipeline_root = result.graph_root
        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = dispatcher.interconnect_params

        # self.pipeline_graph.save_graph(pipeline_root, "output_graph/", "fw_bw_graph")

        if self.transformer_analytical_time_forward is not None:
            print(
                f"Analytical transformer forward time: "
                f"{self.transformer_analytical_time_forward:.1f}s"
            )
        if self.transformer_analytical_time_backward is not None:
            print(
                f"Analytical transformer backward time: "
                f"{self.transformer_analytical_time_backward:.1f}s"
            )

        if self.transformer_forward_root is not None:
            self.transformer_graph.save_graph(
                self.transformer_forward_root,
                "output_graph/",
                "transformer_graph_forward",
            )
        if self.transformer_backward_root is not None:
            self.transformer_graph.save_graph(
                self.transformer_backward_root,
                "output_graph/",
                "transformer_graph_backward",
            )

        self.tot_time = time_fw_bw

        output_file = "LLM_time_results.txt"
        
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            # f.write("Forward Time: {0:.8f} {1}\n".format(time_fw * m, second))
            # f.write("Backward Time: {0:.8f} {1}\n".format(time_bw * m, second))
            f.write("Forward + Backward Time: {0:.8f} {1}\n".format(time_fw_bw * m, second))

            # f.write("Total Time: {0:.8f}\n".format(TC.getTime()))

        return time_fw_bw

    def getTime(self):
        return self.tot_time

    def getReductionTotal(self):
        return getattr(self, "_reduction_total_llm", 0.0)


class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,
        pipeline_graph: Graph,
        pipeline_root: Any,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_graph: Optional[Graph] = None,
        transformer_forward_root: Optional[Any] = None,
        transformer_backward_root: Optional[Any] = None,
    ) -> None:
        self.time_calc = time_calc
        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.interconnect_params = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.flattened_root: Optional[Any] = None

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            return self._run_full_astrasim_flattened()
        raise ValueError(f"Unsupported execution mode: {mode}")

    def _run_pipeline_with_analytical_comm(self, declared_mode: ExecutionMode) -> ExecutionResult:
        timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
            self.pipeline_root,
            self.time_calc.network_model,
            self.interconnect_params,
        )
        # Persist timed root for any downstream consumer
        self.pipeline_root = timed_root
        total_time = self.pipeline_graph.simulate(timed_root)
        return ExecutionResult(total_time=total_time, graph_root=timed_root, mode=declared_mode)

    def _run_hybrid(self) -> ExecutionResult:
        transformer_time = self._run_transformer_astrasim(ExecutionMode.HYBRID)
        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)
        return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID)

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        transformer_time = self._run_transformer_astrasim(ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)
        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)

        dp_count = getattr(self.time_calc, "dp", 1) or 1
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for AstraSim execution")

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(self.pipeline_root, self.time_calc, "./astra_pipeline_output")
        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        return ExecutionResult(total_time=max_sec, graph_root=self.pipeline_root, mode=ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for flattening")
        if not self.transformer_graph:
            raise RuntimeError("Transformer graph metadata is required for flattening")

        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_unflattened")
        # dot = visualize_graph(self.pipeline_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="png", cleanup=True)
        # except Exception as exc:
        #     print(f"[WARN] Failed to render pipeline graph: {exc}")

        flattener = PipelineGraphFlattener(
            pipeline_graph=self.pipeline_graph,
            transformer_graph=self.transformer_graph,
        )
        flattened_root = flattener.build(self.pipeline_root)
        if flattened_root is None:
            raise RuntimeError("Pipeline flattening produced an empty graph")

        setattr(self.time_calc, "flattened_pipeline_root", flattened_root)
        self.pipeline_root = flattened_root

        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_flattened")
        # dot = visualize_graph(flattened_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="png", cleanup=True)
        # except Exception as exc:  # pragma: no cover - visualization best-effort
        #     print(f"[WARN] Failed to render flattened pipeline graph: {exc}")

        unique_hw_ids = self._collect_hw_ids(flattened_root)
        if not unique_hw_ids:
            raise RuntimeError("Flattened pipeline graph exposes no compute nodes with hardware IDs")

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            flattened_root,
            self.time_calc,
            "./astra_pipeline_output_flat",
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings")

        dp_count = max(1, getattr(self.time_calc, "dp", 1))
        expected_rank_count = dp_count * len(unique_hw_ids)
        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration")

        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        setattr(self.time_calc, "flattened_astrasim_per_rank", per_rank_sec)
        setattr(self.time_calc, "flattened_astrasim_total", max_sec)

        return ExecutionResult(
            total_time=max_sec,
            graph_root=flattened_root,
            mode=ExecutionMode.FULL_ASTRASIM_FLATTENED,
        )

    def _collect_hw_ids(self, root: Any) -> Set[int]:
        visited: Set[int] = set()
        hw_ids: Set[int] = set()

        def enqueue_children(obj: Any) -> None:
            for child in getattr(obj, "children", []):
                stack.append(child)

        stack: List[Any]
        if isinstance(root, (list, tuple)):
            stack = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if isinstance(obj, simulate_LLM.Node):
                hw_id = getattr(obj, "hw_id", None)
                if hw_id is not None and hw_id >= 0:
                    hw_ids.add(int(hw_id))

            enqueue_children(obj)

        return hw_ids

    def _run_transformer_astrasim(self, mode: ExecutionMode) -> Optional[TransformerTimings]:
        del mode  # mode currently unused but kept for signature consistency
        if not self.transformer_forward_root or not self.transformer_backward_root:
            return None

        fwd_per_rank, fwd_max = run_astra_simulation_only_onepath(
            self.transformer_forward_root,
            self.time_calc,
            "./astra_transformer_output_forward",
            dp_override=1,
        )
        bwd_per_rank, bwd_max = run_astra_simulation_only_onepath(
            self.transformer_backward_root,
            self.time_calc,
            "./astra_transformer_output_backward",
            dp_override=1,
        )

        self.time_calc.transformer_astrasim_per_rank_forward = fwd_per_rank
        self.time_calc.transformer_astrasim_per_rank_backward = bwd_per_rank
        self.time_calc.transformer_astrasim_time_forward = fwd_max
        self.time_calc.transformer_astrasim_time_backward = bwd_max

        if fwd_max <= 0 or bwd_max <= 0:
            raise RuntimeError("AstraSim transformer execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max)

    def _apply_transformer_time(self, timings: TransformerTimings) -> None:
        if timings.forward <= 0 or timings.backward <= 0:
            raise ValueError("AstraSim transformer times must be positive")

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if "transformer_f" in comp_times:
                comp_times["transformer_f"] = timings.forward
            if "transformer_b" in comp_times:
                comp_times["transformer_b"] = timings.backward

        visited: Set[int] = set()
        roots: List[Any]
        if isinstance(self.pipeline_root, (list, tuple)):
            roots = list(self.pipeline_root)
        else:
            roots = [self.pipeline_root]

        for root in roots:
            self._assign_transformer_durations(root, visited, timings.forward, timings.backward)

    def _assign_transformer_durations(self, node: Any, visited: Set[int], forward_value: float, backward_value: float) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, simulate_LLM.Node):
            if node.name == "transformer":
                node.duration = forward_value
            elif node.name == "transformer_b":
                node.duration = backward_value

        for child in getattr(node, "children", []):
            self._assign_transformer_durations(child, visited, forward_value, backward_value)
