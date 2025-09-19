"""
AstraSim vs DeepFlow comparison functionality.

This module converts DeepFlow graphs (with communication sizes) to AstraSim Chakra ET format
and executes AstraSim simulation for comparison with DeepFlow analytical timing.

Non-mainlined test functionality - designed to be easily removable.
"""

import os
import json
import sys
import time
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Iterable, Set, Optional


# Increase recursion limit for deep transformer graphs (96+ layers)
sys.setrecursionlimit(10000)

# Import DeepFlow components
from astrasim_integration import (
    _new_comp_node, _new_comm_node, write_et_node,
    run_astrasim_analytical, generate_astrasim_configs_from_hw,
    get_remote_memory_path
)
from simulate_LLM import visualize_graph


# Chakra ET dependencies
BASE_DIR = os.path.dirname(__file__)
CHAKRA_PB_DIR = os.path.join(BASE_DIR, 'astra-sim', 'extern', 'graph_frontend', 'chakra', 'schema', 'protobuf')
CHAKRA_UTILS_DIR = os.path.join(BASE_DIR, 'astra-sim', 'extern', 'graph_frontend', 'chakra', 'src', 'third_party', 'utils')
sys.path.insert(0, CHAKRA_PB_DIR)
sys.path.insert(0, CHAKRA_UTILS_DIR)
import et_def_pb2 as pb
from protolib import encodeMessage as chakra_encode, decodeMessage as chakra_decode, openFileRd as chakra_open
from graphviz import Digraph


class _RankTrace:
    """Helper to build AstraSim ET for a single hardware rank."""

    def __init__(self, hw_id: int, rank: int, path: str) -> None:
        self.hw_id = hw_id
        self.rank = rank
        self.path = path
        self.nodes: List[pb.Node] = []

    @property
    def next_id(self) -> int:
        return len(self.nodes)

    def close(self) -> None:
        with open(self.path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            for node in self.nodes:
                write_et_node(fh, node)


def _new_send_node(node_id: int, name: str, size_bytes: int, dst_rank: int, tag: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMM_SEND_NODE
    node.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    node.attr.append(pb.AttributeProto(name="comm_dst", int32_val=int(dst_rank)))
    node.attr.append(pb.AttributeProto(name="comm_tag", int32_val=int(tag)))
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return node


def _new_recv_node(node_id: int, name: str, size_bytes: int, src_rank: int, tag: int) -> pb.Node:
    node = pb.Node()
    node.id = node_id
    node.name = name
    node.type = pb.COMM_RECV_NODE
    node.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    node.attr.append(pb.AttributeProto(name="comm_src", int32_val=int(src_rank)))
    node.attr.append(pb.AttributeProto(name="comm_tag", int32_val=int(tag)))
    node.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return node


def get_collective_type(comm_type_str: str) -> int:
    """Map DeepFlow comm types to AstraSim protobuf enums."""
    mapping = {
        "all_reduce": pb.ALL_REDUCE,
        "all_gather": pb.ALL_GATHER,
        "reduce_scatter": pb.REDUCE_SCATTER,
        "all_to_all": pb.ALL_TO_ALL
    }
    return mapping.get(comm_type_str, pb.ALL_REDUCE)


def _attr_to_dict(node: pb.Node) -> Dict[str, Any]:
    attr_map: Dict[str, Any] = {}
    for attr in node.attr:
        name = attr.name or f"attr_{len(attr_map)}"
        which = attr.WhichOneof("value")
        if not which:
            continue
        raw = getattr(attr, which)
        if hasattr(raw, "values"):
            attr_map[name] = list(raw.values)
        else:
            attr_map[name] = raw
    return attr_map


def _visualize_et_files(et_paths: List[str]) -> None:
    if not et_paths:
        return

    for et_path in et_paths:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for visualization: {exc}")
            continue

        meta = pb.GlobalMetadata()
        if not chakra_decode(fh, meta):
            print(f"[WARN] {et_path} does not contain GlobalMetadata; skipping graph output")
            fh.close()
            continue

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        dot = Digraph(comment=os.path.basename(et_path))
        dot.graph_attr.update({"rankdir": "TB", "fontsize": "10"})

        id_to_node = {int(node.id): node for node in nodes}
        type_color = {
            pb.COMP_NODE: "lightblue",
            pb.COMM_COLL_NODE: "palegreen",
            pb.COMM_SEND_NODE: "khaki",
            pb.COMM_RECV_NODE: "lightsalmon",
        }

        for node in nodes:
            attr_map = _attr_to_dict(node)
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            label_lines = [node.name or f"node_{node.id}", f"id={node.id}", node_type]
            if node.duration_micros:
                label_lines.append(f"dur={node.duration_micros}us")
            if "comm_type" in attr_map:
                try:
                    comm_name = pb.CollectiveCommType.Name(int(attr_map["comm_type"]))
                except ValueError:
                    comm_name = str(attr_map["comm_type"])
                label_lines.append(f"comm={comm_name}")
            if "comm_dst" in attr_map:
                label_lines.append(f"dst={attr_map['comm_dst']}")
            if "comm_src" in attr_map:
                label_lines.append(f"src={attr_map['comm_src']}")
            if "comm_tag" in attr_map:
                label_lines.append(f"tag={attr_map['comm_tag']}")
            if "comm_size" in attr_map:
                label_lines.append(f"bytes={attr_map['comm_size']}")

            color = type_color.get(node.type, "white")
            node_id = str(node.id)
            dot.node(node_id, label="\n".join(label_lines), style="filled", fillcolor=color, shape="box")

        for node in nodes:
            for dep in node.ctrl_deps:
                if dep in id_to_node:
                    dot.edge(str(dep), str(node.id))

        dir_name, base_name = os.path.split(et_path)
        viz_base = base_name + ".viz"
        try:
            output_path = dot.render(viz_base, directory=dir_name or None, format="png", cleanup=True)
            final_png = et_path + ".png"
            if output_path and output_path != final_png:
                try:
                    os.replace(output_path, final_png)
                except FileNotFoundError:
                    # Some graphviz versions return path without creating file when empty graph
                    pass
            print(f"[AstraSim] Saved ET graph visualization to {final_png}")
        except Exception as exc:
            print(f"[WARN] Failed to render Graphviz graph for {et_path}: {exc}")



def _dump_et_text(et_paths: List[str]) -> None:
    if not et_paths:
        return

    for et_path in et_paths:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for text dump: {exc}")
            continue

        meta = pb.GlobalMetadata()
        has_meta = chakra_decode(fh, meta)

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        id_to_node = {int(node.id): node for node in nodes}

        lines: List[str] = []
        lines.append(f"ET file: {os.path.basename(et_path)}")
        if has_meta:
            lines.append(f"GlobalMetadata: version={meta.version}")
        lines.append(f"Nodes: {len(nodes)}")
        lines.append("")

        def fmt_attr_map(node: pb.Node) -> str:
            items: List[str] = []
            for attr in node.attr:
                name = attr.name or "unnamed"
                which = attr.WhichOneof("value")
                if not which:
                    continue
                raw = getattr(attr, which)
                if hasattr(raw, "values"):
                    val = list(raw.values)
                else:
                    val = raw
                items.append(f"{name}={val}")
            return ", ".join(items)

        lines.append("Nodes detail:")
        for node in nodes:
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            attr_str = fmt_attr_map(node)
            if node.type == pb.COMP_NODE:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} dur_us={node.duration_micros} attrs=[{attr_str}]"
                )
            else:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} attrs=[{attr_str}]"
                )
        lines.append("")

        lines.append("Edges (ctrl_deps):")
        for node in nodes:
            if not node.ctrl_deps:
                continue
            deps_str = ", ".join(str(int(d)) for d in node.ctrl_deps if int(d) in id_to_node)
            lines.append(f"- {node.id} <- [{deps_str}]")

        out_path = et_path + ".txt"
        try:
            with open(out_path, "w") as outf:
                outf.write("\n".join(lines) + "\n")
            print(f"[AstraSim] Saved ET text dump to {out_path}")
        except OSError as exc:
            print(f"[WARN] Failed to write ET text dump for {et_path}: {exc}")


def _write_comm_groups_json(base_output_dir: str, dp_count: int, rank_ids: List[int]) -> Optional[str]:
    try:
        dp = int(dp_count)
    except Exception:
        dp = 1
    if dp <= 1:
        return None
    if not rank_ids:
        return None
    rank_ids = sorted(rank_ids)
    total_ranks = len(rank_ids)
    if total_ranks % dp != 0:
        raise ValueError(f"Cannot partition {total_ranks} ranks into {dp} stages evenly")
    # dp-major mapping: ranks are ordered by dp first, then stage
    num_stages = total_ranks // dp
    groups = {}
    for stage_idx in range(num_stages):
        group = []
        for dp_idx in range(dp):
            rank = dp_idx * num_stages + stage_idx
            group.append(rank)
        # AstraSim requires communicator group IDs > 0
        groups[str(stage_idx + 1)] = group
    os.makedirs(base_output_dir, exist_ok=True)
    path = os.path.join(base_output_dir, "comm_groups.json")
    try:
        with open(path, "w") as f:
            json.dump(groups, f, indent=2)
        print(f"[AstraSim] Wrote communicator groups to {path}")
        return path
    except OSError as exc:
        print(f"[WARN] Failed to write comm_groups.json: {exc}")
        return None

def convert_deepflow_graph_to_chakra_et(
    graph_root,
    dp_size: int,
    output_dir: str,
) -> Tuple[str, List[int]]:
    """Convert DeepFlow graph to AstraSim ET format by scheduling per stage and DP rank."""

    from collections import deque
    import itertools

    os.makedirs(output_dir, exist_ok=True)

    def collect_objects(root) -> List[Any]:
        visited: Set[int] = set()
        ordered: List[Any] = []

        def dfs(obj: Any) -> None:
            if id(obj) in visited:
                return
            visited.add(id(obj))
            ordered.append(obj)
            for child in getattr(obj, "children", []):
                dfs(child)

        if isinstance(root, (list, tuple)):
            for item in root:
                dfs(item)
        else:
            dfs(root)
        return ordered

    all_objects = collect_objects(graph_root)
    compute_nodes = [obj for obj in all_objects if getattr(obj, "hw_id", None) is not None and obj.hw_id >= 0]
    if not compute_nodes:
        raise ValueError("DeepFlow graph did not expose any executable compute nodes (hw_id >= 0).")

    stage_ids = sorted({node.hw_id for node in compute_nodes})
    dp_count = max(int(dp_size) if dp_size else 1, 1)

    stage_index = {stage: idx for idx, stage in enumerate(stage_ids)}
    stage_to_ranks: Dict[int, List[int]] = {stage: [] for stage in stage_ids}
    rank_meta: Dict[int, Dict[str, int]] = {}
    rank_traces: Dict[int, _RankTrace] = {}
    num_stages = len(stage_ids)

    # dp-major mapping: rank = dp_idx * num_stages + stage_index
    for dp_idx in range(dp_count):
        for stage in stage_ids:
            rank = dp_idx * num_stages + stage_index[stage]
            rank_meta[rank] = {"stage": stage, "dp": dp_idx}
            path = f"{output_dir}/llm_graph.{rank}.et"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rank_traces[rank] = _RankTrace(stage, rank, path)
            stage_to_ranks[stage].append(rank)

    def rank_for(stage: int, dp_idx: int) -> int:
        return stage_to_ranks[stage][dp_idx]

    collectives_by_parent: Dict[Any, List[Any]] = defaultdict(list)
    edge_stage: Dict[Any, int] = {}
    for obj in all_objects:
        comm = getattr(obj, "comm_type", None)
        if comm and comm != "pipeline":
            stage = None
            parent = None
            for candidate in getattr(obj, "parents", []):
                if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                    stage = candidate.hw_id
                    parent = candidate
                    break
            if stage is None:
                for candidate in getattr(obj, "children", []):
                    if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                        stage = candidate.hw_id
                        break
            if stage is None:
                continue
            edge_stage[obj] = stage
            if parent is not None:
                collectives_by_parent[parent].append(obj)

    pipeline_edge_map: Dict[Tuple[Any, Any], Any] = {}

    def analyze_for_compute(node: Any) -> Tuple[Set[Any], Set[Any], Set[Any]]:
        stage = node.hw_id
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        collective_deps: Set[Any] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(node, "parents", [])]

        while stack:
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            if key in visited:
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if hw is not None and hw >= 0:
                if hw == stage:
                    if not via_collective:
                        stage_deps.add(cur)
                else:
                    pipeline_deps.add(cur)
                    pipeline_edge_map.setdefault((cur, node), None)
                continue

            comm = getattr(cur, "comm_type", None)
            if comm:
                if comm == "pipeline":
                    srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                    if srcs:
                        for src in srcs:
                            if src.hw_id == stage:
                                if not via_collective:
                                    stage_deps.add(src)
                            else:
                                pipeline_deps.add(src)
                                pipeline_edge_map[(src, node)] = cur
                        continue
                else:
                    collective_deps.add(cur)
                    for parent in getattr(cur, "parents", []):
                        stack.append((parent, True))
                    continue

            for parent in getattr(cur, "parents", []):
                stack.append((parent, via_collective))

        stage_deps.discard(node)
        return stage_deps, pipeline_deps, collective_deps

    def analyze_for_collective(edge: Any) -> Tuple[Set[Any], Set[Any]]:
        stage = edge_stage[edge]
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(edge, "parents", [])]

        while stack:
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            if key in visited:
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if hw is not None and hw >= 0:
                if hw == stage:
                    if not via_collective:
                        stage_deps.add(cur)
                else:
                    pipeline_deps.add(cur)
                    pipeline_edge_map[(cur, edge)] = cur
                continue

            comm = getattr(cur, "comm_type", None)
            if comm == "pipeline":
                srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                if srcs:
                    for src in srcs:
                        if src.hw_id == stage:
                            if not via_collective:
                                stage_deps.add(src)
                        else:
                            pipeline_deps.add(src)
                            pipeline_edge_map[(src, edge)] = cur
                    continue
            elif comm:
                for parent in getattr(cur, "parents", []):
                    stack.append((parent, True))
                continue

            for parent in getattr(cur, "parents", []):
                stack.append((parent, via_collective))

        return stage_deps, pipeline_deps

    compute_info: Dict[Any, Dict[str, Any]] = {}
    for node in compute_nodes:
        stage_deps, pipeline_deps, collective_deps = analyze_for_compute(node)
        compute_info[node] = {
            "stage": node.hw_id,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "collective_deps": collective_deps,
            "name": node.name,
        }

    collective_info: Dict[Any, Dict[str, Any]] = {}
    for edge, stage in edge_stage.items():
        stage_deps, pipeline_deps = analyze_for_collective(edge)
        collective_info[edge] = {
            "stage": stage,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "size": int(getattr(edge, "comm_size_bytes", 0)),
            "comm_type": get_collective_type(edge.comm_type),
            "name": edge.name,
        }

    stage_tasks: Dict[int, Set[Any]] = {stage: set() for stage in stage_ids}
    for node in compute_nodes:
        stage_tasks[node.hw_id].add(node)
    for edge, info in collective_info.items():
        stage = info["stage"]
        if stage not in stage_tasks:
            stage_tasks[stage] = set()
            # Ensure mapping present even for stages discovered only via collectives
            stage_idx = stage_index.setdefault(stage, len(stage_index))
            stage_to_ranks[stage] = [dp_idx * num_stages + stage_idx for dp_idx in range(dp_count)]
        stage_tasks[stage].add(edge)

    stage_adj: Dict[int, Dict[Any, Set[Any]]] = {stage: defaultdict(set) for stage in stage_tasks}
    stage_indegree: Dict[int, Dict[Any, int]] = {stage: defaultdict(int) for stage in stage_tasks}

    for stage, tasks in stage_tasks.items():
        for task in tasks:
            stage_indegree[stage].setdefault(task, 0)

    for node, info in compute_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(node)
                stage_indegree[stage][node] += 1
        for edge in info["collective_deps"]:
            stage_edge = collective_info.get(edge, {}).get("stage")
            if stage_edge == stage:
                stage_adj[stage][edge].add(node)
                stage_indegree[stage][node] += 1

    for edge, info in collective_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(edge)
                stage_indegree[stage][edge] += 1


    stage_order: Dict[int, List[Any]] = {}
    for stage, tasks in stage_tasks.items():
        indeg = stage_indegree[stage]
        queue = deque(task for task in tasks if indeg.get(task, 0) == 0)
        order: List[Any] = []
        while queue:
            task = queue.popleft()
            order.append(task)
            for neighbor in stage_adj[stage].get(task, set()):
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(tasks):
            raise RuntimeError(f"Cycle detected in stage {stage} while scheduling")
        stage_order[stage] = order

    compute_et_ids: Dict[Tuple[Any, int], int] = {}
    collective_et_ids: Dict[Tuple[Any, int], int] = {}
    pipeline_recv_cache: Dict[Tuple[Any, Any, int], int] = {}
    tag_counter = itertools.count(start=1)

    def ensure_pipeline(parent: Any, target: Any, dp_idx: int) -> int:
        parent_stage = getattr(parent, "hw_id", None)
        target_stage = collective_info[target]["stage"] if target in collective_info else compute_info[target]["stage"]
        if parent_stage == target_stage:
            if parent in collective_info:
                return collective_et_ids[(parent, rank_for(parent_stage, dp_idx))]
            return compute_et_ids[(parent, rank_for(parent_stage, dp_idx))]

        key = (parent, target, dp_idx)
        cached = pipeline_recv_cache.get(key)
        if cached is not None:
            return cached

        edge_obj = pipeline_edge_map.get((parent, target))
        size = int(getattr(edge_obj, "comm_size_bytes", 0)) if edge_obj else 0
        tag = getattr(edge_obj, "op_id", next(tag_counter)) if edge_obj else next(tag_counter)

        src_rank = rank_for(parent_stage, dp_idx)
        dst_rank = rank_for(target_stage, dp_idx)
        send_trace = rank_traces[src_rank]
        recv_trace = rank_traces[dst_rank]

        send_id = send_trace.next_id
        send_name = f"{getattr(edge_obj, 'name', 'pipeline')}_send_dp{dp_idx}"
        send_node = _new_send_node(send_id, send_name, size, dst_rank, tag)
        if parent in collective_info:
            send_node.ctrl_deps.append(collective_et_ids[(parent, src_rank)])
        else:
            send_node.ctrl_deps.append(compute_et_ids[(parent, src_rank)])
        send_trace.nodes.append(send_node)

        recv_id = recv_trace.next_id
        recv_name = f"{getattr(edge_obj, 'name', 'pipeline')}_recv_dp{dp_idx}"
        recv_node = _new_recv_node(recv_id, recv_name, size, src_rank, tag)
        # recv_node.ctrl_deps.append(send_id)
        recv_trace.nodes.append(recv_node)

        pipeline_recv_cache[key] = recv_id
        return recv_id

    # ignore pipeline deps
    for stage in stage_order:
        order = stage_order[stage]
        for task in order:
            if task in collective_info:
                info = collective_info[task]
                # Skip stage collectives entirely when dp_count <= 1
                if dp_count <= 1:
                    continue
                for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                    trace = rank_traces[rank]
                    deps: List[int] = []
                    for dep in info["stage_deps"]:
                        if dep in collective_info:
                            deps.append(collective_et_ids[(dep, rank_for(collective_info[dep]["stage"], dp_idx))])
                        else:
                            deps.append(compute_et_ids[(dep, rank_for(dep.hw_id, dp_idx))])
                    # for parent in info["pipeline_deps"]:
                    #     deps.append(ensure_pipeline(parent, task, dp_idx))
                    unique_deps = []
                    for dep in deps:
                        if dep not in unique_deps:
                            unique_deps.append(dep)

                    node_id = trace.next_id
                    comm_node = _new_comm_node(
                        node_id,
                        f"{task.name}_{task.op_id}_dp{dp_idx}",
                        info["comm_type"],
                        info["size"],
                    )
                    # Tag with dp communication group name (string id)
                    if dp_count > 1:
                        stage_idx = stage_index[stage]
                        group_id = str(stage_idx + 1)
                        comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=group_id))
                    comm_node.ctrl_deps.extend(unique_deps)
                    trace.nodes.append(comm_node)
                    collective_et_ids[(task, rank)] = node_id
            else:
                info = compute_info[task]
                stage = info["stage"]
                for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                    trace = rank_traces[rank]
                    deps: List[int] = []
                    for parent in info["stage_deps"]:
                        deps.append(compute_et_ids[(parent, rank_for(parent.hw_id, dp_idx))])
                    for edge in info["collective_deps"]:
                        stage_edge = collective_info.get(edge, {}).get("stage")
                        if stage_edge is not None and stage_edge == stage:
                            deps.append(collective_et_ids[(edge, rank)] )
                    # for parent in info["pipeline_deps"]:
                    #     deps.append(ensure_pipeline(parent, task, dp_idx))

                    unique_deps = []
                    for dep in deps:
                        if dep not in unique_deps:
                            unique_deps.append(dep)

                    duration_sec = getattr(task, "duration", 0.0) or 0.0
                    duration_micros = int(round(duration_sec * 1e6)) if duration_sec else 0
                    node_id = trace.next_id
                    comp_node = _new_comp_node(
                        node_id,
                        f"{task.name}_{task.op_id}",
                        max(duration_micros, 0)
                    )
                    comp_node.ctrl_deps.extend(unique_deps)
                    trace.nodes.append(comp_node)
                    compute_et_ids[(task, rank)] = node_id

                    # NOTE: Do not generate collective nodes here. Collectives
                    # are created in their own 'task in collective_info' branch
                    # above to avoid duplication. Here we only reference them
                    # via dependencies when needed.


    # this time with pipeline deps ONLY
    # Attach cross-stage (pipeline) dependencies after all nodes are created.
    for stage, order in stage_order.items():
        for task in order:
            if task in collective_info:
                info = collective_info[task]
            else:
                info = compute_info[task]

            if not info["pipeline_deps"]:
                continue

            for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                # Ensure SEND/RECV nodes exist and collect RECV ids local to this rank
                recv_ids: List[int] = []
                for parent in info["pipeline_deps"]:
                    recv_ids.append(ensure_pipeline(parent, task, dp_idx))

                # Deduplicate
                unique_recv_ids: List[int] = []
                for rid in recv_ids:
                    if rid not in unique_recv_ids:
                        unique_recv_ids.append(rid)

                # Append RECV deps to the already-created node for this task
                if task in collective_info:
                    node_id = collective_et_ids[(task, rank)]
                else:
                    node_id = compute_et_ids[(task, rank)]
                node = rank_traces[rank].nodes[node_id]
                for rid in unique_recv_ids:
                    if rid not in node.ctrl_deps:
                        node.ctrl_deps.append(rid)


    for trace in rank_traces.values():
        trace.close()

    et_prefix = f"{output_dir}/llm_graph"
    rank_ids = sorted(rank_traces.keys())
    print(f"[AstraSim] Generated ET files for ranks {rank_ids}: {et_prefix}.{{0..{len(rank_ids)-1}}}.et")
    return et_prefix, rank_ids

def run_astra_simulation_only(fwd_root, bwd_root, time_calc_obj, output_dir: str = "astra_comparison_output"):
    """
    Run AstraSim simulation on DeepFlow graph and print results.

    Args:
        fwd_root: Forward graph root node
        bwd_root: Backward graph root node
        time_calc_obj: TimeCalculationLLM object with hw_config and dp attributes
        output_dir: Directory for temporary files and results
    """
    print("\n" + "="*60)
    print("ASTRASIM SIMULATION RESULTS")
    print("="*60)

    try:
        # Convert both forward and backward graphs to Chakra ET format
        astrasim_start = time.time()

        # For now, just convert forward graph (can extend to include backward later)
        print(f"[AstraSim] Converting forward graph...")
        # Clean previous forward outputs to avoid stale files
        try:
            fwd_dir = os.path.join(output_dir, "fwd")
            if os.path.isdir(fwd_dir):
                shutil.rmtree(fwd_dir)
        except Exception as exc:
            print(f"[WARN] Failed to clean {fwd_dir}: {exc}")
        fwd_et_prefix, fwd_ranks = convert_deepflow_graph_to_chakra_et(
            fwd_root,
            time_calc_obj.dp,
            f"{output_dir}/fwd",
        )

        print(f"[AstraSim] Converting backward graph...")
        bwd_et_prefix, bwd_ranks = convert_deepflow_graph_to_chakra_et(
            bwd_root,
            time_calc_obj.dp,
            f"{output_dir}/bwd",
        )

        if fwd_ranks != bwd_ranks:
            raise ValueError(
                "Forward and backward graphs map to different hardware IDs. "
                f"Forward: {fwd_ranks}, Backward: {bwd_ranks}"
            )

        rank_count = len(fwd_ranks)
        # Emit ET text dumps for both forward and backward graphs
        _dump_et_text([f"{fwd_et_prefix}.{rank}.et" for rank in fwd_ranks])
        _dump_et_text([f"{bwd_et_prefix}.{rank}.et" for rank in bwd_ranks])

        # Generate AstraSim configuration files using actual hardware config
        print(f"[AstraSim] Generating configuration files...")
        astra_configs = generate_astrasim_configs_from_hw(time_calc_obj.hw_config, output_dir, rank_count)
        remote_memory_json = get_remote_memory_path()
        comm_groups_path = _write_comm_groups_json(output_dir, getattr(time_calc_obj, "dp", 1), fwd_ranks)

        # Run AstraSim simulation on forward graph
        print(f"[AstraSim] Executing forward simulation with {rank_count} ranks...")
        fwd_times, fwd_total = run_astrasim_analytical(
            fwd_et_prefix,
            astra_configs["system_json"],
            astra_configs["network_yaml"],
            remote_memory_json,
            comm_group_json=comm_groups_path
        )

        print(f"[AstraSim] Executing backward simulation with {rank_count} ranks...")
        bwd_times, bwd_total = run_astrasim_analytical(
            bwd_et_prefix,
            astra_configs["system_json"],
            astra_configs["network_yaml"],
            remote_memory_json,
            comm_group_json=comm_groups_path
        )

        conversion_and_sim_time = time.time() - astrasim_start

        # Print results
        print(f"[AstraSim] Forward execution time: {fwd_total:.6f} seconds")
        print(f"[AstraSim] Backward execution time: {bwd_total:.6f} seconds")
        print(f"[AstraSim] Total execution time: {fwd_total + bwd_total:.6f} seconds")
        print(f"[AstraSim] Simulation duration: {conversion_and_sim_time:.3f} seconds")

        print("="*60)

    except Exception as e:
        print(f"[AstraSim] ERROR: Failed to run simulation: {e}")
        print("="*60)
        raise

def run_astra_simulation_only_onepath(fwdbwd_root, time_calc_obj, output_dir: str = "astra_comparison_output"):
    """
    Run AstraSim simulation on DeepFlow graph and print results.

    Args:
        fwdbwd_root: Forward and backward graph root node
        time_calc_obj: TimeCalculationLLM object with hw_config and dp attributes
        output_dir: Directory for temporary files and results
    """
    print("\n" + "="*60)
    print("ASTRASIM SIMULATION RESULTS")
    print("="*60)

    try:
        # Convert both forward and backward graphs to Chakra ET format
        astrasim_start = time.time()

        # For now, just convert forward graph (can extend to include backward later)
        print(f"[AstraSim] Converting graph...")
        # Clean previous forward outputs to avoid stale files
        try:
            fwd_dir = os.path.join(output_dir, "fwd")
            if os.path.isdir(fwd_dir):
                shutil.rmtree(fwd_dir)
        except Exception as exc:
            print(f"[WARN] Failed to clean {fwd_dir}: {exc}")
        fwd_et_prefix, rank_ids = convert_deepflow_graph_to_chakra_et(
            fwdbwd_root,
            time_calc_obj.dp,
            f"{output_dir}/fwd",
        )
        rank_count = len(rank_ids)
        for rank in rank_ids:
            _visualize_et_files([f"{output_dir}/fwd/llm_graph.{rank}.et"])
            _dump_et_text([f"{output_dir}/fwd/llm_graph.{rank}.et"])
        # exit()

        # Generate AstraSim configuration files using actual hardware config
        print(f"[AstraSim] Generating configuration files...")
        astra_configs = generate_astrasim_configs_from_hw(time_calc_obj.hw_config, output_dir, rank_count)
        remote_memory_json = get_remote_memory_path()
        comm_groups_path = _write_comm_groups_json(output_dir, getattr(time_calc_obj, "dp", 1), rank_ids)

        # Run AstraSim simulation on forward graph
        print(f"[AstraSim] Executing forward simulation with {rank_count} ranks...")
        fwd_times, fwd_total = run_astrasim_analytical(
            fwd_et_prefix,
            astra_configs["system_json"],
            astra_configs["network_yaml"],
            remote_memory_json,
            comm_group_json=comm_groups_path
        )


        conversion_and_sim_time = time.time() - astrasim_start

        # Print results
        # include times per node
        print(f"[AstraSim] Times per node: {fwd_times}")
        print(f"[AstraSim] Total execution time: {fwd_total:.6f} seconds")
        print(f"[AstraSim] Simulation duration: {conversion_and_sim_time:.3f} seconds")

        print("="*60)

    except Exception as e:
        print(f"[AstraSim] ERROR: Failed to run simulation: {e}")
        print("="*60)
        raise



if __name__ == "__main__":

    def my_save_graph(roots, output_folder = "output_graph/", filename="graph"):
        dot_fw = visualize_graph(roots, filename=output_folder + filename)
        dot_fw.render(output_folder + filename , format="png", cleanup=True)
        print("graph saved to %s%s.png" % (output_folder , filename ))

    import config
    import pickle
    # exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_hw_path = os.path.expandvars(os.path.expanduser("configs/hardware-config/a100_80GB_tp.yaml"))
    exp_model_path = os.path.expandvars(os.path.expanduser("configs/model-config/LLM.yaml"))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type="LLM")
    with open("fw_bw_graph.pkl", "rb") as f:
        fw_bw_root = pickle.load(f)
    # make a fake object 
    class FakeTimeCalculationLLM:
        def __init__(self, hw_config, model_config, mode):
            self.hw_config = hw_config
            self.model_config = model_config    
            self.mode = mode
            self.dp = 2
    time_calc_obj = FakeTimeCalculationLLM(exp_hw_config, exp_model_config, "LLM")
    my_save_graph(fw_bw_root, "./astra_comparison_output", "fw_bw_graph_astra")
    paths = []
    paths.append("/app/nanocad/projects/deepflow_dev/DeepFlow/DeepFlow_george/astra_cache/workload/all_reduce/2npus_1.50GB/all_reduce_1.50GB.0.et")
    _dump_et_text(paths)
    run_astra_simulation_only_onepath(fw_bw_root, time_calc_obj, "./astra_comparison_output")   
