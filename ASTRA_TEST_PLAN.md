# AstraSim vs DeepFlow Comparison Test Plan

## Overview
This plan outlines how to compare DeepFlow's event-driven simulation against AstraSim by running the same LLM execution graph through both simulators with identical compute times and communication sizes.

## Key Insight
Currently DeepFlow bundles local operations (gradient application, memory copies) with communication operations, and annotates graph edges with communication **time** (already computed). For proper comparison, we need to:
1. Separate local computation from communication operations
2. Decouple size calculation from time calculation
3. Use explicit local computation nodes in the execution graph

## Implementation Approach

### 1. Modify DeepFlow Mainline: 2-Pass Graph Construction with Separate Local Nodes

**Phase 1: Graph Construction with Sizes**
- Modify `TimeCalculationLLM` to build graphs with communication **sizes** and **separate local computation nodes**
- Communication edges contain:
  - `comm_size_bytes`: The actual data size for the communication operation
  - `comm_type`: The type of collective (all_reduce, all_gather, etc.)
  - `participants`: Number of ranks participating
- Add explicit local computation nodes after each communication:
  ```python
  # Instead of bundled: comm_edge(comm_time + local_time)
  # Use explicit: comm_edge(size) -> local_comp_node(local_time)
  ```

**Phase 2: Time Calculation Pass**
- Second pass converts sizes to times using network model
- Preserves existing DeepFlow timing calculations
- Results in complete execution graph with explicit timing

### 2. AstraSim Test Implementation (Non-Mainlined)
Using the new 2-pass graph structure, create dual-path comparison:

**Path A: DeepFlow Analytical**
- Pass the graph with comm sizes to DeepFlow's existing network model
- Use `self.network_model.collective()` to convert sizes to times
- Run DeepFlow's event-driven simulation with computed times
- Record total execution time

**Path B: AstraSim Simulation**
- Convert DeepFlow graph (with comm sizes + comp times) to AstraSim Chakra ET format
- Run AstraSim congestion-aware simulation
- Record total execution time

**Comparison**
- Compare total times and per-operation breakdown
- Analyze differences in communication modeling
- Validate that both simulators respect identical dependency constraints

## Implementation Details

### DeepFlow Mainline Modifications

**Graph Node/Edge Structure Changes:**
```python
# Add to simulate_LLM.py Node class
class Node:
    def __init__(self, name, op_id, hw_id, duration):
        # ... existing fields ...
        self.local_comp_time = 0.0  # For local operations after comm

# Add to simulate_LLM.py Edge class
class Edge:
    def __init__(self, name, op_id, hw_id, duration):
        # ... existing fields ...
        self.comm_size_bytes = 0
        self.comm_type = None        # "all_reduce", "all_gather", etc.
        self.participants = 1
        self.is_communication = False
```

**TimeCalculationLLM Modifications:**
```python
# Modify getDataParallelReduction_LLM() to return sizes and local times separately
def getDataParallelReduction_LLM(self, d, ffn_dim):
    # Calculate communication sizes (existing logic)
    qkv_size = math.ceil(self.precision * d * 3 * d)
    out_size = math.ceil(self.precision * d * d)
    ffn_size = math.ceil(self.precision * ffn_dim * d)

    # Calculate local computation times (existing logic)
    qkv_local = self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
    out_local = self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")
    ffn_local = 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

    # Return structured data instead of bundled time
    return {
        'qkv': {'size': qkv_size, 'local_time': qkv_local},
        'output': {'size': out_size, 'local_time': out_local},
        'ffn': {'size': ffn_size, 'local_time': ffn_local}
    }
```

**Graph Construction with Explicit Local Nodes:**
```python
# In construct_bwd_graph(), create explicit local computation nodes
def construct_bwd_graph(self):
    # ... existing node creation ...

    # For each reduction, create: comm_edge -> local_comp_node
    for layer_id in range(self.num_layer):
        # Communication edge with size metadata
        comm_edge = Edge(f"Reduce_transformer_{layer_id}", op_id, hw_id, 0)
        comm_edge.comm_size_bytes = reduction_data['size']
        comm_edge.comm_type = "all_reduce"
        comm_edge.participants = self.dp
        comm_edge.is_communication = True

        # Local computation node after communication
        local_node = Node(f"Local_comp_{layer_id}", op_id+1, hw_id, reduction_data['local_time'])

        # Chain them: transformer_layer -> comm_edge -> local_node -> next_layer
        prev_layer.add_child(comm_edge)
        comm_edge.add_child(local_node)
        local_node.add_child(next_layer)
```

**2-Pass Time Calculation:**
```python
# New method in TimeCalculationLLM
def convert_sizes_to_times(self, graph):
    """Second pass: convert communication sizes to times"""
    for edge in self._get_all_edges(graph):
        if edge.is_communication:
            comm_time = self.network_model.collective(
                kind=edge.comm_type,
                size_bytes=edge.comm_size_bytes,
                participants=edge.participants,
                ib=self.IBD,
                ll=self.LLD
            )
            edge.duration = comm_time
```

### AstraSim Test Implementation (Detailed Conversion)

**Graph Traversal and Conversion:**
```python
# In astra_comparison.py
def convert_deepflow_graph_to_astrasim(graph, hw_config, output_dir):
    """Convert DeepFlow graph with explicit local nodes to AstraSim ET"""

    # Extract all nodes and edges from DeepFlow graph
    all_nodes = extract_all_nodes(graph)
    all_edges = extract_all_edges(graph)
    dp_size = hw_config.scheduling_param.dp

    # Generate per-rank ET files
    for rank in range(dp_size):
        et_path = f"{output_dir}/llm_graph.{rank}.et"
        with open(et_path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))

            node_id = 0

            # Convert compute nodes (including explicit local comp nodes)
            for node in all_nodes:
                if node.hw_id == rank or node.hw_id == -1:  # This rank or memory
                    astra_node = _new_comp_node(
                        node_id,
                        f"{node.name}_{node.op_id}",
                        int(node.duration * 1e6)  # sec → μs
                    )
                    # Add dependencies from DeepFlow parent/child relationships
                    astra_node.ctrl_deps.extend([
                        parent_mapping[p.op_id] for p in node.parents
                        if p.op_id in parent_mapping
                    ])
                    write_et_node(fh, astra_node)
                    node_id += 1

            # Convert communication edges
            for edge in all_edges:
                if edge.is_communication:
                    # All ranks participate in collective
                    astra_comm = _new_comm_node(
                        node_id,
                        f"{edge.name}_{edge.op_id}",
                        get_collective_type(edge.comm_type),
                        edge.comm_size_bytes
                    )
                    # Add dependencies
                    astra_comm.ctrl_deps.extend([
                        parent_mapping[p.op_id] for p in edge.parents
                        if p.op_id in parent_mapping
                    ])
                    write_et_node(fh, astra_comm)
                    node_id += 1

    return f"{output_dir}/llm_graph"

def get_collective_type(comm_type_str):
    """Map DeepFlow comm types to AstraSim protobuf enums"""
    mapping = {
        "all_reduce": pb.ALL_REDUCE,
        "all_gather": pb.ALL_GATHER,
        "reduce_scatter": pb.REDUCE_SCATTER,
        "all_to_all": pb.ALL_TO_ALL
    }
    return mapping.get(comm_type_str, pb.ALL_REDUCE)
```

**Dual Execution and Comparison:**
```python
def run_dual_simulation_comparison(graph, hw_config, output_dir):
    """Run same graph through both simulators and compare"""

    # Path A: DeepFlow with its network model
    deepflow_start = time.time()
    fw_time = graph.simulate(graph.construct_fwd_graph()[0], 0)
    bw_time = graph.simulate(graph.construct_bwd_graph()[0], graph.lp - 1)
    deepflow_total = fw_time + bw_time
    deepflow_duration = time.time() - deepflow_start

    # Path B: AstraSim simulation
    astrasim_start = time.time()
    et_prefix = convert_deepflow_graph_to_astrasim(graph, hw_config, output_dir)
    astrasim_times, astrasim_total = run_astrasim_simulation(et_prefix, hw_config)
    astrasim_duration = time.time() - astrasim_start

    # Analysis and comparison
    comparison = {
        'deepflow_time': deepflow_total,
        'astrasim_time': astrasim_total,
        'percent_diff': abs(deepflow_total - astrasim_total) / deepflow_total * 100,
        'deepflow_sim_duration': deepflow_duration,
        'astrasim_sim_duration': astrasim_duration,
        'communication_ratio': calculate_comm_ratio(graph),
        'bottleneck_analysis': compare_critical_paths(graph, astrasim_times)
    }

    return comparison
```

### Expected Results
- Both simulators should produce similar total execution times (within 10-30%)
- Communication-heavy workloads may show larger differences due to congestion modeling
- Compute-bound workloads should be nearly identical
- Critical path analysis should be consistent between simulators

## Files to Create/Modify
1. **`astra_comparison.py`** - New file with graph conversion and comparison functions
2. **`time_calculation_LLM.py`** - Modified to store comm sizes and call comparison
3. **Existing infrastructure** - Leverage `astrasim_integration.py` for ET generation

## Success Criteria
- Successful conversion of DeepFlow LLM graphs to AstraSim ET format
- Both simulators execute identical dependency graphs with same compute times
- Meaningful comparison of communication modeling approaches
- Validation that DeepFlow's event-driven scheduling produces realistic results