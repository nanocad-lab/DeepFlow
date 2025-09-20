# full_astrasim_flattened Implementation Plan

## Goal
Implement the `full_astrasim_flattened` execution mode so a single AstraSim run consumes an explicit micro-batch-by-micro-batch graph containing all transformer compute kernels, tensor-parallel communications, data-parallel collectives, and pipeline send/recv edges.

## High-level Steps
1. **Gather transformer metadata for expansion**
   - Surface transformer layer GEMM templates and communication descriptors to the pipeline graph builder.
   - Annotate pipeline transformer nodes with the micro-batch index, layer index, and forward/backward direction so the flattener knows which template to clone.

2. **Author the graph-flattening pass**
   - Introduce a helper (e.g., `FlattenedGraphBuilder`) that takes the pipeline root, transformer template, and parallelism parameters (`dp`, `lp`, `kp1`, `kp2`).
   - For each transformer node in the pipeline graph:
     - Clone the per-layer GEMM sequence for each tensor-parallel rank (`tp_degree = kp1 * kp2`), emitting unique compute node IDs that encode layer, micro-batch, direction, and rank.
     - Insert the corresponding tensor-parallel collectives using existing utilities so metadata (message size, interconnect type) stays consistent.
     - Reattach pre-existing pipeline edges: incoming dependencies feed the first GEMM in the rank group, and outputs from the final GEMM/collective flow back into the pipeline successor nodes.
   - Apply the same expansion to backward nodes, respecting their reversed ordering and comm patterns.
   - Assign hardware IDs deterministically with `hw_id = stage_index * tp_degree + tp_rank`; embedding and softmax nodes follow the rank assignment of their owning stage (usually `tp_rank == 0`).
   - Preserve data-parallel collectives by connecting them to the expanded rank nodes.

3. **Integrate with dispatcher**
   - Replace `NotImplementedError` in `LLMExecutionDispatcher._run_full_astrasim_flattened` with logic that runs the flattener and stores the flattened root.
   - Run `run_astra_simulation_only_onepath` on the flattened graph without applying the existing DP override (the flattened graph already presents all ranks explicitly).
   - Collect per-rank timing results and aggregate statistics analogous to the hierarchical mode.
   - Add lightweight sanity assertions (e.g., non-empty node list, expected rank multiplicity) to catch wiring mistakes without building a full regression harness.

4. **Update ET conversion pipeline**
   - Touch `convert_deepflow_graph_to_chakra_et` as little as possible:
     - Thread the minimum extra metadata (e.g., tensor-parallel degree) needed to translate the new `hw_id` layout.
     - Reuse existing helper code paths, only inserting narrowly scoped conditionals to map `(pipeline_stage, tp_rank, dp_rank)` triples to AstraSim ranks when the flattened mode is active.
     - Avoid reshuffling communicator construction; instead, extend the current logic to accept the enlarged rank lists.
   - Confirm pipeline send/recv edges line up with the new `hw_id`s.

5. **Configuration and caching adjustments**
   - Ensure AstraSim config generation receives `npus_count = dp * lp * tp_degree` derived from the flattened graph.
   - Decide on a naming convention/output directory for flattened ET artifacts (e.g., reuse existing folder with a `_flattened` suffix) for easier debugging.

6. **Validation**
   - Rely on existing integration flows and manual spot checks rather than new regression suites.
   - Use the assertions from step 3 plus ad-hoc comparisons (e.g., total runtime sanity checks) to gain confidence that flattened and hierarchical executions stay in the same ballpark.

## Open Questions / Follow-ups
- Verify whether embedding/softmax should ever be sharded across TP ranks; if so, the flattener must clone their compute and comm logic similarly.
- Confirm communicator naming conventions in the flattened graph remain compatible with downstream tooling (e.g., visualization scripts).
- Evaluate whether additional caching/invalidation hooks are needed when switching between hierarchical and flattened runs.
