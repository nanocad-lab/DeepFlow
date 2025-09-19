# Execution Mode Integration Plan

## Objectives
- Support four DeepFlow↔AstraSim integration styles without falling back to legacy behavior.
- Centralise execution-mode selection in configuration (`execution_backend`).
- Keep DeepFlow’s analytical scheduler as the orchestrator while making data exchanges with AstraSim explicit and testable.
- Document responsibilities, produced artefacts, and validation steps for each mode before touching code.

## Target Execution Modes
1. **Analytical**  
   DeepFlow alone computes all timings. No AstraSim assets are generated.

2. **Hybrid (Comm Oracle)**  
   DeepFlow schedules the full pipeline graph. Communication nodes are re-timed with AstraSim analytical runs. Compute stays analytical.

3. **Hybrid Congestion-Aware**  
   Same as Hybrid, plus a second AstraSim workload that models a transformer block (with TP/DP congestion). The resulting block duration overwrites transformer-node compute times in the pipeline graph prior to the DeepFlow simulation.

4. **Full AstraSim**  
   DeepFlow only generates the two Chakra ET graphs (pipeline + transformer block). AstraSim runs both and provides all timing numbers that replace DeepFlow’s per-node durations.

## High-Level Work Breakdown
1. **Configuration and Plumbing**  
   - Rename `network_backend` → `execution_backend` and drop legacy shims.  
   - Preserve the top-level `model` toggle (`analytical|astra`).  
   - Extend `execution_backend.astra.mode` to accept `hybrid`, `hybrid_congestion`, `full_astrasim`.  
   - Provide validation/helpers so CLI/tests can query `execution_mode` deterministically.

2. **Graph Generation Enhancements**  
   - Retain the existing pipeline graph builder.  
   - Add a dedicated transformer-block generator capable of representing tensor-parallel congestion scenarios (re-usable across Hybrid Congestion-Aware and Full AstraSim).  
   - Ensure node naming/op IDs align between graphs so AstraSim outputs can be mapped back unambiguously.

3. **Orchestration Layer**  
   - Introduce an execution-mode dispatcher (likely in `time_calculation_LLM.py`) that sequences: graph construction, DeepFlow analytical runs, AstraSim invocations, and result stitching.  
   - Cache intermediate ET/config files to avoid redundant generation across modes.

4. **Result Application**  
   - Generalise the “apply AstraSim timings” pathway so it can retime communication nodes, transformer compute nodes, or whole graphs depending on mode.  
   - Add consistency checks (e.g., missing AstraSim outputs) with fail-fast diagnostics.

5. **Testing & Tooling**  
   - Build smoke tests for each execution mode using a minimal LLM config.  
   - Provide comparison utilities to diff mode outputs (latency breakdowns, per-edge durations).  
   - Update documentation and CLI help to reflect new terminology and behaviour.

## Configuration Design (New YAML Layout)
```yaml
execution_backend:
  model: analytical | astra
  astra:
    backend: analytical   # existing knob, remains optional
    mode: hybrid | hybrid_congestion | full_astrasim
    collectives:
      all_reduce: auto
      all_gather: auto
      reduce_scatter: auto
      all_to_all: auto
```
- When `model: analytical`, DeepFlow executes in Analytical mode and the `astra` block is ignored.  
- When `model: astra`, `astra.mode` selects between Hybrid, Hybrid Congestion-Aware, and Full AstraSim behaviour.  
- Additional per-mode parameters (e.g., TP settings for the transformer block graph) will be added under `execution_backend.astra` as follow-on work.

## Open Questions / Follow-Ups
- Define how transformer-block graph parameters (micro-batch size, TP degree) are sourced—derivable from existing model config or new YAML keys.  
- Decide where AstraSim outputs are stored and versioned for reproducibility.  
- Determine whether to allow user-provided ET overrides for advanced use cases.

