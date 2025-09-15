
# DeepFlow ↔ AstraSim Integration Plan (Simplified **b** Path)

**Owner:** DeepFlow LLM contributors  
**Last updated:** 2025‑09‑13 (PT)  
**Scope:** `llm-dev` branch — add an optional network‑timing backend powered by **AstraSim** using **Chakra Execution Traces (ET)**, while keeping DeepFlow’s existing compute model, mapping/routing, and event‑driven scheduler intact.

---



## 0) TL;DR

We will integrate **AstraSim** as a **communication‑time oracle** for DeepFlow’s LLM/LSTM paths. We will do this in **three phases**:

1. **(a) Isolated‑op baseline.** For every DeepFlow **communication edge**, call AstraSim **in isolation** (no concurrency) to convert **bytes + participants + topology** → **time**. This validates units, groupings, algorithm parity, and I/O plumbing.  
2. **Snapshot Collector (built on the (a) schedule).** Run DeepFlow once using (a)’s per‑edge times; record the timeline of comm activity and derive **overlap “snapshots”** (change‑point intervals where the active comm set is stable).  
3. **Static simplified‑(b).** For each snapshot, build a **tiny Chakra ET** that contains **all concurrent comm ops** in that interval. Feed it to AstraSim (Analytical backend) to obtain **contention‑aware** durations. Fold those times back onto edges and run DeepFlow once more to finalize timing.

This achieves topology‑portable, contention‑aware communication times **without** replacing DeepFlow’s scheduler or compute model and **without** complex in‑place rollback during execution.

---

## GEMM‑First Note (M1)

- Focus M1 on GEMMs: When `network_backend.model = astra`, GEMM internal communication in `getR(…)` routes to AstraSim Analytical via `astrasim_integration.run_cache_astrasim` instead of the built‑in ring math.  
- Scope: Only GEMM paths are switched; LLM/LSTM remain on DeepFlow’s analytical model until M2–M4.  
- Config: Use existing YAML keys under `network_backend` (e.g., `model: astra`, `astra.backend: analytical`, `astra.mode: isolated`, and optional `astra.collectives`).  
- Behavior: If AstraSim is unavailable or errors, we automatically fall back to the default calculation to preserve runability.  
- Deliverable: No change in defaults; enabling the flag affects GEMM comms only.

## 1) What these tools are (one page each)

### 1.1 DeepFlow (what & why)
**DeepFlow** is a cross‑stack performance modeling and pathfinding framework from UCLA NanoCAD. It integrates technology nodes, system architecture, memory hierarchy, and **workload graphs** to predict iteration time and explore design spaces. For neural workloads—especially **LLMs**—DeepFlow:  
- **Transforms** the model graph to apply **DP/TP/PP/sequence** parallelism and **emits explicit communication edges**.  
- **Maps** shards to devices, **routes** comm edges over the selected topology, and models contention.  
- **Times compute** via a hierarchical roofline model with memory‑aware tiling (GEMM‑centric), and **schedules** with an **event‑driven** engine to respect resource limits and overlap.  

**Why keep DeepFlow in charge:** We retain its modeling of compute kernels, pipeline/microbatch structure, and event‑driven overlap logic; we only swap the function that turns **comm sizes** into **times**.

### 1.2 AstraSim (what & why)
**AstraSim** is a **distributed ML system simulator** with pluggable backends for **network** (Analytical, ns‑3, Garnet) and **memory/compute** layers. Its **workload layer** consumes **Chakra ET**: a per‑rank DAG of **compute/communication** nodes and dependencies. From the same ET, AstraSim can evaluate different topologies and collective algorithms and returns **cycle counts** (convertible to time).

**Why use AstraSim here:** DeepFlow already knows **what overlaps** and which **participants**/sizes are involved. AstraSim specializes in **how those concurrent comms perform** under a chosen topology/algorithm—capturing **contention** more realistically than simple link‑sharing heuristics.

### 1.3 Chakra ET (the “wire format”)
**Chakra Execution Traces** are a standardized, graph‑based description of ML workloads. We will generate **comm‑only, per‑rank ETs** for each snapshot:  
- **Collectives**: one node per participating rank (AllReduce / AllGather / ReduceScatter / AllToAll), tagged with a **communicator/group id** and **byte size**.  
- **Point‑to‑point** (for PP activations): matching **SEND**/**RECV** nodes between rank pairs with byte sizes.  
- **No incoming deps** inside a snapshot ⇒ all comm nodes **can start concurrently** at t=0 in that snapshot micro‑trace.

### 1.4 Topologies and collectives (practical notes)
- Topologies (Analytical backend): `Ring`, `Switch`, and `FullyConnected` are available; for true 1‑hop per pair, use **FullyConnected** (not Switch).  
- Collective implementations (System layer): `ring`, `halvingDoubling`, `doubleBinaryTree`, and `direct|oneDirect[<window>]`. For FullyConnected fabrics, prefer **direct**; for ring fabrics, prefer **ring**.  
- Units: `comm_size` is in bytes; 1 cycle = 1 ns.

---

## 2) Repositories & docs (for maintainers)

- **DeepFlow (UCLA NanoCAD)**: internal repo — branch `llm-dev`. Papers/tutorials describe the compute model, graph‑transform, and event‑driven simulation philosophy.  
- **AstraSim** (GitHub): `github.com/astra-sim/astra-sim` (docs at `astra-sim.github.io`). Accepts **Chakra ET**; supports **Analytical** (fast), **ns‑3**, **Garnet** backends.  
- **Analytical Network Backend**: `github.com/astra-sim/astra-network-analytical` (supports congestion‑aware and congestion‑unaware modes).  
- **Chakra** (schema/tools): `github.com/mlcommons/chakra` (schema, generators, examples, test cases).

> We will **vendor nothing** here; we call AstraSim as a **sidecar process** and generate **ETs** via a small helper in our adapter. Where possible, we leverage Chakra’s reference builder utilities rather than inventing our own serializers.

---

## 3) Big‑picture integration (Simplified‑b built on (a))

We add a **pluggable network timing backend** with two modes: **`analytical`** (existing DeepFlow math) and **`astra`** (our adapter). The adapter supports **two invocation shapes**:

- **Edge mode (a)**: `estimate_comm_times([edge]) → {edge_id: time}` — used for the first pass to get a complete schedule and to calibrate units.  
- **Snapshot mode (simplified‑b)**: `estimate_comm_times(batch_of_edges) → {edge_id: time}` — used offline per **overlap snapshot** derived from the (a) run.

Configuration is file‑driven (no new CLI switches). The adapter writes and consumes:
- AstraSim network YAML (e.g., `Ring_*.yml`, `FullyConnected_*.yml`).  
- AstraSim system JSON (collective algorithm choices, e.g., `ring` vs `direct`).  
- Per‑rank workload `.et` files for (a) edges or per‑snapshot batches.  
DeepFlow chooses `analytical` vs `astra` and (a) vs snapshot through its own config/YAML, and the adapter launches the AstraSim sidecar with those files.

We do **not** change DeepFlow’s compute model, graph transforms, or global scheduler semantics.

---

## 4) What to add (new files) vs. what to touch (existing)

### 4.1 New module (adapter): `astrasim_integration.py`

**Purpose:** All AstraSim‑specific logic lives here.

**Responsibilities:**
- **ET Builder**: Make **per‑rank Chakra ETs** from either (i) one comm edge (a), or (ii) a **set** of concurrent edges (snapshot). For collectives, create one node per participating rank with `pg_name` and **bytes**; for P2P, create matching SEND/RECV nodes. Omit compute nodes unless needed for ordering.  
- **Communicator Groups**: Maintain a mapping from DeepFlow’s **group descriptors** (DP/TP/PP/sequence groups) to **AstraSim communicator names** and rank lists.  
- **AstraSim Driver**: Construct YAML/JSON configs (network, system, comm‑groups), write per‑rank ETs, and launch the AstraSim sidecar; parse **cycles** (and, where available, **per‑node completion**) into **seconds**; return a dictionary keyed by edge ids. No user‑facing CLI toggles are added; all behavior is driven by config files.  
- **Caching**: Memoize by **edge signature** (a) or **snapshot signature** (simplified‑b) to avoid recomputation across identical layers/microbatches/runs.  
- **Error Handling & Fallback**: If AstraSim fails for a snapshot, fall back to the (a) isolated sums for that window (flag in logs).  
- **No scheduler logic**: This module **does not** decide *when* ops overlap; it only prices the provided batch.

**Inputs required (from DeepFlow):**
- Edge list with: `edge_id`, `op_kind` (AR/AG/RS/A2A/P2P), `size_bytes`, `participants` (rank ids), optional `src/dst` (P2P), and `collective_algo` (semantic hint: ring, tree, etc.).  
- Rank→device mapping (stable IDs).  
- Topology key (to select the right AstraSim network config).

**Outputs:**
- Mapping `{edge_id → time_seconds}` for the provided edge(s).  
- Optional snapshot metadata (e.g., makespan, slowdown factor) for diagnostics.

---

### 4.2 Minimal changes to existing files

> **Design rule:** Keep changes surgical; add a small **interface seam** for “comm size → comm time,” and a **snapshot collector** that runs once after the (a) pass.

1) **`time_calculation.py`** — *central seam*  
   - Introduce a **`NetworkTiming`** interface with two methods:  
     - `estimate_edge(edge) → seconds` (used by the (a) pass),  
     - `estimate_batch(edges) → {edge_id: seconds}` (used by simplified‑b).  
   - Keep current math as the default **`AnalyticalTiming`**; add **`AstraSimTiming`** that delegates to `astrasim_integration.py`.

2) **`simulate.py`** — *snapshot collector harness*  
   - **(a) pass**: run simulation as today, but ask `NetworkTiming.estimate_edge` for every comm edge. Record **edge start & end times** into a timeline.  
   - **Snapshot derivation (offline)**: from the recorded timeline, compute **change points** (any comm start/end) and derive **snapshots** as intervals where the **active set of comm edges is constant**. Optionally partition snapshots by fabric (e.g., intra‑node TP vs. inter‑node DP) to keep ETs small and contention local. Persist a **snapshot manifest** (JSON) for reproducibility.  
   - **Static simplified‑b pass**: iterate snapshots, call `NetworkTiming.estimate_batch(edges_in_snapshot)`, and **overwrite** their per‑edge times in the model with the returned values. Finally, **re-run** the event‑driven simulation **once** using the new times to obtain the final schedule and iteration time.  
   - **Bounded iteration**: Optionally compare the new schedule’s change points to the prior set and re‑extract snapshots **at most once** if memberships shifted materially (avoid infinite refinement).

3) **`LLM_util.py`** — *edge metadata*  
   - Ensure every comm edge carries **all metadata needed** by the adapter: `op_kind`, `size_bytes`, `participants`, optional `src/dst` (P2P), and a stable `edge_id` and `group_id`. This is largely a wiring/labeling change; no semantic changes to graph construction.

4) **`topology.py` / `deviceMapping.py`** — *rank consistency*  
   - Expose the **rank→device** mapping and **communicator definitions** (DP/TP/PP/sequence groups) in a form the adapter can consume. No behavioral changes to mapping or routing.

5) **Configuration (YAML only)** — *mode selection*  
   - Extend DeepFlow’s YAML to select `network_model: {analytical|astra}` and, for AstraSim, `astra_mode: {isolated|snapshot}`. Route all paths (ET dir, logs) via config into `AstraSimTiming`. No new CLI flags are introduced.

> **Unchanged modules:** the hierarchical roofline compute model; graph transforms; device mapping & routing; the core event‑driven scheduler semantics.

---

## 5) Snapshots: definition & derivation

- A **snapshot** is a half‑open interval `[t_k, t_{k+1})` in which the set of **active** communication edges is **constant**. Change points are any comm **start** or **end** observed in the (a) pass.  
- We optionally **partition** a snapshot by:  
  1) **Communicator/fabric** (e.g., separate TP intra‑node collectives from DP inter‑node gradients),  
  2) **Op kind** (keep collectives together; keep unrelated P2P flows separate unless they share links),  
  3) **Link domain** (if DeepFlow exposes a clear intra‑socket vs inter‑socket boundary).  
- **Coalescing small ops**: Optionally group “tiny” edges (< X KiB) into a “small‑ops bucket” per group to keep ETs compact without materially affecting contention.

**Why this works:** Snapshot sets are derived from DeepFlow’s **own** scheduler (with per‑edge isolated times), so the **overlap structure** is preserved. AstraSim then supplies **contention‑aware** durations for those same concurrency sets.

---

## 6) Chakra ET shape for our use

- **Per‑rank ET files:** `{prefix}.{rank}.et`. Each file contains only the nodes relevant to that rank for this snapshot.  
- **Collective nodes:** one node per rank with attributes: `type ∈ {AllReduce, AllGather, ReduceScatter, AllToAll}`, `pg_name`, `comm_size_bytes`.  
- **P2P nodes:** `SEND` on src rank and matching `RECV` on dst rank with `message_size_bytes`.  
- **No incoming dependencies inside the snapshot:** allows all comm to **start at t=0**; AstraSim’s workload & network layers determine inter‑op interactions and **contention** under the given topology.  
- **Communicator‑group configuration:** a JSON mapping from `pg_name` → list of ranks. We generate this once per run from DeepFlow’s existing group definitions.

ET generation: we will reuse/port the Chakra microbenchmark generators in `examples/workload/microbenchmarks/generator_scripts` (AG/A2A/AR/RS) inside the adapter to emit comm‑only ETs. This is sufficient for DeepFlow; no custom collective definitions are required.

> We begin with AstraSim’s **Analytical** network backend for speed and determinism; higher‑fidelity backends (ns‑3/Garnet) remain plug‑compatible if later needed.

---

## 7) Data contracts (adapter ↔ DeepFlow)

**Edge Descriptor (input):**  
- `edge_id` (stable)  
- `op_kind` (AR | AG | RS | A2A | P2P)  
- `size_bytes` (int, ≥ 0)  
- `participants` (list[int] rank ids) — for P2P, also `src_rank`, `dst_rank`  
- `group_id` or `pg_name` (string)  
- `collective_algo` (optional hint: ring, tree, hierarchical, …)  
- `topology_key` (selects a network config in AstraSim)  

**Batch Result (output):**  
- `{edge_id → time_seconds}` for all edges in the batch  
- Optional: `{edge_id → diagnostics}` (e.g., per‑node finish cycles if available; otherwise per‑batch makespan + inflation factor)

---

## 8) Caching, logging, and failure policy

- **Caching:**  
  - (a) mode: key by `(op_kind, size_bytes, group_size, topology_key, algo_hint)`  
  - Snapshot mode: canonicalize a **snapshot signature** (sorted tuples of `(op_kind, size_bytes, participants, optional src/dst)`, plus `topology_key`).  
- **Logging:**  
  - Emit a **snapshot manifest** (intervals, memberships, sizes) and a **run report**: #snapshots, average inflation vs. isolated sum, slowest snapshots.  
  - Record AstraSim versions and backend mode in the run headers for reproducibility.  
- **Failure policy:**  
  - If a snapshot run fails/timeouts, mark it and fall back to the isolated‑sum apportioning for that snapshot; proceed so the overall run completes.

---

## 9) Validation & tests

**Smoke tests (unit):**  
- Canary: 1 MiB AllReduce on N ranks — verify AstraSim cycles scale with N similarly to ring math; verify byte→cycle→time conversions.  
- P2P sanity: single SEND/RECV vs. two overlapping pairs — ensure congestion‑aware analytical mode reports longer makespan in the overlapping case.  
- Group separation: two disjoint collectives in separate groups — ensure no cross‑group interference when topology doesn’t share links.

**Integration tests:**  
- **A‑only run**: end‑to‑end timing matches within ε of the analytical backend when AstraSim is set to **congestion‑unaware** mode (parity check).  
- **Snapshot run**: total iteration time is ≥ A‑only time; changing topology (ring ↔ fully‑connected) changes snapshot makespans in plausible ways and reflects expected hop factors (e.g., 4‑GPU AG → ring ≈ 3× fully‑connected under similar parameters).  
- **Stability**: one‑repass schedule change points differ minimally; if major reordering occurs, bound to at most one recompute.

**Performance tests:**  
- Large L layers × M microbatches: snapshot count growth, cache hit ratio, and AstraSim wall‑time per snapshot. Target: adapter overhead ≪ simulate‑time.

---

## 10) Risks & mitigations

- **Order changes after contention:** Use a **bounded two‑pass** (re‑extract snapshots once), and threshold on “material change.”  
- **Snapshot explosion:** Coalesce tiny ops; partition by fabric; cache aggressively; parallelize snapshot simulation.  
- **Collective mismatch:** Pin AstraSim’s collective settings to match DeepFlow’s semantics; document the chosen algorithms per op kind.  
- **Units drift (bytes/Hz):** Centralize conversions; add a canary test in CI.  
- **Runtime overhead:** Prefer Analytical backend; hash‑cache; lazy‑generate ETs only for unique snapshots.

---

## 11) Milestones & deliverables

1. **M1 – Interface seam**
   - `NetworkTiming` interface; default `AnalyticalTiming` preserved; selection is via DeepFlow YAML (no new CLI).  
   - **Deliverable:** trivial no‑op run parity with current results.

2. **M2 – (a) Isolated AstraSim**
   - `AstraSimTiming.estimate_edge` implemented; ET generation (ported from Chakra generators) and sidecar invocation using YAML/JSON; basic cache; canary tests.  
   - **Deliverable:** End‑to‑end run completes using YAML‑selected AstraSim isolated mode.

3. **M3 – Snapshot Collector**
   - Timeline capture + change‑point analysis; snapshot manifest serialization; batch adapter path (`estimate_batch`).  
   - **Deliverable:** Snapshot JSON emitted for representative LLM configs.

4. **M4 – Static simplified‑(b)**
   - Snapshot → ET builder; AstraSim per‑snapshot run; per‑edge time mapping; one‑pass final reschedule.  
   - **Deliverable:** YAML‑selected snapshot mode yields contention‑aware iteration times.

5. **M5 – Hardening**
   - Caching, thresholds, logging, failure fallback; perf tuning; CI canaries; documentation.

---

## 12) Open questions (to resolve during M2–M3)

- **Per‑node completion vs makespan:** Do we extend AstraSim logging to return **per‑ET‑node finish** cycles (preferred), or do we apportion by inflation factor within a snapshot?  
- **Grouping policy:** Default partitions (DP vs TP vs PP, intra vs inter‑node) — confirm with maintainers to keep traces small but realistic.  
- **Collective variants:** Which algorithms are assumed in DeepFlow’s LLM graphs for AR/AG/RS/A2A? (We should pin AstraSim to the same without introducing new variability for now.)  
- **Topology/collective mapping policy:** Default to `FullyConnected + direct` for 1‑hop fabrics; `Ring + ring` for ring fabrics. Expose choice via YAML.
- **Edge identity across passes:** Confirm a stable edge id survives re‑timing so we can overwrite durations deterministically.

---

## 13) Glossary

- **Edge (comm):** A graph edge emitted by DeepFlow representing a communication operation (collective or P2P).  
- **Snapshot:** A time interval where the set of **active comm edges** is constant; derived from an initial (a) schedule.  
- **ET (Chakra Execution Trace):** Per‑rank DAG describing compute/comm ops and deps; input to AstraSim’s workload layer.  
- **Analytical backend:** AstraSim’s fast network model; supports congestion‑aware and ‑unaware modes.  
- **Participants / ranks:** The ordered list of device ranks involved in a comm op; tied to communicator groups.

---

## 14) Summary

We integrate AstraSim as a **drop‑in network timing backend** behind a simple `NetworkTiming` interface, start with **isolated per‑edge pricing** to validate plumbing, then elevate to **contention‑aware snapshot pricing** based on the schedule we already get from that run. We keep DeepFlow’s strengths—graphing, mapping, compute modeling, event‑driven scheduling—while gaining **topology‑portable** comm timing and realistic **congestion** modeling through compact per‑snapshot ETs.

— End of document —
