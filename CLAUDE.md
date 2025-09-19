# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

DeepFlow is a cross-stack performance modeling framework for neural network workloads, particularly LLMs. It integrates technology nodes, system architecture, memory hierarchy, and workload graphs to predict iteration time and explore design spaces. The repository contains Python-based simulation tools with AstraSim integration for network modeling.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Performance Analysis
The main entry points are:

1. **Performance Prediction (run_perf.py)** - Primary interface for GEMM, LSTM, and LLM analysis:
```bash
python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/[GEMM|LSTM|LLM].yaml --output_dir output
```

2. **Architecture Search (GD_search.py)** - For fixed parallelism strategies:
```bash
python GD_search.py --exp_config configs/[config.yaml] --exp_dir output --debug False --index [index] --batch_size [batch] --hidden_dim [lstm_dim] --data_scale [dataset_scaling_factor] --dp [data_parallel_dim] --lp [layer_parallel_dim] --kp_type [0|1] --kp1 [kp1_dim] --kp2 [kp2_dim] --inter_derate [derate_factor] --intra_derate [derate_factor] --kp1_inter [False|True] --kp2_inter [False|True] --dp_inter [False|True] --lp_inter [False|True] --wafer_dim [package_dim]
```

3. **Main Entry Point (main.py)** - For SLURM-based execution:
```bash
python main.py stand_alone --exp_dir [output_directory] --exp_config configs/[config.yaml]
python main.py arch_search --exp_dir [output_directory] --exp_config configs/[config.yaml]
```

### Test Installation
```bash
python run_perf.py --hardware_config configs/hardware-config/waferscale_20v100_80hbm.yaml --model_config configs/model-config/LSTM.yaml --output_dir output
# Check output: vim output/LSTM/summary_LSTM.txt
```

### Debugging
Use `--debug True` flag with any of the main scripts to activate debugging mode.
Use `--no_launch True` with main.py to see the command without running it.

## Architecture and Key Components

### Core Modules
- **time_calculation.py**: Main simulation engine with NetworkModel class and event-driven scheduler
- **astrasim_integration.py**: AstraSim integration for network modeling (M1 scope: GEMM focus)
- **model.py**: Model definitions (Model_LSTM, Model_GEMM, Model_LLM)
- **config.py**: Configuration parsing for hardware and model parameters
- **hw_component.py**: Hardware modeling (Core, MemoryHierarchy, Network)
- **parallelism.py**: Parallelism strategy implementation
- **topology.py**: Network topology modeling
- **tile.py**: TiledGEMM operations and memory formatting
- **simulate.py**: Graph-based simulation framework

### Configuration System
Configurations are split into two main categories:
- **Hardware configs**: `configs/hardware-config/` (e.g., v100.yaml, a100_80GB.yaml, H100_SXM5_80GB.yaml)
- **Model configs**: `configs/model-config/` (GEMM.yaml, LSTM.yaml, LLM.yaml)

The system supports both old-style configs (configs/old-configs/) and new hardware-specific configs.

### AstraSim Integration
- Execution backend can be switched between `analytical` (DeepFlow's built-in) and `astra` (AstraSim integration)
- Configuration via `execution_backend.model` in YAML files
- Automatic fallback to analytical model if AstraSim fails
- Cached results in `astra_cache/` directory with JSON cache files

### Output Structure
Results are typically saved to the specified output directory with structure:
- `output/[MODEL_TYPE]/summary_[MODEL_TYPE].txt` - Main results summary
- Various intermediate files and logs

### Key Design Patterns
- Event-driven simulation with overlap modeling
- Hierarchical roofline modeling for compute performance
- Pluggable network timing backends
- Configuration-driven execution modes
- SLURM integration for cluster execution

### Development Workflow
1. Configure hardware parameters in `configs/hardware-config/`
2. Set model parameters in `configs/model-config/`
3. Run analysis using `run_perf.py` or other entry points
4. Check results in the output directory
5. For architecture search, use `GD_search.py` or `main.py arch_search`

### Important Notes
- The repository is currently in active development on the `llm-dev` branch
- Main production branch is `master`
- AstraSim integration is work-in-progress with M1 focus on GEMM operations
- LLM mode has limited validation and not all parallelism configurations are supported
- The system includes extensive caching to avoid recomputation across runs
