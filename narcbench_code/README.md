# NARCBench

Code and data for **Detecting Multi-Agent Collusion Through Multi-Agent Interpretability**.

NARCBench is a three-tier benchmark for evaluating multi-agent collusion detection under environment distribution shift. A linear deception probe is trained on committee-deliberation activations (Core) and evaluated zero-shot on structurally novel scenarios (Transfer) and steganographic signalling (Stego).

**Paper**: [arXiv (coming soon)]

**Data**: [HuggingFace](https://huggingface.co/datasets/aaronrose227/narcbench)

## Quick Start

```bash
pip install numpy scikit-learn matplotlib
python reproduce.py
```

This reproduces Table 1 and Figure 2 from the paper. Requires the pre-extracted activation data from the HuggingFace dataset (see Data section below). Output is saved to `results/figures/`.

## Data

The activation data and scenario data are hosted on HuggingFace due to size (~2.6GB). Download and place them in `data/`:

```bash
huggingface-cli download aaronrose227/narcbench --repo-type dataset --local-dir data
```

## Structure

```
narcbench/
├── README.md
├── config.py                          # Shared config (model, paths, vLLM settings)
├── reproduce.py                       # Reproduces Table 1 and Figure 2
├── generation/                        # Scenario generation (requires vLLM server)
│   ├── core.py                        # NARCBench-Core (50 committee deliberation scenarios)
│   ├── transfer.py                    # NARCBench-Transfer (6 structurally distinct types)
│   ├── stego.py                       # NARCBench-Stego (steganographic card counting)
│   └── verify_ground_truth.py         # Ground truth verification
└── extraction/                        # Activation extraction (requires GPU + model weights)
    ├── extract_core.py                # Core activation extraction
    ├── extract_transfer.py            # Transfer activation extraction
    ├── extract_stego.py               # Stego activation extraction
    └── extract_token_traces.py        # Token-level trace extraction
```

## Pipeline

1. **Generate scenarios** (`generation/`): Requires a running vLLM server with Qwen3-32B-AWQ. Each script generates scenario data (prompts, conversations, outcomes) for one benchmark tier.
2. **Extract activations** (`extraction/`): Loads the model locally and extracts hidden-state activations at specified layers.
3. **Evaluate probes** (`reproduce.py`): Trains a deception probe on Core activations and evaluates five probing techniques across all three benchmark tiers.

## Data Format

Each scenario run directory contains:
- `run_config.json` — Scenario setup, agent roles, and coalition membership
- `results.json` — Full conversation transcripts, votes, and outcomes
- `agent_prompts.json` — Exact system and user prompts sent to each agent per round

## Citation

```bibtex
@article{rose2026narcbench,
  title={Detecting Multi-Agent Collusion Through Multi-Agent Interpretability},
  author={Rose, Aaron and Cullen, Carissa and Kaplowitz, Brandon Gary and Schroeder de Witt, Christian},
  year={2026}
}
```

## License

MIT
