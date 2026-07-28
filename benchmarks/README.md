# Voyager Benchmarks

Stage 5.6 freezes the Stage 5.5 ten-agent, 32x32, 300-step scenario and evaluates
three independently trained PPO policies on explicit held-out seed suites.

Run the development suite:

```bash
python examples/run_benchmark.py \
  --manifest benchmarks/manifests/stage5_6_dev.json \
  --output results/benchmark/stage5_6_dev_v1
```

Run or resume the final suite:

```bash
python examples/run_benchmark.py \
  --manifest benchmarks/manifests/stage5_6_final.json \
  --output results/benchmark/stage5_6_final_v1 \
  --resume
```

Deterministic PPO is the official learned-policy result. Stochastic PPO is reported
separately as a diagnostic. Generated episode-level artifacts remain under `results/`;
the final manifest, summary, and CSV tables are copied into `benchmarks/reference/`.
