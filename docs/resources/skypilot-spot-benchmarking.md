# SkyPilot Spot Benchmarking

> **TL;DR:** Use SkyPilot to grab a single spot GPU, sync `pegainfer`, run `scripts/run_snapshot_benchmark.sh` on the remote machine, `rsync` back only the snapshot JSON, then `sky down` the cluster immediately. PrimeIntellect spot catalog entries are not inventory guarantees; expect retries.
>
> **Status:** Active. Next step: reuse this flow for the next GPU baseline instead of rediscovering the bootstrap steps.

---

## Scope

This is the operational playbook for running `bench_serving snapshot` on a rented spot GPU with SkyPilot. It is for the "fresh machine, no preinstalled Rust or uv, maybe missing submodules" case, not for a developer laptop.

The benchmark logic itself is documented in [bench-regression](../areas/bench-regression.md). This doc covers the machine lifecycle around it.

## What Was Verified

Verified on `2026-04-02` against PrimeIntellect spot inventory:

- `RTX6000Ada:1` spot was listed by `sky gpus list` but twice failed to launch with "available right now" inventory errors.
- `L40S:1` spot was also listed but failed the same way.
- `A100-80GB:1` spot eventually launched and completed the workflow.
- `sky exec` hit a PrimeIntellect status-query bug after the cluster was up, so direct `ssh` was used as the fallback control path.

Resulting baseline:

- GPU: `NVIDIA A100-SXM4-80GB`
- Snapshot: `bench_snapshots/a100-sxm4-80gb/qwen3-4b.json`

## Launch Pattern

Use a single-node interactive cluster, not a managed job. The point is to keep control while bootstrapping the remote environment.

Verified command shape:

```bash
.venv/bin/sky launch \
  -c pega-qwen3-4b-a100-80g-1gpu \
  --infra primeintellect \
  --gpus A100-80GB \
  --use-spot \
  --retry-until-up \
  --workdir /Users/mac/code/sky/pegainfer \
  --yes \
  -- nvidia-smi
```

Why this shape:

- `--retry-until-up` matters for spot. Catalog visibility does not mean immediate inventory.
- `--workdir` gets the repo onto the remote machine as `~/sky_workdir`.
- The inline `nvidia-smi` is just a cheap first job to verify the instance is real and the driver stack is usable.

## PrimeIntellect Gotchas

### 1. Listing is not inventory

This command only shows that the SKU exists in the catalog:

```bash
.venv/bin/sky gpus list --infra primeintellect -o json
```

It does **not** mean the spot VM can be provisioned right now.

Two observed failures:

- `datacrunch doesn't have 1RTX6000ADA.10V_SPOT available right now`
- `datacrunch doesn't have 1L40S.20V_SPOT available right now`

If the requested spot SKU matters, leave the launcher on `--retry-until-up`. Do not assume one failure means the SKU is unsupported.

### 2. `sky exec` may fail after the cluster is up

Observed failure against PrimeIntellect:

```text
query_instances() got an unexpected keyword argument 'retry_if_missing'
```

If that happens, use plain `ssh` with the SkyPilot-managed key and continue from there.

## Remote Bootstrap

Once the machine is reachable, switch to SSH and run the repo-local script:

```bash
ssh -T -i /Users/mac/.sky/clients/6f2ae57f/ssh/sky-key root@<remote-ip> \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  'cd ~/sky_workdir && HF_TOKEN=... ./scripts/run_snapshot_benchmark.sh'
```

What the script does:

- installs missing system packages with `apt-get`
- installs Rust via `rustup` if needed
- installs `uv` if needed
- initializes `third_party/flashinfer` submodules if missing
- creates `.venv` and installs `triton` plus `huggingface_hub`
- downloads `Qwen/Qwen3-4B`
- builds `bench_serving`
- runs `snapshot --warmup 5 --iters 20`

Script entrypoint:

```bash
HF_TOKEN=... ./scripts/run_snapshot_benchmark.sh
```

## Why The Script Exists

The remote machine used in the verified run was missing all of the following:

- `cargo`
- `rustc`
- `rustup`
- `uv`
- `pip3`

Also, the initial `workdir` sync did **not** include populated git submodules, so the first build failed on:

```text
flashinfer/norm.cuh: No such file or directory
```

That is why the script explicitly runs:

```bash
git submodule update --init --recursive
```

before building.

## Copy Back Only The Result

Do not sync the model or `target/` back to the local repo. Only pull the benchmark JSON:

```bash
rsync -av \
  -e 'ssh -i /Users/mac/.sky/clients/6f2ae57f/ssh/sky-key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes' \
  root@<remote-ip>:/root/sky_workdir/bench_snapshots/a100-sxm4-80gb/qwen3-4b.json \
  /Users/mac/code/sky/pegainfer/bench_snapshots/a100-sxm4-80gb/
```

This keeps the local repo small and preserves the benchmark artifact that matters for regression tracking.

## Cleanup

Always tear the spot VM down immediately after the artifact is copied back:

```bash
.venv/bin/sky down -y pega-qwen3-4b-a100-80g-1gpu
```

If a failed launch leaves a cluster stuck in `INIT`, retry `sky down`. Some providers reject deletion while the VM is still in their internal "creating" state; waiting briefly and retrying is normal.

## Recommended Flow

1. Pick the spot SKU you actually care about.
2. Launch with `--retry-until-up`.
3. Let SkyPilot sync the repo with `--workdir`.
4. Use `ssh` if `sky exec` is blocked by provider-specific status bugs.
5. Run `HF_TOKEN=... ./scripts/run_snapshot_benchmark.sh`.
6. `rsync` back only `bench_snapshots/<gpu-slug>/<model>.json`.
7. `sky down` the cluster immediately.

## Related Docs

- [bench-regression](../areas/bench-regression.md) for snapshot semantics and thresholds
- [developer-onboarding](./developer-onboarding.md) for the non-SkyPilot local setup path
- [bench-vs-vllm](./bench-vs-vllm.md) for the comparative benchmark workflow
