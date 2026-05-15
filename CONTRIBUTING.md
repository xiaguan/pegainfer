# Contributing to pegainfer

Thanks for your interest in contributing! This guide will get you from "I want to help" to "my PR is in review."

## Quick Links

- **Build & Run** → [CLAUDE.md](CLAUDE.md#build--run)
- **Onboarding** → [docs/resources/developer-onboarding.md](docs/resources/developer-onboarding.md)
- **Coding Style** → [docs/areas/coding-style.md](docs/areas/coding-style.md)

## Branch Naming

Use a prefix that describes the change type:

| Prefix | Example |
|--------|---------|
| eat/ | eat/add-llama-support |
| ix/ | ix/softmax-overflow |
| perf/ | perf/optimize-kv-cache |
| docs/ | docs/add-benchmark-guide |
| chore/ | chore/update-deps |

## Commit Format

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) conventions:

`
type(scope): description

feat(qwen3): add streaming support
fix(scheduler): fix token event ordering
perf(cuda): fuse attention kernels
docs(api): add curl examples
`

## Running Tests

`ash
# Fast unit tests (~9s)
cargo test --release --workspace --lib

# E2E tests (requires GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test e2e
`

**Always use --release** — debug builds are too slow for GPU/CUDA code.

## PR Checklist

Before opening a PR:

- [ ] Tests pass (cargo test --release --workspace --lib)
- [ ] Commit messages follow Commitizen format
- [ ] No debug prints or commented-out code
- [ ] New public APIs have doc comments
- [ ] If changing numerical output, update test baselines in 	est_data/

## Questions?

Open an issue with the question label or start a Discussion.