# DSV4 Overlap Compressor GEMM Decision Candidate

This note records the decision PR for replacing the strict reduction-order
overlap compressor prefill projection with a cuBLAS-backed GEMM projection plus
a route-combine kernel.

## Scope

This branch is a decision candidate, not a strict-hash-ready production PR.

It changes only the overlap compressor prefill path:

- `compressor_overlap_prefill_bf16_hidden_with_dim` allocates F32 projection
  scratch buffers for `x @ wkv.T` and `x @ wgate.T`.
- `deepseek_compressor_overlap_prefill_projected_cuda` computes both
  projections with BF16 input/weights and F32 output, then combines the 8
  overlap routes and runs the existing F32 weighted RMS normalization.
- The old `deepseek_compressor_overlap_prefill_cuda` path remains in the tree
  as the strict-output reference.

It does not change decode compressor, indexer score/top-k, scheduler behavior,
HTTP harness, cache reuse, or collective placement.

## Measured Benefit

All numbers below are from the same 10k synthetic prefill workload used in the
DSV4 prefill tuning thread:

- model: `/data/DeepSeek-V4-Flash`
- prompt tokens: `10580`
- output tokens: `1`
- seed: `42`

Baseline after PR #119:

- 10k direct hash: `39a863e299d2b187`
- 10k TTFT: about `6513-6530ms`
- overlap compressor bucket: about `20575ms / 504 calls`

GEMM/CuTe projection prototype:

- 10k direct hash: `39a863e299d2b187`
- 10k TTFT: about `4034ms`
- old overlap compressor weighted bucket disappears from the top profile;
  the replacement shows GEMM projection plus
  `deepseek_compressor_overlap_combine_projected_kernel`
- route-combine bucket: about `7.43ms / 504 calls`

Replacement cost:

| Component | Calls | Total time |
| --- | ---: | ---: |
| BF16 GEMM projection, main overlap shape | 336 | `130.29ms` |
| BF16 GEMM projection, indexer overlap shape | 672 | `68.51ms` |
| Route-combine kernel | 504 | `7.43ms` |

The measured replacement kernel cost for the two GEMM projection families plus
route combine is about `206.23ms` in the 10k capture. The `1008` GEMM calls are
expected: each overlap compressor call performs two projections, `x @ wkv.T`
and `x @ wgate.T`, and the profile separates the main-compressor and
indexer-compressor shapes.

The Rust-side projected path also adds two F32 projection scratch buffers at the
call site:

- main overlap (`head_dim=512`): two buffers of
  `10580 * 1024 * 4 = 43,335,680` bytes each, about `82.66 MiB` total per rank;
- indexer overlap (`head_dim=128`): two buffers of
  `10580 * 256 * 4 = 10,833,920` bytes each, about `20.66 MiB` total per rank.

The existing weighted/output buffers are not new. The two F32 projection buffers
above are the new memory pressure introduced by this candidate.

The structural performance signal is strong enough to justify a product-level
decision on numeric tolerance. It is not evidence that the branch is ready to
become the default path under the current strict output-hash gate.

## Drift Evidence

The projected path changes accumulation order. The old path reduces each
`(compressed, dim, route)` projection with the current block-reduction tree.
The projected path uses GEMM accumulation for the same dot products.

Observed drift:

- Small representative shape already shows score bit drift:
  `0.6163566` versus `0.61635655`, max score absolute difference about `6e-8`.
- 10k representative shape shows score max absolute difference about
  `1.073e-6` and value max absolute difference about `2.98e-7`.
- Full weighted output first bit difference appears at index `1`, weighted
  max absolute difference about `7.153e-6`.
- Normalized BF16 output first difference appears at index `2735`, with output
  max absolute difference `0.0078125`.

Hash gate:

- Direct short hash remains `6346f03343d75a65`.
- 10k direct hash remains `39a863e299d2b187`.
- HTTP c1/c2/c4/c8 repeated sweep is stable but changes combined hash from
  main `22706877075acde0` to `097097877d134a88`.

The drift is deterministic and localized to projection accumulation order, but
it does change streamed HTTP output under the current hash gate.

## Decision Needed

The engineering question is no longer whether the GEMM/CuTe-shaped projection
can be faster. The open decision is whether DSV4 can accept this class of tiny
numeric drift for a large prefill improvement.

Accepting it requires replacing strict output hash as the only gate for this
path with a quality golden set, such as:

- fixed prompt suite with token-sequence diffs;
- logprob/top-k diffs around the first divergence;
- long-prompt quality examples;
- repeated HTTP c1/c2/c4/c8 stability under the new expected outputs;
- explicit owner signoff for the new tolerance policy.

If this drift is not accepted, keep `deepseek_compressor_overlap_prefill_cuda`
as the default overlap compressor path and do not merge this candidate branch.
