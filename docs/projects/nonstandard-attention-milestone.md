# Milestone: Non-Standard Attention Focus

**Status**: Active
**TL;DR**: pegainfer is focused on inference and serving for non-standard attention models, especially linear attention and sparse attention. This milestone prioritizes model-family readiness, service experience, framework debt repayment, and disciplined correctness/performance evaluation.

---

## Goal

pegainfer should become a strong runtime and serving foundation for non-standard attention models.

The primary battlefield is:

- linear attention
- sparse attention
- hybrid variants that mix dense and non-dense attention patterns

This milestone is not about being a general-purpose LLM serving stack. It is about being good at the class of models that will matter more as attention patterns diversify.

## Direction

- Focus on model-family readiness, not one-off wins on a single model.
- Keep service experience strong: TTFT, TPOT, p99, startup, and correctness should stay competitive for linear-attention models.
- Repay framework-level complexity so new non-standard attention models do not require a fresh end-to-end runtime each time.
- Treat evaluation as a first-class system: performance, correctness drift, and serving behavior must be measured together.

## What Good Looks Like

- A new linear or sparse attention model can be integrated with predictable effort.
- Serving quality for non-standard attention models is stable and credible, not just fast in a narrow benchmark.
- The codebase gets more reusable as model diversity increases, instead of more fragmented.

## Next Action

Turn this milestone into a short execution plan with:

- the target model families to support next
- the service metrics that define acceptable experience
- the framework debt that must be repaid before model coverage expands further
