# Project: Runtime Complexity Paydown

**Status**: Active
**TL;DR**: pegainfer now has enough model-specific runtime branching that complexity itself is becoming a delivery risk. This project exists to reduce fragmentation before support for more non-standard attention families expands further.

---

## Goal

Bring runtime complexity back under control before the next wave of model-family support.

The point is not abstraction for its own sake. The point is to make future model integration more predictable, keep the serving path coherent, and avoid growing a separate runtime for every new attention pattern.

## Focus

- reduce runtime branching across model families
- stabilize the core runtime contracts that serving depends on
- lower the marginal cost of integrating the next non-standard attention model

## Success

- new model-family work starts from a shared runtime foundation, not a fresh execution stack
- runtime behavior is easier to reason about across prefill, decode, state, and serving
- framework debt stops compounding faster than model coverage

## Next Action

Identify the runtime surfaces where fragmentation is already slowing delivery, then turn that into a small, ordered payoff plan.
