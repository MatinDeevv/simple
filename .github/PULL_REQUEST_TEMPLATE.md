# Pull request

## Summary

Brief description of the change and the research or engineering problem it addresses.

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / performance
- [ ] Quantum archive update
- [ ] Breaking change

## Checklist

- [ ] `python -m pytest tests -q` passes locally.
- [ ] `python -m engine.tools.verify_repository --tree head` passes.
- [ ] No raw market data, secrets, or generated artifacts are committed.
- [ ] Quantum or residual-trading changes do not affect the classical state schema, integrator, controller, or trading path.
- [ ] Documentation is updated if public APIs or contracts changed.

## Research note

If this touches models or evaluation, state the predeclared target, comparator, and holdout plan.
