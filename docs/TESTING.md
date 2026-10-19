# Validation guide — Stock-Market

## Commands

From the repository root, after installation:

~~~bash
python -m pytest -q
python -m compileall -q Main.py tests
~~~

## Beyond static checks

Predictions are experimental, not investment advice. No walk-forward evaluation, reproducible random seeds, live-trading support, execution engine, or production monitoring. Yahoo Finance data can change or become temporarily inaccessible. The comparison chart shows raw prices, not comparable normalized performance.

A passing CI run should be reported as the specific checks that passed; it is not proof of all end-to-end features or production security. Record exact reproduction instructions and the environment for failures.

## Pull request review checklist

- [ ] New tests exercise changed behavior and relevant edge cases.
- [ ] Manual verification is described honestly.
- [ ] External services, secrets and missing files are documented.
- [ ] No unrelated changes or data files are committed.
