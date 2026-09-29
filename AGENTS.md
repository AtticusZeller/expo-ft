# EXPO-FT Agent Guide

Read `README.md` and the relevant source before changing this repository. Keep changes scoped to EXPO-FT or Real-Time EXPO-FT as requested; the two paths share code but have different control and training assumptions.

The learner and actor use separate Python environments. The learner uses the root `pyproject.toml`; the actor uses `client/pyproject.toml`. Both require the matching OpenPI and DROID checkouts described in `README.md` before dependency installation or execution.

Do not treat a source checkout as evidence that training, policy evaluation, or robot control works in the current environment. Report checks actually run and the hardware, data, and checkpoints that were available.
