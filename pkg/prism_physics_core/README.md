# Prism Physics Core

Backend-neutral core kernel for Prism's next-generation physics engine.

This crate is engine-agnostic: it does **not** depend on `bevy_ecs` or any
renderer, so it can be reused by servers, tools, and headless simulations.
It deliberately contains **no Unreal Engine source or derived code**; where UE
algorithms are referenced they are re-implemented from published descriptions.

## Core architecture

- `math`: scalar configuration (`Real`), transform/isometry helpers.
- `state`: Structure-of-Arrays (SoA) rigid-body storage with generational handles.
- `dynamics`: semi-implicit Euler free-body integration (real, tested).
- `collider`: analytic collider shapes and a shared shape registry.
- `world`: `PhysicsWorld` bundling storage, colliders and configuration.
- Runtime extension surfaces:
  - `solver`: `Solver` trait + `SolverRegistry` + `IntegrateOnlySolver`.
  - `backend`: `PhysicsBackend` trait + `CpuBackend` reference backend.
  - `driver`: `DriveMode` (Realtime / Offline / Playback) selector.
  - `island`: `IslandSet` + union-find `IslandBuilder`.

XPBD constraint solving and its contact, joint, rigid-body, and soft-body
specializations are implemented in their dedicated modules.
