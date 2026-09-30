//! Optional `wgpu` compute twin of Prism's volumetric-cloud scattering phase
//! functions.
//!
//! Cloud single scattering is driven by an anisotropic phase function that
//! biases light toward the forward direction (the silver-lining and glory
//! response). The `CPU` golden standard for that math lives in
//! [`prism_render_architecture::volumetric::scatter`]; this crate is the `GPU`
//! twin, validated against that reference so a passing real-device parity test
//! is direct evidence the ported kernel computes the same phase values as the
//! reference, not merely that its shader compiles.
//!
//! # Scope
//!
//! [`GpuPhaseEvaluator`] evaluates the full dual-lobe `HG`+`Draine` cloud phase
//! [`dual_lobe_draine_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_draine_phase),
//! which internally composes the `Henyey-Greenstein`, `Draine` and `HG`-`Draine`
//! sub-phases, so a single kernel covers the whole phase stack the ray-march
//! and multi-scatter stages consume.
//!
//! # Portability
//!
//! The phase algebra uses only `sqrt`, `min`, `max` and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The phase functions contain no transcendental call, so `CPU` and `GPU`
//! evaluate the same closed-form algebra. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`), tight enough to catch a genuinely
//! wrong port yet loose enough to admit legal fma contraction. See
//! [`phase`] for the full rationale.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Henyey-Greenstein` / `Draine` (`Jendersie` and
//! `d'Eon` 2023) dual-lobe cloud phase plus `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.
#![forbid(unsafe_code)]

pub mod context;
pub mod phase;

pub use context::{block_on, GpuContext};
pub use phase::{GpuPhaseEvaluator, PhaseQuery};
