//! `GPU` MLS-MPM (Moving Least Squares Material Point Method) kernels.
//!
//! This module ports the CPU golden MLS-MPM solver in
//! [`prism_physics_core::mpm`] to real-device `wgpu` compute pipelines. The
//! CPU path stays the numerical reference: every `GPU` kernel here has a
//! line-for-line CPU twin, and each kernel is anchored by a real-device parity
//! test that runs the shared WGSL against the host golden.
//!
//! The port is built in isolated slices to keep the highest-risk arithmetic
//! (the sqrt-based `SVD` and the constitutive model) verifiable on its own
//! before it is fused into the transfer pipeline:
//!
//! - [`gpu::GpuMpmConstitutive`] — the constitutive probe: fixed-corotated
//!   stress `P Fᵀ`, the polar rotation `R`, and the snow return-mapping,
//!   evaluated per particle in isolation.
//!
//! # Provenance
//!
//! The MLS-MPM transfer (Hu et al. 2018), the affine `P2G`/`G2P` conventions
//! (Jiang et al. 2015), and the fixed-corotated / snow plasticity model
//! (Stomakhin et al. 2013) are standard, publicly documented techniques. No
//! Unreal Engine source or derived code.

pub mod gpu;

pub use gpu::{ConstitutiveOutput, GpuMpmConstitutive};
