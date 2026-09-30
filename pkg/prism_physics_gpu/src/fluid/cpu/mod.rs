//! The `CPU` golden twin of the `GPU` `FLIP`/`APIC` fluid transfers.
//!
//! These modules perform the identical fixed-point scatter, normalisation, and
//! trilinear gather as the device kernels in `shaders/fluid_transfer.wgsl`, so a
//! passing real-device parity test is direct evidence the ported kernel matches
//! the reference. The integer momentum/weight accumulation is exact and
//! order-independent, so the accumulators agree bit-for-bit; only the final
//! division and the gather are floating point, bounded by a tight tolerance.
//!
//! # Provenance
//!
//! Trilinear `P2G`/`G2P` with the `PIC`/`FLIP` blend follows Zhu and Bridson
//! 2005, Bridson, and (for the affine path) Jiang et al. 2015. No Unreal Engine
//! source or derived code.

pub mod fields;
pub mod g2p;
pub mod grid_ops;
pub mod p2g;
pub mod pressure;
pub mod stencil;

pub use fields::GoldenGrid;
pub use g2p::grid_to_particle;
pub use grid_ops::{add_gravity, enforce_solid_faces};
pub use p2g::particle_to_grid;
pub use pressure::PressureConfig;
