//! Real-device `GPU` twin of the Prism opacity-micromap (`OMM`) baker.
//!
//! This crate ports the device-free `CPU` golden in `prism_micromap` to two
//! `wgpu` compute kernels — a per-micro-triangle tri-state classifier and a
//! per-word `DXR` bit packer — and validates them element for element against
//! that golden on a real device. Baking an opacity micromap classifies each
//! micro-triangle of a base triangle as transparent, opaque, or (mixed)
//! unknown so a ray-tracing pipeline can skip the any-hit shader on the
//! unambiguous micro-triangles.
//!
//! Acquisition is best-effort: [`GpuContext::try_headless`] returns [`None`] on
//! a host without a usable adapter so the parity test skips cleanly rather than
//! failing, while running the full dispatch on any machine with a real device.
//!
//! # Provenance
//!
//! The opacity-micromap concept and `DXR` / `VK_EXT_opacity_micromap` bit
//! layout are open hardware specifications; the classification and packing here
//! are classical conservative coverage sampling and bit manipulation. No Unreal
//! Engine source or derived code is used, and nothing in this crate uses AI,
//! machine learning, or neural methods.

pub mod bake;
pub mod buffer;
pub mod context;

pub use bake::{GpuBakedOmm, GpuOmmBaker};
pub use context::GpuContext;
