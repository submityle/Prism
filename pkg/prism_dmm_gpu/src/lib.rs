//! Real-device `GPU` twin of the Prism displaced-micro-map (`DMM`) baker.
//!
//! This crate ports the device-free `CPU` golden in `prism_dmm` to four `wgpu`
//! compute kernels and validates them element for element against that golden
//! on a real device. Baking a displaced micro-map tessellates a base triangle
//! into a regular micro-mesh and stores a per-micro-**vertex** scalar
//! displacement, which a trace- or raster-time evaluator later pushes along an
//! interpolated displacement direction to turn a cheap flat triangle into dense
//! geometric detail.
//!
//! The bake is four dispatches, each mirroring one stage of
//! [`bake_triangle`](prism_dmm::bake_triangle):
//!
//! 1. a per-micro-vertex *sample* pass that decodes the canonical lattice
//!    index, interpolates the micro-vertex `UV`, and bilinearly samples the
//!    displacement height texture;
//! 2. a single-invocation *reduce* pass that scans the sampled heights for the
//!    per-triangle `[min, max]` range (or copies a caller-provided fixed
//!    range);
//! 3. a per-micro-vertex *quantize* pass that normalises each height into
//!    `[0, 1]` and rounds it to an `11-bit` unorm code; and
//! 4. a per-output-word *pack* pass that folds the codes into the raw
//!    little-endian `11-bit` displacement bitstream.
//!
//! Acquisition is best-effort: [`GpuContext::try_headless`] returns [`None`] on
//! a host without a usable adapter so the parity test skips cleanly rather than
//! failing, while running the full dispatch on any machine with a real device.
//!
//! # Provenance
//!
//! The displaced-micro-map concept and `DXR` / `VK_EXT_displacement_micromap`
//! `11-bit` unorm layout are open hardware specifications; the sampling,
//! quantization, and packing here are classical barycentric interpolation,
//! bilinear filtering, round-to-nearest quantization, and bit manipulation. No
//! Unreal Engine source or derived code is used, and nothing in this crate uses
//! AI, machine learning, or neural methods.

pub mod bake;
pub mod buffer;
pub mod context;

pub use bake::{GpuBakedDmm, GpuDmmBaker};
pub use context::GpuContext;
