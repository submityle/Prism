//! Optional `wgpu` compute twin for Prism's §24.1 CPU/GPU math consistency
//! contract (the *shader mirror*).
//!
//! The portable, device-free half of the contract lives in
//! [`prism_math::shader_mirror`]: the `std140` byte-layout packers, the CPU
//! reference op [`quat_rotate_vec3`](prism_math::shader_mirror::quat_rotate_vec3),
//! and the single-sourced WGSL fragment
//! [`WGSL_QUAT_ROTATE`](prism_math::shader_mirror::WGSL_QUAT_ROTATE). This crate
//! is the *real-device* half: it uploads a batch of vectors, dispatches a
//! compute kernel whose rotation function **is** `WGSL_QUAT_ROTATE` (composed at
//! runtime, never re-typed), reads the result back, and the parity tests diff
//! the GPU output against the CPU reference.
//!
//! # Why a tolerance, not bit-exactness
//!
//! The kernel does floating-point arithmetic. Metal compiles WGSL under
//! fast-math, so the shader compiler may contract `a + b * c` into one `fma`
//! and reassociate, rounding differently from the CPU's separate multiply and
//! add. The §24.1 contract is therefore defined as a *tolerance* round-trip:
//! the two sides must agree within a small absolute+relative epsilon, which is
//! exactly what rejects a genuine algorithm/operand-order/layout drift while
//! tolerating last-ULP FMA rounding.
//!
//! # Graceful skip
//!
//! [`GpuContext::try_headless`] returns [`None`] on a host with no usable
//! adapter so the parity suite skips rather than fails on a device-less CI
//! image, while running the full dispatch on any real `GPU`.
//!
//! Provenance: standard `wgpu` compute orchestration; the quaternion rotation
//! is the classic `v + 2w(q x v) + 2 q x (q x v)` form. No neural, learned, or
//! data-driven components. No Unreal Engine or Unity source or derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod easing;
pub mod f16;
pub mod fractal;
pub mod frustum_cull;
pub mod hilbert;
pub mod hsl;
pub mod morton;
pub mod octahedral;
pub mod oklab;
pub mod overlap;
pub mod pack16;
pub mod pack8;
pub mod perlin;
pub mod projection;
pub mod quat;
pub mod quat_interp;
pub mod raycast;
pub mod raytri;
pub mod sh;
pub mod simplex;
pub mod skinning;
pub mod spline;
pub mod srgb;
pub mod surface;
pub mod temperature;
pub mod view;
pub mod xyz;

pub use context::{block_on, GpuContext};
pub use easing::{Ease, GpuEasing};
pub use f16::GpuF16Pack;
pub use fractal::{GpuFractal, NoiseSource};
pub use frustum_cull::GpuFrustumCull;
pub use hilbert::GpuHilbert;
pub use hsl::GpuHsl;
pub use morton::GpuMorton;
pub use octahedral::GpuOctahedral;
pub use oklab::GpuOklab;
pub use overlap::GpuOverlap;
pub use pack16::GpuPack16;
pub use pack8::GpuPack8;
pub use perlin::GpuPerlin;
pub use projection::{GpuProjection, ProjectionKind};
pub use quat::GpuQuatRotate;
pub use quat_interp::GpuQuatInterp;
pub use raycast::{GpuAabb, GpuRay, GpuRayCast, GpuRayHit};
pub use raytri::{GpuRayTri, GpuTri, GpuTriHit};
pub use sh::{GpuSh3Eval, SH3_COEFFS};
pub use simplex::GpuSimplex;
pub use skinning::{GpuDualQuatSkin, Influence, MAX_INFLUENCES};
pub use spline::{GpuSpline, Spline, SplineSample};
pub use srgb::GpuSrgbTransfer;
pub use surface::{GpuSurface, Surface};
pub use temperature::GpuTemperature;
pub use view::{GpuView, ViewKind};
pub use xyz::GpuXyz;
