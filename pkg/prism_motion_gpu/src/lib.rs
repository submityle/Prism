//! Optional `wgpu` compute backend for Prism's motion-vector dilation.
//!
//! Temporal rendering (`TAA`, motion blur, temporal upsamplers, temporal
//! denoisers) reads a per-pixel screen-space velocity field. Thin, fast-moving
//! foreground silhouettes tear unless the foreground's motion vector is
//! "bled" one or two pixels outward so edge pixels reproject along the occluder
//! rather than the background they briefly cover. The classic fix is
//! **closest-depth dilation**: each output pixel adopts the velocity of the
//! strictly-nearest neighbor inside a small clamped square. The complete,
//! device-free reference — the field types, the tie-breaking, and the
//! edge-clamping semantics — lives in
//! [`prism_render_architecture::motion::dilation`].
//!
//! This crate is the real-device half of that stage: a compute kernel where one
//! invocation produces one output pixel by scanning its clamped neighborhood in
//! the identical row-major order and replacing the incumbent only when a
//! neighbor is *strictly* closer under the active [`DepthOrder`]. Because the
//! kernel does nothing but depth comparisons and whole-vector copies — it never
//! does arithmetic on a velocity — the device output equals the golden
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
//! bit-for-bit.
//!
//! # Exact parity
//!
//! Every output velocity is a verbatim copy of some input velocity, and the
//! only derived quantity is the depth comparison, which uses the same strict
//! inequality in the same scan order as the golden. There is therefore a single
//! deterministic winner per pixel and no floating-point rounding enters the
//! chosen value, so the parity tests assert exact equality (`f32::to_bits` per
//! component) rather than a tolerance. The golden `dilate_closest_depth` is the
//! authority the device output is checked against directly.
//!
//! Provenance: the closest-depth dilation technique is standard temporal-AA
//! practice (von Reeuwijk/Karis-era TAA, documented widely). Classical,
//! data-oblivious integer-indexed gather with float comparisons; no neural,
//! learned, or data-driven components. No Unreal Engine source or derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod dilation;

pub use context::GpuContext;
pub use dilation::GpuDilate;
