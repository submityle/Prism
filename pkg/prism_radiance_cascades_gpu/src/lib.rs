//! Optional `wgpu` compute backend for Prism's Radiance Cascades global
//! illumination.
//!
//! Radiance cascades compute diffuse/low-gloss GI by sampling radiance on a
//! hierarchy of probe grids, trading spatial for angular resolution as the
//! radial interval grows, then merging the shells front-to-back into a single
//! full-range directional field at cascade 0. The deterministic contract for
//! that solve — hierarchy dimensioning, the `over` interval composite, the
//! bilinear/angular merge, and the directional resolve — lives device-free in
//! [`prism_render_architecture`](prism_render_architecture::lighting::radiance_cascades).
//!
//! This crate is the real-device half: three compute kernels
//! (`gather`/`merge`/`resolve`) that reproduce that CPU golden on a `GPU`.
//! Device acquisition is best-effort via [`GpuContext::try_headless`], so the
//! parity tests exercise a real Apple `M`-series (or other native) `GPU` when
//! present and skip cleanly otherwise.
//!
//! # Float parity
//!
//! A `GPU` twin can only match a `CPU` golden bit-for-bit when every operation
//! is identical. Two deliberate design choices keep the two sides in agreement:
//!
//! 1. **Directions are precomputed on the host** with the golden's exact
//!    `bin_angle` + [`glam::Vec2::from_angle`] and uploaded as a buffer, so the
//!    shader never evaluates its own trigonometry and both sides consume
//!    byte-identical direction vectors.
//! 2. **The scene sampler is a pure `+ - * /` rational** of the probe origin,
//!    direction, and interval bounds — no `sin`/`cos`/`exp` — so the only
//!    residual divergence is a possible fused multiply-add on the device. The
//!    parity tests therefore compare with a small relative/absolute tolerance
//!    rather than exact equality.
//!
//! Provenance: the hierarchy/`over`/merge/resolve algorithm is Alexander
//! Sannikov's *Radiance Cascades* (2023); compositing is Porter–Duff "over"
//! (1984). Clean-room classical implementation with no neural, learned, or
//! data-driven components. No Unreal Engine source or derived code.

pub mod buffer;
pub mod context;
pub mod solve;

pub use context::GpuContext;
pub use solve::{GpuInterval, GpuRadianceCascades};
