//! Temporal resolve toolkit (TAA-grade): motion-vector reprojection, bicubic
//! Catmull-Rom history sampling, neighborhood color clipping (AABB / variance),
//! and disocclusion detection shared by TAA and temporal GI accumulation.
//!
//! # Conventions
//! * Color-space clamping operates in a luma-chroma basis (YCoCg) to limit ghosting
//!   while preserving chroma; all weights are energy-preserving where documented.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).
//!
//! * [`reproject`] — motion-vector reprojection, bounds classification, and
//!   depth/normal/velocity disocclusion detection plus clamp-to-edge bilinear /
//!   nearest history fetch封装.
//! * [`catmull_rom`] — sharp bicubic Catmull-Rom history resampling with the
//!   Jimenez 5-tap optimization and optional ringing (negative-lobe) clamping.
//! * [`clip`] — YCoCg transform, 3x3 neighborhood min/max + variance (mean±γσ)
//!   bounding boxes, ray-based history clipping, and luma-weighted blending.

pub mod catmull_rom;
pub mod clip;
pub mod reproject;
