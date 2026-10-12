//! # `prism_camera`
//!
//! Engine-neutral camera math for Prism's own renderer.
//!
//! This crate is the in-house replacement for the projection, view, and
//! frustum role that `bevy_camera` plays inside Bevy's render stack. It is
//! deliberately free of any engine, ECS, or graphics-backend type so the RHI
//! render path can own its camera model without a `bevy_*` dependency:
//! everything here is plain `f32` math over [`prism_math`].
//!
//! ## Layout
//! - [`Projection`] — perspective (vertical FOV / aspect / near / far) and
//!   orthographic (explicit bounds or symmetric size) clip-space matrices.
//! - [`View`] — a world-space pose inverted into a world-to-camera matrix.
//! - [`Camera`] — a [`View`] plus a [`Projection`], exposing
//!   [`Camera::view_projection`] and [`Camera::frustum`].
//! - [`Frustum`] — the six clip planes extracted from a view-projection with
//!   the Gribb–Hartmann method, plus sphere/AABB classification.
//! - [`planetest`] — the inside / outside / intersecting classification the
//!   [`Frustum`] queries are built on.
//!
//! ## Conventions
//! Matrices are column-major and multiply on the left (`m * v`). The camera
//! looks down `-Z` in view space with `+Y` up (right-handed), and the default
//! projections map clip-space depth to `[0, 1]` (Vulkan / D3D / Metal /
//! WebGPU), matching Prism's RHI.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

mod camera;
mod frustum;
mod projection;
mod view;

pub mod planetest;

pub use camera::Camera;
pub use frustum::Frustum;
pub use projection::{Orthographic, Perspective, Projection};
pub use view::View;

#[cfg(test)]
mod tests;
