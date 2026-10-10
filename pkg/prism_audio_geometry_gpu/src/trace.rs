//! Host-side mirror of the ray primitives both acoustics shaders share.
//!
//! The direct and reflection kernels each embed the same handful of helpers in
//! `WGSL`: a unit-quaternion rotation, a Moller-Trumbore ray/triangle test, and
//! a brute-force nearest-hit scan over the scene. This module reproduces that
//! arithmetic on the host so each kernel's `CPU` twin
//! ([`crate::direct::cpu_direct`], [`crate::reflection::cpu_reflection`]) is a
//! line-for-line port of the device code and the parity tests have something
//! exact to compare against.
//!
//! Determinism matters here: the twin must track the device within a tight
//! tolerance, so every square root goes through [`bevy_math::ops::sqrt`] and the
//! vector algebra uses [`bevy_math::Vec3`], matching the built-in `sqrt`,
//! `dot`, and `cross` the shaders call.
//!
//! # Provenance
//!
//! Original work; the classic Moller-Trumbore ray/triangle test and a
//! brute-force nearest-hit scan already implemented on the `CPU` in
//! [`prism_audio_geometry`]; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumed by [`crate::direct`] and [`crate::reflection`] to port their
//! `WGSL` entry points to the host for the golden comparison in
//! [`crate::backend`].

use bevy_math::Vec3;

use crate::scene_upload::GpuTriangle;

/// Ray/triangle determinant floor; mirrors `TRI_EPSILON` in both shaders.
pub(crate) const TRI_EPSILON: f32 = 1.0e-7;

/// Coincident-source distance floor; mirrors `COINCIDENT` in both shaders and
/// the `1e-6` guard in `Listener::localize`.
pub(crate) const COINCIDENT: f32 = 1.0e-6;

/// The result of a nearest-hit scan over the scene triangles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Hit {
    /// Whether any triangle was hit within the queried distance.
    pub(crate) valid: bool,
    /// Ray parameter (metres along a unit direction) of the nearest hit.
    pub(crate) t: f32,
    /// Index of the nearest hit triangle in scene order.
    pub(crate) index: usize,
}

/// Reads the first three lanes of a padded device vector as a [`Vec3`].
#[must_use]
pub(crate) fn vec3(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Conjugate of a unit quaternion `[x, y, z, w]`, which equals its inverse.
#[must_use]
pub(crate) fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by the quaternion `q` (`[x, y, z, w]`), matching the shader's
/// `quat_rotate` and glam's quaternion-times-vector arithmetic.
#[must_use]
pub(crate) fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let b = Vec3::new(q[0], q[1], q[2]);
    let w = q[3];
    let b2 = b.dot(b);
    v * (w * w - b2) + b * (v.dot(b) * 2.0) + b.cross(v) * (w * 2.0)
}

/// Moller-Trumbore ray/triangle test, double-sided.
///
/// Returns the ray parameter `t` when the ray from `origin` along the unit
/// `dir` strikes triangle `a`/`b`/`c` within `[0, max_distance]`, mirroring the
/// shader's `ray_triangle` and the `CPU` `ray_cast` arithmetic.
#[must_use]
pub(crate) fn ray_triangle(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    max_distance: f32,
) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = dir.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < TRI_EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    if t >= 0.0 && t <= max_distance {
        Some(t)
    } else {
        None
    }
}

/// Nearest triangle the ray hits no farther than `max_distance`, scanning every
/// triangle in index order with a strict `<` tie-break to the lowest index.
///
/// Mirrors the shader's `first_hit`, which loops over `params.triangle_count`;
/// the caller passes the real triangle slice so the host bound matches.
#[must_use]
pub(crate) fn first_hit(
    triangles: &[GpuTriangle],
    origin: Vec3,
    dir: Vec3,
    max_distance: f32,
) -> Hit {
    if max_distance <= 0.0 {
        return Hit {
            valid: false,
            t: max_distance,
            index: 0,
        };
    }
    let mut found = false;
    let mut best_t = max_distance;
    let mut best_index = 0usize;
    for (index, tri) in triangles.iter().enumerate() {
        let hit = ray_triangle(
            origin,
            dir,
            vec3(tri.a),
            vec3(tri.b),
            vec3(tri.c),
            max_distance,
        );
        if let Some(t) = hit
            && (!found || t < best_t)
        {
            found = true;
            best_t = t;
            best_index = index;
        }
    }
    Hit {
        valid: found,
        t: best_t,
        index: best_index,
    }
}
