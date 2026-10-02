//! `CPU` mirror of the surfel spatial-filter producer kernel.
//!
//! [`filter_center`] is the scalar twin of `surfel_spatial_filter_main` in
//! `shaders/surfel_spatial_filter.wesl`: given the dispatch
//! [`GpuSpatialFilterParams`], one [`GpuSpatialCenter`] and the shared
//! [`GpuSpatialNeighbor`] buffer it reduces the centre's neighbour slice into a
//! filtered [`GpuSpatialResult`] exactly as the kernel does. It is a *faithful
//! op-for-op transcription* of the shader rather than a call into the
//! [`spatial_filter`](crate::gi::surface_cache::integration::spatial_filter)
//! golden, so the parity test it anchors is an independent cross-check that
//! fails if either the shader or the golden drifts.
//!
//! # Bit-exact parity
//!
//! The geometric weight is radial coverage x off-plane falloff x normal
//! agreement; the transcendentals go through [`bevy_math::ops`] (`exp` and
//! `powf`), the exact `libm` routines the golden uses, so the mirror and the
//! golden agree bit-for-bit. The per-centre reduction sums the centre (unit
//! weight) and then each kept neighbour in buffer order — the identical order
//! and associativity as the golden's `for` loop — so no reordering can perturb
//! the running `f32` sum. `sanitize_rgb` is idempotent, matching the golden.
//!
//! Provenance: standard bilateral surfel reuse driven by a geometric weight; no
//! Unreal Engine source or derived code.

use bevy_math::{ops, Vec3};

use crate::gi::surface_cache::gpu::abi::{
    GpuSpatialCenter, GpuSpatialFilterParams, GpuSpatialNeighbor, GpuSpatialResult,
};

/// Smallest surfel radius; keeps the axial falloff division finite. Mirrors
/// `MIN_RADIUS` in the surfel golden and the shader.
const MIN_RADIUS: f32 = 1.0e-6;

/// Replace a non-finite scalar with zero (golden `finite_or_zero`).
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Sanitise a linear-RGB triple finite and non-negative (golden `sanitize_rgb`).
#[inline]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(c.x),
        finite_or_zero(c.y),
        finite_or_zero(c.z),
    )
    .max(Vec3::ZERO)
}

/// Normalise a vector, falling back to `+Z` for zero-length / non-finite input,
/// mirroring the golden `safe_normalize`.
#[inline]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Numerically safe `exp` saturating for large magnitudes (golden `stable_exp`).
#[inline]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

/// In-plane (radial) distance from the centre disc to `point`, mirroring the
/// golden `Surfel::radial_distance`.
#[inline]
fn radial_distance(pos: Vec3, normal: Vec3, point: Vec3) -> f32 {
    let delta = point - pos;
    let axial = delta.dot(normal);
    let in_plane = delta - normal * axial;
    let len_sq = in_plane.length_squared();
    if len_sq > 0.0 {
        len_sq.sqrt()
    } else {
        0.0
    }
}

/// Radial coverage weight `max(0, 1 - (r / radius)^2)` (golden `radial_weight`).
#[inline]
fn radial_weight(pos: Vec3, normal: Vec3, radius: f32, point: Vec3) -> f32 {
    let t = (radial_distance(pos, normal, point) / radius).clamp(0.0, 1.0);
    (1.0 - t * t).clamp(0.0, 1.0)
}

/// Axial coverage weight `exp(-|axial| / (radius * tol))` (golden
/// `axial_weight`).
#[inline]
fn axial_weight(pos: Vec3, normal: Vec3, radius: f32, point: Vec3, tol: f32) -> f32 {
    let axial = (point - pos).dot(normal).abs();
    let scale = radius * tol.max(0.0);
    let denom = scale.max(MIN_RADIUS);
    stable_exp(-axial / denom)
}

/// Orientation-agreement weight `max(0, dot(a, b))^sharpness` (golden
/// `normal_consistency`).
#[inline]
fn normal_consistency(a: Vec3, b: Vec3, sharpness: f32) -> f32 {
    let cosine = safe_normalize(a).dot(safe_normalize(b)).clamp(0.0, 1.0);
    ops::powf(cosine, sharpness.max(0.0)).clamp(0.0, 1.0)
}

/// Geometric reuse weight of `neighbor` against the centre disc, mirroring the
/// golden `Surfel::geometric_weight` op-for-op.
#[inline]
fn geometric_weight(
    center: &GpuSpatialCenter,
    neighbor: &GpuSpatialNeighbor,
    params: &GpuSpatialFilterParams,
) -> f32 {
    let pos = Vec3::new(center.pos_x, center.pos_y, center.pos_z);
    let normal = Vec3::new(center.normal_x, center.normal_y, center.normal_z);
    let nb_pos = Vec3::new(neighbor.pos_x, neighbor.pos_y, neighbor.pos_z);
    let nb_normal = Vec3::new(neighbor.normal_x, neighbor.normal_y, neighbor.normal_z);
    let w_radial = radial_weight(pos, normal, center.radius, nb_pos);
    if w_radial <= 0.0 {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, center.radius, nb_pos, params.axial_tolerance);
    let w_normal = normal_consistency(normal, nb_normal, params.normal_sharpness);
    (w_radial * w_axial * w_normal).clamp(0.0, 1.0)
}

/// Filter one centre surfel against its neighbour slice, the scalar twin of
/// `surfel_spatial_filter_main`.
///
/// `neighbors` is the shared neighbour buffer; the centre's
/// `[neighbor_offset, neighbor_offset + neighbor_count)` slice is reduced in
/// order. The centre contributes unit weight so the filter degrades to identity
/// when every neighbour is incompatible; the result radiance is sanitised
/// finite and non-negative.
///
/// # Panics
///
/// Panics if the centre's neighbour slice extends past the end of `neighbors`.
#[must_use]
pub fn filter_center(
    params: &GpuSpatialFilterParams,
    center: &GpuSpatialCenter,
    neighbors: &[GpuSpatialNeighbor],
) -> GpuSpatialResult {
    let center_rgb = sanitize_rgb(Vec3::new(
        center.radiance_x,
        center.radiance_y,
        center.radiance_z,
    ));

    let mut sum = center_rgb;
    let mut weight = 1.0_f32;
    let start = center.neighbor_offset as usize;
    let end = start + center.neighbor_count as usize;
    for neighbor in &neighbors[start..end] {
        let w = geometric_weight(center, neighbor, params);
        if w <= 0.0 {
            continue;
        }
        let nb_rgb = sanitize_rgb(Vec3::new(
            neighbor.radiance_x,
            neighbor.radiance_y,
            neighbor.radiance_z,
        ));
        sum += nb_rgb * w;
        weight += w;
    }

    let out_rgb = if weight <= 1.0e-12 {
        center_rgb
    } else {
        sanitize_rgb(sum / weight)
    };

    GpuSpatialResult {
        radiance_x: out_rgb.x,
        radiance_y: out_rgb.y,
        radiance_z: out_rgb.z,
        total_weight: weight,
    }
}
