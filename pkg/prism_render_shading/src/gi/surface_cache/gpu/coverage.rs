//! `CPU` mirror of the surfel coverage-gather producer kernel.
//!
//! [`gather_point`] is the scalar twin of `surfel_coverage_main` in
//! `shaders/surfel_coverage.wesl`: given the dispatch
//! [`GpuSurfelCoverageParams`], one [`GpuCoveragePoint`] and the shared
//! [`GpuCoverageSurfel`] buffer it reduces the point's candidate slice into a
//! gathered [`GpuCoverageResult`] exactly as the kernel does. It is a *faithful
//! op-for-op transcription* of the shader rather than a call into the
//! [`Surfel::coverage`](crate::gi::surface_cache::surfel::Surfel::coverage)
//! golden, so the parity test it anchors is an independent cross-check that
//! fails if either the shader or the golden drifts.
//!
//! # Bit-exact parity
//!
//! The coverage weight is radial coverage x off-plane falloff x point/surfel
//! normal agreement; the transcendentals go through [`bevy_math::ops`] (`exp`
//! and `powf`), the exact `libm` routines the golden uses, so the mirror and
//! the golden agree bit-for-bit. The per-point reduction sums each covering
//! surfel in buffer order — the identical order and associativity as the
//! golden's gather loop — so no reordering can perturb the running `f32` sum.
//! The shading point is sanitised once up front with the idempotent
//! `sanitize_vec`, matching the golden which sanitises it inside every distance
//! query.
//!
//! Provenance: standard surfel-coverage gather driven by a coverage kernel; no
//! Unreal Engine source or derived code.

use bevy_math::{ops, Vec3};

use crate::gi::surface_cache::gpu::abi::{
    GpuCoveragePoint, GpuCoverageResult, GpuCoverageSurfel, GpuSurfelCoverageParams,
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

/// Replace any non-finite component of a position with zero (golden
/// `sanitize_vec`).
#[inline]
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(v.x),
        finite_or_zero(v.y),
        finite_or_zero(v.z),
    )
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

/// In-plane (radial) distance from the surfel disc to `point`, mirroring the
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

/// Full coverage weight of `surfel` for a shading point, mirroring the golden
/// `Surfel::coverage` op-for-op.
#[inline]
fn coverage_weight(
    surfel: &GpuCoverageSurfel,
    point: Vec3,
    point_normal: Vec3,
    params: &GpuSurfelCoverageParams,
) -> f32 {
    let pos = Vec3::new(surfel.pos_x, surfel.pos_y, surfel.pos_z);
    let normal = Vec3::new(surfel.normal_x, surfel.normal_y, surfel.normal_z);
    let w_radial = radial_weight(pos, normal, surfel.radius, point);
    if w_radial <= 0.0 {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, surfel.radius, point, params.axial_tolerance);
    let w_normal = normal_consistency(normal, point_normal, params.normal_sharpness);
    (w_radial * w_axial * w_normal).clamp(0.0, 1.0)
}

/// Gather one shading point over its candidate-surfel slice, the scalar twin of
/// `surfel_coverage_main`.
///
/// `surfels` is the shared candidate buffer; the point's
/// `[surfel_offset, surfel_offset + surfel_count)` slice is reduced in order.
/// Each covering surfel contributes its cached radiance weighted by its
/// coverage; the result radiance is the coverage-weighted mean, sanitised
/// finite and non-negative. A point that no surfel covers resolves to all-zero
/// radiance with zero total weight.
///
/// # Panics
///
/// Panics if the point's surfel slice extends past the end of `surfels`.
#[must_use]
pub fn gather_point(
    params: &GpuSurfelCoverageParams,
    point: &GpuCoveragePoint,
    surfels: &[GpuCoverageSurfel],
) -> GpuCoverageResult {
    let pos = sanitize_vec(Vec3::new(point.pos_x, point.pos_y, point.pos_z));
    let point_normal = Vec3::new(point.normal_x, point.normal_y, point.normal_z);

    let mut sum = Vec3::ZERO;
    let mut weight = 0.0_f32;
    let start = point.surfel_offset as usize;
    let end = start + point.surfel_count as usize;
    for surfel in &surfels[start..end] {
        let w = coverage_weight(surfel, pos, point_normal, params);
        if w <= 0.0 {
            continue;
        }
        let rgb = sanitize_rgb(Vec3::new(
            surfel.radiance_x,
            surfel.radiance_y,
            surfel.radiance_z,
        ));
        sum += rgb * w;
        weight += w;
    }

    let out_rgb = if weight <= 1.0e-12 {
        Vec3::ZERO
    } else {
        sanitize_rgb(sum / weight)
    };

    GpuCoverageResult {
        radiance_x: out_rgb.x,
        radiance_y: out_rgb.y,
        radiance_z: out_rgb.z,
        total_weight: weight,
    }
}
