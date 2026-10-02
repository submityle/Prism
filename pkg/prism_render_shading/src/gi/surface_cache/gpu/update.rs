//! `CPU` mirror of the surfel-update (temporal-integration) producer kernel.
//!
//! [`update_entry`] is the scalar twin of `surfel_update_main` in
//! `shaders/surfel_update.wesl`: given the dispatch
//! [`GpuSurfelUpdateParams`] and one [`GpuSurfelUpdateInput`] it advances the
//! surfel's cached radiance by one frame exactly as the kernel does, returning
//! a [`GpuSurfelUpdateResult`]. It is a *faithful op-for-op transcription* of
//! the shader rather than a call into the
//! [`integrate_radiance`](crate::gi::surface_cache::integration::integrate_radiance)
//! golden, so the parity test it anchors is an independent cross-check that
//! fails if either the shader or the golden drifts.
//!
//! # Bit-exact parity
//!
//! The arithmetic is a closed-form sequence of `+ - * /`, `sqrt` (via
//! [`Vec3::length`]), `min`/`max` and finiteness guards in the portable scalar
//! subset — no transcendental, no reorderable reduction, no fused multiply-add.
//! The confidence-weighted blend is `prev.lerp(sample, alpha)`, and `glam`
//! defines [`Vec3::lerp`] as `self * (1 - s) + rhs * s`, which is *exactly* the
//! `WGSL` `mix(a, b, t)` the kernel uses — so the kernel, this mirror and the
//! golden evaluate the identical closed form in the identical order and agree
//! bit-for-bit. `sanitize_rgb` is idempotent, so the kernel's single sanitise
//! of the reseed path matches the golden's `SurfelCacheEntry::from_sample`
//! double sanitise.
//!
//! Provenance: standard confidence-weighted temporal accumulation with
//! disocclusion reset; no Unreal Engine source or derived code.

use bevy_math::Vec3;

use crate::gi::surface_cache::gpu::abi::{
    GpuSurfelUpdateInput, GpuSurfelUpdateParams, GpuSurfelUpdateResult,
};

/// Replace a non-finite scalar with zero, mirroring the shader `finite_or_zero`
/// and the golden `finite_or_zero`.
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Sanitise a linear-RGB triple: non-finite components become zero and the
/// result is clamped non-negative, mirroring the shader `sanitize_rgb` and the
/// golden `sanitize_rgb`.
#[inline]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(c.x),
        finite_or_zero(c.y),
        finite_or_zero(c.z),
    )
    .max(Vec3::ZERO)
}

/// Whether the current surfel geometry drifted far enough from the previous one
/// to invalidate the history, mirroring the shader `is_disoccluded` and the
/// golden `is_disoccluded` op-for-op.
#[inline]
fn is_disoccluded(input: &GpuSurfelUpdateInput, params: &GpuSurfelUpdateParams) -> bool {
    let prev_pos = Vec3::new(input.prev_pos_x, input.prev_pos_y, input.prev_pos_z);
    let curr_pos = Vec3::new(input.curr_pos_x, input.curr_pos_y, input.curr_pos_z);
    let offset = (curr_pos - prev_pos).length();
    if !offset.is_finite() {
        return true;
    }
    let radius = input.prev_radius.max(input.curr_radius);
    let max_offset = radius * params.position_tolerance.max(0.0);
    if offset > max_offset {
        return true;
    }
    let prev_normal = Vec3::new(
        input.prev_normal_x,
        input.prev_normal_y,
        input.prev_normal_z,
    );
    let curr_normal = Vec3::new(
        input.curr_normal_x,
        input.curr_normal_y,
        input.curr_normal_z,
    );
    let cos = prev_normal.dot(curr_normal);
    if !cos.is_finite() || cos < params.normal_tolerance {
        return true;
    }
    false
}

/// Advance one surfel's temporal accumulation by a frame, the scalar twin of
/// `surfel_update_main`.
///
/// Reseeds a fresh `(sample, 1)` entry on first sight (`has_history == 0`) or on
/// disocclusion; otherwise blends the previous radiance toward the sample with
/// weight `1 / min(prev_count + 1, max(max_samples, 1))` and advances the
/// confidence by one (capped). The radiance is sanitised finite and
/// non-negative on store.
#[must_use]
pub fn update_entry(
    params: &GpuSurfelUpdateParams,
    input: &GpuSurfelUpdateInput,
) -> GpuSurfelUpdateResult {
    let sample = sanitize_rgb(Vec3::new(
        input.new_radiance_x,
        input.new_radiance_y,
        input.new_radiance_z,
    ));

    if input.has_history == 0 || is_disoccluded(input, params) {
        return GpuSurfelUpdateResult::new(sample, 1);
    }

    let cap = params.max_samples.max(1);
    let n = (input.prev_count + 1).min(cap);
    let alpha = 1.0 / n as f32;
    let prev_radiance = sanitize_rgb(Vec3::new(
        input.prev_radiance_x,
        input.prev_radiance_y,
        input.prev_radiance_z,
    ));
    let blended = sanitize_rgb(prev_radiance.lerp(sample, alpha));

    GpuSurfelUpdateResult::new(blended, n)
}
