//! Runtime lookup: trilinear interpolation of a baked parameter field and a
//! one-pole smoother, both safe to run on the audio callback thread.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "runtime lookup" of design section 43: a listener position
//! is blended from its eight surrounding probes with the trilinear weights of
//! [`crate::grid`], and the per-block jumps are smoothed with a one-pole
//! filter. Everything here is allocation free, lock free, and panic free, so
//! it meets the real-time contract; scalar fields blend linearly and the two
//! angles blend on the unit circle.

use bevy_math::{ops, Vec3};

use crate::encoding::PerceptualParams;
use crate::field::ParameterField;

/// A read-only, allocation-free view over a [`ParameterField`] that answers
/// listener-position queries by trilinear interpolation.
///
/// Hold one per baked field; it borrows the field and performs no mutation, so
/// it is cheap to construct every block.
#[derive(Debug, Clone, Copy)]
pub struct ParameterLookup<'a> {
    field: &'a ParameterField,
}

impl<'a> ParameterLookup<'a> {
    /// Binds a lookup to `field`.
    #[must_use]
    #[inline]
    pub fn new(field: &'a ParameterField) -> Self {
        Self { field }
    }

    /// The field this lookup reads.
    #[must_use]
    #[inline]
    pub fn field(&self) -> &ParameterField {
        self.field
    }

    /// Trilinearly interpolates the field at the world-space listener position.
    ///
    /// Scalars blend by the eight trilinear corner weights; azimuth and
    /// elevation blend through weighted `(cos, sin)` accumulation so the wrap
    /// at `+/-pi` introduces no artefact. The call allocates nothing, takes no
    /// locks, and never panics (out-of-grid positions clamp to the boundary).
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Vec3;
    /// # use prism_audio_wave::field::{BitDepth, ParameterFieldBuilder};
    /// # use prism_audio_wave::grid::{Aabb, ProbeGrid};
    /// # use prism_audio_wave::lookup::ParameterLookup;
    /// let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 2, 2, 2);
    /// let field = ParameterFieldBuilder::new(grid).build(BitDepth::Sixteen);
    /// let lookup = ParameterLookup::new(&field);
    /// let p = lookup.sample(Vec3::splat(1.0));
    /// assert!(p.direct_gain >= 0.0 && p.direct_gain <= 1.0);
    /// ```
    #[must_use]
    pub fn sample(&self, world_pos: Vec3) -> PerceptualParams {
        let grid = self.field.grid();
        let corners = grid.trilinear(world_pos).corner_weights(grid);

        let mut direct_gain = 0.0_f32;
        let mut direct_cutoff_hz = 0.0_f32;
        let mut rt60_s = 0.0_f32;
        let mut wet_gain = 0.0_f32;
        let mut drr_db = 0.0_f32;
        let mut az_x = 0.0_f32;
        let mut az_y = 0.0_f32;
        let mut el_x = 0.0_f32;
        let mut el_y = 0.0_f32;

        for (index, weight) in corners {
            let p = self.field.decode(index);
            direct_gain += weight * p.direct_gain;
            direct_cutoff_hz += weight * p.direct_cutoff_hz;
            rt60_s += weight * p.rt60_s;
            wet_gain += weight * p.wet_gain;
            drr_db += weight * p.drr_db;
            az_x += weight * ops::cos(p.azimuth);
            az_y += weight * ops::sin(p.azimuth);
            el_x += weight * ops::cos(p.elevation);
            el_y += weight * ops::sin(p.elevation);
        }

        let azimuth = resolve_angle(az_x, az_y);
        let elevation = resolve_angle(el_x, el_y);

        PerceptualParams {
            direct_gain,
            direct_cutoff_hz,
            rt60_s,
            wet_gain,
            azimuth,
            elevation,
            drr_db,
        }
    }
}

/// Recovers an angle from accumulated `(cos, sin)` weight sums, defaulting a
/// vanishing resultant to straight ahead.
#[must_use]
#[inline]
fn resolve_angle(x: f32, y: f32) -> f32 {
    if x * x + y * y <= 1.0e-20 {
        0.0
    } else {
        ops::atan2(y, x)
    }
}

/// A per-parameter one-pole smoother that eases the control-rate lookup output
/// toward each new target, avoiding zipper noise.
///
/// The update is `cur += coeff * (target - cur)` per scalar, with angles eased
/// on the unit circle. It is allocation free and real-time safe.
#[derive(Debug, Clone, Copy)]
pub struct WaveParamSmoother {
    coeff: f32,
    current: PerceptualParams,
}

impl WaveParamSmoother {
    /// Builds a smoother with per-block coefficient `coeff` in `[0, 1]`
    /// (`1` snaps instantly, small values glide slowly), initialised to
    /// [`PerceptualParams::OPEN`].
    #[must_use]
    pub fn new(coeff: f32) -> Self {
        Self {
            coeff: coeff.clamp(0.0, 1.0),
            current: PerceptualParams::OPEN,
        }
    }

    /// The smoothing coefficient.
    #[must_use]
    #[inline]
    pub fn coeff(&self) -> f32 {
        self.coeff
    }

    /// The current smoothed value.
    #[must_use]
    #[inline]
    pub fn current(&self) -> PerceptualParams {
        self.current
    }

    /// Snaps the smoother directly to `params` (use on teleports and resets).
    #[inline]
    pub fn reset(&mut self, params: PerceptualParams) {
        self.current = params;
    }

    /// Advances one block toward `target` and returns the new smoothed value.
    ///
    /// Real-time safe: no allocation, no locks, no panics.
    pub fn process(&mut self, target: PerceptualParams) -> PerceptualParams {
        // `lerp` eases every scalar by `coeff` and both angles on the circle.
        self.current = self.current.lerp(&target, self.coeff);
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{BitDepth, ParameterFieldBuilder};
    use crate::grid::{Aabb, ProbeGrid};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn eight_cell_field() -> ParameterField {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 2, 2, 2);
        let mut builder = ParameterFieldBuilder::new(grid);
        // Probe 0 (origin) fully occluded, probe 7 (far corner) open.
        builder.set_probe(0, PerceptualParams::OCCLUDED);
        builder.set_probe(7, PerceptualParams::OPEN);
        builder.build(BitDepth::Sixteen)
    }

    #[test]
    fn sample_at_probe_matches_decode() {
        let field = eight_cell_field();
        let lookup = ParameterLookup::new(&field);
        let grid = *field.grid();
        let pos = grid.probe_position(1, 1, 1);
        let s = lookup.sample(pos);
        let decoded = field.decode(grid.linear_index(1, 1, 1));
        assert!(approx(s.direct_gain, decoded.direct_gain, 1e-3));
    }

    #[test]
    fn sample_blends_between_probes() {
        let field = eight_cell_field();
        let lookup = ParameterLookup::new(&field);
        // Mid-grid should land between the occluded origin and open far corner.
        let mid = lookup.sample(Vec3::splat(1.0));
        assert!(mid.direct_gain > 0.0 && mid.direct_gain < 1.0, "{}", mid.direct_gain);
    }

    #[test]
    fn out_of_grid_clamps_without_panic() {
        let field = eight_cell_field();
        let lookup = ParameterLookup::new(&field);
        let far = lookup.sample(Vec3::splat(1000.0));
        let corner = field.decode(field.probe_count() - 1);
        assert!(approx(far.direct_gain, corner.direct_gain, 1e-3));
    }

    #[test]
    fn smoother_converges_to_target() {
        let mut sm = WaveParamSmoother::new(0.5);
        sm.reset(PerceptualParams::OCCLUDED);
        let target = PerceptualParams::OPEN;
        let mut last = sm.current();
        for _ in 0..64 {
            last = sm.process(target);
        }
        assert!(approx(last.direct_gain, target.direct_gain, 1e-2));
        assert!(approx(last.wet_gain, target.wet_gain, 1e-2));
    }

    #[test]
    fn smoother_coeff_one_snaps() {
        let mut sm = WaveParamSmoother::new(1.0);
        let out = sm.process(PerceptualParams::OCCLUDED);
        assert!(approx(out.direct_gain, 0.0, 1e-6));
    }
}
