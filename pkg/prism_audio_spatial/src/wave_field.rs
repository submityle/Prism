//! Baked wave-acoustics perceptual-parameter field (design section 43).
//!
//! Geometric acoustics (ray occlusion and image-source reflections) assumes the
//! wavelength is small against the scene, so it misrepresents low-frequency
//! diffraction, soft occlusion, coupled rooms, and realistic reverberant tails.
//! The next-generation answer, pioneered by Microsoft Project Acoustics, is to
//! solve the wave equation *offline* on static geometry and store a compact
//! grid of **perceptual parameters** that the runtime interpolates per frame.
//!
//! This module is the runtime half of that pipeline. It does **not** solve the
//! wave equation: that belongs to an offline baking tool. Instead it owns the
//! honest runtime contract -- a regular [`WaveProbeGrid`] of already-baked
//! [`WaveParameters`], a deterministic trilinear interpolation at the listener
//! position, and a [`WaveFieldBackend`] that turns the interpolated parameters
//! into the same [`PropagationPath`] vocabulary every other backend speaks.
//! The baked grid is supplied by the caller, mirroring how
//! [`HrirSource`](https://en.wikipedia.org/wiki/Head-related_transfer_function)
//! style data is loaded elsewhere: no simulation is faked, only interpolated.
//!
//! # What a probe stores
//!
//! Each probe encodes the perceptual summary of one baked source heard from a
//! listener cell, aligned one-to-one with the engine's existing parameter
//! buses (design sections 14, 16, 17): direct-path blocking
//! ([`OcclusionFactors`]), the rendered direct gain and three-band colour, the
//! dominant early arrival (gain, colour, world-space direction, extra delay),
//! and the late reverberant send (wet gain and `RT60`). There is no raw impulse
//! response: storing one per probe would be prohibitively large, exactly the
//! problem perceptual encoding solves.
//!
//! # Real-time safety
//!
//! Baking happens offline. [`WaveProbeGrid::sample`] and
//! [`WaveFieldBackend::query`] run at control rate: they only read the grid and
//! interpolate scalars, so they allocate nothing, never lock, and never panic.
//! The sample-rate DSP (delay lines, filters, reverb) still runs on the audio
//! thread from the paths this backend reports.
//!
//! # Provenance
//!
//! The perceptual-encoding idea is the publicly documented approach of Project
//! Acoustics / Triton. This module contains **no** Microsoft, Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code, and no machine learning: it is an original implementation of a
//! probe-grid interpolator from that public knowledge.

use alloc::vec::Vec;

use bevy_math::{Vec3, ops};
use prism_audio_core::math::Sample;

use crate::band_spectrum::BandGains;
use crate::doppler::SPEED_OF_SOUND_MPS;
use crate::geometry::{Emitter, Listener};
use crate::occlusion::OcclusionFactors;
use crate::propagation::{
    FULL_BAND_CUTOFF_HZ, PathKind, PropagationBackend, PropagationPath, PropagationSummary,
};

/// A direction shorter than the square root of this is treated as degenerate
/// and replaced by a fallback, matching the geometry module's convention.
const DIRECTION_EPSILON_SQ: Sample = 1.0e-12;

/// Linear interpolation `a + (a_to_b) * t`, built from add/sub/mul only so it
/// is bit-reproducible across targets.
#[inline]
#[must_use]
fn lerp(a: Sample, b: Sample, t: Sample) -> Sample {
    a + (b - a) * t
}

/// Normalises `v`, falling back to `fallback` when `v` is too short to have a
/// well-defined direction. Routes the length through [`bevy_math::ops`] so the
/// result is deterministic.
#[inline]
#[must_use]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.dot(v);
    if len_sq <= DIRECTION_EPSILON_SQ {
        fallback
    } else {
        v / ops::sqrt(len_sq)
    }
}

/// Interpolates two three-band spectra component-wise.
#[inline]
#[must_use]
fn lerp_bands(a: BandGains, b: BandGains, t: Sample) -> BandGains {
    let a = a.bands();
    let b = b.bands();
    BandGains::new([
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ])
}

/// Interpolates two blocking summaries component-wise.
#[inline]
#[must_use]
fn lerp_occlusion(a: OcclusionFactors, b: OcclusionFactors, t: Sample) -> OcclusionFactors {
    OcclusionFactors::new(
        lerp(a.obstruction, b.obstruction, t),
        lerp(a.occlusion, b.occlusion, t),
    )
}

/// Interpolates two directions as a normalised blend, falling back to local
/// forward (`-Z`) when the blend collapses.
#[inline]
#[must_use]
fn lerp_direction(a: Vec3, b: Vec3, t: Sample) -> Vec3 {
    normalize_or(a + (b - a) * t, Vec3::NEG_Z)
}

/// The late reverberant send a wave probe prescribes for one source.
///
/// This feeds the auxiliary-send / reverb bus (design section 17); it is not a
/// discrete arrival, so [`WaveFieldBackend::query`] does not emit it as a
/// [`PropagationPath`]. Query it separately with [`WaveFieldBackend::reverb`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveReverb {
    /// Linear send level into the reverberant field, in `[0, 1]`.
    pub wet_gain: Sample,
    /// Reverberation time `RT60` in seconds (non-negative), driving the decay
    /// of the reverberant field the send feeds.
    pub rt60_seconds: Sample,
}

impl WaveReverb {
    /// A dry send: no reverberant energy and no decay.
    pub const DRY: Self = Self {
        wet_gain: 0.0,
        rt60_seconds: 0.0,
    };
}

/// The baked perceptual parameters of one source heard from one point.
///
/// Every field is a quantity the runtime render path already consumes, so a
/// probe is a direct snapshot of the engine's parameter buses rather than an
/// intermediate acoustic representation. All constructors clamp inputs into
/// their physical ranges, so an interpolated or deserialised probe is always
/// well formed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveParameters {
    /// Blocking of the direct line of sight, in the shared
    /// [`OcclusionFactors`] vocabulary (design section 14). Reported verbatim
    /// as [`PropagationSummary::direct`].
    pub occlusion: OcclusionFactors,
    /// Rendered broadband gain of the direct arrival, in `[0, 1]`.
    pub direct_gain: Sample,
    /// Three-band colour of the direct arrival (material and air filtering).
    pub direct_bands: BandGains,
    /// Low-pass corner (Hz) colouring the direct arrival;
    /// [`FULL_BAND_CUTOFF_HZ`] means unfiltered.
    pub direct_cutoff_hz: Sample,
    /// Rendered broadband gain of the dominant early arrival, in `[0, 1]`. Zero
    /// means the probe has no significant early energy.
    pub early_gain: Sample,
    /// Three-band colour of the early arrival.
    pub early_bands: BandGains,
    /// World-space unit direction the early energy arrives from. Rotated into
    /// the listener frame when the backend reports it.
    pub early_direction: Vec3,
    /// Extra delay of the early arrival beyond the direct path, in seconds
    /// (non-negative).
    pub early_delay_seconds: Sample,
    /// Late reverberant send level, in `[0, 1]`.
    pub reverb_gain: Sample,
    /// Reverberation time `RT60` in seconds (non-negative).
    pub reverb_time_seconds: Sample,
}

impl WaveParameters {
    /// Free-field parameters: fully open, unity direct gain, flat colour, and
    /// no early or reverberant energy. The correct baseline for an unbaked or
    /// anechoic cell.
    pub const ANECHOIC: Self = Self {
        occlusion: OcclusionFactors::OPEN,
        direct_gain: 1.0,
        direct_bands: BandGains::UNITY,
        direct_cutoff_hz: FULL_BAND_CUTOFF_HZ,
        early_gain: 0.0,
        early_bands: BandGains::SILENT,
        early_direction: Vec3::NEG_Z,
        early_delay_seconds: 0.0,
        reverb_gain: 0.0,
        reverb_time_seconds: 0.0,
    };

    /// Builds a probe, clamping each field into its physical range: gains into
    /// `[0, 1]`, delays and `RT60` to non-negative, the direct corner to a
    /// positive value, and the early direction to a unit vector (falling back
    /// to `-Z` when degenerate). A non-finite scalar becomes a safe default.
    #[must_use]
    pub fn new(
        occlusion: OcclusionFactors,
        direct_gain: Sample,
        direct_bands: BandGains,
        direct_cutoff_hz: Sample,
        early_gain: Sample,
        early_bands: BandGains,
        early_direction: Vec3,
        early_delay_seconds: Sample,
        reverb_gain: Sample,
        reverb_time_seconds: Sample,
    ) -> Self {
        Self {
            occlusion,
            direct_gain: clamp_unit(direct_gain),
            direct_bands,
            direct_cutoff_hz: sanitise_cutoff(direct_cutoff_hz),
            early_gain: clamp_unit(early_gain),
            early_bands,
            early_direction: normalize_or(early_direction, Vec3::NEG_Z),
            early_delay_seconds: non_negative(early_delay_seconds),
            reverb_gain: clamp_unit(reverb_gain),
            reverb_time_seconds: non_negative(reverb_time_seconds),
        }
    }

    /// The late reverberant send this probe prescribes.
    #[inline]
    #[must_use]
    pub fn reverb(&self) -> WaveReverb {
        WaveReverb {
            wet_gain: self.reverb_gain,
            rt60_seconds: self.reverb_time_seconds,
        }
    }

    /// Interpolates two probes field-by-field at `t` in `[0, 1]`. Directions
    /// blend as a renormalised vector; every other field blends linearly.
    #[must_use]
    fn lerp(a: Self, b: Self, t: Sample) -> Self {
        Self {
            occlusion: lerp_occlusion(a.occlusion, b.occlusion, t),
            direct_gain: lerp(a.direct_gain, b.direct_gain, t),
            direct_bands: lerp_bands(a.direct_bands, b.direct_bands, t),
            direct_cutoff_hz: lerp(a.direct_cutoff_hz, b.direct_cutoff_hz, t),
            early_gain: lerp(a.early_gain, b.early_gain, t),
            early_bands: lerp_bands(a.early_bands, b.early_bands, t),
            early_direction: lerp_direction(a.early_direction, b.early_direction, t),
            early_delay_seconds: lerp(a.early_delay_seconds, b.early_delay_seconds, t),
            reverb_gain: lerp(a.reverb_gain, b.reverb_gain, t),
            reverb_time_seconds: lerp(a.reverb_time_seconds, b.reverb_time_seconds, t),
        }
    }
}

impl Default for WaveParameters {
    #[inline]
    fn default() -> Self {
        Self::ANECHOIC
    }
}

/// Clamps a gain into the physical `[0, 1]` amplitude range; a non-finite input
/// becomes `0`.
#[inline]
#[must_use]
fn clamp_unit(gain: Sample) -> Sample {
    if gain.is_finite() {
        gain.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a time to be non-negative; a non-finite input becomes `0`.
#[inline]
#[must_use]
fn non_negative(seconds: Sample) -> Sample {
    if seconds.is_finite() {
        seconds.max(0.0)
    } else {
        0.0
    }
}

/// Clamps a low-pass corner to a positive value, defaulting a non-positive or
/// non-finite input to [`FULL_BAND_CUTOFF_HZ`] (unfiltered).
#[inline]
#[must_use]
fn sanitise_cutoff(hz: Sample) -> Sample {
    if hz.is_finite() && hz > 0.0 {
        hz
    } else {
        FULL_BAND_CUTOFF_HZ
    }
}

/// Resolves a listener coordinate on one axis into the bracketing cell indices
/// and the fractional position inside the cell.
///
/// `dim` is the probe count along the axis. A single-probe axis collapses to
/// cell `(0, 0)` with zero fraction. Positions outside `[lo, hi]` clamp to the
/// boundary cell (nearest-value extrapolation), so a listener leaving the baked
/// volume degrades gracefully rather than reading out of bounds.
#[inline]
#[must_use]
fn axis_cell(p: Sample, lo: Sample, hi: Sample, dim: usize) -> (usize, usize, Sample) {
    if dim <= 1 {
        return (0, 0, 0.0);
    }
    let span = hi - lo;
    let t = if span > 0.0 {
        ((p - lo) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scaled = t * ((dim - 1) as Sample);
    let base = scaled.floor();
    let i0 = (base as usize).min(dim - 2);
    let frac = (scaled - (i0 as Sample)).clamp(0.0, 1.0);
    (i0, i0 + 1, frac)
}

/// A regular grid of baked [`WaveParameters`] encoding one source heard across
/// a bounded listener volume.
///
/// The grid spans the axis-aligned box `[min, max]` with `dims` probes per
/// axis. Probes are stored in `x`-fastest, then `y`, then `z` order. A caller
/// (an offline bake tool or a loaded asset) supplies the probe data; this type
/// only validates and interpolates it.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveProbeGrid {
    /// World-space position of the baked source this grid encodes.
    source_position: Vec3,
    /// Lower corner of the baked listener volume, in metres.
    min: Vec3,
    /// Upper corner of the baked listener volume, in metres.
    max: Vec3,
    /// Probe counts along `x`, `y`, and `z`; each is at least `1`.
    dims: [usize; 3],
    /// Flattened probes, length `dims[0] * dims[1] * dims[2]`, in `x`-fastest
    /// then `y` then `z` order.
    probes: Vec<WaveParameters>,
}

impl WaveProbeGrid {
    /// Builds a grid, returning [`None`] when the data is inconsistent: any
    /// axis has zero probes, the probe count does not equal the product of
    /// `dims`, a bound is non-finite, or `max` is below `min` on any axis.
    #[must_use]
    pub fn new(
        source_position: Vec3,
        min: Vec3,
        max: Vec3,
        dims: [usize; 3],
        probes: Vec<WaveParameters>,
    ) -> Option<Self> {
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return None;
        }
        let expected = dims[0]
            .checked_mul(dims[1])
            .and_then(|xy| xy.checked_mul(dims[2]))?;
        if probes.len() != expected {
            return None;
        }
        if !source_position.is_finite() || !min.is_finite() || !max.is_finite() {
            return None;
        }
        if max.x < min.x || max.y < min.y || max.z < min.z {
            return None;
        }
        Some(Self {
            source_position,
            min,
            max,
            dims,
            probes,
        })
    }

    /// Builds a grid whose every probe holds the same parameters, useful for an
    /// anechoic baseline or a test fixture. Returns [`None`] on the same
    /// degenerate `dims` or bounds as [`Self::new`].
    #[must_use]
    pub fn uniform(
        source_position: Vec3,
        min: Vec3,
        max: Vec3,
        dims: [usize; 3],
        params: WaveParameters,
    ) -> Option<Self> {
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return None;
        }
        let count = dims[0]
            .checked_mul(dims[1])
            .and_then(|xy| xy.checked_mul(dims[2]))?;
        let mut probes = Vec::with_capacity(count);
        for _ in 0..count {
            probes.push(params);
        }
        Self::new(source_position, min, max, dims, probes)
    }

    /// World-space position of the baked source.
    #[inline]
    #[must_use]
    pub fn source_position(&self) -> Vec3 {
        self.source_position
    }

    /// Probe counts along `x`, `y`, and `z`.
    #[inline]
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Lower and upper corners of the baked listener volume.
    #[inline]
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.min, self.max)
    }

    /// Total number of stored probes.
    #[inline]
    #[must_use]
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Borrows the probe at integer lattice coordinates, or [`None`] when any
    /// coordinate is out of range.
    #[must_use]
    pub fn probe(&self, ix: usize, iy: usize, iz: usize) -> Option<&WaveParameters> {
        if ix >= self.dims[0] || iy >= self.dims[1] || iz >= self.dims[2] {
            return None;
        }
        self.probes.get(self.flat_index(ix, iy, iz))
    }

    /// Flattens lattice coordinates into the storage index (`x`-fastest).
    #[inline]
    #[must_use]
    fn flat_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * self.dims[1] + iy) * self.dims[0] + ix
    }

    /// Reads a probe by lattice coordinates, saturating any coordinate to the
    /// last probe on its axis so interpolation never indexes out of bounds.
    #[inline]
    #[must_use]
    fn probe_saturating(&self, ix: usize, iy: usize, iz: usize) -> WaveParameters {
        let ix = ix.min(self.dims[0] - 1);
        let iy = iy.min(self.dims[1] - 1);
        let iz = iz.min(self.dims[2] - 1);
        self.probes[self.flat_index(ix, iy, iz)]
    }

    /// Trilinearly interpolates the baked parameters at a world-space listener
    /// `position`. Positions outside the baked volume clamp to the boundary
    /// (nearest-value extrapolation); the result is always a valid probe.
    #[must_use]
    pub fn sample(&self, position: Vec3) -> WaveParameters {
        let (x0, x1, fx) = axis_cell(position.x, self.min.x, self.max.x, self.dims[0]);
        let (y0, y1, fy) = axis_cell(position.y, self.min.y, self.max.y, self.dims[1]);
        let (z0, z1, fz) = axis_cell(position.z, self.min.z, self.max.z, self.dims[2]);

        let c000 = self.probe_saturating(x0, y0, z0);
        let c100 = self.probe_saturating(x1, y0, z0);
        let c010 = self.probe_saturating(x0, y1, z0);
        let c110 = self.probe_saturating(x1, y1, z0);
        let c001 = self.probe_saturating(x0, y0, z1);
        let c101 = self.probe_saturating(x1, y0, z1);
        let c011 = self.probe_saturating(x0, y1, z1);
        let c111 = self.probe_saturating(x1, y1, z1);

        let x00 = WaveParameters::lerp(c000, c100, fx);
        let x10 = WaveParameters::lerp(c010, c110, fx);
        let x01 = WaveParameters::lerp(c001, c101, fx);
        let x11 = WaveParameters::lerp(c011, c111, fx);

        let y0 = WaveParameters::lerp(x00, x10, fy);
        let y1 = WaveParameters::lerp(x01, x11, fy);

        WaveParameters::lerp(y0, y1, fz)
    }
}

/// A [`PropagationBackend`] driven by a baked [`WaveProbeGrid`].
///
/// It interpolates the grid at the listener and reports the direct arrival plus
/// (when present) the dominant early arrival as [`PropagationPath`]s, in the
/// same vocabulary the geometric and free-field backends use, so the hybrid
/// arbiter and the real-time voice consume it uniformly. The late reverberant
/// send is a separate query ([`Self::reverb`]) because it is not a discrete
/// path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveFieldBackend<'grid> {
    /// The baked probe grid this backend interpolates.
    grid: &'grid WaveProbeGrid,
}

impl<'grid> WaveFieldBackend<'grid> {
    /// Wraps a baked grid as a propagation backend.
    #[inline]
    #[must_use]
    pub fn new(grid: &'grid WaveProbeGrid) -> Self {
        Self { grid }
    }

    /// The baked grid this backend reads.
    #[inline]
    #[must_use]
    pub fn grid(&self) -> &WaveProbeGrid {
        self.grid
    }

    /// Interpolates the baked parameters at the `listener`'s position.
    #[inline]
    #[must_use]
    pub fn parameters(&self, listener: &Listener) -> WaveParameters {
        self.grid.sample(listener.position)
    }

    /// The late reverberant send prescribed at the `listener`'s position.
    #[inline]
    #[must_use]
    pub fn reverb(&self, listener: &Listener) -> WaveReverb {
        self.parameters(listener).reverb()
    }
}

impl PropagationBackend for WaveFieldBackend<'_> {
    fn query(
        &self,
        listener: &Listener,
        emitter: &Emitter,
        paths: &mut [PropagationPath],
    ) -> PropagationSummary {
        let params = self.parameters(listener);
        if paths.is_empty() {
            return PropagationSummary {
                direct: params.occlusion,
                path_count: 0,
            };
        }

        let local = listener.localize(emitter);
        let direct_delay = local.distance / SPEED_OF_SOUND_MPS;

        paths[0] = PropagationPath {
            kind: PathKind::Direct,
            delay_seconds: direct_delay,
            gain: params.direct_gain,
            cutoff_hz: params.direct_cutoff_hz,
            bands: params.direct_bands,
            direction: local.direction,
        };
        let mut path_count = 1;

        if params.early_gain > 0.0 && paths.len() >= 2 {
            let world_dir = params.early_direction;
            let local_dir = normalize_or(listener.orientation.inverse() * world_dir, local.direction);
            paths[1] = PropagationPath {
                kind: PathKind::Reflection,
                delay_seconds: direct_delay + params.early_delay_seconds,
                gain: params.early_gain,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                bands: params.early_bands,
                direction: local_dir,
            };
            path_count = 2;
        }

        PropagationSummary {
            direct: params.occlusion,
            path_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Quat;
    use core::f32::consts::FRAC_PI_2;

    fn blocked_probe() -> WaveParameters {
        WaveParameters::new(
            OcclusionFactors::new(0.5, 0.5),
            0.4,
            BandGains::new([0.9, 0.5, 0.1]),
            2_000.0,
            0.3,
            BandGains::new([0.6, 0.4, 0.2]),
            Vec3::X,
            0.01,
            0.5,
            1.2,
        )
    }

    #[test]
    fn anechoic_is_open_and_unity() {
        let p = WaveParameters::ANECHOIC;
        assert_eq!(p.occlusion, OcclusionFactors::OPEN);
        assert_eq!(p.direct_gain, 1.0);
        assert_eq!(p.direct_bands, BandGains::UNITY);
        assert_eq!(p.reverb_gain, 0.0);
    }

    #[test]
    fn new_clamps_out_of_range_fields() {
        let p = WaveParameters::new(
            OcclusionFactors::OPEN,
            5.0,
            BandGains::UNITY,
            -10.0,
            -1.0,
            BandGains::UNITY,
            Vec3::ZERO,
            -3.0,
            2.0,
            -4.0,
        );
        assert_eq!(p.direct_gain, 1.0);
        assert_eq!(p.direct_cutoff_hz, FULL_BAND_CUTOFF_HZ);
        assert_eq!(p.early_gain, 0.0);
        assert_eq!(p.early_direction, Vec3::NEG_Z);
        assert_eq!(p.early_delay_seconds, 0.0);
        assert_eq!(p.reverb_gain, 1.0);
        assert_eq!(p.reverb_time_seconds, 0.0);
    }

    #[test]
    fn grid_rejects_mismatched_probe_count() {
        let probes = alloc::vec![WaveParameters::ANECHOIC; 3];
        assert!(WaveProbeGrid::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE, [2, 2, 2], probes).is_none());
    }

    #[test]
    fn grid_rejects_zero_dimension() {
        assert!(WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ONE,
            [0, 1, 1],
            WaveParameters::ANECHOIC
        )
        .is_none());
    }

    #[test]
    fn grid_rejects_inverted_bounds() {
        assert!(WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::ONE,
            Vec3::ZERO,
            [2, 2, 2],
            WaveParameters::ANECHOIC
        )
        .is_none());
    }

    #[test]
    fn uniform_grid_samples_the_same_everywhere() {
        let probe = blocked_probe();
        let grid =
            WaveProbeGrid::uniform(Vec3::ZERO, Vec3::ZERO, Vec3::splat(4.0), [3, 3, 3], probe)
                .expect("valid grid");
        let a = grid.sample(Vec3::new(1.0, 2.0, 3.0));
        let b = grid.sample(Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(a, b);
        assert_eq!(a.direct_gain, probe.direct_gain);
    }

    #[test]
    fn trilinear_midpoint_averages_two_corners() {
        // A 2x1x1 grid: open at x=0, half-gain at x=1.
        let open = WaveParameters::ANECHOIC;
        let dark = WaveParameters::new(
            OcclusionFactors::new(1.0, 1.0),
            0.0,
            BandGains::SILENT,
            FULL_BAND_CUTOFF_HZ,
            0.0,
            BandGains::SILENT,
            Vec3::NEG_Z,
            0.0,
            0.0,
            0.0,
        );
        let grid = WaveProbeGrid::new(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            [2, 1, 1],
            alloc::vec![open, dark],
        )
        .expect("valid grid");
        let mid = grid.sample(Vec3::new(1.0, 0.0, 0.0));
        assert!((mid.direct_gain - 0.5).abs() < 1.0e-6);
        assert!((mid.occlusion.obstruction - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn sampling_outside_bounds_clamps_to_boundary() {
        let open = WaveParameters::ANECHOIC;
        let dark = WaveParameters::new(
            OcclusionFactors::new(1.0, 1.0),
            0.0,
            BandGains::SILENT,
            FULL_BAND_CUTOFF_HZ,
            0.0,
            BandGains::SILENT,
            Vec3::NEG_Z,
            0.0,
            0.0,
            0.0,
        );
        let grid = WaveProbeGrid::new(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            [2, 1, 1],
            alloc::vec![open, dark],
        )
        .expect("valid grid");
        let far_left = grid.sample(Vec3::new(-100.0, 0.0, 0.0));
        let far_right = grid.sample(Vec3::new(100.0, 0.0, 0.0));
        assert_eq!(far_left.direct_gain, open.direct_gain);
        assert_eq!(far_right.direct_gain, dark.direct_gain);
    }

    #[test]
    fn single_probe_axis_is_degenerate() {
        let (i0, i1, frac) = axis_cell(5.0, 0.0, 10.0, 1);
        assert_eq!((i0, i1), (0, 0));
        assert_eq!(frac, 0.0);
    }

    #[test]
    fn probe_accessor_bounds_check() {
        let grid =
            WaveProbeGrid::uniform(Vec3::ZERO, Vec3::ZERO, Vec3::ONE, [2, 2, 2], blocked_probe())
                .expect("valid grid");
        assert!(grid.probe(1, 1, 1).is_some());
        assert!(grid.probe(2, 0, 0).is_none());
        assert_eq!(grid.probe_count(), 8);
    }

    #[test]
    fn backend_reports_direct_path_from_baked_gain() {
        let grid = WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            [2, 2, 2],
            blocked_probe(),
        )
        .expect("valid grid");
        let backend = WaveFieldBackend::new(&grid);
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -3.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert_eq!(summary.path_count, 2);
        assert_eq!(paths[0].kind, PathKind::Direct);
        assert!((paths[0].gain - 0.4).abs() < 1.0e-6);
        // 3 m / 343 m/s.
        assert!((paths[0].delay_seconds - 3.0 / 343.0).abs() < 1.0e-6);
    }

    #[test]
    fn backend_omits_early_path_when_gain_is_zero() {
        let grid = WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            [2, 2, 2],
            WaveParameters::ANECHOIC,
        )
        .expect("valid grid");
        let backend = WaveFieldBackend::new(&grid);
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -3.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert_eq!(summary.path_count, 1);
        assert_eq!(summary.direct, OcclusionFactors::OPEN);
    }

    #[test]
    fn early_direction_rotates_into_listener_frame() {
        // Early energy arrives from world +X; a listener yawed +90 deg about Y
        // maps world +X onto local -Z (forward).
        let mut probe = WaveParameters::ANECHOIC;
        probe = WaveParameters::new(
            probe.occlusion,
            probe.direct_gain,
            probe.direct_bands,
            probe.direct_cutoff_hz,
            0.5,
            BandGains::UNITY,
            Vec3::X,
            0.0,
            0.0,
            0.0,
        );
        let grid = WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            [2, 2, 2],
            probe,
        )
        .expect("valid grid");
        let backend = WaveFieldBackend::new(&grid);
        let listener = Listener::new(Vec3::ZERO, Quat::from_rotation_y(FRAC_PI_2), Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -3.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 2];
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert_eq!(summary.path_count, 2);
        // The listener faces world `-X`, so a world `+X` early direction arrives
        // from directly behind the head, i.e. local `+Z` (local forward is `-Z`).
        let dir = paths[1].direction;
        assert!((dir.x - 0.0).abs() < 1.0e-5);
        assert!((dir.z - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn empty_path_buffer_still_reports_direct_blocking() {
        let grid = WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            [2, 2, 2],
            blocked_probe(),
        )
        .expect("valid grid");
        let backend = WaveFieldBackend::new(&grid);
        let summary = backend.query(&Listener::default(), &Emitter::default(), &mut []);
        assert_eq!(summary.path_count, 0);
        assert_eq!(summary.direct, OcclusionFactors::new(0.5, 0.5));
    }

    #[test]
    fn reverb_query_reads_baked_send() {
        let grid = WaveProbeGrid::uniform(
            Vec3::ZERO,
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            [2, 2, 2],
            blocked_probe(),
        )
        .expect("valid grid");
        let backend = WaveFieldBackend::new(&grid);
        let reverb = backend.reverb(&Listener::default());
        assert!((reverb.wet_gain - 0.5).abs() < 1.0e-6);
        assert!((reverb.rt60_seconds - 1.2).abs() < 1.0e-6);
    }

    #[test]
    fn sampling_is_deterministic() {
        let open = WaveParameters::ANECHOIC;
        let dark = blocked_probe();
        let grid = WaveProbeGrid::new(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(2.0, 2.0, 2.0),
            [2, 1, 1],
            alloc::vec![open, dark],
        )
        .expect("valid grid");
        let p = Vec3::new(0.73, 1.0, 1.0);
        assert_eq!(grid.sample(p), grid.sample(p));
    }

    #[test]
    fn midpoint_direction_blends_and_renormalises() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let mid = lerp_direction(a, b, 0.5);
        assert!((mid.length() - 1.0).abs() < 1.0e-6);
        assert!((mid.x - mid.y).abs() < 1.0e-6);
    }
}
