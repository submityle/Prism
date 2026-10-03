//! All-round ambisonic decoding to an arbitrary array (design section 49.3).
//!
//! `AllRAD` decodes an Ambisonic (HOA) field to an irregular loudspeaker array
//! in two stages. First the field is decoded to a dense, near-uniform set of
//! *virtual* loudspeakers whose geometry is well conditioned for a sampling
//! decode. Then each virtual speaker signal is re-panned onto the real,
//! possibly sparse and irregular, array with vector-base amplitude panning.
//! The composed gain matrix is energy-normalised once at construction so a unit
//! plane wave produces unit total output power on average over direction.
//!
//! The virtual grid here is a Fibonacci sphere, a deterministic quasi-uniform
//! point set. It is not a spherical t-design; it is an easily reproduced
//! near-uniform approximation that keeps the virtual decode well conditioned.
//! Low and high frequency bands reuse the dual-band (basic and max-rE)
//! Ambisonic decoder from `prism_audio_spatial`.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the published all-round ambisonic decoding method (Zotter
//! and Frank 2012) and the Fibonacci-sphere point distribution.
//!
//! # Relationship
//!
//! Reuses `prism_audio_spatial::hoa_decode::{DualBandDecoder, DecodeBand}` for
//! the per-virtual-speaker decode and `prism_audio_object::pan::vbap` for the
//! virtual-to-real resampling. Consumes [`crate::array::layout::ArrayLayout`]
//! and is routed from [`crate::array::decode`] for HOA scene audio.

use alloc::vec::Vec;
use core::f32::consts::PI;

use bevy_math::{ops, Vec3};
use prism_audio_core::math::Sample;
use prism_audio_object::pan::vbap::vbap_gains;
use prism_audio_spatial::hoa::{encode_hoa, MAX_HOA_CHANNELS};
use prism_audio_spatial::hoa_decode::{DecodeBand, DualBandDecoder};

use crate::array::layout::ArrayLayout;

/// A deterministic quasi-uniform set of virtual loudspeaker directions.
///
/// The directions are generated with the Fibonacci-sphere mapping, which
/// spaces points by the golden angle so they cover the sphere evenly without
/// clustering. The grid is used only as an internal decode intermediary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VirtualSpeakerGrid {
    directions: Vec<Vec3>,
}

impl VirtualSpeakerGrid {
    /// Builds a Fibonacci-sphere grid of `count` directions.
    #[must_use]
    pub fn fibonacci(count: usize) -> Self {
        let mut directions = Vec::with_capacity(count);
        if count == 0 {
            return Self { directions };
        }
        let golden_angle = PI * (3.0 - ops::sqrt(5.0));
        let n = count as Sample;
        for index in 0..count {
            let i = index as Sample;
            let y = 1.0 - (i + 0.5) / n * 2.0;
            let radius = ops::sqrt((1.0 - y * y).max(0.0));
            let theta = golden_angle * i;
            let x = ops::cos(theta) * radius;
            let z = ops::sin(theta) * radius;
            directions.push(Vec3::new(x, y, z));
        }
        Self { directions }
    }

    /// Builds a grid from explicit unit directions.
    #[must_use]
    pub fn from_directions(directions: Vec<Vec3>) -> Self {
        Self { directions }
    }

    /// The virtual directions.
    #[must_use]
    pub fn directions(&self) -> &[Vec3] {
        &self.directions
    }

    /// The number of virtual loudspeakers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Whether the grid is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }
}

/// An energy-normalised `AllRAD` decoder for one array and Ambisonic order.
///
/// Construction precomputes, for every virtual loudspeaker, the amplitude
/// panning gains onto the real directional speakers, then calibrates a single
/// scalar so unit-power plane waves map to unit total output power on average.
///
/// # Examples
///
/// ```
/// use prism_audio_immersive::array::allrad::{AllRadDecoder, VirtualSpeakerGrid};
/// use prism_audio_immersive::array::layout::ArrayLayout;
/// use prism_audio_object::bed::BedLayout;
/// use prism_audio_spatial::hoa::{encode_hoa, MAX_HOA_CHANNELS};
/// use prism_audio_spatial::hoa_decode::DecodeBand;
/// use bevy_math::Vec3;
///
/// let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
/// let grid = VirtualSpeakerGrid::fibonacci(100);
/// let decoder = AllRadDecoder::new(&layout, 1, &grid);
/// let mut coeffs = [0.0f32; MAX_HOA_CHANNELS];
/// encode_hoa(Vec3::new(0.0, 0.0, -1.0), 1, &mut coeffs);
/// let feeds = decoder.decode(DecodeBand::High, &coeffs);
/// assert_eq!(feeds.len(), layout.len());
/// ```
#[derive(Clone, Debug)]
pub struct AllRadDecoder {
    slot_count: usize,
    directional_slots: Vec<usize>,
    virtual_dirs: Vec<Vec3>,
    gain_matrix: Vec<Vec<Sample>>,
    decoder: DualBandDecoder,
    norm: Sample,
}

impl AllRadDecoder {
    /// Builds a decoder for `layout` at Ambisonic `order` using `grid` virtual
    /// loudspeakers.
    #[must_use]
    pub fn new(layout: &ArrayLayout, order: usize, grid: &VirtualSpeakerGrid) -> Self {
        let directional_slots = layout.directional_indices();
        let real_dirs = layout.directional_directions();
        let virtual_dirs: Vec<Vec3> = grid.directions().to_vec();
        let mut gain_matrix = Vec::with_capacity(virtual_dirs.len());
        for &dir in &virtual_dirs {
            gain_matrix.push(vbap_gains(dir, &real_dirs));
        }
        let decoder = DualBandDecoder::new(order);
        let mut decoder_out = Self {
            slot_count: layout.len(),
            directional_slots,
            virtual_dirs,
            gain_matrix,
            decoder,
            norm: 1.0,
        };
        decoder_out.norm = decoder_out.calibrate(order);
        decoder_out
    }

    /// The number of layout slots (equal to the output length).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// The number of virtual loudspeakers.
    #[must_use]
    pub fn virtual_count(&self) -> usize {
        self.virtual_dirs.len()
    }

    /// The calibrated energy-normalisation scalar.
    #[must_use]
    pub fn normalization(&self) -> Sample {
        self.norm
    }

    /// Decodes one Ambisonic frame `coeffs` with `band` into `out`, writing one
    /// gain per layout slot. Only `min(slot_count, out.len())` slots are
    /// written, so a short `out` never panics. Allocation, lock, and panic
    /// free.
    pub fn decode_into(&self, band: DecodeBand, coeffs: &[Sample], out: &mut [Sample]) {
        let limit = self.slot_count.min(out.len());
        for cell in out.iter_mut().take(limit) {
            *cell = 0.0;
        }
        for (virtual_index, &dir) in self.virtual_dirs.iter().enumerate() {
            let signal = self.decoder.decode(band, coeffs, dir);
            let row = &self.gain_matrix[virtual_index];
            for (local_index, &slot) in self.directional_slots.iter().enumerate() {
                if slot < limit
                    && let Some(&gain) = row.get(local_index)
                {
                    out[slot] += gain * signal;
                }
            }
        }
        for &slot in &self.directional_slots {
            if slot < limit {
                out[slot] *= self.norm;
            }
        }
    }

    /// Decodes one Ambisonic frame `coeffs` with `band`, returning one gain per
    /// layout slot.
    #[must_use]
    pub fn decode(&self, band: DecodeBand, coeffs: &[Sample]) -> Vec<Sample> {
        let mut out = Vec::new();
        out.resize(self.slot_count, 0.0);
        self.decode_into(band, coeffs, &mut out);
        out
    }

    /// Measures the mean total output power for unit plane waves over the
    /// virtual directions (with the current gain matrix) and returns the scalar
    /// that drives that mean to one.
    fn calibrate(&self, order: usize) -> Sample {
        if self.virtual_dirs.is_empty() {
            return 1.0;
        }
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let mut accum_power = 0.0 as Sample;
        for &dir in &self.virtual_dirs {
            encode_hoa(dir, order, &mut coeffs);
            let mut raw = Vec::new();
            raw.resize(self.slot_count, 0.0);
            for (virtual_index, &vdir) in self.virtual_dirs.iter().enumerate() {
                let signal = self.decoder.decode(DecodeBand::High, &coeffs, vdir);
                let row = &self.gain_matrix[virtual_index];
                for (local_index, &slot) in self.directional_slots.iter().enumerate() {
                    if let Some(&gain) = row.get(local_index) {
                        raw[slot] += gain * signal;
                    }
                }
            }
            let power: Sample = raw.iter().map(|&g| g * g).sum();
            accum_power += power;
        }
        let mean_power = accum_power / self.virtual_dirs.len() as Sample;
        if mean_power > 1.0e-12 {
            1.0 / ops::sqrt(mean_power)
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_object::bed::BedLayout;
    use prism_audio_spatial::hoa::hoa_channel_count;

    use crate::array::layout::ArrayLayout;

    const EPS: Sample = 1.0e-3;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn fibonacci_points_are_unit_length() {
        let grid = VirtualSpeakerGrid::fibonacci(64);
        assert_eq!(grid.len(), 64);
        for &dir in grid.directions() {
            assert!(close(dir.length(), 1.0));
        }
    }

    #[test]
    fn fibonacci_zero_is_empty() {
        let grid = VirtualSpeakerGrid::fibonacci(0);
        assert!(grid.is_empty());
    }

    #[test]
    fn decode_length_matches_slot_count() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
        let grid = VirtualSpeakerGrid::fibonacci(64);
        let decoder = AllRadDecoder::new(&layout, 1, &grid);
        let coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let out = decoder.decode(DecodeBand::High, &coeffs);
        assert_eq!(out.len(), layout.len());
    }

    #[test]
    fn plane_wave_energy_is_near_unity() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
        let grid = VirtualSpeakerGrid::fibonacci(100);
        let order = 1;
        let decoder = AllRadDecoder::new(&layout, order, &grid);
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let dir = Vec3::new(0.0, 0.0, -1.0);
        encode_hoa(dir, order, &mut coeffs);
        let gains = decoder.decode(DecodeBand::High, &coeffs);
        let power: Sample = gains.iter().map(|&g| g * g).sum();
        assert!(power > 0.3 && power < 3.0, "power was {power}");
    }

    #[test]
    fn short_output_slice_does_not_panic() {
        let layout = ArrayLayout::from_bed(BedLayout::Stereo);
        let grid = VirtualSpeakerGrid::fibonacci(32);
        let decoder = AllRadDecoder::new(&layout, 1, &grid);
        let coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let mut out = [0.0 as Sample; 1];
        decoder.decode_into(DecodeBand::Low, &coeffs, &mut out);
        assert_eq!(hoa_channel_count(1), 4);
    }
}
