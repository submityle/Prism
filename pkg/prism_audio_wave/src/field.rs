//! Parameter field: the compact, quantised, per-probe storage of perceptual
//! parameters produced by a bake and consumed by the runtime lookup.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "compact perceptual parameter field" and the "data-scale
//! governance" (grid-resolution tiers, quantisation bit depth, and a
//! compression/streaming surface) of design section 43. Each probe of a
//! [`ProbeGrid`] stores a fixed-point encoding of a
//! [`PerceptualParams`]; [`crate::lookup`] decodes and trilinearly
//! interpolates them at runtime.

use alloc::vec;
use alloc::vec::Vec;

use crate::encoding::PerceptualParams;
use crate::grid::{Aabb, ProbeGrid};

/// Number of scalar parameters stored per probe (the fields of
/// [`PerceptualParams`]).
pub const FIELDS_PER_PROBE: usize = 7;

/// Quantisation bit depth for the stored parameter codes.
///
/// Deeper codes trade storage for precision; eight bits is adequate for
/// coarse ambience, sixteen bits is effectively lossless for these ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BitDepth {
    /// Eight bits per parameter (256 levels).
    Eight,
    /// Twelve bits per parameter (4096 levels).
    Twelve,
    /// Sixteen bits per parameter (65536 levels).
    Sixteen,
}

impl BitDepth {
    /// The number of bits per stored parameter.
    #[must_use]
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Twelve => 12,
            BitDepth::Sixteen => 16,
        }
    }

    /// The number of distinct quantisation levels (`2^bits`).
    #[must_use]
    #[inline]
    pub fn levels(self) -> u32 {
        1u32 << self.bits()
    }
}

/// Probe-grid resolution tier, mapping a physical spacing to probe counts.
///
/// Coarser tiers cut storage and bake time at the cost of spatial detail; the
/// tier chooses a target spacing that [`GridTier::probe_counts_for`] turns into
/// per-axis probe counts for a given bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GridTier {
    /// Coarse tier: two-metre probe spacing.
    Coarse,
    /// Medium tier: one-metre probe spacing.
    Medium,
    /// Fine tier: half-metre probe spacing.
    Fine,
}

impl GridTier {
    /// Target probe spacing in metres for this tier.
    #[must_use]
    #[inline]
    pub fn spacing_m(self) -> f32 {
        match self {
            GridTier::Coarse => 2.0,
            GridTier::Medium => 1.0,
            GridTier::Fine => 0.5,
        }
    }

    /// Per-axis probe counts covering `bounds` at this tier's spacing.
    ///
    /// Each axis gets `ceil(extent / spacing) + 1` probes (so both faces are
    /// sampled), clamped to at least `1`.
    #[must_use]
    pub fn probe_counts_for(self, bounds: &Aabb) -> [u32; 3] {
        let spacing = self.spacing_m().max(1.0e-3);
        let size = bounds.size();
        let axis = |extent: f32| -> u32 {
            if extent <= 0.0 {
                return 1;
            }
            // Integer ceil without float rounding surprises.
            let cells = (extent / spacing) as u32;
            let covered = cells as f32 * spacing;
            let cells = if covered < extent { cells + 1 } else { cells };
            cells + 1
        };
        [axis(size.x), axis(size.y), axis(size.z)]
    }
}

/// Per-parameter `(min, max)` encoding ranges.
///
/// Quantisation maps each parameter linearly from its range onto the integer
/// code space; a tight range keeps the fixed-point error small.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ProbeRanges {
    /// Direct-gain range.
    pub direct_gain: (f32, f32),
    /// Direct low-pass corner range in Hz.
    pub direct_cutoff_hz: (f32, f32),
    /// Reverberation-time (`RT60`) range in seconds.
    pub rt60_s: (f32, f32),
    /// Wet-send gain range.
    pub wet_gain: (f32, f32),
    /// Azimuth range in radians.
    pub azimuth: (f32, f32),
    /// Elevation range in radians.
    pub elevation: (f32, f32),
    /// Direct-to-reverberant ratio (`DRR`) range in decibels.
    pub drr_db: (f32, f32),
}

impl ProbeRanges {
    /// A set of generous default ranges covering the full expected span of
    /// each parameter.
    pub const DEFAULT: Self = Self {
        direct_gain: (0.0, 1.0),
        direct_cutoff_hz: (20.0, 20_000.0),
        rt60_s: (0.0, 10.0),
        wet_gain: (0.0, 1.0),
        azimuth: (-core::f32::consts::PI, core::f32::consts::PI),
        elevation: (
            -core::f32::consts::FRAC_PI_2,
            core::f32::consts::FRAC_PI_2,
        ),
        drr_db: (-60.0, 60.0),
    };

    /// Returns the `(min, max)` pair for the parameter at slot `i`
    /// (`0..FIELDS_PER_PROBE`), with slots beyond the last clamped to the last.
    #[must_use]
    fn slot(&self, i: usize) -> (f32, f32) {
        match i {
            0 => self.direct_gain,
            1 => self.direct_cutoff_hz,
            2 => self.rt60_s,
            3 => self.wet_gain,
            4 => self.azimuth,
            5 => self.elevation,
            _ => self.drr_db,
        }
    }
}

impl Default for ProbeRanges {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Decomposes a [`PerceptualParams`] into its seven scalar slots in storage
/// order.
#[must_use]
fn params_to_slots(p: &PerceptualParams) -> [f32; FIELDS_PER_PROBE] {
    [
        p.direct_gain,
        p.direct_cutoff_hz,
        p.rt60_s,
        p.wet_gain,
        p.azimuth,
        p.elevation,
        p.drr_db,
    ]
}

/// Reassembles a [`PerceptualParams`] from its seven scalar slots.
#[must_use]
fn slots_to_params(s: [f32; FIELDS_PER_PROBE]) -> PerceptualParams {
    PerceptualParams {
        direct_gain: s[0],
        direct_cutoff_hz: s[1],
        rt60_s: s[2],
        wet_gain: s[3],
        azimuth: s[4],
        elevation: s[5],
        drr_db: s[6],
    }
}

/// Quantises `v` from `[min, max]` to an integer code in `[0, levels - 1]`.
#[must_use]
fn quantize(v: f32, min: f32, max: f32, levels: u32) -> u16 {
    let span = max - min;
    let last = levels.saturating_sub(1).max(1);
    if span <= 0.0 {
        return 0;
    }
    let t = ((v - min) / span).clamp(0.0, 1.0);
    // `t * last` is in `[0, last]`; add 0.5 and truncate for round-to-nearest
    // without routing through a float intrinsic.
    let idx = (t * last as f32 + 0.5) as u32;
    idx.min(last) as u16
}

/// Dequantises an integer `code` back to a value in `[min, max]`.
#[must_use]
fn dequantize(code: u16, min: f32, max: f32, levels: u32) -> f32 {
    let span = max - min;
    let last = levels.saturating_sub(1).max(1);
    min + (code as f32 / last as f32) * span
}

/// A baked, quantised parameter field over a probe grid.
///
/// Storage is `FIELDS_PER_PROBE` integer codes per probe plus the shared
/// [`ProbeRanges`]; decoding is allocation free, so the runtime lookup can
/// read any probe without touching the heap.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterField {
    grid: ProbeGrid,
    ranges: ProbeRanges,
    bit_depth: BitDepth,
    // `probe_count * FIELDS_PER_PROBE` codes, probe-major.
    data: Vec<u16>,
}

impl ParameterField {
    /// The probe grid the field is sampled on.
    #[must_use]
    #[inline]
    pub fn grid(&self) -> &ProbeGrid {
        &self.grid
    }

    /// The quantisation bit depth of the stored codes.
    #[must_use]
    #[inline]
    pub fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// The per-parameter encoding ranges.
    #[must_use]
    #[inline]
    pub fn ranges(&self) -> ProbeRanges {
        self.ranges
    }

    /// Number of probes in the field.
    #[must_use]
    #[inline]
    pub fn probe_count(&self) -> usize {
        self.grid.probe_count()
    }

    /// Decodes probe `index` back into a [`PerceptualParams`].
    ///
    /// This performs no allocation and never panics; an out-of-range index
    /// clamps to the last probe.
    #[must_use]
    pub fn decode(&self, index: usize) -> PerceptualParams {
        let count = self.probe_count();
        if count == 0 {
            return PerceptualParams::OPEN;
        }
        let probe = index.min(count - 1);
        let base = probe * FIELDS_PER_PROBE;
        let levels = self.bit_depth.levels();
        let mut slots = [0.0_f32; FIELDS_PER_PROBE];
        for (i, slot) in slots.iter_mut().enumerate() {
            let (min, max) = self.ranges.slot(i);
            *slot = dequantize(self.data[base + i], min, max, levels);
        }
        slots_to_params(slots)
    }

    /// The number of bytes [`ParameterField::to_packed_bytes`] produces.
    #[must_use]
    pub fn packed_byte_len(&self) -> usize {
        let value_count = self.data.len();
        let data_bits = value_count * self.bit_depth.bits() as usize;
        HEADER_BYTES + data_bits.div_ceil(8)
    }

    /// Serialises the field into a self-describing, bit-packed byte stream.
    ///
    /// The codes are packed at the field's native bit depth (so an eight-bit
    /// field is half the size of a sixteen-bit one), prefixed by a small
    /// header describing the grid, ranges, and depth. This is the
    /// compression/streaming surface of design section 43; the result round
    /// trips through [`ParameterField::from_packed_bytes`].
    #[must_use]
    pub fn to_packed_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.packed_byte_len());
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.push(self.bit_depth.bits() as u8);
        let bounds = self.grid.bounds();
        let counts = self.grid.counts();
        for c in counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for v in [
            bounds.min.x,
            bounds.min.y,
            bounds.min.z,
            bounds.max.x,
            bounds.max.y,
            bounds.max.z,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..FIELDS_PER_PROBE {
            let (min, max) = self.ranges.slot(i);
            out.extend_from_slice(&min.to_le_bytes());
            out.extend_from_slice(&max.to_le_bytes());
        }
        // Bit-pack the codes, least-significant-bit first.
        let bits = self.bit_depth.bits();
        let mut acc = 0u32;
        let mut nbits = 0u32;
        for &code in &self.data {
            acc |= (code as u32) << nbits;
            nbits += bits;
            while nbits >= 8 {
                out.push((acc & 0xFF) as u8);
                acc >>= 8;
                nbits -= 8;
            }
        }
        if nbits > 0 {
            out.push((acc & 0xFF) as u8);
        }
        out
    }

    /// Reconstructs a field from the byte stream produced by
    /// [`ParameterField::to_packed_bytes`].
    ///
    /// Returns `None` if the stream is truncated or carries a bad header.
    #[must_use]
    pub fn from_packed_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_BYTES {
            return None;
        }
        if bytes[0..4] != MAGIC || bytes[4] != FORMAT_VERSION {
            return None;
        }
        let bit_depth = match bytes[5] {
            8 => BitDepth::Eight,
            12 => BitDepth::Twelve,
            16 => BitDepth::Sixteen,
            _ => return None,
        };
        let mut cur = 6;
        let read_u32 = |cur: &mut usize| -> u32 {
            let v = u32::from_le_bytes([
                bytes[*cur],
                bytes[*cur + 1],
                bytes[*cur + 2],
                bytes[*cur + 3],
            ]);
            *cur += 4;
            v
        };
        let read_f32 = |cur: &mut usize| -> f32 {
            let v = f32::from_le_bytes([
                bytes[*cur],
                bytes[*cur + 1],
                bytes[*cur + 2],
                bytes[*cur + 3],
            ]);
            *cur += 4;
            v
        };
        let nx = read_u32(&mut cur);
        let ny = read_u32(&mut cur);
        let nz = read_u32(&mut cur);
        let min = bevy_math::Vec3::new(
            read_f32(&mut cur),
            read_f32(&mut cur),
            read_f32(&mut cur),
        );
        let max = bevy_math::Vec3::new(
            read_f32(&mut cur),
            read_f32(&mut cur),
            read_f32(&mut cur),
        );
        let mut slots = [(0.0_f32, 0.0_f32); FIELDS_PER_PROBE];
        for slot in &mut slots {
            let lo = read_f32(&mut cur);
            let hi = read_f32(&mut cur);
            *slot = (lo, hi);
        }
        let ranges = ProbeRanges {
            direct_gain: slots[0],
            direct_cutoff_hz: slots[1],
            rt60_s: slots[2],
            wet_gain: slots[3],
            azimuth: slots[4],
            elevation: slots[5],
            drr_db: slots[6],
        };
        let grid = ProbeGrid::new(Aabb::new(min, max), nx, ny, nz);
        let value_count = grid.probe_count() * FIELDS_PER_PROBE;
        let bits = bit_depth.bits();
        let mut data = vec![0u16; value_count];
        let mut acc = 0u32;
        let mut nbits = 0u32;
        let mask = ((1u64 << bits) - 1) as u32;
        let mut byte = cur;
        for code in &mut data {
            while nbits < bits {
                let next = *bytes.get(byte)? as u32;
                acc |= next << nbits;
                nbits += 8;
                byte += 1;
            }
            *code = (acc & mask) as u16;
            acc >>= bits;
            nbits -= bits;
        }
        Some(Self {
            grid,
            ranges,
            bit_depth,
            data,
        })
    }
}

/// Four-byte stream magic (`"PWF1"`).
const MAGIC: [u8; 4] = *b"PWF1";
/// Stream format version.
const FORMAT_VERSION: u8 = 1;
/// Fixed header size: magic + version + depth + 3 counts + 6 bounds floats +
/// 14 range floats.
const HEADER_BYTES: usize = 4 + 1 + 1 + 3 * 4 + 6 * 4 + FIELDS_PER_PROBE * 2 * 4;

/// Builder that accumulates per-probe [`PerceptualParams`] before quantising
/// them into a finished [`ParameterField`].
///
/// The builder keeps full-precision parameters so [`ParameterFieldBuilder::build`]
/// can derive tight encoding ranges from the actual data.
#[derive(Debug, Clone)]
pub struct ParameterFieldBuilder {
    grid: ProbeGrid,
    probes: Vec<PerceptualParams>,
}

impl ParameterFieldBuilder {
    /// Starts a builder for `grid`, with every probe initialised to
    /// [`PerceptualParams::OPEN`].
    #[must_use]
    pub fn new(grid: ProbeGrid) -> Self {
        let probes = vec![PerceptualParams::OPEN; grid.probe_count()];
        Self { grid, probes }
    }

    /// The probe grid being built.
    #[must_use]
    #[inline]
    pub fn grid(&self) -> &ProbeGrid {
        &self.grid
    }

    /// Stores `params` at probe `index`. Out-of-range indices are ignored.
    pub fn set_probe(&mut self, index: usize, params: PerceptualParams) {
        if let Some(slot) = self.probes.get_mut(index) {
            *slot = params;
        }
    }

    /// Reads back the full-precision parameters at probe `index`, clamping an
    /// out-of-range index to the last probe.
    #[must_use]
    pub fn probe(&self, index: usize) -> PerceptualParams {
        if self.probes.is_empty() {
            return PerceptualParams::OPEN;
        }
        self.probes[index.min(self.probes.len() - 1)]
    }

    /// Derives tight per-parameter ranges from the stored probes, padding each
    /// span slightly so no range collapses to a single level.
    #[must_use]
    fn derive_ranges(&self) -> ProbeRanges {
        let mut lo = [f32::INFINITY; FIELDS_PER_PROBE];
        let mut hi = [f32::NEG_INFINITY; FIELDS_PER_PROBE];
        for p in &self.probes {
            let slots = params_to_slots(p);
            for i in 0..FIELDS_PER_PROBE {
                lo[i] = lo[i].min(slots[i]);
                hi[i] = hi[i].max(slots[i]);
            }
        }
        let pair = |i: usize| -> (f32, f32) {
            if !lo[i].is_finite() || !hi[i].is_finite() {
                return ProbeRanges::DEFAULT.slot(i);
            }
            let mut a = lo[i];
            let mut b = hi[i];
            if (b - a) <= 1.0e-6 {
                // Pad a degenerate range symmetrically.
                let pad = a.abs().max(1.0) * 1.0e-3 + 1.0e-3;
                a -= pad;
                b += pad;
            }
            (a, b)
        };
        ProbeRanges {
            direct_gain: pair(0),
            direct_cutoff_hz: pair(1),
            rt60_s: pair(2),
            wet_gain: pair(3),
            azimuth: pair(4),
            elevation: pair(5),
            drr_db: pair(6),
        }
    }

    /// Quantises every probe at `bit_depth` using automatically derived ranges.
    #[must_use]
    pub fn build(&self, bit_depth: BitDepth) -> ParameterField {
        let ranges = self.derive_ranges();
        self.build_with_ranges(bit_depth, ranges)
    }

    /// Quantises every probe at `bit_depth` using caller-supplied `ranges`.
    #[must_use]
    pub fn build_with_ranges(&self, bit_depth: BitDepth, ranges: ProbeRanges) -> ParameterField {
        let levels = bit_depth.levels();
        let mut data = vec![0u16; self.probes.len() * FIELDS_PER_PROBE];
        for (p, chunk) in self
            .probes
            .iter()
            .zip(data.chunks_mut(FIELDS_PER_PROBE))
        {
            let slots = params_to_slots(p);
            for (i, dst) in chunk.iter_mut().enumerate() {
                let (min, max) = ranges.slot(i);
                *dst = quantize(slots[i], min, max, levels);
            }
        }
        ParameterField {
            grid: self.grid,
            ranges,
            bit_depth,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn demo_grid() -> ProbeGrid {
        ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(4.0)), 3, 3, 3)
    }

    #[test]
    fn bit_depth_levels() {
        assert_eq!(BitDepth::Eight.levels(), 256);
        assert_eq!(BitDepth::Twelve.levels(), 4096);
        assert_eq!(BitDepth::Sixteen.levels(), 65_536);
    }

    #[test]
    fn grid_tier_counts_cover_bounds() {
        let bounds = Aabb::new(Vec3::ZERO, Vec3::new(4.0, 2.0, 1.0));
        let coarse = GridTier::Coarse.probe_counts_for(&bounds);
        let fine = GridTier::Fine.probe_counts_for(&bounds);
        // Finer spacing never yields fewer probes.
        for a in 0..3 {
            assert!(fine[a] >= coarse[a]);
            assert!(coarse[a] >= 1);
        }
        // 4 m at 2 m spacing -> 2 cells -> 3 probes.
        assert_eq!(coarse[0], 3);
    }

    #[test]
    fn quantize_dequantize_round_trips_within_step() {
        let levels = BitDepth::Sixteen.levels();
        for v in [0.0_f32, 0.25, 0.5, 0.9999, 1.0] {
            let code = quantize(v, 0.0, 1.0, levels);
            let back = dequantize(code, 0.0, 1.0, levels);
            assert!(approx(back, v, 1.0 / (levels - 1) as f32 + 1e-6));
        }
    }

    #[test]
    fn field_decode_recovers_params() {
        let grid = demo_grid();
        let mut builder = ParameterFieldBuilder::new(grid);
        let p = PerceptualParams {
            direct_gain: 0.75,
            direct_cutoff_hz: 4_000.0,
            rt60_s: 1.2,
            wet_gain: 0.4,
            azimuth: 0.6,
            elevation: -0.3,
            drr_db: 5.0,
        };
        builder.set_probe(5, p);
        let field = builder.build(BitDepth::Sixteen);
        let decoded = field.decode(5);
        assert!(approx(decoded.direct_gain, p.direct_gain, 1e-3));
        assert!(approx(decoded.direct_cutoff_hz, p.direct_cutoff_hz, 2.0));
        assert!(approx(decoded.rt60_s, p.rt60_s, 1e-2));
        assert!(approx(decoded.azimuth, p.azimuth, 1e-3));
        assert!(approx(decoded.elevation, p.elevation, 1e-3));
    }

    #[test]
    fn out_of_range_decode_clamps() {
        let field = ParameterFieldBuilder::new(demo_grid()).build(BitDepth::Eight);
        let a = field.decode(field.probe_count() + 100);
        let b = field.decode(field.probe_count() - 1);
        assert_eq!(a, b);
    }

    #[test]
    fn packed_bytes_round_trip() {
        let grid = demo_grid();
        let mut builder = ParameterFieldBuilder::new(grid);
        for i in 0..builder.grid().probe_count() {
            let t = i as f32 / 26.0;
            builder.set_probe(
                i,
                PerceptualParams {
                    direct_gain: t,
                    direct_cutoff_hz: 1_000.0 + 3_000.0 * t,
                    rt60_s: 0.5 + t,
                    wet_gain: 1.0 - t,
                    azimuth: -1.0 + 2.0 * t,
                    elevation: 0.5 - t,
                    drr_db: -10.0 + 20.0 * t,
                },
            );
        }
        for depth in [BitDepth::Eight, BitDepth::Twelve, BitDepth::Sixteen] {
            let field = builder.build(depth);
            let bytes = field.to_packed_bytes();
            assert_eq!(bytes.len(), field.packed_byte_len());
            let restored = ParameterField::from_packed_bytes(&bytes).expect("round trip");
            assert_eq!(restored, field);
        }
    }

    #[test]
    fn eight_bit_is_smaller_than_sixteen_bit() {
        let field8 = ParameterFieldBuilder::new(demo_grid()).build(BitDepth::Eight);
        let field16 = ParameterFieldBuilder::new(demo_grid()).build(BitDepth::Sixteen);
        assert!(field8.packed_byte_len() < field16.packed_byte_len());
    }

    #[test]
    fn bad_header_is_rejected() {
        assert!(ParameterField::from_packed_bytes(&[0, 1, 2]).is_none());
        assert!(ParameterField::from_packed_bytes(b"XXXX\x01\x08").is_none());
    }
}
