//! `GPU`-side packing and atlas layout for a multi-cascade ocean spectrum.
//!
//! [`build_cascade_spectra`](super::initial_spectrum::build_cascade_spectra)
//! draws several band-limited [`OceanSpectrumField`]s — one per cascade — the
//! way `UE5` Water, `Crest` and `WaveWorks` stack several `FFT` patches to
//! resolve waves across scales. Those fields are pure `CPU` data: a flat
//! per-cascade `h0` amplitude pair plus the cascade's spatial patch size.
//!
//! The `GPU` spectral pass cannot bind a ragged list of per-cascade buffers and
//! textures; it wants one contiguous amplitude pool and one output atlas whose
//! tiles it addresses by a per-cascade offset. This module is the single
//! responsibility that turns the ragged `CPU` cascade set into that flat,
//! device-friendly layout without inventing or dropping any amplitude:
//!
//! * [`CascadeAtlasLayout`] is the float-free index arithmetic — where cascade
//!   `c`'s amplitudes begin in the concatenated `h0` pool and where its
//!   `N x N` displacement/normal tile sits in the stacked atlas texture.
//! * [`PackedCascades`] is that layout plus the concatenated `h0` / `h0_neg`
//!   amplitude pools (as the `[f32; 2]` lanes the shader reads) and the
//!   per-cascade patch sizes the spectral uniform needs.
//!
//! The layout is a vertical stack: cascade `c` owns rows `[c*N, (c+1)*N)` of an
//! `N`-wide, `(N*cascade_count)`-tall atlas. Stacking vertically keeps each
//! cascade's row-major `N x N` tile contiguous, so a tile's texel `(x, y)`
//! maps to atlas texel `(x, c*N + y)` with a single row bias and no per-column
//! arithmetic. The concatenated amplitude pools mirror that order exactly, so
//! cascade `c`'s amplitudes occupy the half-open element range
//! `[c*N*N, (c+1)*N*N)`.

use alloc::vec::Vec;

use super::initial_spectrum::OceanSpectrumField;

/// Float-free index arithmetic for a stacked multi-cascade spectrum atlas.
///
/// A layout is fully described by the shared per-cascade resolution `N` and the
/// cascade count `M`. Every offset it returns is a plain element or texel
/// index, so the host can size buffers and textures and the recorder can pick a
/// cascade's slice without touching a float. All accessors saturate rather than
/// overflow, and an out-of-range cascade index clamps to the last cascade so a
/// caller can never address past the atlas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CascadeAtlasLayout {
    /// Shared per-cascade grid resolution `N` (each cascade is `N x N`).
    resolution: u32,
    /// Number of stacked cascades `M` (at least one for a live ocean).
    cascade_count: u32,
}

impl CascadeAtlasLayout {
    /// Builds a layout for `cascade_count` cascades of resolution `resolution`.
    ///
    /// A zero resolution or zero cascade count is a degenerate (no-op) layout:
    /// every size and offset it reports is zero, matching the honest empty
    /// ocean the rest of the water stack produces for a calm sea.
    #[must_use]
    pub fn new(resolution: u32, cascade_count: u32) -> Self {
        Self {
            resolution,
            cascade_count,
        }
    }

    /// Shared per-cascade resolution `N`.
    #[must_use]
    pub fn resolution(self) -> u32 {
        self.resolution
    }

    /// Number of stacked cascades `M`.
    #[must_use]
    pub fn cascade_count(self) -> u32 {
        self.cascade_count
    }

    /// Whether the layout carries no cascade or no grid (a degenerate ocean).
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.resolution == 0 || self.cascade_count == 0
    }

    /// Complex amplitudes in one cascade (`N * N`), saturating on overflow.
    #[must_use]
    pub fn texels_per_cascade(self) -> u32 {
        self.resolution.saturating_mul(self.resolution)
    }

    /// Complex amplitudes across every cascade (`M * N * N`), saturating.
    #[must_use]
    pub fn total_texels(self) -> u32 {
        self.texels_per_cascade().saturating_mul(self.cascade_count)
    }

    /// First amplitude element of cascade `c` in the concatenated `h0` pool
    /// (`c * N * N`). An out-of-range index clamps to the last cascade.
    #[must_use]
    pub fn h0_offset(self, cascade: u32) -> u32 {
        self.clamped(cascade)
            .saturating_mul(self.texels_per_cascade())
    }

    /// Width of the stacked atlas texture (`N`).
    #[must_use]
    pub fn atlas_width(self) -> u32 {
        self.resolution
    }

    /// Height of the stacked atlas texture (`M * N`), saturating on overflow.
    #[must_use]
    pub fn atlas_height(self) -> u32 {
        self.resolution.saturating_mul(self.cascade_count)
    }

    /// Top-left atlas texel `(x, y)` of cascade `c`'s `N x N` tile.
    ///
    /// The stack is vertical, so `x` is always `0` and `y` is `c * N`. An
    /// out-of-range index clamps to the last cascade's tile.
    #[must_use]
    pub fn tile_origin(self, cascade: u32) -> (u32, u32) {
        (0, self.clamped(cascade).saturating_mul(self.resolution))
    }

    /// Clamps a cascade index into `[0, cascade_count - 1]`, or `0` when the
    /// layout is degenerate, so no accessor can address past the atlas.
    #[must_use]
    fn clamped(self, cascade: u32) -> u32 {
        if self.cascade_count == 0 {
            0
        } else {
            cascade.min(self.cascade_count - 1)
        }
    }
}

/// A ragged `CPU` cascade set flattened into the contiguous pools and atlas
/// layout the `GPU` spectral pass binds.
///
/// The amplitude pools are row-major per cascade and concatenated in cascade
/// order (coarsest first, matching
/// [`build_cascade_spectra`](super::initial_spectrum::build_cascade_spectra)),
/// so cascade `c` occupies the element range described by
/// [`CascadeAtlasLayout::h0_offset`]. `patch_sizes[c]` is the spatial patch `L`
/// (m) the spectral uniform feeds cascade `c`'s inverse `FFT`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackedCascades {
    /// The index arithmetic describing this packing.
    pub layout: CascadeAtlasLayout,
    /// Concatenated `h0(+k)` amplitudes as `[real, imaginary]` lanes.
    pub h0: Vec<[f32; 2]>,
    /// Concatenated `h0(-k)` amplitudes as `[real, imaginary]` lanes.
    pub h0_neg: Vec<[f32; 2]>,
    /// Per-cascade spatial patch size `L` (m), in cascade order.
    pub patch_sizes: Vec<f32>,
}

/// Flattens a `CPU` cascade set into device-friendly [`PackedCascades`].
///
/// Returns [`None`] for a set that cannot be packed into one atlas: an empty
/// set, a set whose cascades disagree on resolution (a stacked atlas needs one
/// shared `N`), or a set with any degenerate (zero-resolution or empty) field.
/// This is an honest no-op refusal rather than a silently truncated or padded
/// pack that would desynchronise the amplitude pools from the atlas tiles.
///
/// On success the concatenated pools carry exactly `M * N * N` amplitudes each
/// and `patch_sizes` carries one entry per cascade, so the returned layout's
/// [`total_texels`](CascadeAtlasLayout::total_texels) equals both pool lengths.
#[must_use]
pub fn pack_cascades(fields: &[OceanSpectrumField]) -> Option<PackedCascades> {
    let first = fields.first()?;
    let resolution = first.resolution;
    if resolution == 0 {
        return None;
    }
    let texels = (resolution as usize).checked_mul(resolution as usize)?;

    // Every cascade must share the resolution and carry a full grid, or the
    // stacked atlas tiles and the concatenated pools would drift apart.
    for field in fields {
        if field.resolution != resolution
            || field.h0.len() != texels
            || field.h0_neg.len() != texels
        {
            return None;
        }
    }

    let cascade_count = fields.len();
    let mut h0 = Vec::with_capacity(cascade_count * texels);
    let mut h0_neg = Vec::with_capacity(cascade_count * texels);
    let mut patch_sizes = Vec::with_capacity(cascade_count);
    for field in fields {
        for amp in &field.h0 {
            h0.push([amp.re, amp.im]);
        }
        for amp in &field.h0_neg {
            h0_neg.push([amp.re, amp.im]);
        }
        patch_sizes.push(field.patch_size);
    }

    Some(PackedCascades {
        layout: CascadeAtlasLayout::new(resolution, cascade_count as u32),
        h0,
        h0_neg,
        patch_sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::initial_spectrum::build_cascade_spectra;
    use crate::water::spectrum::{SpectrumKind, SpectrumParams};
    use crate::water::Vec2;

    fn sea(wind_speed: f32) -> SpectrumParams {
        SpectrumParams {
            kind: SpectrumKind::Phillips,
            wind: Vec2::new(wind_speed, 0.0),
            amplitude: 0.5,
            peak_enhancement: 1.0,
            min_wavelength: 0.2,
            directional_exponent: 2,
        }
    }

    #[test]
    fn layout_offsets_stack_vertically_and_tile_contiguously() {
        let layout = CascadeAtlasLayout::new(64, 4);
        assert_eq!(layout.texels_per_cascade(), 64 * 64);
        assert_eq!(layout.total_texels(), 4 * 64 * 64);
        assert_eq!(layout.atlas_width(), 64);
        assert_eq!(layout.atlas_height(), 4 * 64);
        for c in 0..4 {
            assert_eq!(layout.h0_offset(c), c * 64 * 64);
            assert_eq!(layout.tile_origin(c), (0, c * 64));
        }
    }

    #[test]
    fn layout_clamps_out_of_range_cascade_to_the_last_tile() {
        let layout = CascadeAtlasLayout::new(32, 3);
        // Index 9 is past the end; it clamps to cascade 2 rather than
        // addressing past the atlas.
        assert_eq!(layout.h0_offset(9), 2 * 32 * 32);
        assert_eq!(layout.tile_origin(9), (0, 2 * 32));
    }

    #[test]
    fn degenerate_layout_is_all_zero_and_empty() {
        let empty_cascades = CascadeAtlasLayout::new(64, 0);
        assert!(empty_cascades.is_empty());
        assert_eq!(empty_cascades.total_texels(), 0);
        assert_eq!(empty_cascades.atlas_height(), 0);
        assert_eq!(empty_cascades.h0_offset(0), 0);

        let empty_grid = CascadeAtlasLayout::new(0, 4);
        assert!(empty_grid.is_empty());
        assert_eq!(empty_grid.total_texels(), 0);
        assert_eq!(empty_grid.atlas_width(), 0);
    }

    #[test]
    fn pack_concatenates_cascades_in_order_with_matching_lengths() {
        let fields = build_cascade_spectra(32, 256.0, 4.0, 4, sea(12.0), 7);
        assert_eq!(fields.len(), 4);
        let packed = pack_cascades(&fields).expect("a uniform cascade set packs");

        assert_eq!(packed.layout.resolution(), 32);
        assert_eq!(packed.layout.cascade_count(), 4);
        assert_eq!(packed.h0.len(), 4 * 32 * 32);
        assert_eq!(packed.h0_neg.len(), 4 * 32 * 32);
        assert_eq!(packed.patch_sizes.len(), 4);

        // The concatenated pool length equals the layout's total, and each
        // cascade's slice lands at its layout offset with the source amplitudes.
        assert_eq!(packed.h0.len() as u32, packed.layout.total_texels());
        for (c, field) in fields.iter().enumerate() {
            let base = packed.layout.h0_offset(c as u32) as usize;
            assert_eq!(packed.h0[base][0], field.h0[0].re);
            assert_eq!(packed.h0[base][1], field.h0[0].im);
        }
    }

    #[test]
    fn pack_records_geometrically_shrinking_patch_sizes() {
        let fields = build_cascade_spectra(32, 256.0, 4.0, 4, sea(12.0), 7);
        let packed = pack_cascades(&fields).expect("packs");
        // Cascade 0 is the coarsest (largest patch); each finer cascade shrinks
        // the patch by the ratio, so the recorded sizes are strictly decreasing.
        for pair in packed.patch_sizes.windows(2) {
            assert!(
                pair[0] > pair[1],
                "patch {} must exceed the finer {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn pack_refuses_an_empty_or_mismatched_set() {
        assert!(pack_cascades(&[]).is_none());

        // Two cascades of different resolution cannot share one atlas.
        let mut mixed = build_cascade_spectra(32, 256.0, 4.0, 1, sea(12.0), 7);
        mixed.extend(build_cascade_spectra(16, 64.0, 4.0, 1, sea(12.0), 8));
        assert_eq!(mixed.len(), 2);
        assert!(pack_cascades(&mixed).is_none());
    }

    #[test]
    fn single_cascade_packs_to_one_full_tile() {
        let fields = build_cascade_spectra(48, 400.0, 4.0, 1, sea(9.0), 3);
        let packed = pack_cascades(&fields).expect("a lone cascade packs");
        assert_eq!(packed.layout.cascade_count(), 1);
        assert_eq!(packed.layout.atlas_height(), 48);
        assert_eq!(packed.h0.len(), 48 * 48);
        assert_eq!(packed.patch_sizes.len(), 1);
    }
}
