//! The per-cascade evolve → butterfly → assemble schedule for the spectral
//! ocean, built on the separable inverse-`FFT` plan in [`super::fft_plan`].
//!
//! The reference `water_spectrum_ifft` in `water_ocean.wesl` does everything in
//! one `O(N^4)` pass: it advances every complex amplitude to the frame time and
//! then direct-sums the inverse transform for eight real output fields at once
//! (height, the two horizontal `choppiness` displacements, the two surface
//! slopes, and the three horizontal displacement gradients that feed the folding
//! Jacobian). A shipping ocean (`WaveWorks`, `Crest`, `UE5` Water) cannot afford
//! that quartic sum: it factors the transform into a separable radix-2 butterfly
//! `FFT` and splits the single pass into three stages the host ping-pongs:
//!
//!   1. **Evolve** — advance `h0(+k)`/`h0(-k)` to the frame time and pack the
//!      eight real field spectra into [`SPECTRAL_COMPLEX_FIELD_COUNT`] complex
//!      spectra (two real fields per complex buffer; see [`field_slot`]).
//!   2. **Butterfly** — the [`super::fft_plan`] pass list, run once per packed
//!      complex buffer (each buffer inverse-transformed independently by the
//!      proven single-grid butterfly), turning each spectrum into its spatial
//!      field. The four buffers run four full pass lists back to back.
//!   3. **Assemble** — unpack the transformed complex buffers back into the
//!      eight real fields, apply `choppiness`, and write the displacement and
//!      normal textures exactly as the reference kernel's tail does.
//!
//! Packing two real fields into one complex inverse `FFT` is exact here: each
//! field's spectrum is Hermitian (so its inverse transform is real), the inverse
//! transform is linear, and for two real fields `a`, `b` the packed spectrum
//! `a + i b` inverts to `ifft(a) + i ifft(b)` — a complex field whose real part
//! is `a`'s spatial signal and whose imaginary part is `b`'s. Eight real fields
//! therefore need only four complex transforms.
//!
//! Like [`super::fft_plan`] this is float-free integer bookkeeping: given the
//! grid edge `n` and cascade count it produces the exact ordered stage list, so
//! the scene crate's `GPU` dispatcher drives a deterministic, `CPU`-testable
//! schedule. An out-of-contract edge (`n <= 1` or non-power-of-two) yields an
//! empty plan — a deterministic skip, never a partial evolve/assemble with no
//! transform between them.

use alloc::vec::Vec;

use super::fft_plan::{inverse_fft2_pass_count, plan_inverse_fft2, FftPass};

/// The eight real output fields the spectral ocean evaluates per texel, matching
/// the accumulators the reference `water_spectrum_ifft` direct-sums.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpectralRealField {
    /// Surface height `h(x)` (the `y` displacement).
    Height,
    /// Horizontal `choppiness` displacement along `x` (`Dx`).
    DisplacementX,
    /// Horizontal `choppiness` displacement along `z` (`Dz`).
    DisplacementZ,
    /// Surface slope along `x` (feeds the surface normal).
    SlopeX,
    /// Surface slope along `z` (feeds the surface normal).
    SlopeZ,
    /// Horizontal displacement gradient `dDx/dx` (folding Jacobian term).
    GradXx,
    /// Horizontal displacement gradient `dDz/dz` (folding Jacobian term).
    GradZz,
    /// Horizontal displacement gradient `dDx/dz` = `dDz/dx` (Jacobian cross
    /// term; `xz`-symmetric, so one field serves both).
    GradXz,
}

/// The number of real output fields the spectral pass produces.
pub const SPECTRAL_REAL_FIELD_COUNT: usize = 8;

/// The number of complex inverse-`FFT` buffers the real fields pack into (two
/// real fields per complex transform).
pub const SPECTRAL_COMPLEX_FIELD_COUNT: usize = SPECTRAL_REAL_FIELD_COUNT / 2;

impl SpectralRealField {
    /// Every real field in a stable order, for packing tables and exhaustiveness
    /// tests.
    pub const ALL: [SpectralRealField; SPECTRAL_REAL_FIELD_COUNT] = [
        SpectralRealField::Height,
        SpectralRealField::DisplacementX,
        SpectralRealField::DisplacementZ,
        SpectralRealField::SlopeX,
        SpectralRealField::SlopeZ,
        SpectralRealField::GradXx,
        SpectralRealField::GradZz,
        SpectralRealField::GradXz,
    ];
}

/// Which half of a packed complex buffer a real field occupies.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComplexPart {
    /// The real component of the packed complex value.
    Real,
    /// The imaginary component of the packed complex value.
    Imag,
}

/// The packed location of one real field: which complex buffer, and which
/// component of it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FieldSlot {
    /// The index of the complex buffer (`0..SPECTRAL_COMPLEX_FIELD_COUNT`).
    pub complex_index: u32,
    /// The component of that complex buffer this real field occupies.
    pub part: ComplexPart,
}

/// The packed slot of a real field: two consecutive fields share a complex
/// buffer, the first as its real part and the second as its imaginary part.
///
/// The packing is a fixed bijection over the eight fields (verified in tests),
/// so the evolve stage knows where to write each spectrum and the assemble stage
/// knows where to read each spatial field back.
#[must_use]
pub fn field_slot(field: SpectralRealField) -> FieldSlot {
    let ordinal = SpectralRealField::ALL
        .iter()
        .position(|&f| f == field)
        .unwrap_or(0) as u32;
    let complex_index = ordinal / 2;
    let part = if ordinal.is_multiple_of(2) {
        ComplexPart::Real
    } else {
        ComplexPart::Imag
    };
    FieldSlot {
        complex_index,
        part,
    }
}

/// One stage of the spectral schedule for a single cascade.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpectralStage {
    /// Advance the initial amplitudes to the frame time and pack the eight field
    /// spectra into the [`SPECTRAL_COMPLEX_FIELD_COUNT`] complex buffers.
    Evolve,
    /// One separable inverse-`FFT` pass over a single packed complex buffer.
    /// Each of the [`SPECTRAL_COMPLEX_FIELD_COUNT`] buffers runs a full pass
    /// list with the proven single-grid butterfly; `complex_index` selects
    /// which buffer this pass transforms.
    Butterfly {
        /// The packed complex buffer this pass transforms
        /// (`0..SPECTRAL_COMPLEX_FIELD_COUNT`).
        complex_index: u32,
        /// The butterfly pass to run on that buffer.
        pass: FftPass,
    },
    /// Unpack the transformed buffers into the eight real fields and write the
    /// displacement and normal textures.
    Assemble,
}

/// One stage tagged with the cascade it belongs to, for the full multi-cascade
/// ocean schedule.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpectralPass {
    /// The cascade index this stage runs for (`0..cascades`).
    pub cascade: u32,
    /// The stage to run.
    pub stage: SpectralStage,
}

/// The number of stages [`plan_cascade_spectral`] emits for grid edge `n`:
/// evolve, the butterfly pass list run once per packed complex buffer, and
/// assemble. Returns `0` for an out-of-contract edge (so the whole cascade is
/// skipped, never a bare evolve/assemble with no transform).
#[must_use]
pub fn cascade_spectral_stage_count(n: u32) -> usize {
    let fft = inverse_fft2_pass_count(n);
    if fft == 0 {
        return 0;
    }
    2 + SPECTRAL_COMPLEX_FIELD_COUNT * fft
}

/// The stage list for one cascade: [`SpectralStage::Evolve`], then the
/// [`super::fft_plan`] butterfly passes run once per packed complex buffer (in
/// ascending `complex_index` order), then [`SpectralStage::Assemble`].
///
/// Empty for an out-of-contract edge.
#[must_use]
pub fn plan_cascade_spectral(n: u32) -> Vec<SpectralStage> {
    let fft = plan_inverse_fft2(n);
    if fft.is_empty() {
        return Vec::new();
    }
    let mut stages = Vec::with_capacity(2 + SPECTRAL_COMPLEX_FIELD_COUNT * fft.len());
    stages.push(SpectralStage::Evolve);
    for complex_index in 0..SPECTRAL_COMPLEX_FIELD_COUNT as u32 {
        for &pass in &fft {
            stages.push(SpectralStage::Butterfly {
                complex_index,
                pass,
            });
        }
    }
    stages.push(SpectralStage::Assemble);
    stages
}

/// The total stage count for a `cascades`-deep ocean at grid edge `n`.
#[must_use]
pub fn ocean_spectral_pass_count(n: u32, cascades: u32) -> usize {
    cascade_spectral_stage_count(n) * cascades as usize
}

/// The full multi-cascade ocean schedule: the per-cascade stage list repeated
/// for every cascade, each stage tagged with its cascade index.
///
/// Cascades run back to back in ascending index order. Empty when the edge is
/// out of contract or `cascades == 0`.
#[must_use]
pub fn plan_ocean_spectral(n: u32, cascades: u32) -> Vec<SpectralPass> {
    let per_cascade = plan_cascade_spectral(n);
    if per_cascade.is_empty() || cascades == 0 {
        return Vec::new();
    }
    let mut passes = Vec::with_capacity(per_cascade.len() * cascades as usize);
    for cascade in 0..cascades {
        for &stage in &per_cascade {
            passes.push(SpectralPass { cascade, stage });
        }
    }
    passes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::gpu::fft_plan::FftEntry;

    #[test]
    fn eight_real_fields_pack_into_four_complex_buffers() {
        assert_eq!(SPECTRAL_REAL_FIELD_COUNT, 8);
        assert_eq!(SPECTRAL_COMPLEX_FIELD_COUNT, 4);
        assert_eq!(SpectralRealField::ALL.len(), SPECTRAL_REAL_FIELD_COUNT);
    }

    #[test]
    fn field_packing_is_a_bijection_over_the_complex_slots() {
        let mut seen = [[false, false]; SPECTRAL_COMPLEX_FIELD_COUNT];
        for field in SpectralRealField::ALL {
            let slot = field_slot(field);
            assert!(
                (slot.complex_index as usize) < SPECTRAL_COMPLEX_FIELD_COUNT,
                "complex index in range"
            );
            let part = match slot.part {
                ComplexPart::Real => 0,
                ComplexPart::Imag => 1,
            };
            assert!(
                !seen[slot.complex_index as usize][part],
                "each (buffer, part) slot is used at most once"
            );
            seen[slot.complex_index as usize][part] = true;
        }
        // Every one of the four buffers has both halves filled.
        for slots in seen {
            assert!(slots[0] && slots[1], "both halves of every buffer are used");
        }
    }

    #[test]
    fn consecutive_fields_share_a_buffer_as_real_then_imag() {
        assert_eq!(
            field_slot(SpectralRealField::Height),
            FieldSlot {
                complex_index: 0,
                part: ComplexPart::Real
            }
        );
        assert_eq!(
            field_slot(SpectralRealField::DisplacementX),
            FieldSlot {
                complex_index: 0,
                part: ComplexPart::Imag
            }
        );
        assert_eq!(
            field_slot(SpectralRealField::GradXz),
            FieldSlot {
                complex_index: 3,
                part: ComplexPart::Imag
            }
        );
    }

    #[test]
    fn cascade_stage_count_is_evolve_plus_fft_plus_assemble() {
        // N = 16 → 11 butterfly passes × 4 buffers + 2 → 46 stages;
        // N = 256 → 19 × 4 + 2 → 78 stages.
        assert_eq!(cascade_spectral_stage_count(16), 46);
        assert_eq!(cascade_spectral_stage_count(256), 78);
    }

    #[test]
    fn out_of_contract_edges_plan_no_stages() {
        assert_eq!(cascade_spectral_stage_count(0), 0);
        assert_eq!(cascade_spectral_stage_count(1), 0);
        assert_eq!(cascade_spectral_stage_count(3), 0);
        assert!(plan_cascade_spectral(3).is_empty());
        assert!(plan_ocean_spectral(3, 4).is_empty());
    }

    #[test]
    fn cascade_plan_brackets_the_fft_with_evolve_and_assemble() {
        let stages = plan_cascade_spectral(16);
        assert_eq!(stages.len(), cascade_spectral_stage_count(16));
        assert_eq!(stages.first(), Some(&SpectralStage::Evolve));
        assert_eq!(stages.last(), Some(&SpectralStage::Assemble));
        // Everything between the brackets is a butterfly pass.
        for stage in &stages[1..stages.len() - 1] {
            assert!(matches!(stage, SpectralStage::Butterfly { .. }));
        }
    }

    #[test]
    fn cascade_plan_preserves_the_fft_pass_order() {
        let n = 16u32;
        let stages = plan_cascade_spectral(n);
        let single = plan_inverse_fft2(n);
        // Each packed complex buffer runs the full pass list, in ascending
        // order, exactly matching the single-grid inverse-`FFT` plan.
        for complex_index in 0..SPECTRAL_COMPLEX_FIELD_COUNT as u32 {
            let group: Vec<FftPass> = stages
                .iter()
                .filter_map(|s| match s {
                    SpectralStage::Butterfly {
                        complex_index: ci,
                        pass,
                    } if *ci == complex_index => Some(*pass),
                    _ => None,
                })
                .collect();
            assert_eq!(group, single, "buffer runs the full inverse-FFT plan");
            // The first transform pass is the row bit-reversal, matching fft_plan.
            assert_eq!(group[0].entry, FftEntry::BitReversal);
        }
    }

    #[test]
    fn cascade_plan_runs_each_buffer_back_to_back_in_ascending_order() {
        let stages = plan_cascade_spectral(16);
        // The complex_index of the butterfly stages never decreases: every
        // buffer's full pass list runs before the next buffer starts.
        let indices: Vec<u32> = stages
            .iter()
            .filter_map(|s| match s {
                SpectralStage::Butterfly { complex_index, .. } => Some(*complex_index),
                _ => None,
            })
            .collect();
        assert_eq!(
            indices.len(),
            SPECTRAL_COMPLEX_FIELD_COUNT * plan_inverse_fft2(16).len()
        );
        for pair in indices.windows(2) {
            assert!(pair[0] <= pair[1], "buffers run in ascending index order");
        }
        assert_eq!(indices.first(), Some(&0));
        assert_eq!(
            indices.last(),
            Some(&(SPECTRAL_COMPLEX_FIELD_COUNT as u32 - 1))
        );
    }

    #[test]
    fn ocean_plan_repeats_each_cascade_in_ascending_order() {
        let n = 16u32;
        let cascades = 3u32;
        let passes = plan_ocean_spectral(n, cascades);
        assert_eq!(passes.len(), ocean_spectral_pass_count(n, cascades));
        assert_eq!(passes.len(), cascade_spectral_stage_count(n) * 3);

        let per = cascade_spectral_stage_count(n);
        // Each cascade's slice equals the single-cascade stage list.
        let single = plan_cascade_spectral(n);
        for c in 0..cascades {
            let start = c as usize * per;
            let slice = &passes[start..start + per];
            for (pass, stage) in slice.iter().zip(single.iter()) {
                assert_eq!(pass.cascade, c);
                assert_eq!(pass.stage, *stage);
            }
        }
    }

    #[test]
    fn zero_cascades_plans_no_passes() {
        assert!(plan_ocean_spectral(16, 0).is_empty());
        assert_eq!(ocean_spectral_pass_count(16, 0), 0);
    }

    #[test]
    fn ocean_plan_is_deterministic() {
        assert_eq!(plan_ocean_spectral(64, 4), plan_ocean_spectral(64, 4));
    }
}
