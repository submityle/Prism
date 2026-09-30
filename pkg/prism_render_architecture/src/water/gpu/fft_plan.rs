//! Deterministic ping-pong schedule for the separable inverse butterfly `FFT`.
//!
//! The spectral ocean (`Tessendorf`) inverts a grid of time-advanced complex
//! amplitudes into a spatial displacement field. Evaluated as a direct sum that
//! inverse transform is `O(N^4)` for an `N*N` patch — the shape the reference
//! `water_ocean.wesl` spells out only as a compilable ground truth. Every
//! shipping ocean (`WaveWorks`, `Crest`, `UE5` Water) instead runs a separable
//! radix-2 `Cooley-Tukey` butterfly `FFT`: `O(N log N)` per line, `O(N^2 log N)`
//! for the whole patch.
//!
//! On a `GPU` that butterfly is not a single dispatch: it is a fixed sequence of
//! tiny passes the host ping-pongs between two complex buffers. This module is
//! the dependency-free, float-free planner for that sequence. Given the grid
//! edge `n` it produces the exact, ordered list of passes — a bit-reversal
//! reorder plus `log2(n)` butterfly stages along the rows (axis 0), the same
//! along the columns (axis 1), then one final `1/(N*N)` normalize — matching the
//! row-then-column, normalise-once order the `CPU` golden
//! [`crate::water::fft`] `transform2`/`ifft2` uses, and the per-pass uniform
//! ([`FftPassParams`]) the device twin `water_butterfly.wesl` consumes.
//!
//! The whole plan is integer bookkeeping with no `GPU` handles, no floats and no
//! wall clock, so the schedule the scene crate uploads is deterministic and
//! `CPU`-testable pass-for-pass. Non-power-of-two (and degenerate `n <= 1`)
//! edges yield an empty plan — a deterministic no-op rather than a panic,
//! matching the crate's "skip, do not crash" contract for out-of-contract
//! sizes, and mirroring [`crate::water::fft`]'s identity behaviour there.

use alloc::vec::Vec;

use crate::water::fft::is_power_of_two;

/// The traversal axis of one separable transform pass.
///
/// Row-major storage: a [`Row`](FftAxis::Row) pass walks `x` within a row
/// (`line * n + pos`); a [`Column`](FftAxis::Column) pass walks `y` within a
/// column (`pos * n + line`). The inverse of an `N*N` grid transforms every row
/// first, then every column, exactly as the `CPU` golden `transform2` does.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FftAxis {
    /// Axis 0: transform along `x` within each row.
    Row,
    /// Axis 1: transform along `y` within each column.
    Column,
}

impl FftAxis {
    /// The device-side axis selector uploaded in [`FftPassParams::axis`]
    /// (`0` = row, `1` = column), matching `water_butterfly.wesl`'s `lin_index`.
    #[must_use]
    pub fn index(self) -> u32 {
        match self {
            FftAxis::Row => 0,
            FftAxis::Column => 1,
        }
    }
}

/// Which `water_butterfly.wesl` entry point a scheduled pass dispatches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FftEntry {
    /// `water_fft_bitrev`: decimation-in-time reorder seeding one axis
    /// (`dst[pos] = src[reverse_bits(pos)]`). Consumes `n`, `axis`, `log2n`.
    BitReversal,
    /// `water_fft_stage`: one radix-2 inverse butterfly stage of span `len`
    /// along one axis. Consumes `n`, `axis`, `len`.
    Butterfly,
    /// `water_fft_normalize`: the single `1/(N*N)` inverse scale. Consumes `n`.
    Normalize,
}

/// The per-pass uniform payload the host uploads, byte-compatible with the
/// `FftParams` uniform in `water_butterfly.wesl`.
///
/// Every pass carries all four integer fields so the payload is a fixed shape,
/// but each entry point reads only the subset documented on [`FftEntry`]: the
/// bit-reversal ignores `len`, and the normalize ignores both `axis` and `len`
/// (they are set to a stable `0` there so the plan is deterministic).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FftPassParams {
    /// The grid edge `N` (elements per row/column).
    pub n: u32,
    /// The traversal axis selector (`0` = row, `1` = column).
    pub axis: u32,
    /// The current butterfly stage span (`2, 4, …, N`); `0` for the reorder and
    /// normalize passes, which do not read it.
    pub len: u32,
    /// `log2(N)`, the bit width the reorder reverses and the stage count.
    pub log2n: u32,
}

/// One scheduled compute pass of the separable inverse butterfly `FFT`: which
/// entry point to dispatch and the uniform payload to bind for it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FftPass {
    /// The `water_butterfly.wesl` entry point this pass dispatches.
    pub entry: FftEntry,
    /// The uniform payload to upload before dispatching `entry`.
    pub params: FftPassParams,
}

/// The number of passes [`plan_inverse_fft2`] emits for grid edge `n`.
///
/// A valid power-of-two edge yields `2 * (1 + log2(n)) + 1` passes: for each of
/// the two axes a bit-reversal plus `log2(n)` butterfly stages, then one shared
/// normalize (`N = 16 → 11`, `N = 256 → 19`). Degenerate `n <= 1` and
/// non-power-of-two edges yield `0`.
#[must_use]
pub fn inverse_fft2_pass_count(n: u32) -> usize {
    if n <= 1 || !is_power_of_two(n as usize) {
        return 0;
    }
    let log2n = n.trailing_zeros() as usize;
    2 * (1 + log2n) + 1
}

/// Builds the deterministic, ordered ping-pong pass list for the separable
/// inverse butterfly `FFT` of an `n * n` complex grid.
///
/// The order is the canonical separable inverse: for the rows (axis 0) a
/// bit-reversal reorder then `log2(n)` butterfly stages with the span doubling
/// `2, 4, …, n`, the same for the columns (axis 1), then a single `1/(N*N)`
/// normalize — identical to the `CPU` golden [`crate::water::fft`] `ifft2`. The
/// host binds [`FftPass::params`] as the pass uniform, ping-ponging the two
/// complex buffers between passes.
///
/// Returns an empty plan for degenerate (`n <= 1`) or non-power-of-two edges,
/// so an out-of-contract size is a deterministic no-op rather than a panic.
#[must_use]
pub fn plan_inverse_fft2(n: u32) -> Vec<FftPass> {
    let count = inverse_fft2_pass_count(n);
    let mut passes = Vec::with_capacity(count);
    if count == 0 {
        return passes;
    }

    let log2n = n.trailing_zeros();
    for axis in [FftAxis::Row, FftAxis::Column] {
        passes.push(FftPass {
            entry: FftEntry::BitReversal,
            params: FftPassParams {
                n,
                axis: axis.index(),
                len: 0,
                log2n,
            },
        });
        let mut len = 2u32;
        while len <= n {
            passes.push(FftPass {
                entry: FftEntry::Butterfly,
                params: FftPassParams {
                    n,
                    axis: axis.index(),
                    len,
                    log2n,
                },
            });
            len *= 2;
        }
    }
    passes.push(FftPass {
        entry: FftEntry::Normalize,
        params: FftPassParams {
            n,
            axis: 0,
            len: 0,
            log2n,
        },
    });

    passes
}

/// The physical source and destination complex buffer a single ping-pong pass
/// reads from and writes to.
///
/// A grid keeps two resident complex buffers: buffer `0` seeds the spectrum and
/// buffer `1` is scratch. Every scheduled pass reads one and writes the other,
/// so the routing is pure parity of the pass position within the plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FftPingPong {
    /// The complex buffer index (`0` or `1`) the pass reads from.
    pub src: u32,
    /// The complex buffer index (`0` or `1`) the pass writes to.
    pub dst: u32,
}

/// The physical `src`/`dst` complex buffer selection for the pass at `ordinal`
/// in a [`plan_inverse_fft2`] pass list.
///
/// Every pass ping-pongs, so pass `0` reads the seed buffer `0` and writes the
/// scratch buffer `1`, and each subsequent pass flips. The dispatch recorder
/// uses this to bind the correct physical buffer as the read and read-write
/// binding of `water_butterfly.wesl` without re-deriving the parity itself.
#[must_use]
pub fn fft_pass_ping_pong(ordinal: usize) -> FftPingPong {
    let src = (ordinal & 1) as u32;
    FftPingPong { src, dst: src ^ 1 }
}

/// The physical complex buffer index holding the finished transform after the
/// full `n * n` inverse plan runs — where the assemble stage reads each grid.
///
/// A valid power-of-two edge always schedules an odd pass count
/// (`2 * (1 + log2(n)) + 1`), so the result always lands in the scratch buffer
/// `1`; a degenerate or non-power-of-two edge plans no passes and leaves the
/// result in the seed buffer `0`.
#[must_use]
pub fn fft_result_buffer(n: u32) -> u32 {
    (inverse_fft2_pass_count(n) & 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_index_maps_row_to_zero_and_column_to_one() {
        assert_eq!(FftAxis::Row.index(), 0);
        assert_eq!(FftAxis::Column.index(), 1);
    }

    #[test]
    fn pass_count_matches_the_two_axes_plus_normalize_formula() {
        assert_eq!(inverse_fft2_pass_count(2), 5);
        assert_eq!(inverse_fft2_pass_count(4), 7);
        assert_eq!(inverse_fft2_pass_count(8), 9);
        assert_eq!(inverse_fft2_pass_count(16), 11);
        assert_eq!(inverse_fft2_pass_count(256), 19);
    }

    #[test]
    fn pass_count_is_zero_for_degenerate_and_non_power_of_two_edges() {
        assert_eq!(inverse_fft2_pass_count(0), 0);
        assert_eq!(inverse_fft2_pass_count(1), 0);
        assert_eq!(inverse_fft2_pass_count(3), 0);
        assert_eq!(inverse_fft2_pass_count(6), 0);
        assert_eq!(inverse_fft2_pass_count(255), 0);
    }

    #[test]
    fn plan_length_equals_the_declared_pass_count() {
        for &n in &[0u32, 1, 2, 3, 4, 8, 16, 64, 256] {
            assert_eq!(plan_inverse_fft2(n).len(), inverse_fft2_pass_count(n));
        }
    }

    #[test]
    fn degenerate_and_non_power_of_two_edges_plan_no_passes() {
        assert!(plan_inverse_fft2(0).is_empty());
        assert!(plan_inverse_fft2(1).is_empty());
        assert!(plan_inverse_fft2(3).is_empty());
        assert!(plan_inverse_fft2(6).is_empty());
    }

    #[test]
    fn plan_is_deterministic() {
        assert_eq!(plan_inverse_fft2(16), plan_inverse_fft2(16));
    }

    #[test]
    fn every_pass_carries_the_grid_edge_and_log2n() {
        let n = 16u32;
        let log2n = n.trailing_zeros();
        for pass in plan_inverse_fft2(n) {
            assert_eq!(pass.params.n, n);
            assert_eq!(pass.params.log2n, log2n);
        }
    }

    #[test]
    fn plan_starts_with_a_row_bit_reversal_and_ends_with_normalize() {
        let plan = plan_inverse_fft2(16);
        let first = plan.first().expect("non-empty plan");
        assert_eq!(first.entry, FftEntry::BitReversal);
        assert_eq!(first.params.axis, FftAxis::Row.index());

        let last = plan.last().expect("non-empty plan");
        assert_eq!(last.entry, FftEntry::Normalize);
    }

    #[test]
    fn exactly_one_normalize_and_two_bit_reversals_are_scheduled() {
        let plan = plan_inverse_fft2(64);
        let normalizes = plan
            .iter()
            .filter(|p| p.entry == FftEntry::Normalize)
            .count();
        let bitrevs = plan
            .iter()
            .filter(|p| p.entry == FftEntry::BitReversal)
            .count();
        assert_eq!(normalizes, 1);
        assert_eq!(bitrevs, 2);
    }

    #[test]
    fn each_axis_runs_log2n_butterfly_stages() {
        let n = 256u32;
        let log2n = n.trailing_zeros() as usize;
        let plan = plan_inverse_fft2(n);
        for axis in [FftAxis::Row, FftAxis::Column] {
            let stages = plan
                .iter()
                .filter(|p| p.entry == FftEntry::Butterfly && p.params.axis == axis.index())
                .count();
            assert_eq!(stages, log2n);
        }
    }

    #[test]
    fn butterfly_spans_double_from_two_to_n_within_each_axis() {
        let n = 32u32;
        let plan = plan_inverse_fft2(n);
        for axis in [FftAxis::Row, FftAxis::Column] {
            let spans: Vec<u32> = plan
                .iter()
                .filter(|p| p.entry == FftEntry::Butterfly && p.params.axis == axis.index())
                .map(|p| p.params.len)
                .collect();
            let mut expected = Vec::new();
            let mut len = 2u32;
            while len <= n {
                expected.push(len);
                len *= 2;
            }
            assert_eq!(spans, expected);
        }
    }

    #[test]
    fn all_row_passes_precede_all_column_passes() {
        let plan = plan_inverse_fft2(64);
        // Everything up to (but excluding) the trailing normalize is axis work;
        // the rows must be a contiguous prefix of that, the columns the suffix.
        let axis_passes = &plan[..plan.len() - 1];
        let first_column = axis_passes
            .iter()
            .position(|p| p.params.axis == FftAxis::Column.index())
            .expect("a column pass exists");
        assert!(
            axis_passes[..first_column]
                .iter()
                .all(|p| p.params.axis == FftAxis::Row.index()),
            "every pass before the first column pass is a row pass"
        );
        assert!(
            axis_passes[first_column..]
                .iter()
                .all(|p| p.params.axis == FftAxis::Column.index()),
            "every pass from the first column pass on is a column pass"
        );
    }

    #[test]
    fn bit_reversal_and_normalize_passes_carry_no_stage_span() {
        for pass in plan_inverse_fft2(16) {
            match pass.entry {
                FftEntry::BitReversal | FftEntry::Normalize => assert_eq!(pass.params.len, 0),
                FftEntry::Butterfly => assert!(pass.params.len >= 2),
            }
        }
    }

    #[test]
    fn ping_pong_reads_the_seed_then_flips_every_pass() {
        // Pass 0 reads the seed buffer 0 and writes scratch 1; each subsequent
        // pass swaps the roles, so src is exactly the pass-position parity.
        for ordinal in 0..8usize {
            let route = fft_pass_ping_pong(ordinal);
            let expected_src = (ordinal & 1) as u32;
            assert_eq!(route.src, expected_src);
            assert_eq!(route.dst, expected_src ^ 1);
            assert_ne!(route.src, route.dst);
        }
    }

    #[test]
    fn ping_pong_src_dst_are_always_the_two_valid_buffers() {
        for ordinal in 0..16usize {
            let route = fft_pass_ping_pong(ordinal);
            assert!(route.src < 2);
            assert!(route.dst < 2);
        }
    }

    #[test]
    fn result_buffer_is_the_scratch_for_every_valid_edge() {
        // A valid power-of-two edge always plans an odd pass count, so the
        // finished transform always lands in the scratch buffer 1.
        for &n in &[2u32, 4, 8, 16, 64, 256, 512] {
            assert_eq!(fft_result_buffer(n), 1, "n = {n}");
        }
    }

    #[test]
    fn result_buffer_is_the_seed_for_empty_plans() {
        // Degenerate and non-power-of-two edges plan no passes, so the result
        // trivially stays in the seed buffer 0.
        for &n in &[0u32, 1, 3, 24] {
            assert_eq!(fft_result_buffer(n), 0, "n = {n}");
        }
    }

    #[test]
    fn result_buffer_matches_the_parity_of_the_scheduled_pass_count() {
        for &n in &[2u32, 4, 16, 256] {
            let count = inverse_fft2_pass_count(n);
            let route = fft_pass_ping_pong(count.saturating_sub(1));
            // The last pass writes into the result buffer; that destination is
            // exactly what `fft_result_buffer` reports for the grid edge.
            assert_eq!(route.dst, fft_result_buffer(n), "n = {n}");
        }
    }
}
