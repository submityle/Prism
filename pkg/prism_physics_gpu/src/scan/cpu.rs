//! `CPU` golden twins for the `GPU` exclusive scan and stream compaction.
//!
//! [`cpu_exclusive_scan`] runs the sequential exclusive prefix sum the `WGSL`
//! kernels reproduce in parallel, and [`cpu_compact`] runs the gather the
//! scatter kernel reproduces. Because unsigned integer addition is associative
//! and wraps identically on both sides, a passing real-device parity test is
//! direct evidence that the ported kernels compute the same prefix offsets and
//! compacted stream as this reference — bit-for-bit, not merely that the
//! shaders compiled.
//!
//! # Wrapping semantics
//!
//! `WGSL` `u32` addition wraps on overflow, so the twin folds with
//! [`u32::wrapping_add`]. For the dominant use — scanning a stream of `0`/`1`
//! keep flags — the running total never exceeds the element count and no
//! wrapping occurs, but matching the wrap keeps the parity exact for arbitrary
//! `u32` inputs as well.
//!
//! # Provenance
//!
//! The exclusive prefix sum and flag-driven stream compaction are classical,
//! openly published parallel primitives. This module contains no Unreal Engine
//! source or derived code.

/// Computes the exclusive prefix sum of `values` and returns it alongside the
/// grand total.
///
/// Entry `i` of the returned vector is the sum of `values[0..i]`, and the
/// returned scalar is the sum of every element (both with wrapping `u32`
/// addition). An empty input yields an empty vector and a zero total.
#[must_use]
pub fn cpu_exclusive_scan(values: &[u32]) -> (Vec<u32>, u32) {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0u32;
    for &v in values {
        out.push(acc);
        acc = acc.wrapping_add(v);
    }
    (out, acc)
}

/// Gathers the entries of `data` whose corresponding `flags` entry is non-zero,
/// preserving input order.
///
/// The compacted vector has one entry per set flag, in the same relative order
/// as in `data`; its length equals the number of non-zero flags, which is also
/// the grand total returned by [`cpu_exclusive_scan`] applied to the flags when
/// every flag is `0` or `1`.
///
/// # Panics
///
/// Panics if `data` and `flags` do not have the same length.
#[must_use]
pub fn cpu_compact(data: &[u32], flags: &[u32]) -> Vec<u32> {
    assert!(
        data.len() == flags.len(),
        "data and flags must have equal length"
    );
    data.iter()
        .zip(flags)
        .filter_map(|(&d, &f)| if f != 0 { Some(d) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_scan_is_empty_with_zero_total() {
        let (out, total) = cpu_exclusive_scan(&[]);
        assert!(out.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn exclusive_scan_shifts_and_accumulates() {
        let (out, total) = cpu_exclusive_scan(&[3, 1, 7, 0, 4]);
        assert_eq!(out, vec![0, 3, 4, 11, 11]);
        assert_eq!(total, 15);
    }

    #[test]
    fn scan_of_flags_gives_output_offsets() {
        // A 0/1 flag stream scans to the destination index of each kept entry.
        let flags = [1, 0, 1, 1, 0, 1];
        let (offsets, total) = cpu_exclusive_scan(&flags);
        assert_eq!(offsets, vec![0, 1, 1, 2, 3, 3]);
        assert_eq!(total, 4);
    }

    #[test]
    fn scan_wraps_like_the_device() {
        let (out, total) = cpu_exclusive_scan(&[u32::MAX, 2]);
        assert_eq!(out, vec![0, u32::MAX]);
        // u32::MAX + 2 wraps to 1.
        assert_eq!(total, 1);
    }

    #[test]
    fn compact_keeps_flagged_entries_in_order() {
        let data = [10, 20, 30, 40, 50, 60];
        let flags = [1, 0, 1, 1, 0, 1];
        assert_eq!(cpu_compact(&data, &flags), vec![10, 30, 40, 60]);
    }

    #[test]
    fn compact_of_all_zero_flags_is_empty() {
        assert_eq!(cpu_compact(&[1, 2, 3], &[0, 0, 0]), Vec::<u32>::new());
    }

    #[test]
    fn compact_length_matches_scan_total() {
        let data = [5, 6, 7, 8, 9];
        let flags = [0, 1, 1, 0, 1];
        let (_, total) = cpu_exclusive_scan(&flags);
        assert_eq!(cpu_compact(&data, &flags).len(), total as usize);
    }
}
