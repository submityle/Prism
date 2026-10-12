//! Checked host-side arithmetic shared by batched compute kernels.
//!
//! Every kernel uses a `u32` element index in WGSL.  Converting a Rust
//! `usize` with `as u32` would silently wrap on oversized inputs, causing the
//! shader's bounds guard to process the wrong prefix (or no elements at all).
//! Keep the conversion and workgroup tiling in one small, device-independent
//! module so every host wrapper has the same contract.

/// Maximum batch length accepted by the standard 64-thread batch dispatches.
///
/// Keeping this bound in the shared host helper means oversized inputs fail
/// before any GPU buffer is allocated. Every batch wrapper in this crate uses
/// a 64-thread workgroup and WebGPU guarantees at most 65,535 groups per
/// dispatch dimension.
const MAX_BATCH: usize = (MAX_WORKGROUPS_PER_DIMENSION as usize) * 64;

/// Maximum workgroup count guaranteed for one compute-dispatch dimension by
/// WebGPU's baseline limits.
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Converts a host batch length to the WGSL count type.
///
/// # Panics
///
/// Panics with a stable message when the batch cannot be represented by the
/// shader's `u32` index space.  Silently truncating here would be a data
/// corruption bug, not a recoverable GPU condition.
#[inline]
pub(crate) fn checked_count(n: usize) -> u32 {
    assert!(
        n <= MAX_BATCH,
        "GPU batch requires more than {MAX_WORKGROUPS_PER_DIMENSION} workgroups at 64 threads per group"
    );
    let count = u32::try_from(n).expect("GPU batch length exceeds the WGSL u32 index space");
    count
}

/// Returns the number of 64-thread workgroups required for `n` elements.
///
/// Empty batches must be handled before resource creation; this helper makes
/// that invariant explicit and never produces a zero-sized dispatch for a
/// non-empty batch.
#[inline]
pub(crate) fn checked_workgroup_count(n: usize, workgroup_size: u32) -> u32 {
    assert!(workgroup_size > 0, "GPU workgroup size must be non-zero");
    let count = checked_count(n);
    let groups = count.div_ceil(workgroup_size);
    assert!(
        groups <= MAX_WORKGROUPS_PER_DIMENSION,
        "GPU batch requires {groups} workgroups, exceeding the portable limit of {MAX_WORKGROUPS_PER_DIMENSION}"
    );
    groups
}

#[cfg(test)]
mod tests {
    use super::{checked_count, checked_workgroup_count, MAX_BATCH, MAX_WORKGROUPS_PER_DIMENSION};

    #[test]
    fn count_is_checked_at_the_wgsl_boundary() {
        assert_eq!(checked_count(0), 0);
        assert_eq!(checked_count(MAX_BATCH), MAX_WORKGROUPS_PER_DIMENSION * 64);
    }

    #[test]
    #[should_panic(expected = "more than 65535 workgroups")]
    fn count_rejects_batches_that_would_exceed_dispatch_limit() {
        let _ = checked_count(MAX_BATCH + 1);
    }

    #[test]
    fn workgroups_are_zero_only_for_empty_batches() {
        assert_eq!(checked_workgroup_count(0, 64), 0);
        assert_eq!(checked_workgroup_count(1, 64), 1);
        assert_eq!(checked_workgroup_count(64, 64), 1);
        assert_eq!(checked_workgroup_count(65, 64), 2);
    }

    #[test]
    #[should_panic(expected = "more than 65535 workgroups")]
    fn workgroups_reject_non_portable_dispatches() {
        let _ = checked_workgroup_count((MAX_WORKGROUPS_PER_DIMENSION as usize + 1) * 64, 64);
    }

    #[test]
    #[should_panic(expected = "GPU workgroup size must be non-zero")]
    fn workgroups_reject_zero_size() {
        let _ = checked_workgroup_count(1, 0);
    }
}
