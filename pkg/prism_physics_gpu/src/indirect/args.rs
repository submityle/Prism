//! The `GPU`-driven indirect-dispatch argument block and its `CPU` golden twin.
//!
//! A `GPU`-driven pipeline must decide *how many* workgroups to launch for a
//! stage from a count that only exists in device memory — the number of
//! candidate pairs a broad phase emitted, the number of particles still awake,
//! the number of fragments a fracture produced. Reading that count back to the
//! host to pick a dispatch size would stall the pipeline on a round-trip every
//! stage. Instead a tiny kernel turns the device-side count into a
//! [`DispatchArgs`] block that `dispatch_workgroups_indirect` consumes directly,
//! so the host never learns the count and never blocks.
//!
//! [`cpu_dispatch_args`] is the exact golden twin of that kernel: it maps a
//! count and a workgroup size to the same `[x, y, z]` grid, so a passing parity
//! test is direct evidence the kernel computes the launch the host would have.
//!
//! # Grid shape
//!
//! The natural launch is one-dimensional: `ceil(count / workgroup_size)`
//! workgroups along `x`. A device caps each dimension at
//! `max_compute_workgroups_per_dimension` (commonly `65_535`), so once the
//! one-dimensional group total would exceed that cap the grid tiles into a
//! second dimension: `x` saturates at the cap and `y` holds the remaining rows.
//! The row width is published in the fourth word (`groups_x`) so a consumer
//! kernel can rebuild a linear invocation index as
//! `(gid.y * groups_x + gid.x) * workgroup_size + local` and guard it against
//! the true count. When the launch fits in one dimension the width equals `x`,
//! so the same linearisation is correct in both regimes.
//!
//! # Known boundary
//!
//! The tiling is **two-dimensional**: `x` saturates at the per-dimension cap
//! and `y` carries the rows. It therefore only covers up to
//! `max_dim * max_dim` workgroups; past that, `y` would itself exceed the
//! device cap (`total > max_dim^2` — reachable only with `workgroup_size == 1`
//! and a count within `~131k` of `u32::MAX` under the default `65_535` cap) and
//! the indirect dispatch would be rejected. Covering the full `u32` range needs
//! a third grid dimension plus a matching `groups_xy` stride for the consumer
//! linearisation, which would change the consumer contract; until that lands the
//! caller must split such a launch. This is an honest limitation, not covered up
//! by a wrong `y`.
//!
//! # Provenance
//!
//! Indirect compute dispatch and ceil-division grid sizing are standard,
//! openly published `GPU`-driven techniques. No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};

/// The default per-dimension workgroup cap assumed when a caller does not
/// supply the device's own `max_compute_workgroups_per_dimension`.
///
/// `65_535` is the floor the `WebGPU` specification guarantees and the value
/// almost every desktop and mobile adapter reports, so it is a safe tiling
/// threshold when the real limit is not yet known.
pub const DEFAULT_MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// A `dispatch_workgroups_indirect` argument block: the `[x, y, z]` workgroup
/// grid plus the published row width used to linearise a tiled launch.
///
/// The first three words are laid out exactly as `wgpu` expects an indirect
/// dispatch buffer to begin, so the block can be bound as the indirect source
/// at offset `0`. The fourth word is additional state a consumer kernel reads
/// to reconstruct a linear invocation index; the indirect draw ignores it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DispatchArgs {
    /// Workgroup count along `x`.
    pub x: u32,
    /// Workgroup count along `y` (`1` for a one-dimensional launch).
    pub y: u32,
    /// Workgroup count along `z` (always `1` for these launches).
    pub z: u32,
    /// The `x`-row width a consumer multiplies `gid.y` by to linearise a tiled
    /// launch; equal to `x` whenever the launch is one-dimensional.
    pub groups_x: u32,
}

impl DispatchArgs {
    /// The empty launch: no workgroups, so an indirect dispatch over it is a
    /// no-op. Returned for a zero count.
    pub const EMPTY: DispatchArgs = DispatchArgs {
        x: 0,
        y: 1,
        z: 1,
        groups_x: 0,
    };

    /// The total number of workgroups the grid launches, `x * y` (with `z`
    /// always `1`).
    #[must_use]
    pub const fn total_groups(self) -> u32 {
        self.x * self.y
    }
}

/// Computes the indirect-dispatch grid for `count` work items at
/// `workgroup_size` lanes per group, tiling into `y` once the one-dimensional
/// group total would exceed `max_workgroups_per_dim`.
///
/// This is the golden twin of `shaders/indirect_args.wgsl`'s `build_args`
/// entry point and must stay bit-identical to it. A `workgroup_size` or
/// `max_workgroups_per_dim` of `0` is treated as `1` so the result is always a
/// well-formed grid rather than a division trap.
#[must_use]
pub fn cpu_dispatch_args(
    count: u32,
    workgroup_size: u32,
    max_workgroups_per_dim: u32,
) -> DispatchArgs {
    let wg = workgroup_size.max(1);
    let total = count.div_ceil(wg);
    if total == 0 {
        return DispatchArgs::EMPTY;
    }
    let max_dim = max_workgroups_per_dim.max(1);
    if total <= max_dim {
        DispatchArgs {
            x: total,
            y: 1,
            z: 1,
            groups_x: total,
        }
    } else {
        let x = max_dim;
        let y = total.div_ceil(max_dim);
        DispatchArgs {
            x,
            y,
            z: 1,
            groups_x: x,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_count_is_the_empty_launch() {
        assert_eq!(
            cpu_dispatch_args(0, 256, DEFAULT_MAX_WORKGROUPS_PER_DIM),
            DispatchArgs::EMPTY
        );
        assert_eq!(
            cpu_dispatch_args(0, 256, DEFAULT_MAX_WORKGROUPS_PER_DIM).total_groups(),
            0
        );
    }

    #[test]
    fn exact_multiple_rounds_without_a_remainder_group() {
        let a = cpu_dispatch_args(1024, 256, DEFAULT_MAX_WORKGROUPS_PER_DIM);
        assert_eq!(
            a,
            DispatchArgs {
                x: 4,
                y: 1,
                z: 1,
                groups_x: 4
            }
        );
    }

    #[test]
    fn remainder_adds_one_group() {
        let a = cpu_dispatch_args(1025, 256, DEFAULT_MAX_WORKGROUPS_PER_DIM);
        assert_eq!(
            a,
            DispatchArgs {
                x: 5,
                y: 1,
                z: 1,
                groups_x: 5
            }
        );
    }

    #[test]
    fn single_item_launches_one_group() {
        assert_eq!(
            cpu_dispatch_args(1, 256, DEFAULT_MAX_WORKGROUPS_PER_DIM),
            DispatchArgs {
                x: 1,
                y: 1,
                z: 1,
                groups_x: 1
            }
        );
    }

    #[test]
    fn zero_workgroup_size_is_treated_as_one() {
        // Degenerate size must not divide by zero; one lane per group means one
        // group per item.
        assert_eq!(
            cpu_dispatch_args(7, 0, DEFAULT_MAX_WORKGROUPS_PER_DIM),
            DispatchArgs {
                x: 7,
                y: 1,
                z: 1,
                groups_x: 7
            }
        );
    }

    #[test]
    fn launch_at_the_cap_stays_one_dimensional() {
        // Exactly `max_dim` groups must not spill into a second row.
        let max_dim = 8u32;
        let count = max_dim * 256; // 2048 items -> 8 groups at size 256.
        let a = cpu_dispatch_args(count, 256, max_dim);
        assert_eq!(
            a,
            DispatchArgs {
                x: 8,
                y: 1,
                z: 1,
                groups_x: 8
            }
        );
    }

    #[test]
    fn launch_past_the_cap_tiles_into_a_second_dimension() {
        let max_dim = 8u32;
        // 9 groups' worth of work: one past the cap, so it tiles to a 8x2 grid.
        let count = 9 * 256;
        let a = cpu_dispatch_args(count, 256, max_dim);
        assert_eq!(
            a,
            DispatchArgs {
                x: 8,
                y: 2,
                z: 1,
                groups_x: 8
            }
        );
        // The grid must cover every needed group.
        assert!(a.total_groups() >= count.div_ceil(256));
    }

    #[test]
    fn tiled_grid_width_lets_a_consumer_cover_exactly_the_count() {
        let max_dim = 4u32;
        let wg = 64u32;
        let count = 1000u32;
        let a = cpu_dispatch_args(count, wg, max_dim);
        // Linearising `(y, x)` with the published width must reach every item.
        let covered = a.total_groups() * wg;
        assert!(covered >= count);
        // ...and the previous row must not already have covered it (tight).
        assert!((a.y - 1) * a.groups_x * wg < count);
    }
}
