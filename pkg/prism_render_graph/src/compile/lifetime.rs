//! Resource lifetime intervals over the scheduled order.
//!
//! Once passes are linearized, every resource occupies a contiguous span from
//! the first scheduled position that touches it to the last. That interval is
//! all the aliaser needs: two transients whose intervals do not overlap can
//! share physical memory. Only the scheduled passes count, so a resource used
//! exclusively by culled passes has no interval and is never realized.

use alloc::vec;
use alloc::vec::Vec;

use crate::pass::PassNode;

/// First/last scheduled position of every resource, indexed by resource index.
///
/// `None` means no surviving pass touches the resource. Positions are indices
/// into the schedule produced by [`schedule`](super::schedule::schedule), not
/// original pass indices.
pub(crate) struct Lifetimes {
    /// Per-texture `(first, last)` scheduled position.
    pub textures: Vec<Option<(usize, usize)>>,
    /// Per-buffer `(first, last)` scheduled position.
    pub buffers: Vec<Option<(usize, usize)>>,
}

/// Computes the live interval of each resource across the scheduled `order`.
pub(crate) fn analyze(
    passes: &[PassNode],
    order: &[usize],
    texture_count: usize,
    buffer_count: usize,
) -> Lifetimes {
    let mut textures = vec![None; texture_count];
    let mut buffers = vec![None; buffer_count];

    for (pos, &pidx) in order.iter().enumerate() {
        let pass = &passes[pidx];
        for acc in &pass.texture_accesses {
            extend(&mut textures[acc.resource.get() as usize], pos);
        }
        for acc in &pass.buffer_accesses {
            extend(&mut buffers[acc.resource.get() as usize], pos);
        }
    }

    Lifetimes { textures, buffers }
}

/// Widens an interval to include `pos`.
fn extend(slot: &mut Option<(usize, usize)>, pos: usize) {
    match slot {
        None => *slot = Some((pos, pos)),
        Some((first, last)) => {
            if pos < *first {
                *first = pos;
            }
            if pos > *last {
                *last = pos;
            }
        }
    }
}
