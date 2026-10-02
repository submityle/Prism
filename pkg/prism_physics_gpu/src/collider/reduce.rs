//! Per-group manifold reduction: the deterministic rule that collapses a
//! proxy's broad-phase candidate contacts down to the single deepest one.
//!
//! Every mesh collider in this layer (sphere, capsule, OBB, …) runs the same
//! three-stage pipeline — broad phase, narrow phase, reduction — and only the
//! narrow-phase test differs. This module is the shared reduction stage, the
//! single source of truth all those colliders and their device twins call so
//! the tie-break rule cannot drift between shapes or between CPU and GPU.
//!
//! # Contract
//!
//! `contacts` is the flat, group-major batch the narrow phase produced: the
//! first `group_len[0]` entries belong to proxy 0, the next `group_len[1]` to
//! proxy 1, and so on. Each entry is the narrow-phase result for one
//! `(proxy, triangle)` pair, in the same order the pairs were submitted. The
//! caller must have submitted each group's pairs in **ascending triangle
//! index**.
//!
//! The returned vector has one entry per group, in group order: the deepest
//! contact in that group, or [`None`] when the group has no penetrating pair.
//!
//! # Determinism and the tie-break
//!
//! A proxy resting in a crease touches several triangles at the same depth. To
//! make the winner independent of broad-phase traversal order — so a device
//! twin can match the CPU golden lane for lane — a candidate replaces the
//! current best only when it is **strictly deeper**. Combined with the caller's
//! ascending-triangle-index submission, ties therefore resolve to the
//! **smallest triangle index**, deterministically.
//!
//! Provenance: a deepest-point reduction with a lowest-index tie-break is
//! textbook. No Unreal Engine source or derived code.

use crate::narrowphase::Contact;

/// Reduces a group-major batch of per-pair contacts to the single deepest
/// contact per group.
///
/// See the [module documentation](self) for the batch layout, the ascending
/// triangle-index precondition, and the strict-deeper tie-break that makes the
/// result deterministic.
///
/// The sum of `group_len` must equal `contacts.len()`; any trailing contacts
/// beyond the groups described by `group_len` are ignored.
pub(super) fn deepest_per_group(
    contacts: &[Option<Contact>],
    group_len: &[usize],
) -> Vec<Option<Contact>> {
    let mut out: Vec<Option<Contact>> = Vec::with_capacity(group_len.len());
    let mut cursor = 0usize;
    for &len in group_len {
        let group = &contacts[cursor..cursor + len];
        let mut best: Option<Contact> = None;
        for c in group.iter().flatten() {
            match best {
                // Strict-deeper: an equal-depth candidate never displaces the
                // incumbent, so the ascending-index order keeps the smaller
                // triangle index on ties.
                Some(b) if c.depth <= b.depth => {}
                _ => best = Some(*c),
            }
        }
        out.push(best);
        cursor += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Builds a contact carrying just the fields the reduction inspects: the
    /// triangle index `b` and the `depth`. The rest are placeholders.
    fn contact(b: u32, depth: f32) -> Contact {
        Contact {
            a: 0,
            b,
            normal: Vec3::Z,
            depth,
            point: Vec3::ZERO,
        }
    }

    #[test]
    fn empty_group_reports_none() {
        let out = deepest_per_group(&[], &[0]);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_none());
    }

    #[test]
    fn group_of_only_misses_reports_none() {
        let contacts = vec![None, None, None];
        let out = deepest_per_group(&contacts, &[3]);
        assert!(out[0].is_none());
    }

    #[test]
    fn picks_the_strictly_deepest_contact() {
        let contacts = vec![
            Some(contact(5, 0.1)),
            Some(contact(9, 0.4)),
            Some(contact(2, 0.3)),
        ];
        let out = deepest_per_group(&contacts, &[3]);
        let c = out[0].expect("one of the three must win");
        assert_eq!(c.b, 9);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn equal_depth_tie_keeps_the_first_submitted() {
        // Caller submits ascending triangle index, so the first entry is the
        // smallest index; a strict-deeper rule must keep it on an exact tie.
        let contacts = vec![Some(contact(3, 0.25)), Some(contact(7, 0.25))];
        let out = deepest_per_group(&contacts, &[2]);
        assert_eq!(
            out[0].expect("tie still contacts").b,
            3,
            "exact depth tie must keep the first (smallest-index) candidate"
        );
    }

    #[test]
    fn splits_a_flat_batch_across_groups_in_order() {
        // Three groups of lengths 2, 0, 1: group 0 picks the deeper of two,
        // group 1 is empty (None), group 2 passes its single contact through.
        let contacts = vec![
            Some(contact(0, 0.1)),
            Some(contact(1, 0.6)),
            Some(contact(4, 0.2)),
        ];
        let out = deepest_per_group(&contacts, &[2, 0, 1]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].expect("group 0 contacts").b, 1);
        assert!(out[1].is_none());
        assert_eq!(out[2].expect("group 2 contacts").b, 4);
    }

    #[test]
    fn mixed_hits_and_misses_within_a_group() {
        let contacts = vec![None, Some(contact(2, 0.15)), None, Some(contact(8, 0.05))];
        let out = deepest_per_group(&contacts, &[4]);
        assert_eq!(out[0].expect("group contacts").b, 2);
    }
}
