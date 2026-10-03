//! Constraint-graph colouring for intra-island parallel solving.
//!
//! The per-island XPBD pass solves a single island's joints and contacts with
//! Gauss-Seidel: each constraint reads and writes the *current* poses of the
//! two bodies it couples, so neighbouring constraints (those sharing a body)
//! must be solved one after another. Across whole *islands* that dependency is
//! already broken — islands touch disjoint dynamic bodies, so
//! [`parallel_solve`](super::parallel_solve) runs each island on its own worker
//! thread. A single *large* island (a tall stack, a dense pile) still collapses
//! to one thread, which is the remaining scalability wall this module removes.
//!
//! The classic fix, used by Jolt, Rapier and `PhysX`, is **constraint-graph
//! colouring**: partition one island's constraints into colours so that no two
//! constraints in the same colour share a *dynamic* body. Every constraint in a
//! colour then writes a disjoint set of dynamic bodies, so the whole colour can
//! be relaxed in parallel with no read-after-write hazard. Applying colours one
//! after another (Gauss-Seidel *across* colours, Jacobi *within* a colour)
//! reproduces a valid Gauss-Seidel sweep whose per-constraint result does not
//! depend on the intra-colour order, because within one colour the constraints
//! never touch the same dynamic body.
//!
//! Static and kinematic bodies are deliberately **not** treated as conflicts:
//! the position solve never writes them (every apply path is guarded by
//! `is_dynamic`), so many constraints may share the same static separator and
//! still land in one colour without racing. Only dynamic bodies gate a colour.
//!
//! The colouring is a deterministic greedy pass — constraints are visited in
//! ascending index order and each takes the lowest colour not already used by a
//! previously-coloured constraint sharing one of its dynamic bodies. A fixed
//! constraint list therefore always yields the same colours, the same
//! colour-major order and the same offsets, which is what makes a colour-ordered
//! solve reproducible and lets a CPU reference and a GPU per-colour dispatch
//! agree.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Greedy
//! graph colouring for parallel Gauss-Seidel is a standard, publicly documented
//! technique; the constraint-graph schedule mirrors the crate's own
//! [`VbdColoring`](crate::vbd::coloring::VbdColoring) vertex colouring.

/// The dynamic bodies one constraint couples, for colouring purposes.
///
/// A rigid contact or joint touches at most two bodies. Only the *dynamic* ones
/// gate a colour (statics/kinematics are read-only separators and are omitted),
/// so this records between zero and two dynamic body slots. The slots are local
/// solver slots, matching whatever index space the caller colours in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicBodies {
    slots: [usize; 2],
    len: u8,
}

impl DynamicBodies {
    /// A constraint that couples no dynamic body (e.g. a static-vs-static pair
    /// that was not filtered earlier). It conflicts with nothing.
    #[must_use]
    pub const fn none() -> DynamicBodies {
        DynamicBodies {
            slots: [0, 0],
            len: 0,
        }
    }

    /// A constraint that couples a single dynamic body (the other side is a
    /// static or kinematic separator).
    #[must_use]
    pub const fn one(slot: usize) -> DynamicBodies {
        DynamicBodies {
            slots: [slot, 0],
            len: 1,
        }
    }

    /// A constraint that couples two dynamic bodies.
    #[must_use]
    pub const fn two(a: usize, b: usize) -> DynamicBodies {
        DynamicBodies {
            slots: [a, b],
            len: 2,
        }
    }

    /// The dynamic body slots this constraint couples (length `0..=2`).
    #[must_use]
    pub fn slots(&self) -> &[usize] {
        &self.slots[..self.len as usize]
    }
}

/// A proper colouring of an island's constraint graph.
///
/// Produced by [`color_constraints`]. Besides the per-constraint colour it
/// carries a *colour-major ordering* (`order`) split into contiguous runs by
/// `offsets`, so a caller iterates one colour's constraints as
/// `order[offsets[c]..offsets[c + 1]]`. That layout is what both a
/// colour-ordered CPU reference and a per-colour parallel dispatch consume.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintColoring {
    /// `colors[c]` is the colour assigned to constraint `c` (input order).
    colors: Vec<u32>,
    /// Number of distinct colours (serial passes a colour-ordered sweep runs).
    /// `0` only when there are no constraints.
    color_count: u32,
    /// Constraint indices listed colour-major: all of colour `0` (ascending by
    /// constraint index), then colour `1`, and so on. Length equals the
    /// constraint count.
    order: Vec<u32>,
    /// Prefix offsets into `order`; colour `c` occupies
    /// `order[offsets[c]..offsets[c + 1]]`. Length is `color_count + 1` (or a
    /// single `0` when empty).
    offsets: Vec<u32>,
}

impl ConstraintColoring {
    /// The number of colours, i.e. the number of serial passes a colour-ordered
    /// sweep performs per position iteration.
    #[must_use]
    pub fn color_count(&self) -> u32 {
        self.color_count
    }

    /// Returns `true` when the colouring covers no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// The colour assigned to `constraint`, or `None` when the index is out of
    /// range.
    #[must_use]
    pub fn color_of(&self, constraint: usize) -> Option<u32> {
        self.colors.get(constraint).copied()
    }

    /// The per-constraint colour array (length equals the constraint count).
    #[must_use]
    pub fn colors(&self) -> &[u32] {
        &self.colors
    }

    /// The colour-major constraint order (length equals the constraint count).
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The prefix offsets into [`order`](Self::order); length is
    /// `color_count + 1` whenever there is at least one constraint.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// The half-open range into [`order`](Self::order) for `color`, or an empty
    /// range when the colour index is out of bounds (never panics).
    #[must_use]
    pub fn color_range(&self, color: usize) -> core::ops::Range<usize> {
        if color + 1 < self.offsets.len() {
            let start = self.offsets[color] as usize;
            let end = self.offsets[color + 1] as usize;
            start..end
        } else {
            0..0
        }
    }

    /// The constraints belonging to `color` in ascending index order, or an
    /// empty slice when the colour index is out of bounds.
    #[must_use]
    pub fn color_members(&self, color: usize) -> &[u32] {
        let range = self.color_range(color);
        &self.order[range]
    }

    /// Verifies the colouring is *proper*: within every colour the dynamic
    /// bodies touched by any two constraints are disjoint.
    ///
    /// This is the invariant that makes per-colour parallel relaxation free of
    /// write-after-write hazards. It is intended for tests and debug assertions;
    /// it runs in O(constraints x bodies-per-constraint) with a reusable scratch.
    #[must_use]
    pub fn is_proper(&self, constraints: &[DynamicBodies], body_slot_count: usize) -> bool {
        // For each dynamic body slot, the colour that last claimed it. A proper
        // colouring never claims the same slot twice within one colour.
        let mut claimed_by: Vec<u32> = vec![u32::MAX; body_slot_count];
        for color in 0..self.color_count as usize {
            for &ci in self.color_members(color) {
                let Some(constraint) = constraints.get(ci as usize) else {
                    return false;
                };
                for &slot in constraint.slots() {
                    if slot >= body_slot_count {
                        continue;
                    }
                    if claimed_by[slot] == color as u32 {
                        return false;
                    }
                    claimed_by[slot] = color as u32;
                }
            }
        }
        true
    }
}

/// Greedily colours an island's constraint graph so that no two constraints in
/// one colour share a dynamic body.
///
/// `constraints[c]` lists the dynamic body slots constraint `c` couples;
/// `body_slot_count` is the number of body slots in the index space those slots
/// address (used only to size an internal scratch — slots at or beyond it are
/// ignored, which also lets callers pass static separators safely as unlisted
/// bodies). The pass visits constraints in ascending index order and assigns
/// each the lowest colour not already used by a previously-coloured constraint
/// sharing one of its dynamic bodies, so a constraint with no dynamic conflicts
/// always lands in colour `0`. The result is deterministic for a fixed input.
///
/// An empty constraint set yields an empty colouring with `color_count == 0`.
#[must_use]
pub fn color_constraints(
    constraints: &[DynamicBodies],
    body_slot_count: usize,
) -> ConstraintColoring {
    let count = constraints.len();
    if count == 0 {
        return ConstraintColoring {
            colors: Vec::new(),
            color_count: 0,
            order: Vec::new(),
            offsets: vec![0],
        };
    }

    // Colours already used by previously-coloured constraints on each dynamic
    // body slot. Because every two constraints sharing a slot must differ, the
    // colours recorded for one slot are always distinct.
    let mut used_by_body: Vec<Vec<u32>> = vec![Vec::new(); body_slot_count];
    let mut colors = vec![0u32; count];
    let mut max_color = 0u32;

    for (c, constraint) in constraints.iter().enumerate() {
        // A constraint needs at most (distinct neighbour colours + 1) colours,
        // so a boolean scratch of that size always contains the lowest free one.
        let upper = constraint
            .slots()
            .iter()
            .filter(|&&slot| slot < body_slot_count)
            .map(|&slot| used_by_body[slot].len())
            .sum::<usize>()
            + 1;
        let mut forbidden = vec![false; upper + 1];
        for &slot in constraint.slots() {
            if slot >= body_slot_count {
                continue;
            }
            for &used in &used_by_body[slot] {
                if (used as usize) < forbidden.len() {
                    forbidden[used as usize] = true;
                }
            }
        }
        let mut chosen = 0u32;
        while (chosen as usize) < forbidden.len() && forbidden[chosen as usize] {
            chosen += 1;
        }
        colors[c] = chosen;
        max_color = max_color.max(chosen);
        for &slot in constraint.slots() {
            if slot < body_slot_count {
                used_by_body[slot].push(chosen);
            }
        }
    }

    let color_count = max_color + 1;

    // Count each colour, then lay constraints out colour-major (ascending within
    // each colour) with a running per-colour cursor.
    let mut offsets = vec![0u32; color_count as usize + 1];
    for &c in &colors {
        offsets[c as usize + 1] += 1;
    }
    for i in 0..color_count as usize {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut order = vec![0u32; count];
    for (c, &color) in colors.iter().enumerate() {
        let slot = cursor[color as usize];
        order[slot as usize] = c as u32;
        cursor[color as usize] = slot + 1;
    }

    ConstraintColoring {
        colors,
        color_count,
        order,
        offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_constraints_color_to_empty_coloring() {
        let coloring = color_constraints(&[], 0);
        assert!(coloring.is_empty());
        assert_eq!(coloring.color_count(), 0);
        assert_eq!(coloring.offsets(), &[0]);
        assert!(coloring.order().is_empty());
        assert!(coloring.is_proper(&[], 0));
    }

    #[test]
    fn constraints_without_dynamic_conflicts_all_get_color_zero() {
        // Four constraints each coupling a single, distinct dynamic body: no two
        // share a body, so one colour suffices.
        let constraints = [
            DynamicBodies::one(0),
            DynamicBodies::one(1),
            DynamicBodies::one(2),
            DynamicBodies::one(3),
        ];
        let coloring = color_constraints(&constraints, 4);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0, 0, 0]);
        assert_eq!(coloring.color_members(0), &[0, 1, 2, 3]);
        assert!(coloring.is_proper(&constraints, 4));
    }

    #[test]
    fn constraints_with_no_dynamic_bodies_never_conflict() {
        // Static-vs-static pairs touch no dynamic body; they all share colour 0.
        let constraints = [DynamicBodies::none(), DynamicBodies::none()];
        let coloring = color_constraints(&constraints, 4);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0]);
        assert!(coloring.is_proper(&constraints, 4));
    }

    #[test]
    fn two_constraints_sharing_a_body_need_two_colors() {
        // Both couple dynamic body 0, so they must differ.
        let constraints = [DynamicBodies::two(0, 1), DynamicBodies::two(0, 2)];
        let coloring = color_constraints(&constraints, 3);
        assert_eq!(coloring.color_count(), 2);
        assert_ne!(coloring.color_of(0), coloring.color_of(1));
        assert!(coloring.is_proper(&constraints, 3));
    }

    #[test]
    fn static_separators_do_not_force_extra_colors() {
        // Two contacts against the same static ground (slot 10, outside the
        // dynamic index space of 2 bodies) couple distinct dynamic bodies, so
        // they share a colour even though both touch the static separator.
        let constraints = [DynamicBodies::one(0), DynamicBodies::one(1)];
        let coloring = color_constraints(&constraints, 2);
        assert_eq!(coloring.color_count(), 1);
        assert!(coloring.is_proper(&constraints, 2));
    }

    #[test]
    fn chain_of_contacts_colors_like_a_path_graph() {
        // A tall stack: contacts (0,1),(1,2),(2,3),(3,4). Each shares a body with
        // its neighbour, so a greedy pass alternates two colours along the path.
        let constraints = [
            DynamicBodies::two(0, 1),
            DynamicBodies::two(1, 2),
            DynamicBodies::two(2, 3),
            DynamicBodies::two(3, 4),
        ];
        let coloring = color_constraints(&constraints, 5);
        assert_eq!(coloring.color_count(), 2);
        assert_eq!(coloring.colors(), &[0, 1, 0, 1]);
        assert!(coloring.is_proper(&constraints, 5));
    }

    #[test]
    fn star_of_contacts_needs_a_color_per_edge() {
        // A hub body 0 touched by four others forces four colours: every contact
        // shares body 0, so no two may coincide.
        let constraints = [
            DynamicBodies::two(0, 1),
            DynamicBodies::two(0, 2),
            DynamicBodies::two(0, 3),
            DynamicBodies::two(0, 4),
        ];
        let coloring = color_constraints(&constraints, 5);
        assert_eq!(coloring.color_count(), 4);
        assert_eq!(coloring.colors(), &[0, 1, 2, 3]);
        assert!(coloring.is_proper(&constraints, 5));
    }

    #[test]
    fn coloring_is_deterministic() {
        let constraints = [
            DynamicBodies::two(0, 1),
            DynamicBodies::two(1, 2),
            DynamicBodies::two(0, 2),
            DynamicBodies::one(3),
            DynamicBodies::two(3, 4),
        ];
        let a = color_constraints(&constraints, 5);
        let b = color_constraints(&constraints, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn every_colour_bucket_is_internally_independent_on_a_dense_graph() {
        // A deterministic pseudo-random dense graph; the proper-colouring
        // invariant must hold for every bucket.
        let mut constraints = Vec::new();
        let bodies = 24usize;
        let mut seed = 0x9e37_79b9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as usize
        };
        for _ in 0..200 {
            let a = next() % bodies;
            let mut b = next() % bodies;
            if a == b {
                b = (b + 1) % bodies;
            }
            constraints.push(DynamicBodies::two(a, b));
        }
        let coloring = color_constraints(&constraints, bodies);
        assert!(
            coloring.is_proper(&constraints, bodies),
            "greedy colouring produced a non-proper bucket"
        );
        // Every constraint is placed exactly once in the colour-major order.
        assert_eq!(coloring.order().len(), constraints.len());
        let mut seen = vec![false; constraints.len()];
        for &c in coloring.order() {
            assert!(!seen[c as usize], "constraint {c} listed twice");
            seen[c as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn out_of_range_slots_are_ignored() {
        // Slots at or beyond body_slot_count act like static separators.
        let constraints = [DynamicBodies::two(0, 99), DynamicBodies::two(1, 99)];
        let coloring = color_constraints(&constraints, 2);
        assert_eq!(coloring.color_count(), 1);
        assert!(coloring.is_proper(&constraints, 2));
    }
}
