//! Threaded intra-island contact solving over a coloured constraint graph.
//!
//! [`graph_color`](super::graph_color) partitions one island's contacts into
//! colours so that no two contacts in a colour write the same *dynamic* body.
//! That independence is what this module turns into thread-level parallelism:
//! a whole colour is relaxed at once, every contact in it on its own worker,
//! because their writes land in disjoint dynamic bodies.
//!
//! # Why this is bit-identical to the serial colour sweep
//!
//! Within one colour the contacts touch disjoint dynamic bodies, so solving
//! contact *k* never reads or writes a body that contact *j* in the same colour
//! touches. Relaxing them one after another (serial) therefore produces exactly
//! the same per-body result as relaxing them all from the colour's entry state
//! (parallel). This module captures each contact's two bodies into an owned
//! two-body snapshot at the start of the colour, solves each snapshot with the
//! *unmodified* [`solve_constraint_positions`](super::contact_constraint)
//! kernel on worker threads, then writes the dynamic results back. Because the
//! kernel and its inputs are byte-for-byte the ones the serial path would use,
//! the floating-point output matches the serial colour sweep bit for bit; the
//! only freedom — the order of contacts *within* a colour — cannot change any
//! value, since those contacts share no mutable body.
//!
//! Colours are still applied in order (Gauss-Seidel across colours), so each
//! colour reads the poses the previous colour produced.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Parallel
//! Gauss-Seidel over a coloured constraint graph is a standard, publicly
//! documented technique (Jolt, Rapier, `PhysX`); the schedule mirrors the
//! crate's own [`VbdColoring`](crate::vbd::coloring::VbdColoring).

use super::contact_constraint::{self, ContactConstraint};
use super::graph_color::ConstraintColoring;
use crate::collider::PhysicsMaterial;
use crate::state::body::{BodyKind, MassProperties};
use crate::state::view::BodySolverView;
use glam::{Quat, Vec3};
use rayon::prelude::*;

/// The read-only state of one body, captured for an isolated two-body solve.
#[derive(Clone, Copy)]
struct BodySnapshot {
    position: Vec3,
    orientation: Quat,
    prev_position: Vec3,
    prev_orientation: Quat,
    mass: MassProperties,
    kind: BodyKind,
    material: PhysicsMaterial,
    is_sensor: bool,
    active: bool,
}

/// One contact lifted out of the shared columns for an independent solve.
///
/// `constraint` is a clone whose body slots have been remapped to the local
/// pair `(0, 1)`, so it can be solved against a two-element [`BodySolverView`].
struct IsolatedContact {
    /// Index of this contact in the island contact slice.
    index: usize,
    /// Global/local island slot of body `a`.
    slot_a: usize,
    /// Global/local island slot of body `b`.
    slot_b: usize,
    constraint: ContactConstraint,
    a: BodySnapshot,
    b: BodySnapshot,
}

/// The solved result of one [`IsolatedContact`], ready to scatter back.
struct SolvedContact {
    index: usize,
    slot_a: usize,
    slot_b: usize,
    a_dynamic: bool,
    b_dynamic: bool,
    position_a: Vec3,
    orientation_a: Quat,
    position_b: Vec3,
    orientation_b: Quat,
    constraint: ContactConstraint,
}

/// Captures a contact and its two bodies into an owned, `Send` unit of work.
fn lift(
    view: &BodySolverView<'_>,
    contacts: &[ContactConstraint],
    index: usize,
) -> IsolatedContact {
    let c = &contacts[index];
    let snap = |slot: usize| BodySnapshot {
        position: view.positions[slot],
        orientation: view.orientations[slot],
        prev_position: view.prev_positions[slot],
        prev_orientation: view.prev_orientations[slot],
        mass: view.mass_props[slot],
        kind: view.kinds[slot],
        material: view.materials[slot],
        is_sensor: view.is_sensor[slot],
        active: view.active[slot],
    };
    let mut remapped = c.clone();
    remapped.slot_a = 0;
    remapped.slot_b = 1;
    IsolatedContact {
        index,
        slot_a: c.slot_a,
        slot_b: c.slot_b,
        constraint: remapped,
        a: snap(c.slot_a),
        b: snap(c.slot_b),
    }
}

/// Solves one isolated contact on a private two-body view and reports the
/// result. This runs the exact same kernel as the serial path.
fn solve_isolated(work: IsolatedContact, alpha_tilde: f32) -> SolvedContact {
    let IsolatedContact {
        index,
        slot_a,
        slot_b,
        mut constraint,
        a,
        b,
    } = work;

    // Two-body columns: local slot 0 is body a, local slot 1 is body b.
    let mut positions = [a.position, b.position];
    let mut orientations = [a.orientation, b.orientation];
    let mut prev_positions = [a.prev_position, b.prev_position];
    let mut prev_orientations = [a.prev_orientation, b.prev_orientation];
    // The position solve never touches velocities, damping, sleeping or
    // colliders, so those columns are inert placeholders.
    let mut linear_velocities = [Vec3::ZERO; 2];
    let mut angular_velocities = [Vec3::ZERO; 2];
    let mut sleeping = [false; 2];
    let mut sleep_timers = [0.0f32; 2];
    let mass_props = [a.mass, b.mass];
    let kinds = [a.kind, b.kind];
    let colliders = [None, None];
    let materials = [a.material, b.material];
    let is_sensor = [a.is_sensor, b.is_sensor];
    let linear_damping = [0.0f32; 2];
    let angular_damping = [0.0f32; 2];
    let active = [a.active, b.active];

    let mut view = BodySolverView {
        positions: &mut positions,
        orientations: &mut orientations,
        prev_positions: &mut prev_positions,
        prev_orientations: &mut prev_orientations,
        linear_velocities: &mut linear_velocities,
        angular_velocities: &mut angular_velocities,
        mass_props: &mass_props,
        kinds: &kinds,
        colliders: &colliders,
        materials: &materials,
        is_sensor: &is_sensor,
        linear_damping: &linear_damping,
        angular_damping: &angular_damping,
        sleeping: &mut sleeping,
        sleep_timers: &mut sleep_timers,
        active: &active,
    };

    let a_dynamic = view.is_dynamic(0);
    let b_dynamic = view.is_dynamic(1);
    contact_constraint::solve_constraint_positions(&mut view, &mut constraint, alpha_tilde);

    // Restore the real island slots so the scattered constraint stays valid.
    constraint.slot_a = slot_a;
    constraint.slot_b = slot_b;

    SolvedContact {
        index,
        slot_a,
        slot_b,
        a_dynamic,
        b_dynamic,
        position_a: positions[0],
        orientation_a: orientations[0],
        position_b: positions[1],
        orientation_b: orientations[1],
        constraint,
    }
}

/// Writes one solved contact's dynamic bodies and updated working state back
/// into the shared island columns.
fn scatter(
    view: &mut BodySolverView<'_>,
    contacts: &mut [ContactConstraint],
    solved: SolvedContact,
) {
    if solved.a_dynamic {
        view.positions[solved.slot_a] = solved.position_a;
        view.orientations[solved.slot_a] = solved.orientation_a;
    }
    if solved.b_dynamic {
        view.positions[solved.slot_b] = solved.position_b;
        view.orientations[solved.slot_b] = solved.orientation_b;
    }
    // The constraint carries its own accumulated Lagrange state; keep it.
    contacts[solved.index] = solved.constraint;
}

/// Runs one position-solve iteration over `contacts`, relaxing each colour of
/// `coloring` in parallel across worker threads.
///
/// `order` is the colour-major constraint order from
/// [`ConstraintColoring::order`]; `coloring.color_range(colour)` slices the
/// contacts that make up each colour. Colours are applied in ascending order so
/// the sweep stays Gauss-Seidel across colours and bit-identical to the serial
/// colour sweep.
pub(super) fn solve_contacts_parallel(
    view: &mut BodySolverView<'_>,
    contacts: &mut [ContactConstraint],
    coloring: &ConstraintColoring,
    order: &[usize],
    alpha_tilde: f32,
) {
    for colour in 0..coloring.color_count() as usize {
        let members = &order[coloring.color_range(colour)];
        if members.is_empty() {
            continue;
        }
        // Lift the whole colour into owned, independent work units captured
        // from the colour's entry state, solve them on worker threads, then
        // scatter the disjoint dynamic results back.
        let work: Vec<IsolatedContact> = members.iter().map(|&i| lift(view, contacts, i)).collect();
        let solved: Vec<SolvedContact> = work
            .into_par_iter()
            .map(|w| solve_isolated(w, alpha_tilde))
            .collect();
        for s in solved {
            scatter(view, contacts, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::contact::{ContactManifold, ContactPoint};
    use crate::solver::xpbd::contact_constraint;
    use crate::solver::xpbd::graph_color;
    use crate::solver::xpbd::XpbdConfig;
    use crate::state::body::{BodyDesc, MassProperties};
    use crate::state::storage::BodyStorage;

    fn unit_mass() -> MassProperties {
        MassProperties {
            inv_mass: 1.0,
            inv_inertia: Vec3::ONE,
        }
    }

    /// A vertical chain of `n` dynamic boxes, each penetrating the one below.
    /// Consecutive contacts share the middle bodies, so the constraint graph
    /// needs two colours — exactly the case parallel colour solving targets.
    fn chain(n: usize) -> (BodyStorage, Vec<ContactManifold>) {
        let mut s = BodyStorage::new();
        for i in 0..n {
            s.insert(
                BodyDesc::dynamic_at(Vec3::new(0.0, i as f32 * 0.9, 0.0))
                    .with_mass_properties(unit_mass()),
            );
        }
        let h = |slot: usize| s.handle_at_slot(slot).expect("live slot");
        let mut ms = Vec::new();
        for i in 0..n - 1 {
            let y = (i as f32 + 0.5) * 0.9;
            let mut m = ContactManifold::new(h(i), h(i + 1), Vec3::Y);
            m.push(ContactPoint::new(
                Vec3::new(0.0, y + 0.05, 0.0),
                Vec3::new(0.0, y - 0.05, 0.0),
                0.1,
            ));
            ms.push(m);
        }
        (s, ms)
    }

    fn dynamic_bodies(
        view: &BodySolverView<'_>,
        c: &ContactConstraint,
    ) -> graph_color::DynamicBodies {
        match (view.is_dynamic(c.slot_a), view.is_dynamic(c.slot_b)) {
            (true, true) => graph_color::DynamicBodies::two(c.slot_a, c.slot_b),
            (true, false) => graph_color::DynamicBodies::one(c.slot_a),
            (false, true) => graph_color::DynamicBodies::one(c.slot_b),
            (false, false) => graph_color::DynamicBodies::none(),
        }
    }

    /// The threaded colour solve must be *bit-for-bit* identical to a serial
    /// colour sweep driven through `solve_positions_indexed`. Both relax the
    /// colours in the same order; within a colour the contacts write disjoint
    /// dynamic bodies, so parallel and serial cannot differ. This is the
    /// anti-fake-implementation guarantee for `solve_contacts_parallel`.
    #[test]
    fn parallel_colour_solve_is_bit_identical_to_serial_colour_sweep() {
        let n = 7;
        let h = 1.0 / 120.0;
        let config = XpbdConfig {
            position_iterations: 4,
            ..XpbdConfig::default()
        };
        let alpha_tilde = contact_constraint::contact_alpha_tilde(&config, h);

        // Build the coloured schedule once from a throwaway view.
        let (coloring, order) = {
            let mut s = chain(n).0;
            let ms = chain(n).1;
            let view = s.solver_view_mut();
            let contacts = ContactConstraint::build(&view, &ms);
            let bodies: Vec<_> = contacts.iter().map(|c| dynamic_bodies(&view, c)).collect();
            let coloring = graph_color::color_constraints(&bodies, view.slot_count());
            assert!(
                coloring.color_count() >= 2,
                "chain must need multiple colours"
            );
            let order: Vec<usize> = coloring.order().iter().map(|&i| i as usize).collect();
            (coloring, order)
        };

        // Serial reference: relax each colour in turn via the indexed kernel.
        let serial = {
            let mut s = chain(n).0;
            let ms = chain(n).1;
            let mut view = s.solver_view_mut();
            let mut contacts = ContactConstraint::build(&view, &ms);
            for _ in 0..config.position_iterations {
                for colour in 0..coloring.color_count() as usize {
                    let members: Vec<usize> = order[coloring.color_range(colour)].to_vec();
                    contact_constraint::solve_positions_indexed(
                        &mut view,
                        &mut contacts,
                        &members,
                        &config,
                        h,
                    );
                }
            }
            (view.positions.to_vec(), view.orientations.to_vec())
        };

        // Threaded path: relax each colour in parallel.
        let parallel = {
            let mut s = chain(n).0;
            let ms = chain(n).1;
            let mut view = s.solver_view_mut();
            let mut contacts = ContactConstraint::build(&view, &ms);
            for _ in 0..config.position_iterations {
                solve_contacts_parallel(&mut view, &mut contacts, &coloring, &order, alpha_tilde);
            }
            (view.positions.to_vec(), view.orientations.to_vec())
        };

        assert_eq!(serial.0, parallel.0, "positions must match bit for bit");
        assert_eq!(serial.1, parallel.1, "orientations must match bit for bit");
        // And the solve must have done real work: the chain spread out.
        assert!(
            parallel.0[1].y - parallel.0[0].y > 0.2,
            "contacts did not separate the chain"
        );
    }
}
