//! Parallel island solving for the XPBD solver.
//!
//! Islands are connected components of dynamic bodies; two islands never share
//! a dynamic body, so their constraint solves are independent. This module
//! turns that independence into thread-level parallelism: each active island is
//! copied into an owned [`IslandScratch`] with its body columns and locally
//! re-indexed constraints, the scratches are solved in parallel with rayon, and
//! the results are scattered back into the global solver view.
//!
//! # Bit-exactness
//!
//! The parallel path is numerically identical to the serial per-island path,
//! bit for bit, for three reasons:
//!
//! 1. Each island's operation sequence is unchanged: for every position
//!    iteration it solves the same joints then the same contacts, in the same
//!    order, then recovers velocities and runs the same velocity pass.
//! 2. Gathering copies body columns bitwise and only remaps global slot indices
//!    to local ones; remapping an index never changes a floating-point result.
//! 3. Islands touch disjoint dynamic bodies, so no thread ever observes another
//!    thread's writes. Free awake bodies (in no active island) are recovered
//!    with the same finite-difference formula through
//!    [`recover_velocities_excluding`](super::integrate::recover_velocities_excluding).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is an
//! original scatter/gather parallelisation layer over the crate's own
//! per-island XPBD solve.

use super::config::XpbdConfig;
use super::contact_constraint::{self, ContactConstraint};
use super::graph_color::{self, DynamicBodies};
use super::island_solve::SolveIslands;
use super::{integrate, joint_constraint, velocity_solve};
use crate::collider::{ColliderHandle, PhysicsMaterial};
use crate::joint::Joint;
use crate::state::body::{BodyKind, MassProperties};
use crate::state::handle::BodyHandle;
use crate::state::view::BodySolverView;
use glam::{Quat, Vec3};
use rayon::prelude::*;

/// An owned copy of a single island's solver state.
///
/// Every column mirrors the matching [`BodySolverView`] slice but is indexed by
/// a *local* slot: local slot `l` maps to global slot `slots[l]`. Contacts and
/// joints are cloned with their body references rewritten to local slots so the
/// solve never touches the global storage until [`IslandScratch::scatter`].
#[derive(Clone)]
struct IslandScratch {
    positions: Vec<Vec3>,
    orientations: Vec<Quat>,
    prev_positions: Vec<Vec3>,
    prev_orientations: Vec<Quat>,
    linear_velocities: Vec<Vec3>,
    angular_velocities: Vec<Vec3>,
    mass_props: Vec<MassProperties>,
    kinds: Vec<BodyKind>,
    colliders: Vec<Option<ColliderHandle>>,
    materials: Vec<PhysicsMaterial>,
    is_sensor: Vec<bool>,
    linear_damping: Vec<f32>,
    angular_damping: Vec<f32>,
    sleeping: Vec<bool>,
    sleep_timers: Vec<f32>,
    active: Vec<bool>,
    /// Global slot for each local index, sorted and de-duplicated.
    slots: Vec<usize>,
    /// Island contacts, re-indexed to local slots.
    contacts: Vec<ContactConstraint>,
    /// Island joints, re-indexed to local slots.
    joints: Vec<Joint>,
}

impl IslandScratch {
    /// Copies one island out of the global view into an owned scratch.
    fn gather(
        view: &BodySolverView<'_>,
        islands: &SolveIslands,
        island: usize,
        constraints: &[ContactConstraint],
        joints: &[&Joint],
    ) -> IslandScratch {
        // Union of every global slot the island's constraints and joints touch,
        // including static separators (contacts/joints may reference statics).
        let mut slots: Vec<usize> = Vec::new();
        for &ci in islands.contacts(island) {
            let c = &constraints[ci];
            slots.push(c.slot_a);
            slots.push(c.slot_b);
        }
        for &ji in islands.joints(island) {
            let j = joints[ji];
            slots.push(j.anchor_a.body.index() as usize);
            slots.push(j.anchor_b.body.index() as usize);
        }
        slots.sort_unstable();
        slots.dedup();

        let n = slots.len();
        let mut scratch = IslandScratch {
            positions: Vec::with_capacity(n),
            orientations: Vec::with_capacity(n),
            prev_positions: Vec::with_capacity(n),
            prev_orientations: Vec::with_capacity(n),
            linear_velocities: Vec::with_capacity(n),
            angular_velocities: Vec::with_capacity(n),
            mass_props: Vec::with_capacity(n),
            kinds: Vec::with_capacity(n),
            colliders: Vec::with_capacity(n),
            materials: Vec::with_capacity(n),
            is_sensor: Vec::with_capacity(n),
            linear_damping: Vec::with_capacity(n),
            angular_damping: Vec::with_capacity(n),
            sleeping: Vec::with_capacity(n),
            sleep_timers: Vec::with_capacity(n),
            active: Vec::with_capacity(n),
            slots,
            contacts: Vec::new(),
            joints: Vec::new(),
        };
        for &g in &scratch.slots {
            scratch.positions.push(view.positions[g]);
            scratch.orientations.push(view.orientations[g]);
            scratch.prev_positions.push(view.prev_positions[g]);
            scratch.prev_orientations.push(view.prev_orientations[g]);
            scratch.linear_velocities.push(view.linear_velocities[g]);
            scratch.angular_velocities.push(view.angular_velocities[g]);
            scratch.mass_props.push(view.mass_props[g]);
            scratch.kinds.push(view.kinds[g]);
            scratch.colliders.push(view.colliders[g]);
            scratch.materials.push(view.materials[g]);
            scratch.is_sensor.push(view.is_sensor[g]);
            scratch.linear_damping.push(view.linear_damping[g]);
            scratch.angular_damping.push(view.angular_damping[g]);
            scratch.sleeping.push(view.sleeping[g]);
            scratch.sleep_timers.push(view.sleep_timers[g]);
            scratch.active.push(view.active[g]);
        }

        // Re-index the island's contacts onto local slots.
        for &ci in islands.contacts(island) {
            let mut c = constraints[ci].clone();
            c.slot_a = scratch.local_of(c.slot_a);
            c.slot_b = scratch.local_of(c.slot_b);
            scratch.contacts.push(c);
        }
        // Re-index the island's joints onto local slots. The joint solver reads
        // only the handle index (never the generation), so preserving the
        // original generation while rewriting the index is sufficient.
        for &ji in islands.joints(island) {
            let mut j = *joints[ji];
            let la = scratch.local_of(j.anchor_a.body.index() as usize);
            let lb = scratch.local_of(j.anchor_b.body.index() as usize);
            j.anchor_a.body = BodyHandle::new(la as u32, j.anchor_a.body.generation());
            j.anchor_b.body = BodyHandle::new(lb as u32, j.anchor_b.body.generation());
            scratch.joints.push(j);
        }
        scratch
    }

    /// Maps a global slot to its local index within this scratch.
    fn local_of(&self, global: usize) -> usize {
        self.slots
            .binary_search(&global)
            .expect("gathered slot must be present in the scratch")
    }

    /// Runs the full per-island XPBD solve on the owned scratch.
    ///
    /// This mirrors the serial per-island pass exactly: for each position
    /// iteration it solves the island joints then the island contacts, then it
    /// recovers velocities and applies the velocity-level restitution and
    /// dynamic-friction solve.
    fn solve(&mut self, config: &XpbdConfig, h: f32) {
        let mut view = BodySolverView {
            positions: &mut self.positions,
            orientations: &mut self.orientations,
            prev_positions: &mut self.prev_positions,
            prev_orientations: &mut self.prev_orientations,
            linear_velocities: &mut self.linear_velocities,
            angular_velocities: &mut self.angular_velocities,
            mass_props: &self.mass_props,
            kinds: &self.kinds,
            colliders: &self.colliders,
            materials: &self.materials,
            is_sensor: &self.is_sensor,
            linear_damping: &self.linear_damping,
            angular_damping: &self.angular_damping,
            sleeping: &mut self.sleeping,
            sleep_timers: &mut self.sleep_timers,
            active: &self.active,
        };
        let iterations = config.position_iterations.max(1);
        if config.parallel_within_island {
            // Colour the island once per sub-step (the constraint graph is
            // fixed across the position iterations) and relax one colour at a
            // time. Joints and contacts are coloured separately so the natural
            // "all joints, then all contacts" phase ordering is preserved; only
            // the order *within* each phase changes, from natural index order
            // to colour-major order.
            let joint_bodies: Vec<DynamicBodies> = self
                .joints
                .iter()
                .map(|j| {
                    Self::dynamic_bodies(
                        &self.kinds,
                        &self.active,
                        j.anchor_a.body.index() as usize,
                        j.anchor_b.body.index() as usize,
                    )
                })
                .collect();
            let contact_bodies: Vec<DynamicBodies> = self
                .contacts
                .iter()
                .map(|c| Self::dynamic_bodies(&self.kinds, &self.active, c.slot_a, c.slot_b))
                .collect();
            let slot_count = self.slots.len();
            let joint_coloring = graph_color::color_constraints(&joint_bodies, slot_count);
            let contact_coloring = graph_color::color_constraints(&contact_bodies, slot_count);
            // Flatten the colour-major order once; `color_range` then slices the
            // constraints that make up each colour.
            let joint_order: Vec<usize> =
                joint_coloring.order().iter().map(|&i| i as usize).collect();
            let contact_order: Vec<usize> = contact_coloring
                .order()
                .iter()
                .map(|&i| i as usize)
                .collect();
            for _ in 0..iterations {
                for colour in 0..joint_coloring.color_count() as usize {
                    for &ji in &joint_order[joint_coloring.color_range(colour)] {
                        joint_constraint::solve_joint(&mut view, &self.joints[ji], h);
                    }
                }
                for colour in 0..contact_coloring.color_count() as usize {
                    let members = &contact_order[contact_coloring.color_range(colour)];
                    contact_constraint::solve_positions_indexed(
                        &mut view,
                        &mut self.contacts,
                        members,
                        config,
                        h,
                    );
                }
            }
        } else {
            for _ in 0..iterations {
                for joint in &self.joints {
                    joint_constraint::solve_joint(&mut view, joint, h);
                }
                contact_constraint::solve_positions(&mut view, &mut self.contacts, config, h);
            }
        }
        integrate::recover_velocities(&mut view, h);
        velocity_solve::solve(&mut view, &self.contacts, config, h);
    }

    /// Classifies the dynamic bodies a constraint between `slot_a`/`slot_b`
    /// couples for colouring: only live dynamic locals gate a colour, since the
    /// position solve never writes static or kinematic separators.
    fn dynamic_bodies(
        kinds: &[BodyKind],
        active: &[bool],
        slot_a: usize,
        slot_b: usize,
    ) -> DynamicBodies {
        let is_dyn = |slot: usize| {
            active.get(slot).copied().unwrap_or(false)
                && kinds.get(slot).copied() == Some(BodyKind::Dynamic)
        };
        match (is_dyn(slot_a), is_dyn(slot_b)) {
            (true, true) => DynamicBodies::two(slot_a, slot_b),
            (true, false) => DynamicBodies::one(slot_a),
            (false, true) => DynamicBodies::one(slot_b),
            (false, false) => DynamicBodies::none(),
        }
    }

    /// Writes the solved dynamic-body columns back into the global view.
    ///
    /// Only live dynamic locals are scattered; static and kinematic separators
    /// are read-only during the solve and keep their global pose untouched.
    fn scatter(&self, view: &mut BodySolverView<'_>) {
        for l in 0..self.slots.len() {
            if self.kinds[l] != BodyKind::Dynamic || !self.active[l] {
                continue;
            }
            let g = self.slots[l];
            view.positions[g] = self.positions[l];
            view.orientations[g] = self.orientations[l];
            view.linear_velocities[g] = self.linear_velocities[l];
            view.angular_velocities[g] = self.angular_velocities[l];
        }
    }
}

/// Solves the active islands in parallel and recovers the free bodies.
///
/// `active` is the list of island indices that still have an awake body (the
/// output of [`classify_and_wake`](super::sleep_solve::classify_and_wake)).
/// Each active island is gathered into an owned scratch, the scratches are
/// solved concurrently (or serially when `use_threads` is `false`), and their
/// dynamic results are scattered back. Free awake dynamic bodies that belong to
/// no active island are then recovered globally with the same formula the
/// serial path uses, so the whole pass is bit-identical to the serial solve.
#[expect(
    clippy::too_many_arguments,
    reason = "the parallel island pass needs the full view, island partition, constraint and joint slices, config, step, and the threading toggle"
)]
pub fn solve_islands_parallel(
    view: &mut BodySolverView<'_>,
    islands: &SolveIslands,
    active: &[usize],
    constraints: &[ContactConstraint],
    joints: &[&Joint],
    config: &XpbdConfig,
    h: f32,
    use_threads: bool,
) {
    let mut scratches: Vec<IslandScratch> = active
        .iter()
        .map(|&island| IslandScratch::gather(view, islands, island, constraints, joints))
        .collect();

    if use_threads {
        scratches
            .par_iter_mut()
            .for_each(|scratch| scratch.solve(config, h));
    } else {
        for scratch in &mut scratches {
            scratch.solve(config, h);
        }
    }

    // Mark every dynamic member of an active island so the free-body recovery
    // below skips the bodies the scratches already handled.
    let mut in_island = vec![false; view.slot_count()];
    for &island in active {
        for &slot in islands.members(island) {
            if slot < in_island.len() {
                in_island[slot] = true;
            }
        }
    }

    for scratch in &scratches {
        scratch.scatter(view);
    }
    integrate::recover_velocities_excluding(view, h, &in_island);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::contact::{ContactManifold, ContactPoint};
    use crate::sleep::SleepConfig;
    use crate::solver::xpbd::sleep_solve;
    use crate::state::body::{BodyDesc, MassProperties};
    use crate::state::storage::BodyStorage;

    fn unit_mass() -> MassProperties {
        MassProperties {
            inv_mass: 1.0,
            inv_inertia: Vec3::ONE,
        }
    }

    /// Two dynamic boxes overlapping along Y, plus a second isolated pair far
    /// away, plus a lone free-falling body with no contacts.
    fn scenario() -> BodyStorage {
        let mut s = BodyStorage::new();
        // Island A: slots 0,1 penetrating.
        s.insert(BodyDesc::dynamic_at(Vec3::new(0.0, 0.0, 0.0)).with_mass_properties(unit_mass()));
        s.insert(BodyDesc::dynamic_at(Vec3::new(0.0, 0.9, 0.0)).with_mass_properties(unit_mass()));
        // Island B: slots 2,3 penetrating, far away.
        s.insert(BodyDesc::dynamic_at(Vec3::new(50.0, 0.0, 0.0)).with_mass_properties(unit_mass()));
        s.insert(BodyDesc::dynamic_at(Vec3::new(50.0, 0.9, 0.0)).with_mass_properties(unit_mass()));
        // Free body: slot 4, no contacts.
        s.insert(
            BodyDesc::dynamic_at(Vec3::new(-30.0, 5.0, 0.0)).with_mass_properties(unit_mass()),
        );
        s
    }

    fn manifolds(s: &BodyStorage) -> Vec<ContactManifold> {
        let h = |slot: usize| s.handle_at_slot(slot).expect("live slot");
        let mut a = ContactManifold::new(h(0), h(1), Vec3::Y);
        a.push(ContactPoint::new(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.4, 0.0),
            0.1,
        ));
        let mut b = ContactManifold::new(h(2), h(3), Vec3::Y);
        b.push(ContactPoint::new(
            Vec3::new(50.0, 0.5, 0.0),
            Vec3::new(50.0, 0.4, 0.0),
            0.1,
        ));
        vec![a, b]
    }

    #[test]
    fn gather_remaps_slots_to_local() {
        let mut s = scenario();
        let ms = manifolds(&s);
        let view = s.solver_view_mut();
        let constraints = ContactConstraint::build(&view, &ms);
        let islands = SolveIslands::build(&view, &constraints, &[]);
        // Island B references global slots 2 and 3.
        let b_island = (0..islands.island_count())
            .find(|&i| islands.members(i) == [2usize, 3])
            .expect("island B present");
        let scratch = IslandScratch::gather(&view, &islands, b_island, &constraints, &[]);
        assert_eq!(scratch.slots, vec![2, 3]);
        assert_eq!(scratch.contacts.len(), 1);
        // The single contact must now address local slots 0 and 1.
        assert_eq!(scratch.contacts[0].slot_a, 0);
        assert_eq!(scratch.contacts[0].slot_b, 1);
        assert_eq!(scratch.positions[0], Vec3::new(50.0, 0.0, 0.0));
        assert_eq!(scratch.positions[1], Vec3::new(50.0, 0.9, 0.0));
    }

    /// Solving with worker threads must produce byte-identical bodies to the
    /// single-threaded path.
    #[test]
    fn threaded_matches_serial_bit_for_bit() {
        let h = 1.0 / 60.0;
        let run = |use_threads: bool| -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>, Vec<Vec3>) {
            let mut s = scenario();
            let ms = manifolds(&s);
            let mut view = s.solver_view_mut();
            let constraints = ContactConstraint::build(&view, &ms);
            let islands = SolveIslands::build(&view, &constraints, &[]);
            let active =
                sleep_solve::classify_and_wake(&mut view, &islands, &SleepConfig::default());
            solve_islands_parallel(
                &mut view,
                &islands,
                &active,
                &constraints,
                &[],
                &XpbdConfig::default(),
                h,
                use_threads,
            );
            (
                view.positions.to_vec(),
                view.orientations.to_vec(),
                view.linear_velocities.to_vec(),
                view.angular_velocities.to_vec(),
            )
        };
        let serial = run(false);
        let threaded = run(true);
        assert_eq!(serial.0, threaded.0, "positions diverged");
        assert_eq!(serial.1, threaded.1, "orientations diverged");
        assert_eq!(serial.2, threaded.2, "linear velocities diverged");
        assert_eq!(serial.3, threaded.3, "angular velocities diverged");
        // The overlapping pairs must actually have been pushed apart, proving
        // the solve did real work rather than trivially matching.
        assert!(
            serial.0[1].y - serial.0[0].y > 0.9,
            "island A did not separate"
        );
    }

    /// A single tall stack is one dense island: colouring is the only way to
    /// parallelise it. The colour-ordered sweep reorders Gauss-Seidel so it is
    /// *not* bit-identical to the natural-index sweep, but it must converge to
    /// the same rest state. This proves the intra-island colouring path does
    /// real, correct work rather than silently falling back.
    fn chained_stack(n: usize) -> BodyStorage {
        let mut s = BodyStorage::new();
        // Boxes stacked along Y, each penetrating the one below by 0.1.
        for i in 0..n {
            s.insert(
                BodyDesc::dynamic_at(Vec3::new(0.0, i as f32 * 0.9, 0.0))
                    .with_mass_properties(unit_mass()),
            );
        }
        s
    }

    fn chain_manifolds(s: &BodyStorage, n: usize) -> Vec<ContactManifold> {
        let h = |slot: usize| s.handle_at_slot(slot).expect("live slot");
        let mut out = Vec::new();
        for i in 0..n - 1 {
            let y = (i as f32 + 0.5) * 0.9;
            let mut m = ContactManifold::new(h(i), h(i + 1), Vec3::Y);
            m.push(ContactPoint::new(
                Vec3::new(0.0, y + 0.05, 0.0),
                Vec3::new(0.0, y - 0.05, 0.0),
                0.1,
            ));
            out.push(m);
        }
        out
    }

    #[test]
    fn intra_island_colouring_matches_natural_order_closely() {
        let n = 6;
        let h = 1.0 / 60.0;
        let run = |within: bool| -> Vec<Vec3> {
            let mut s = chained_stack(n);
            let ms = chain_manifolds(&s, n);
            let mut view = s.solver_view_mut();
            let constraints = ContactConstraint::build(&view, &ms);
            let islands = SolveIslands::build(&view, &constraints, &[]);
            // The whole stack must be a single island, otherwise this test is
            // not exercising intra-island colouring at all.
            let awake = (0..islands.island_count())
                .filter(|&i| !islands.members(i).is_empty())
                .count();
            assert_eq!(awake, 1, "chained stack should form exactly one island");
            let active =
                sleep_solve::classify_and_wake(&mut view, &islands, &SleepConfig::default());
            let config = XpbdConfig {
                position_iterations: 8,
                parallel_within_island: within,
                ..XpbdConfig::default()
            };
            // Advance several sub-steps so the stack settles.
            for _ in 0..20 {
                solve_islands_parallel(
                    &mut view,
                    &islands,
                    &active,
                    &constraints,
                    &[],
                    &config,
                    h,
                    true,
                );
            }
            view.positions.to_vec()
        };
        let natural = run(false);
        let coloured = run(true);
        assert_eq!(natural.len(), coloured.len());
        // Reordering Gauss-Seidel is not bit-identical, so require closeness,
        // not equality — mirrors the VBD colour-vs-natural-order golden.
        for (i, (nat, col)) in natural.iter().zip(&coloured).enumerate() {
            assert!(
                (*nat - *col).length() < 1e-3,
                "body {i} diverged between natural and coloured sweeps: {nat:?} vs {col:?}"
            );
        }
        // And the colouring must have done real separation work: the stack is
        // monotonically increasing in Y and the first gap opened up.
        for i in 1..coloured.len() {
            assert!(
                coloured[i].y > coloured[i - 1].y,
                "coloured stack not ordered at body {i}"
            );
        }
        assert!(
            coloured[1].y - coloured[0].y > 0.2,
            "coloured stack did not separate"
        );
    }
}
