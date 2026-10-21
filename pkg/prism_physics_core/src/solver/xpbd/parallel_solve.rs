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

use super::color_solve;
use super::config::XpbdConfig;
use super::contact_constraint::{self, ContactConstraint};
use super::graph_color::{self, DynamicBodies};
use super::island_solve::SolveIslands;
use super::{integrate, joint_constraint, velocity_solve};
use crate::collider::{ColliderHandle, PhysicsMaterial};
use crate::dynamics::assembly::JointLoadAccumulator;
use crate::joint::ragdoll::RagdollConstraint;
use crate::joint::{Joint, JointHandle};
use crate::state::body::{BodyKind, SolverMassProperties};
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
    gravity_overrides: Vec<Option<Vec3>>,
    mass_props: Vec<SolverMassProperties>,
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
    /// Original generational handles for `joints`. The solver rewrites body
    /// slots for the scratch view, but load reports must retain the source
    /// handle so recycled joint slots can never receive stale reactions.
    joint_handles: Vec<JointHandle>,
    /// Source entry for each joint, used to return owned winding history.
    joint_indices: Vec<usize>,
    /// Island ragdoll links, re-indexed to local slots.
    ragdolls: Vec<RagdollConstraint>,
}

impl IslandScratch {
    /// Copies one island out of the global view into an owned scratch.
    fn gather(
        view: &BodySolverView<'_>,
        islands: &SolveIslands,
        island: usize,
        constraints: &[ContactConstraint],
        joints: &[(JointHandle, &mut Joint)],
        ragdolls: &[RagdollConstraint],
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
            let (_, j) = &joints[ji];
            slots.push(j.anchor_a.body.index() as usize);
            slots.push(j.anchor_b.body.index() as usize);
        }
        for &ri in islands.ragdolls(island) {
            let r = &ragdolls[ri];
            slots.push(r.body_a.index() as usize);
            slots.push(r.body_b.index() as usize);
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
            gravity_overrides: Vec::with_capacity(n),
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
            joint_handles: Vec::new(),
            joint_indices: Vec::new(),
            ragdolls: Vec::new(),
        };
        for &g in &scratch.slots {
            scratch.positions.push(view.positions[g]);
            scratch.orientations.push(view.orientations[g]);
            scratch.prev_positions.push(view.prev_positions[g]);
            scratch.prev_orientations.push(view.prev_orientations[g]);
            scratch.linear_velocities.push(view.linear_velocities[g]);
            scratch.angular_velocities.push(view.angular_velocities[g]);
            scratch.gravity_overrides.push(view.gravity_overrides[g]);
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
            let (handle, source) = &joints[ji];
            let mut j = **source;
            let la = scratch.local_of(j.anchor_a.body.index() as usize);
            let lb = scratch.local_of(j.anchor_b.body.index() as usize);
            j.anchor_a.body = BodyHandle::new(la as u32, j.anchor_a.body.generation());
            j.anchor_b.body = BodyHandle::new(lb as u32, j.anchor_b.body.generation());
            j.gear_state.remap_bodies(j.anchor_a.body, j.anchor_b.body);
            j.revolute_state.remap_bodies(j.anchor_a.body, j.anchor_b.body);
            scratch.joints.push(j);
            scratch.joint_handles.push(*handle);
            scratch.joint_indices.push(ji);
        }
        for &ri in islands.ragdolls(island) {
            let mut r = ragdolls[ri];
            let la = scratch.local_of(r.body_a.index() as usize);
            let lb = scratch.local_of(r.body_b.index() as usize);
            r.body_a = BodyHandle::new(la as u32, r.body_a.generation());
            r.body_b = BodyHandle::new(lb as u32, r.body_b.generation());
            scratch.ragdolls.push(r);
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
    fn solve(&mut self, config: &XpbdConfig, h: f32) -> JointLoadAccumulator {
        let mut loads = JointLoadAccumulator::new();
        let mut view = BodySolverView {
            positions: &mut self.positions,
            orientations: &mut self.orientations,
            prev_positions: &mut self.prev_positions,
            prev_orientations: &mut self.prev_orientations,
            linear_velocities: &mut self.linear_velocities,
            angular_velocities: &mut self.angular_velocities,
            gravity_overrides: &self.gravity_overrides,
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
                        joint_constraint::solve_joint_with_load(
                            &mut view,
                            self.joint_handles[ji],
                            &mut self.joints[ji],
                            h,
                            &mut loads,
                        );
                    }
                }
                for ragdoll in &self.ragdolls {
                    joint_constraint::solve_ragdoll(&mut view, ragdoll, h);
                }
                // Contacts in a colour write disjoint dynamic bodies, so the
                // whole colour is relaxed across worker threads. This is
                // bit-identical to the serial colour sweep (see `color_solve`).
                let alpha_tilde = contact_constraint::contact_alpha_tilde(config, h);
                color_solve::solve_contacts_parallel(
                    &mut view,
                    &mut self.contacts,
                    &contact_coloring,
                    &contact_order,
                    alpha_tilde,
                );
            }
        } else {
            for _ in 0..iterations {
                for (index, joint) in self.joints.iter_mut().enumerate() {
                    joint_constraint::solve_joint_with_load(
                        &mut view,
                        self.joint_handles[index],
                        joint,
                        h,
                        &mut loads,
                    );
                }
                for ragdoll in &self.ragdolls {
                    joint_constraint::solve_ragdoll(&mut view, ragdoll, h);
                }
                contact_constraint::solve_positions(&mut view, &mut self.contacts, config, h);
            }
        }
        integrate::recover_velocities(&mut view, h);
        for ragdoll in &self.ragdolls {
            joint_constraint::solve_ragdoll_velocity(&mut view, ragdoll, h);
        }
        velocity_solve::solve(&mut view, &self.contacts, config, h);
        loads
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
    fn scatter(&self, view: &mut BodySolverView<'_>, joints: &mut [(JointHandle, &mut Joint)]) {
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
        for (local, &index) in self.joint_indices.iter().enumerate() {
            let target = &mut *joints[index].1;
            target.gear_state = self.joints[local].gear_state;
            target
                .gear_state
                .remap_bodies(target.anchor_a.body, target.anchor_b.body);
            target.revolute_state = self.joints[local].revolute_state;
            target
                .revolute_state
                .remap_bodies(target.anchor_a.body, target.anchor_b.body);
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
    joints: &mut [(JointHandle, &mut Joint)],
    ragdolls: &[RagdollConstraint],
    config: &XpbdConfig,
    h: f32,
    use_threads: bool,
) -> JointLoadAccumulator {
    let mut scratches: Vec<IslandScratch> = active
        .iter()
        .map(|&island| IslandScratch::gather(view, islands, island, constraints, joints, ragdolls))
        .collect();

    // `scratches` is built in stable active-island order. Collect the local
    // reports in that same order, rather than reducing inside rayon workers,
    // so load reporting has the same deterministic reduction as the serial
    // path regardless of worker completion order.
    let island_loads: Vec<JointLoadAccumulator> = if use_threads {
        scratches
            .par_iter_mut()
            .map(|scratch| scratch.solve(config, h))
            .collect()
    } else {
        scratches
            .iter_mut()
            .map(|scratch| scratch.solve(config, h))
            .collect()
    };

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
        scratch.scatter(view, joints);
    }
    integrate::recover_velocities_excluding(view, h, &in_island);

    let mut loads = JointLoadAccumulator::new();
    for island in &island_loads {
        loads
            .merge(island)
            .expect("all XPBD islands in one sub-step share the same finite dt");
    }
    loads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::contact::{ContactManifold, ContactPoint};
    use crate::dynamics::assembly::{BreakLimit, BreakableJointSet, SolverJointLoad};
    use crate::joint::{DistanceJoint, JointAnchor, JointDesc, JointStorage};
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
        let scratch = IslandScratch::gather(&view, &islands, b_island, &constraints, &[], &[]);
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
                &mut [],
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

    /// Interleaved global slots and a shared static separator exercise both
    /// scratch remapping and the preservation of generational joint handles.
    fn reaction_scenario() -> (BodyStorage, JointStorage, Vec<JointHandle>) {
        let mut bodies = BodyStorage::new();
        let ground = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let mut members = [[BodyHandle::INVALID; 3]; 3];
        for link in 0..3 {
            for (island, handles) in members.iter_mut().enumerate() {
                let mut desc = BodyDesc::dynamic_at(Vec3::new(
                    island as f32 * 7.0 + 0.4 + link as f32 * 1.3,
                    0.5 + link as f32 * 0.4,
                    0.2 * island as f32,
                ))
                .with_mass_properties(unit_mass());
                desc.orientation = Quat::from_rotation_z(0.15 * (link + 1) as f32)
                    * Quat::from_rotation_x(0.1 * (island + 1) as f32);
                handles[link] = bodies.insert(desc);
            }
        }
        // An unconstrained body must not create a spurious load report.
        bodies.insert(BodyDesc::dynamic_at(Vec3::new(-20.0, 3.0, 0.0)));
        let mut joints = JointStorage::new();
        let retired = joints.insert(JointDesc::fixed(
            JointAnchor::at_center(ground),
            JointAnchor::at_center(members[0][0]),
        ));
        assert!(joints.remove(retired));
        let mut handles = Vec::new();
        // Insert by joint family, not island, so gathering must retain the
        // original non-contiguous joint handles for each local scratch.
        for link in 0..3 {
            for (island, member) in members.iter().enumerate() {
                let desc = match link {
                    0 => JointDesc::fixed(
                        JointAnchor::at_point(ground, Vec3::X * (island as f32 * 7.0)),
                        JointAnchor::at_point(member[0], Vec3::new(-0.2, 0.1, 0.0)),
                    ),
                    1 => JointDesc::spherical(
                        JointAnchor::at_point(member[0], Vec3::new(0.3, 0.2, 0.0)),
                        JointAnchor::at_point(member[1], Vec3::new(-0.3, 0.0, 0.1)),
                    ),
                    _ => JointDesc::distance(
                        member[1],
                        Vec3::new(0.2, 0.1, 0.0),
                        member[2],
                        Vec3::new(-0.2, 0.0, 0.1),
                        DistanceJoint::rigid(0.6).with_compliance(1.0e-5),
                    ),
                };
                handles.push(joints.insert(desc));
            }
        }
        assert_eq!(handles[0].index(), retired.index());
        assert_ne!(handles[0].generation(), retired.generation());
        joints.insert(
            JointDesc::fixed(
                JointAnchor::at_center(ground),
                JointAnchor::at_center(members[0][2]),
            )
            .with_enabled(false),
        );
        (bodies, joints, handles)
    }

    fn reaction_loads(
        bodies: &mut BodyStorage,
        joints: &JointStorage,
        config: &XpbdConfig,
        h: f32,
        scratch_threads: Option<bool>,
    ) -> Vec<SolverJointLoad> {
        let mut joints = joints.clone();
        let mut entries: Vec<_> = joints.active_entries_mut().collect();
        let joint_refs: Vec<_> = entries.iter().map(|(_, joint)| &**joint).collect();
        let mut view = bodies.solver_view_mut();
        let islands = SolveIslands::build(&view, &[], &joint_refs);
        assert_eq!(islands.island_count(), 3);
        let active = sleep_solve::classify_and_wake(&mut view, &islands, &SleepConfig::default());
        assert_eq!(active.len(), 3);
        for &island in &active {
            assert_eq!(islands.joints(island).len(), 3);
        }
        let loads = if let Some(use_threads) = scratch_threads {
            solve_islands_parallel(
                &mut view,
                &islands,
                &active,
                &[],
                &mut entries,
                &[],
                config,
                h,
                use_threads,
            )
        } else {
            // Independent in-place serial path: no gather/scatter and no
            // worker reduction. Match the production serial iteration order.
            assert!(!config.parallel_within_island);
            let mut loads = JointLoadAccumulator::new();
            for _ in 0..config.position_iterations.max(1) {
                for &island in &active {
                    for &index in islands.joints(island) {
                        let (handle, joint) = &mut entries[index];
                        joint_constraint::solve_joint_with_load(
                            &mut view, *handle, joint, h, &mut loads,
                        );
                    }
                }
            }
            integrate::recover_velocities(&mut view, h);
            loads
        };
        loads.loads().expect("finite measured reactions")
    }

    fn assert_reaction_bits_equal(expected: &[SolverJointLoad], actual: &[SolverJointLoad]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(expected.joint, actual.joint, "joint handle remapping");
            assert_eq!(
                expected.load.linear_impulse.to_array().map(f32::to_bits),
                actual.load.linear_impulse.to_array().map(f32::to_bits),
                "linear reaction for {:?}",
                expected.joint,
            );
            assert_eq!(
                expected.load.angular_impulse.to_array().map(f32::to_bits),
                actual.load.angular_impulse.to_array().map(f32::to_bits),
                "pure angular reaction for {:?}",
                expected.joint,
            );
            assert_eq!(
                expected.load.substep_dt.to_bits(),
                actual.load.substep_dt.to_bits()
            );
        }
    }

    /// Policies straddle the measured forces, and separately straddle angular
    /// reactions, so equality cannot pass just because everything survives or
    /// everything breaks. The last mode crosses a fatigue-reduced threshold
    /// only on the third boundary.
    fn assert_break_decisions_equal(
        joints: &JointStorage,
        expected: &[SolverJointLoad],
        actual: &[SolverJointLoad],
        h: f32,
    ) {
        for mode in 0..3 {
            let mut policies = BreakableJointSet::default();
            for (index, report) in expected.iter().enumerate() {
                let force = report.load.linear_impulse.length() / h;
                let torque = report.load.angular_impulse.length() / h;
                assert!(force.is_finite() && force > 0.0);
                let limit = match mode {
                    0 => BreakLimit::new(force * if index % 2 == 0 { 0.75 } else { 1.25 }, 1.0e30),
                    1 => BreakLimit::new(1.0e30, if torque > 0.0 { torque * 0.75 } else { 1.0 }),
                    _ => BreakLimit::new(force * 1.25, 1.0e30).with_fatigue(0.1 / h),
                };
                policies.register(report.joint, limit).unwrap();
            }
            let mut parallel_policies = policies.clone();
            let mut serial_joints = joints.clone();
            let mut parallel_joints = joints.clone();
            let boundaries = if mode == 2 { 3 } else { 1 };
            for boundary in 0..boundaries {
                let serial_events = policies
                    .submit_solver_boundary(h, expected, &mut serial_joints)
                    .unwrap();
                let parallel_events = parallel_policies
                    .submit_solver_boundary(h, actual, &mut parallel_joints)
                    .unwrap();
                assert_eq!(
                    serial_events, parallel_events,
                    "break mode {mode}, boundary {boundary}"
                );
                assert_eq!(policies, parallel_policies, "fatigue/policy state");
                let expected_count = match mode {
                    0 => expected
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0)
                        .count(),
                    1 => expected
                        .iter()
                        .filter(|load| load.load.angular_impulse != Vec3::ZERO)
                        .count(),
                    _ => {
                        if boundary == 2 {
                            expected.len()
                        } else {
                            0
                        }
                    }
                };
                assert_eq!(serial_events.len(), expected_count);
                for (serial, parallel) in serial_events.iter().zip(&parallel_events) {
                    assert_eq!(serial.force.to_bits(), parallel.force.to_bits());
                    assert_eq!(serial.torque.to_bits(), parallel.torque.to_bits());
                }
                for report in expected {
                    assert_eq!(
                        serial_joints.get(report.joint),
                        parallel_joints.get(report.joint)
                    );
                    let should_break = match mode {
                        0 => {
                            expected
                                .iter()
                                .position(|load| load.joint == report.joint)
                                .unwrap()
                                % 2
                                == 0
                        }
                        1 => report.load.angular_impulse != Vec3::ZERO,
                        _ => boundary == 2,
                    };
                    assert_eq!(
                        serial_joints.get(report.joint).unwrap().enabled,
                        !should_break
                    );
                }
            }
        }
    }

    #[test]
    fn joint_reaction_loads_and_breaks_match_in_place_serial_across_islands() {
        let (bodies, joints, handles) = reaction_scenario();
        let pools: Vec<_> = [1, 2, 4]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        for h in [1.0 / 60.0, 1.0 / 120.0] {
            let mut first_iteration = None;
            for position_iterations in [1, 4, 9] {
                let config = XpbdConfig {
                    position_iterations,
                    parallel_within_island: false,
                    ..XpbdConfig::default()
                };
                let expected = reaction_loads(&mut bodies.clone(), &joints, &config, h, None);
                assert_eq!(
                    expected.len(),
                    handles.len(),
                    "all enabled joints must report"
                );
                for (report, handle) in expected.iter().zip(&handles) {
                    assert_eq!(report.joint, *handle);
                    assert!(report.load.linear_impulse.length() > 0.0);
                    if handles[..3].contains(handle) {
                        assert!(
                            report.load.angular_impulse.length() > 0.0,
                            "fixed joint has angular reaction"
                        );
                    } else {
                        assert_eq!(
                            report.load.angular_impulse,
                            Vec3::ZERO,
                            "free angular DOFs must not report lever-arm torque"
                        );
                    }
                }
                if let Some(first) = &first_iteration {
                    assert_ne!(
                        &expected, first,
                        "later iterations must contribute to reaction load"
                    );
                } else {
                    first_iteration = Some(expected.clone());
                }
                let scratch_serial =
                    reaction_loads(&mut bodies.clone(), &joints, &config, h, Some(false));
                assert_reaction_bits_equal(&expected, &scratch_serial);
                for pool in &pools {
                    let actual = pool.install(|| {
                        reaction_loads(&mut bodies.clone(), &joints, &config, h, Some(true))
                    });
                    assert_reaction_bits_equal(&expected, &actual);
                    assert_break_decisions_equal(&joints, &expected, &actual, h);
                }
            }
        }
    }

    #[test]
    fn coloured_joint_reaction_loads_and_breaks_match_serial_scratch() {
        let (bodies, joints, handles) = reaction_scenario();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let h = 1.0 / 60.0;
        for position_iterations in [1, 4, 9] {
            // Colour-major Gauss-Seidel may differ from natural order; compare
            // threaded vs serial using the same colour schedule, not a tolerance
            // that could conceal dropped reaction contributions.
            let config = XpbdConfig {
                position_iterations,
                parallel_within_island: true,
                ..XpbdConfig::default()
            };
            let expected = reaction_loads(&mut bodies.clone(), &joints, &config, h, Some(false));
            assert_eq!(expected.len(), handles.len());
            let actual = pool
                .install(|| reaction_loads(&mut bodies.clone(), &joints, &config, h, Some(true)));
            assert_reaction_bits_equal(&expected, &actual);
            assert_break_decisions_equal(&joints, &expected, &actual, h);
        }
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
                    &mut [],
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
