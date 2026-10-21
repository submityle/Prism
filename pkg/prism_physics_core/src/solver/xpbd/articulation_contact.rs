//! Mixed reduced/maximal-coordinate contacts. Real collider witnesses feed a
//! projected velocity solve and a separate compliant positional solve. The
//! Jacobian and articulated factorization are frozen for one small substep,
//! then rebuilt at the next boundary. Retained storage is linear in the sum
//! of tree sizes and contacts, not tree size times contact count.

use std::collections::BTreeMap;

use glam::{Mat3, Quat, Vec3};
use prism_physics_geometry::{generate_pairs, Aabb, DynamicBvh};

use super::{rigid, XpbdConfig};
use crate::collide::generate_contacts_in;
use crate::collider::{ColliderHandle, ColliderShape, PhysicsMaterial};
use crate::math::transform::Isometry;
use crate::reduced::articulation::ArticulationImpulseSolver;
use crate::reduced::{ArticulationHandle, ContactImpulseResponse, SpatialMotion};
use crate::world::PhysicsWorld;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    Rigid(usize),
    Tree(usize, Option<usize>),
}

struct Attachment {
    endpoint: Endpoint,
    shape: ColliderHandle,
    pose: Isometry,
    material: PhysicsMaterial,
    self_collision: bool,
}

struct Tree {
    handle: ArticulationHandle,
    mass: ArticulationImpulseSolver,
    velocity: ContactImpulseResponse,
    displacement: ContactImpulseResponse,
    response: ContactImpulseResponse,
    limit_projection: LimitProjection,
    touched: bool,
}

struct Rigid {
    slot: usize,
    pose: Isometry,
    inverse_mass: Mat3,
    inverse_inertia: Mat3,
    velocity: SpatialMotion,
    displacement: SpatialMotion,
    dynamic: bool,
    touched: bool,
}

struct Contact {
    a: Endpoint,
    b: Endpoint,
    point_a: Vec3,
    point_b: Vec3,
    basis: [Vec3; 3],
    inverse_mass: [[f32; 3]; 3],
    penetration: f32,
    friction: f32,
    restitution: f32,
    bounce: f32,
    impulse: [f32; 3],
    position_impulse: f32,
}

struct LimitRow {
    tree: usize,
    dof: usize,
    sign: f32,
    effective_mass: f32,
    impulse: f32,
    position_impulse: f32,
}

#[derive(Clone, Copy)]
struct LimitAxis {
    dof: usize,
    sign: f32,
}

struct LimitProjection {
    axes: Vec<LimitAxis>,
    delassus: Vec<f32>,
    rhs: Vec<f32>,
    lambda: Vec<f32>,
    generalized: Vec<f32>,
    correction: ContactImpulseResponse,
}

impl LimitProjection {
    fn empty(dofs: usize) -> Self {
        Self {
            axes: Vec::new(),
            delassus: Vec::new(),
            rhs: Vec::new(),
            lambda: Vec::new(),
            generalized: vec![0.0; dofs],
            correction: zero_state(dofs),
        }
    }
}

enum Response {
    Rigid(usize, SpatialMotion),
    Tree(usize),
}

fn allowed_pair(a: &Attachment, b: &Attachment, rigids: &[Rigid], world: &PhysicsWorld) -> bool {
    match (a.endpoint, b.endpoint) {
        // Revisit ordinary contacts too: a tree can push a rigid body against
        // a third body in the same substep. Leaving those out treats that
        // support as free space until the next frame. With no attachments the
        // entire mixed pass is absent, preserving the normal rigid-only path.
        (Endpoint::Rigid(a), Endpoint::Rigid(b)) => {
            match (
                world.bodies.handle_at_slot(rigids[a].slot),
                world.bodies.handle_at_slot(rigids[b].slot),
            ) {
                (Some(a), Some(b)) => a != b,
                _ => false,
            }
        }
        (Endpoint::Tree(a_tree, a_link), Endpoint::Tree(b_tree, b_link)) if a_tree == b_tree => {
            a_link != b_link && a.self_collision && b.self_collision
        }
        _ => true,
    }
}

fn root(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

/// Only re-solve rigid contacts in islands reached by a tree. Static supports
/// are read-only separators: a shared ground must not wake unrelated piles.
fn retain_coupled_islands(contacts: &mut Vec<Contact>, trees: &[Tree], rigids: &[Rigid]) {
    let node = |endpoint| match endpoint {
        Endpoint::Tree(index, _) => Some(rigids.len() + index),
        Endpoint::Rigid(index) => rigids[index].dynamic.then_some(index),
    };
    let count = rigids.len() + trees.len();
    let mut parents: Vec<_> = (0..count).collect();
    let mut sizes = vec![1usize; count];
    for contact in contacts.iter() {
        if let (Some(a), Some(b)) = (node(contact.a), node(contact.b)) {
            let (mut a, mut b) = (root(&mut parents, a), root(&mut parents, b));
            if a != b {
                if sizes[a] < sizes[b] {
                    core::mem::swap(&mut a, &mut b);
                }
                parents[b] = a;
                sizes[a] += sizes[b];
            }
        }
    }
    let mut coupled = vec![false; count];
    for index in 0..trees.len() {
        coupled[root(&mut parents, rigids.len() + index)] = true;
    }
    contacts.retain(|contact| {
        [contact.a, contact.b]
            .into_iter()
            .filter_map(node)
            .any(|node| coupled[root(&mut parents, node)])
    });
}

fn candidate_pairs(attachments: &[Attachment], world: &PhysicsWorld) -> Vec<(usize, usize)> {
    let mut bvh = DynamicBvh::with_capacity(attachments.len());
    let mut finite = Vec::new();
    let mut planes = Vec::new();
    for (index, attachment) in attachments.iter().enumerate() {
        let shape = world.shapes.get(attachment.shape).expect("validated shape");
        if matches!(shape, ColliderShape::Plane { .. }) {
            planes.push(index);
            continue;
        }
        let (min, max) = shape.local_aabb();
        let corners: [Vec3; 8] = core::array::from_fn(|corner| {
            attachment.pose.transform_point(Vec3::new(
                if corner & 1 == 0 { min.x } else { max.x },
                if corner & 2 == 0 { min.y } else { max.y },
                if corner & 4 == 0 { min.z } else { max.z },
            ))
        });
        if let Some(aabb) = Aabb::from_points(&corners) {
            bvh.insert(aabb, index as u64);
            finite.push(index);
        }
    }
    let mut pairs: Vec<_> = generate_pairs(&bvh)
        .into_iter()
        .map(|pair| {
            let (a, b) = (pair.a as usize, pair.b as usize);
            (a.min(b), a.max(b))
        })
        .collect();
    for &plane in &planes {
        for &body in &finite {
            pairs.push((plane.min(body), plane.max(body)));
        }
    }
    // Stable contact order is independent of BVH rotation/rebalancing.
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn endpoint_response(
    endpoint: Endpoint,
    point: Vec3,
    impulse: Vec3,
    trees: &mut [Tree],
    rigids: &[Rigid],
    project_limits: bool,
) -> Response {
    match endpoint {
        Endpoint::Rigid(index) => {
            let body = &rigids[index];
            Response::Rigid(
                index,
                SpatialMotion {
                    linear: body.inverse_mass * impulse,
                    angular: body.inverse_inertia * (point - body.pose.translation).cross(impulse),
                },
            )
        }
        Endpoint::Tree(index, link) => {
            let tree = &mut trees[index];
            tree.mass
                .solve_world_impulses_into(&[(link, point, impulse)], &mut tree.response);
            if project_limits {
                project_limit_response(tree);
            }
            Response::Tree(index)
        }
    }
}

fn pair_response(
    contact: &Contact,
    impulse: Vec3,
    trees: &mut [Tree],
    rigids: &[Rigid],
    project_limits: bool,
) -> [Option<Response>; 2] {
    // Two endpoints on one tree share all generalized DOFs. Sum their signed
    // external impulses BEFORE applying M^-1; independent endpoint masses
    // would omit cross terms and can even resolve a locked self contact.
    if let (Endpoint::Tree(a, a_link), Endpoint::Tree(b, b_link)) = (contact.a, contact.b) {
        if a == b {
            let tree = &mut trees[a];
            tree.mass.solve_world_impulses_into(
                &[
                    (a_link, contact.point_a, -impulse),
                    (b_link, contact.point_b, impulse),
                ],
                &mut tree.response,
            );
            if project_limits {
                project_limit_response(tree);
            }
            return [Some(Response::Tree(a)), None];
        }
    }
    [
        Some(endpoint_response(
            contact.a,
            contact.point_a,
            -impulse,
            trees,
            rigids,
            project_limits,
        )),
        Some(endpoint_response(
            contact.b,
            contact.point_b,
            impulse,
            trees,
            rigids,
            project_limits,
        )),
    ]
}

fn response_at(
    endpoint: Endpoint,
    point: Vec3,
    responses: &[Option<Response>; 2],
    trees: &[Tree],
    rigids: &[Rigid],
) -> Vec3 {
    for response in responses.iter().flatten() {
        match (endpoint, response) {
            (Endpoint::Rigid(index), Response::Rigid(owner, motion)) if index == *owner => {
                return motion.linear
                    + motion.angular.cross(point - rigids[index].pose.translation);
            }
            (Endpoint::Tree(index, link), Response::Tree(owner)) if index == *owner => {
                return trees[index]
                    .mass
                    .point_velocity(&trees[index].response, link, point);
            }
            _ => {}
        }
    }
    Vec3::ZERO
}

fn response_relative_velocity(
    contact: &Contact,
    responses: &[Option<Response>; 2],
    trees: &[Tree],
    rigids: &[Rigid],
) -> Vec3 {
    response_at(contact.b, contact.point_b, responses, trees, rigids)
        - response_at(contact.a, contact.point_a, responses, trees, rigids)
}

fn velocity_at(
    endpoint: Endpoint,
    point: Vec3,
    trees: &[Tree],
    rigids: &[Rigid],
    positional: bool,
) -> Vec3 {
    match endpoint {
        Endpoint::Rigid(index) => {
            let body = &rigids[index];
            let motion = if positional {
                body.displacement
            } else {
                body.velocity
            };
            motion.linear + motion.angular.cross(point - body.pose.translation)
        }
        Endpoint::Tree(index, link) => {
            let tree = &trees[index];
            tree.mass.point_velocity(
                if positional {
                    &tree.displacement
                } else {
                    &tree.velocity
                },
                link,
                point,
            )
        }
    }
}

fn relative_velocity(
    contact: &Contact,
    trees: &[Tree],
    rigids: &[Rigid],
    positional: bool,
) -> Vec3 {
    velocity_at(contact.b, contact.point_b, trees, rigids, positional)
        - velocity_at(contact.a, contact.point_a, trees, rigids, positional)
}

fn add_motion(target: &mut SpatialMotion, delta: SpatialMotion) {
    target.linear += delta.linear;
    target.angular += delta.angular;
}

fn apply(
    contact: &Contact,
    impulse: Vec3,
    trees: &mut [Tree],
    rigids: &mut [Rigid],
    positional: bool,
) {
    if impulse == Vec3::ZERO {
        return;
    }
    let responses = pair_response(contact, impulse, trees, rigids, false);
    for response in responses.into_iter().flatten() {
        match response {
            Response::Rigid(index, delta) => {
                let body = &mut rigids[index];
                if !body.dynamic {
                    continue;
                }
                body.touched = true;
                add_motion(
                    if positional {
                        &mut body.displacement
                    } else {
                        &mut body.velocity
                    },
                    delta,
                );
            }
            Response::Tree(index) => {
                let tree = &mut trees[index];
                tree.touched = true;
                let target = if positional {
                    &mut tree.displacement
                } else {
                    &mut tree.velocity
                };
                for (value, delta) in target
                    .generalized_velocity_change
                    .iter_mut()
                    .zip(&tree.response.generalized_velocity_change)
                {
                    *value += delta;
                }
                add_motion(
                    &mut target.base_velocity_change,
                    tree.response.base_velocity_change,
                );
            }
        }
    }
}

/// Spectral pseudoinverse of the symmetric tangential block. Locked tangent
/// directions have zero response, not an artificial identity response. f64
/// intermediates avoid loss of the smaller eigenvalue for anisotropic masses.
fn tangent_response(k: [[f32; 3]; 3], rhs: [f32; 2]) -> [f32; 2] {
    let (a, b, d) = (k[1][1] as f64, k[1][2] as f64, k[2][2] as f64);
    let angle = 0.5 * (2.0 * b).atan2(a - d);
    let (s, c) = angle.sin_cos();
    let vectors = [[c, s], [-s, c]];
    let scale = a.abs().max(d.abs()).max(b.abs());
    let mut result = [0.0_f64; 2];
    for v in vectors {
        let eigenvalue = a * v[0] * v[0] + 2.0 * b * v[0] * v[1] + d * v[1] * v[1];
        if eigenvalue > scale * 1.0e-10 && eigenvalue > 0.0 {
            let value = (rhs[0] as f64 * v[0] + rhs[1] as f64 * v[1]) / eigenvalue;
            result[0] += v[0] * value;
            result[1] += v[1] * value;
        }
    }
    [result[0] as f32, result[1] as f32]
}

fn zero_state(dofs: usize) -> ContactImpulseResponse {
    ContactImpulseResponse {
        generalized_velocity_change: vec![0.0; dofs],
        base_velocity_change: SpatialMotion::default(),
    }
}

fn project_limit_response(tree: &mut Tree) {
    let count = tree.limit_projection.axes.len();
    if count == 0 {
        return;
    }
    let projection = &mut tree.limit_projection;
    for (rhs, axis) in projection.rhs.iter_mut().zip(&projection.axes) {
        *rhs = axis.sign * tree.response.generalized_velocity_change[axis.dof];
    }
    if projection.rhs.iter().all(|value| *value >= -1.0e-7) {
        return;
    }

    projection.lambda.fill(0.0);
    let sweeps = count.saturating_mul(8).clamp(8, 64);
    for _ in 0..sweeps {
        let mut largest_change = 0.0_f32;
        for row in 0..count {
            let diagonal = projection.delassus[row * count + row];
            if diagonal <= 1.0e-8 {
                continue;
            }
            let mut residual = projection.rhs[row];
            for col in 0..count {
                if col != row {
                    residual += projection.delassus[row * count + col] * projection.lambda[col];
                }
            }
            let next = (-residual / diagonal).max(0.0);
            largest_change = largest_change.max((next - projection.lambda[row]).abs());
            projection.lambda[row] = next;
        }
        if largest_change <= 1.0e-7 {
            break;
        }
    }

    projection.generalized.fill(0.0);
    for (axis, &magnitude) in projection.axes.iter().zip(&projection.lambda) {
        projection.generalized[axis.dof] += axis.sign * magnitude;
    }
    tree.mass
        .solve_generalized_impulses_into(&projection.generalized, &mut projection.correction);
    for (value, correction) in tree
        .response
        .generalized_velocity_change
        .iter_mut()
        .zip(&projection.correction.generalized_velocity_change)
    {
        *value += correction;
    }
    add_motion(
        &mut tree.response.base_velocity_change,
        projection.correction.base_velocity_change,
    );
}

fn build_limit_rows(trees: &mut [Tree], world: &PhysicsWorld) -> Vec<LimitRow> {
    let mut rows = Vec::new();
    for (tree_index, tree) in trees.iter_mut().enumerate() {
        let Some(model) = world.articulations.get(tree.handle) else {
            continue;
        };
        let mut axes = Vec::new();
        let mut offset = 0;
        for link in model.links() {
            let dofs = link.joint.kind.dofs();
            if link.joint.limits.enabled {
                for local in 0..dofs {
                    let dof = offset + local;
                    let lower = model.positions()[dof];
                    let lower_bound = link.joint.limits.lower[local];
                    let upper_bound = link.joint.limits.upper[local];
                    let tolerance = 1.0e-5_f32;
                    for (sign, active) in [
                        (1.0, lower <= lower_bound + tolerance),
                        (-1.0, lower >= upper_bound - tolerance),
                    ] {
                        if !active {
                            continue;
                        }
                        axes.push(LimitAxis { dof, sign });
                    }
                }
            }
            offset += dofs;
        }
        let count = axes.len();
        let dofs = model.dof_count();
        let mut projection = LimitProjection {
            axes,
            delassus: vec![0.0; count * count],
            rhs: vec![0.0; count],
            lambda: vec![0.0; count],
            generalized: vec![0.0; dofs],
            correction: zero_state(dofs),
        };
        for col in 0..count {
            projection.generalized.fill(0.0);
            let axis = projection.axes[col];
            projection.generalized[axis.dof] = axis.sign;
            tree.mass.solve_generalized_impulses_into(
                &projection.generalized,
                &mut projection.correction,
            );
            for row in 0..count {
                let target = projection.axes[row];
                projection.delassus[row * count + col] =
                    target.sign * projection.correction.generalized_velocity_change[target.dof];
            }
        }
        for row in 0..count {
            for col in 0..row {
                let value = 0.5
                    * (projection.delassus[row * count + col]
                        + projection.delassus[col * count + row]);
                projection.delassus[row * count + col] = value;
                projection.delassus[col * count + row] = value;
            }
        }
        for (axis_index, axis) in projection.axes.iter().copied().enumerate() {
            let effective_mass = projection.delassus[axis_index * count + axis_index];
            if effective_mass.is_finite() && effective_mass > 1.0e-8 {
                rows.push(LimitRow {
                    tree: tree_index,
                    dof: axis.dof,
                    sign: axis.sign,
                    effective_mass,
                    impulse: 0.0,
                    position_impulse: 0.0,
                });
            }
        }
        tree.limit_projection = projection;
    }
    rows
}

fn apply_limit_impulse(row: &mut LimitRow, magnitude: f32, trees: &mut [Tree], positional: bool) {
    if !magnitude.is_finite() || magnitude == 0.0 {
        return;
    }
    let tree = &mut trees[row.tree];
    tree.limit_projection.generalized.fill(0.0);
    tree.limit_projection.generalized[row.dof] = row.sign * magnitude;
    let target = if positional {
        &mut tree.displacement
    } else {
        &mut tree.velocity
    };
    tree.mass
        .solve_generalized_impulses_into(&tree.limit_projection.generalized, &mut tree.response);
    for (value, delta) in target
        .generalized_velocity_change
        .iter_mut()
        .zip(&tree.response.generalized_velocity_change)
    {
        *value += *delta;
    }
    target.base_velocity_change.angular += tree.response.base_velocity_change.angular;
    target.base_velocity_change.linear += tree.response.base_velocity_change.linear;
    tree.touched = true;
}

fn solve_limit_rows_velocity(rows: &mut [LimitRow], trees: &mut [Tree], world: &PhysicsWorld) {
    for row in rows {
        if world.articulations.get(trees[row.tree].handle).is_none() {
            continue;
        }
        let velocity = trees[row.tree].velocity.generalized_velocity_change[row.dof];
        let inward = row.sign * velocity;
        let next = (row.impulse - inward / row.effective_mass).max(0.0);
        let delta = next - row.impulse;
        if delta != 0.0 {
            apply_limit_impulse(row, delta, trees, false);
            row.impulse = next;
        }
    }
}

fn solve_limit_rows_position(rows: &mut [LimitRow], trees: &mut [Tree], world: &PhysicsWorld) {
    for row in rows {
        let Some(model) = world.articulations.get(trees[row.tree].handle) else {
            continue;
        };
        let q = model.positions()[row.dof]
            + trees[row.tree].displacement.generalized_velocity_change[row.dof];
        let link = model
            .links()
            .iter()
            .scan(0usize, |offset, link| {
                let start = *offset;
                *offset += link.joint.kind.dofs();
                Some((start, link))
            })
            .find(|(start, link)| row.dof >= *start && row.dof < *start + link.joint.kind.dofs());
        let Some((start, link)) = link else {
            continue;
        };
        let local = row.dof - start;
        let bound = if row.sign > 0.0 {
            link.joint.limits.lower[local]
        } else {
            link.joint.limits.upper[local]
        };
        let separation = if row.sign > 0.0 { q - bound } else { bound - q };
        let next = (row.position_impulse - separation / row.effective_mass).max(0.0);
        let delta = next - row.position_impulse;
        if delta != 0.0 {
            apply_limit_impulse(row, delta, trees, true);
            row.position_impulse = next;
        }
    }
}

pub(super) fn solve(world: &mut PhysicsWorld, config: &XpbdConfig, h: f32) {
    if world.articulation_colliders.is_empty() {
        return;
    }
    // Public storages can be edited directly; revalidate generational owners
    // and geometry at the actual consumption boundary, not only at attach.
    let PhysicsWorld {
        articulations,
        shapes,
        articulation_colliders,
        ..
    } = world;
    articulation_colliders.retain(|desc| {
        shapes.get(desc.shape).is_some()
            && articulations
                .get(desc.articulation)
                .is_some_and(|tree| desc.link.is_none_or(|link| link < tree.link_count()))
    });
    // A degraded tree stops solving, but still owns its attachments until an
    // explicit removal or successful atomic handoff. In particular, a rejected
    // handoff must be retryable without losing any of its collision geometry.
    if !articulation_colliders.iter().any(|(_, desc)| {
        articulations
            .get(desc.articulation)
            .is_some_and(|tree| tree.broken_joint().is_none())
    }) {
        return;
    }

    let mut trees = Vec::new();
    let mut tree_indices = BTreeMap::new();
    let mut attachments = Vec::new();
    let mut rigids = Vec::new();
    {
        let view = world.bodies.solver_view_mut();
        for slot in 0..view.slot_count() {
            if !view.is_active(slot) || view.is_sensor(slot) {
                continue;
            }
            let Some(shape) =
                view.colliders[slot].filter(|shape| world.shapes.get(*shape).is_some())
            else {
                continue;
            };
            let pose = Isometry::new(view.positions[slot], view.orientations[slot]);
            let dynamic = view.is_dynamic(slot);
            let mass = view.mass_props[slot];
            attachments.push(Attachment {
                endpoint: Endpoint::Rigid(rigids.len()),
                shape,
                pose,
                material: view.materials[slot],
                self_collision: false,
            });
            rigids.push(Rigid {
                slot,
                pose,
                inverse_mass: if dynamic {
                    rigid::inv_mass_world(mass.inv_mass, pose.rotation)
                } else {
                    Mat3::ZERO
                },
                inverse_inertia: if dynamic {
                    rigid::inv_inertia_world(mass.inv_inertia, pose.rotation)
                } else {
                    Mat3::ZERO
                },
                velocity: SpatialMotion {
                    linear: view.linear_velocities[slot],
                    angular: view.angular_velocities[slot],
                },
                displacement: SpatialMotion::default(),
                dynamic,
                touched: false,
            });
        }
    }
    for (_, desc) in world.articulation_colliders.iter() {
        if world
            .articulations
            .get(desc.articulation)
            .expect("validated owner")
            .broken_joint()
            .is_some()
        {
            continue;
        }
        let key = (desc.articulation.index(), desc.articulation.generation());
        let index = *tree_indices.entry(key).or_insert_with(|| {
            let model = world
                .articulations
                .get(desc.articulation)
                .expect("validated owner");
            let index = trees.len();
            trees.push(Tree {
                handle: desc.articulation,
                mass: ArticulationImpulseSolver::new(model),
                velocity: ContactImpulseResponse {
                    generalized_velocity_change: model.velocities().to_vec(),
                    base_velocity_change: model.base_velocity(),
                },
                displacement: zero_state(model.dof_count()),
                response: zero_state(model.dof_count()),
                limit_projection: LimitProjection::empty(model.dof_count()),
                touched: false,
            });
            index
        });
        attachments.push(Attachment {
            endpoint: Endpoint::Tree(index, desc.link),
            shape: desc.shape,
            pose: trees[index].mass.body_pose(desc.link).mul(&desc.local_pose),
            material: desc.material,
            self_collision: desc.self_collision,
        });
    }
    let mut limits = build_limit_rows(&mut trees, world);
    let pairs = candidate_pairs(&attachments, world);
    let mut contacts = Vec::new();
    for (a, b) in pairs {
        let (a, b) = (&attachments[a], &attachments[b]);
        if !allowed_pair(a, b, &rigids, world) {
            continue;
        }
        let manifolds = generate_contacts_in(
            world.shapes.get(a.shape).expect("validated shape"),
            &a.pose,
            world.shapes.get(b.shape).expect("validated shape"),
            &b.pose,
            &world.shapes,
        );
        for manifold in manifolds {
            if let (Endpoint::Rigid(ai), Endpoint::Rigid(bi)) = (a.endpoint, b.endpoint) {
                let ah = world
                    .bodies
                    .handle_at_slot(rigids[ai].slot)
                    .expect("live rigid");
                let bh = world
                    .bodies
                    .handle_at_slot(rigids[bi].slot)
                    .expect("live rigid");
                if !world.collision_exclusions().allows_leaf_pair(
                    ah,
                    bh,
                    manifold.leaf_self_collision[0],
                    manifold.leaf_self_collision[1],
                ) {
                    continue;
                }
            }
            // Reduced attachments own their policy, overriding any nested
            // shape metadata. Rigid compounds resolve the exact contacted leaf.
            let material_at = |attachment: &Attachment, side: usize| match attachment.endpoint {
                Endpoint::Rigid(_) => manifold.leaf_materials[side].unwrap_or(attachment.material),
                Endpoint::Tree(_, _) => attachment.material,
            };
            let material = material_at(a, 0).combine(material_at(b, 1));
            let n = manifold.normal;
            if !n.is_finite() || n.length_squared() < 0.5 {
                continue;
            }
            let n = n.normalize();
            let axis = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
                Vec3::X
            } else if n.y.abs() <= n.z.abs() {
                Vec3::Y
            } else {
                Vec3::Z
            };
            let t = n.cross(axis).normalize();
            for point in manifold.points() {
                if !point.point_a.is_finite()
                    || !point.point_b.is_finite()
                    || !point.penetration.is_finite()
                {
                    continue;
                }
                let contact = Contact {
                    a: a.endpoint,
                    b: b.endpoint,
                    point_a: point.point_a,
                    point_b: point.point_b,
                    basis: [n, t, n.cross(t)],
                    inverse_mass: [[0.0; 3]; 3],
                    penetration: point.penetration.max(0.0),
                    friction: material.friction,
                    restitution: material.restitution,
                    bounce: 0.0,
                    impulse: [0.0; 3],
                    position_impulse: 0.0,
                };
                contacts.push(contact);
            }
        }
    }
    retain_coupled_islands(&mut contacts, &trees, &rigids);
    contacts.retain_mut(|contact| {
        for col in 0..3 {
            let response = pair_response(contact, contact.basis[col], &mut trees, &rigids, true);
            let delta = response_relative_velocity(contact, &response, &trees, &rigids);
            for row in 0..3 {
                contact.inverse_mass[row][col] = contact.basis[row].dot(delta);
            }
        }
        for row in 0..3 {
            for col in 0..row {
                let value = 0.5 * (contact.inverse_mass[row][col] + contact.inverse_mass[col][row]);
                contact.inverse_mass[row][col] = value;
                contact.inverse_mass[col][row] = value;
            }
        }
        let closing = relative_velocity(contact, &trees, &rigids, false).dot(contact.basis[0]);
        if closing < -config.restitution_threshold.max(0.0) {
            contact.bounce = -closing * contact.restitution;
        }
        contact.inverse_mass.iter().flatten().all(|x| x.is_finite())
            && contact.inverse_mass[0][0] > 0.0
    });
    // A contact stores only its 3x3 Delassus block. Generalized responses are
    // evaluated on demand, never cached densely per witness.
    for _ in 0..config.articulation_contact_iterations.max(1) {
        solve_limit_rows_velocity(&mut limits, &mut trees, world);
        for contact in &mut contacts {
            let v = relative_velocity(contact, &trees, &rigids, false);
            let k = contact.inverse_mass;
            let next_normal = (contact.impulse[0]
                + (contact.bounce - v.dot(contact.basis[0])) / k[0][0])
                .max(0.0);
            let dn = next_normal - contact.impulse[0];
            let tangent = tangent_response(
                k,
                [
                    -v.dot(contact.basis[1]) - k[1][0] * dn,
                    -v.dot(contact.basis[2]) - k[2][0] * dn,
                ],
            );
            let mut next_tangent = [
                contact.impulse[1] + tangent[0],
                contact.impulse[2] + tangent[1],
            ];
            let length = next_tangent[0].hypot(next_tangent[1]);
            let limit = contact.friction * next_normal;
            if length > limit && length > 0.0 {
                for value in &mut next_tangent {
                    *value *= limit / length;
                }
            }
            let delta = contact.basis[0] * dn
                + contact.basis[1] * (next_tangent[0] - contact.impulse[1])
                + contact.basis[2] * (next_tangent[1] - contact.impulse[2]);
            contact.impulse = [next_normal, next_tangent[0], next_tangent[1]];
            apply(contact, delta, &mut trees, &mut rigids, false);
            // The contact block was evaluated in the active-limit feasible
            // space. Applying its raw articulated response followed by the
            // matching accumulated limit reaction is the same block response,
            // while still allowing a later PGS decrement to release that
            // unilateral reaction.
            solve_limit_rows_velocity(&mut limits, &mut trees, world);
        }
    }
    let alpha = config.contact_compliance.max(0.0) / (h * h);
    for _ in 0..config.articulation_contact_iterations.max(1) {
        solve_limit_rows_position(&mut limits, &mut trees, world);
        for contact in &mut contacts {
            let separated = relative_velocity(contact, &trees, &rigids, true).dot(contact.basis[0]);
            let delta = (contact.penetration - separated - alpha * contact.position_impulse)
                / (contact.inverse_mass[0][0] + alpha);
            let next = (contact.position_impulse + delta).max(0.0);
            let impulse = contact.basis[0] * (next - contact.position_impulse);
            contact.position_impulse = next;
            apply(contact, impulse, &mut trees, &mut rigids, true);
            solve_limit_rows_position(&mut limits, &mut trees, world);
        }
    }
    {
        let view = world.bodies.solver_view_mut();
        for rigid in rigids.iter().filter(|rigid| rigid.touched && rigid.dynamic) {
            let slot = rigid.slot;
            view.positions[slot] = rigid.pose.translation + rigid.displacement.linear;
            view.orientations[slot] = (Quat::from_scaled_axis(rigid.displacement.angular)
                * rigid.pose.rotation)
                .normalize();
            view.linear_velocities[slot] = rigid.velocity.linear;
            view.angular_velocities[slot] = rigid.velocity.angular;
            view.sleeping[slot] = false;
            view.sleep_timers[slot] = 0.0;
        }
    }
    for tree in trees.iter().filter(|tree| tree.touched) {
        world
            .articulations
            .get_mut(tree.handle)
            .expect("owner lives through substep")
            .commit_contact_state(&tree.velocity, &tree.displacement);
    }
}
