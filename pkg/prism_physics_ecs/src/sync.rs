use prism_app::EngineClocks;
use prism_physics_core::{PhysicsEvent, Solver};

use crate::*;

fn issue(diagnostics: &mut PhysicsDiagnostics, entity: Option<Entity>, issue: PhysicsIssue) {
    diagnostics.issues.push(PhysicsDiagnostic { entity, issue });
}

fn physics_vec(v: prism_math::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn native_vec(v: Vec3) -> prism_math::Vec3 {
    prism_math::Vec3::new(v.x, v.y, v.z)
}

fn physics_quat(q: prism_math::Quat) -> Quat {
    Quat::from_array(q.to_array())
}

fn valid_body(body: RigidBody) -> bool {
    body.mass.inv_mass.is_finite()
        && body.mass.inv_mass >= 0.0
        && body.mass.inv_inertia.is_finite()
        && body.mass.inv_inertia.min_element() >= 0.0
        && body.linear_damping.is_finite()
        && body.linear_damping >= 0.0
        && body.angular_damping.is_finite()
        && body.angular_damping >= 0.0
}

fn valid_velocity(velocity: Velocity) -> bool {
    velocity.linear.is_finite() && velocity.angular.is_finite()
}

fn validated_pose(world: &World, entity: Entity) -> Result<Transform, PhysicsIssue> {
    let pose = *world
        .get::<Transform>(entity)
        .ok_or(PhysicsIssue::MissingTransform)?;
    if !world.relation_targets::<ChildOf>(entity).is_empty() {
        return Err(PhysicsIssue::ParentedBody);
    }
    if pose.scale != prism_math::Vec3::ONE {
        return Err(PhysicsIssue::NonUnitScale);
    }
    if !physics_vec(pose.translation).is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-4
    {
        return Err(PhysicsIssue::InvalidAuthoring);
    }
    Ok(pose)
}

fn resolve_collider(
    physics: &mut PhysicsWorld,
    collider: Option<Collider>,
) -> Result<Option<ColliderHandle>, PhysicsIssue> {
    let Some(collider) = collider else {
        return Ok(None);
    };
    let material = collider.material;
    if !material.friction.is_finite()
        || material.friction < 0.0
        || !material.restitution.is_finite()
        || !(0.0..=1.0).contains(&material.restitution)
    {
        return Err(PhysicsIssue::InvalidAuthoring);
    }
    let shape = match collider.shape {
        CollisionShape::Registered(handle) => {
            return physics
                .shapes
                .get(handle)
                .map(|_| Some(handle))
                .ok_or(PhysicsIssue::MissingShape);
        }
        CollisionShape::Analytic(shape) => shape,
    };
    let valid = match shape {
        ColliderShape::Sphere { radius } => radius.is_finite() && radius > 0.0,
        ColliderShape::Cuboid { half_extents } => {
            half_extents.is_finite() && half_extents.min_element() > 0.0
        }
        ColliderShape::Capsule {
            radius,
            half_height,
        } => radius.is_finite() && radius > 0.0 && half_height.is_finite() && half_height >= 0.0,
        ColliderShape::Plane { normal, offset } => {
            normal.is_finite()
                && offset.is_finite()
                && (normal.length_squared() - 1.0).abs() <= 1e-4
        }
        _ => return Err(PhysicsIssue::UnsupportedShape),
    };
    if !valid {
        return Err(PhysicsIssue::InvalidAuthoring);
    }
    // The core registry is append-only. Deduplicate analytic authoring across
    // edits/rebindings rather than leaking one shape allocation per fixed tick.
    for index in 0..physics.shapes.len() {
        let handle = ColliderHandle(index as u32);
        if physics.shapes.get(handle) == Some(&shape) {
            return Ok(Some(handle));
        }
    }
    Ok(Some(physics.shapes.insert(shape)))
}

fn retire_body(
    physics: &mut PhysicsWorld,
    bindings: &mut PhysicsBindings,
    diagnostics: &mut PhysicsDiagnostics,
    entity: Entity,
) -> bool {
    let Some(binding) = bindings.bodies.get_mut(&entity) else {
        return true;
    };
    let handle = binding.handle;
    // Capture graph edges before core removal erases them. These include
    // disabled joints created through PhysicsWorld::spawn_joint.
    let mut joints: Vec<_> = physics
        .assemblies
        .connections()
        .filter(|c| c.a == handle || c.b == handle)
        .filter_map(|c| c.joint)
        .collect();
    if physics.bodies.contains(handle) && !physics.remove_body(handle) {
        binding.active = false;
        issue(diagnostics, Some(entity), PhysicsIssue::RemovalRefused);
        return false;
    }
    // Also cover enabled joints inserted directly into joint storage. Disabled
    // raw-storage joints have no public complete iterator and remain caller-owned.
    joints.extend(
        physics
            .joints
            .active_entries()
            .filter(|(_, j)| j.anchor_a.body == handle || j.anchor_b.body == handle)
            .map(|(h, _)| h),
    );
    joints.sort_by_key(|h| (h.index(), h.generation()));
    joints.dedup();
    for joint in joints {
        physics.breakable_joints.unregister(joint);
        physics.joints.remove(joint);
    }
    bindings.bodies.remove(&entity);
    let key = (handle.index(), handle.generation());
    bindings.entities.remove(&key);
    bindings.retired.insert(key, entity);
    true
}

fn retire_soft(physics: &mut PhysicsWorld, bindings: &mut PhysicsBindings, entity: Entity) {
    if let Some(binding) = bindings.soft.remove(&entity) {
        physics.remove_soft_body(binding.handle);
        bindings
            .soft_entities
            .remove(&(binding.handle.index(), binding.handle.generation()));
    }
}

pub(crate) fn reconcile(world: &mut World) {
    let mut physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
    let mut bindings = world.remove_resource::<PhysicsBindings>().unwrap();
    let mut diagnostics = world.remove_resource::<PhysicsDiagnostics>().unwrap();
    let signals = std::mem::take(world.resource_mut::<RemovalSignals>());
    diagnostics.tick = diagnostics.tick.wrapping_add(1);
    diagnostics.simulated_dt = 0.0;
    diagnostics.issues.clear();

    let old_entities: Vec<_> = bindings.bodies.keys().copied().collect();
    // Always retry a previously refused removal, even if authoring was repaired.
    for entity in old_entities {
        if signals.rigid.contains(&entity)
            || world.get::<RigidBody>(entity).is_none()
            || !bindings.bodies[&entity].active
        {
            retire_body(&mut physics, &mut bindings, &mut diagnostics, entity);
        }
    }

    let query = world.query::<(Entity, &RigidBody)>();
    let mut authored: Vec<_> = query.iter(world).map(|(e, b)| (e, *b)).collect();
    authored.sort_by_key(|(e, _)| *e);
    for (entity, body) in authored {
        // Do not overwrite the retained entry after core refused removal.
        if bindings.bodies.get(&entity).is_some_and(|b| !b.active) {
            continue;
        }
        let inputs = (|| {
            let pose = validated_pose(world, entity)?;
            let velocity = world.get::<Velocity>(entity).copied();
            if !valid_body(body) || velocity.is_some_and(|v| !valid_velocity(v)) {
                return Err(PhysicsIssue::InvalidAuthoring);
            }
            let collider = world.get::<Collider>(entity).copied();
            let shape = resolve_collider(&mut physics, collider)?;
            Ok((pose, velocity, collider, shape))
        })();
        let (pose, velocity, collider, shape) = match inputs {
            Ok(inputs) => inputs,
            Err(error) => {
                issue(&mut diagnostics, Some(entity), error);
                retire_body(&mut physics, &mut bindings, &mut diagnostics, entity);
                continue;
            }
        };

        let mut spawn_velocity = velocity.unwrap_or_default();
        let mut spawn_pose = pose;
        if let Some(old) = bindings.bodies.get(&entity).cloned() {
            let stale = !physics.bodies.contains(old.handle);
            if stale || body != old.authored {
                if stale {
                    issue(&mut diagnostics, Some(entity), PhysicsIssue::StaleBody);
                } else {
                    // Preserve live resource-side edits when rebuilding immutable
                    // mass/kind/damping fields, unless ECS authored an override.
                    if pose == old.pose {
                        spawn_pose.translation =
                            native_vec(physics.bodies.position(old.handle).unwrap());
                        let q = physics.bodies.orientation(old.handle).unwrap();
                        spawn_pose.rotation = prism_math::Quat::from_xyzw(q.x, q.y, q.z, q.w);
                    }
                    if velocity == old.velocity && !signals.velocity.contains(&entity) {
                        spawn_velocity = Velocity {
                            linear: physics.bodies.linear_velocity(old.handle).unwrap(),
                            angular: physics.bodies.angular_velocity(old.handle).unwrap(),
                        };
                    }
                }
                if !retire_body(&mut physics, &mut bindings, &mut diagnostics, entity) {
                    continue;
                }
            }
        }

        if !bindings.bodies.contains_key(&entity) {
            let state = BodyDesc {
                kind: body.kind,
                position: physics_vec(spawn_pose.translation),
                orientation: physics_quat(spawn_pose.rotation),
                linear_velocity: spawn_velocity.linear,
                angular_velocity: spawn_velocity.angular,
                mass_properties: body.mass,
                linear_damping: body.linear_damping,
                angular_damping: body.angular_damping,
                ccd: body.ccd,
                collider: shape,
                material: collider.map_or(PhysicsMaterial::default(), |c| c.material),
                is_sensor: collider.is_some_and(|c| c.sensor),
            };
            let handle = physics.spawn(state);
            bindings
                .entities
                .insert((handle.index(), handle.generation()), entity);
            bindings.bodies.insert(
                entity,
                BodyBinding {
                    handle,
                    authored: body,
                    pose,
                    velocity,
                    active: true,
                },
            );
        } else {
            let binding = bindings.bodies.get_mut(&entity).unwrap();
            let handle = binding.handle;
            if pose != binding.pose {
                let position = physics_vec(pose.translation);
                let orientation = physics_quat(pose.rotation);
                physics.bodies.set_position(handle, position);
                physics.bodies.set_prev_position(handle, position);
                physics.bodies.set_orientation(handle, orientation);
                physics.bodies.set_prev_orientation(handle, orientation);
                physics.bodies.wake(handle);
            }
            if velocity != binding.velocity || signals.velocity.contains(&entity) {
                let velocity = velocity.unwrap_or_default();
                physics.bodies.set_linear_velocity(handle, velocity.linear);
                physics
                    .bodies
                    .set_angular_velocity(handle, velocity.angular);
                physics.bodies.wake(handle);
            }
            let material = collider.map_or(PhysicsMaterial::default(), |c| c.material);
            let sensor = collider.is_some_and(|c| c.sensor);
            if physics.bodies.collider(handle) != shape
                || physics.bodies.material(handle) != Some(material)
                || physics.bodies.is_sensor(handle) != Some(sensor)
            {
                physics.bodies.set_collider(handle, shape);
                physics.bodies.set_material(handle, material);
                physics.bodies.set_sensor(handle, sensor);
                physics.bodies.wake(handle);
            }
            binding.pose = pose;
            binding.velocity = velocity;
        }
    }

    let old_soft: Vec<_> = bindings.soft.keys().copied().collect();
    for entity in old_soft {
        if world.get::<SoftBodyBinding>(entity).is_none() || signals.soft.contains(&entity) {
            retire_soft(&mut physics, &mut bindings, entity);
        }
    }
    let query = world.query::<(Entity, &SoftBodyBinding)>();
    let mut soft: Vec<_> = query.iter(world).map(|(e, b)| (e, b.0.clone())).collect();
    soft.sort_by_key(|(e, _)| *e);
    for (entity, authored) in soft {
        let particles = &authored.particles;
        if !authored.config.gravity.is_finite()
            || !authored.config.damping.is_finite()
            || authored.config.damping < 0.0
            || particles
                .positions()
                .iter()
                .chain(particles.prev_positions())
                .chain(particles.velocities())
                .any(|p| !p.is_finite())
            || particles
                .inverse_masses()
                .iter()
                .any(|m| !m.is_finite() || *m < 0.0)
        {
            issue(
                &mut diagnostics,
                Some(entity),
                PhysicsIssue::InvalidAuthoring,
            );
            retire_soft(&mut physics, &mut bindings, entity);
            continue;
        }
        let replace = bindings
            .soft
            .get(&entity)
            .is_none_or(|b| b.authored != authored || !physics.soft_bodies.contains(b.handle));
        if replace {
            if bindings
                .soft
                .get(&entity)
                .is_some_and(|b| !physics.soft_bodies.contains(b.handle))
            {
                issue(&mut diagnostics, Some(entity), PhysicsIssue::StaleSoftBody);
            }
            retire_soft(&mut physics, &mut bindings, entity);
            let handle = physics.spawn_soft_body(authored.clone());
            bindings
                .soft_entities
                .insert((handle.index(), handle.generation()), entity);
            bindings
                .soft
                .insert(entity, SoftBinding { handle, authored });
        }
    }
    world.insert_resource(physics);
    world.insert_resource(bindings);
    world.insert_resource(diagnostics);
}

fn contact_key(a: BodyHandle, b: BodyHandle) -> (u64, u64) {
    let key = |h: BodyHandle| (u64::from(h.index()) << 32) | u64::from(h.generation());
    let (a, b) = (key(a), key(b));
    (a.min(b), a.max(b))
}

fn contacts(
    physics: &mut PhysicsWorld,
    bindings: &mut PhysicsBindings,
    tick: u64,
) -> Vec<PhysicsContactEvent> {
    let mut raw = physics.drain_contact_events();
    for event in &raw {
        match *event {
            PhysicsEvent::TriggerEntered { sensor, other } => {
                bindings
                    .trigger_roles
                    .insert(contact_key(sensor, other), (sensor, other));
            }
            PhysicsEvent::TriggerExited { sensor, other } => {
                bindings.trigger_roles.remove(&contact_key(sensor, other));
            }
            _ => {}
        }
    }
    // Core tracks overlap by unordered pair, not sensor role. Supply role
    // transitions so downstream consumers can balance enter/exit per endpoint.
    for roles in bindings.trigger_roles.values_mut() {
        let (a, b) = *roles;
        let a_sensor = physics.bodies.is_sensor(a) == Some(true);
        let b_sensor = physics.bodies.is_sensor(b) == Some(true);
        let next = if a_sensor && b_sensor {
            if (a.index(), a.generation()) < (b.index(), b.generation()) {
                (a, b)
            } else {
                (b, a)
            }
        } else if a_sensor {
            (a, b)
        } else {
            (b, a)
        };
        if next != *roles {
            raw.push(PhysicsEvent::TriggerExited {
                sensor: a,
                other: b,
            });
            raw.push(PhysicsEvent::TriggerEntered {
                sensor: next.0,
                other: next.1,
            });
            *roles = next;
        }
    }
    let endpoint = |body: BodyHandle| ContactEndpoint {
        body,
        entity: bindings.entity(body).or_else(|| {
            bindings
                .retired
                .get(&(body.index(), body.generation()))
                .copied()
        }),
    };
    let mut events: Vec<_> = raw
        .into_iter()
        .map(|event| {
            let (kind, a, b) = match event {
                PhysicsEvent::CollisionStarted(pair) => (
                    ContactEventKind::CollisionStarted,
                    pair.first(),
                    pair.second(),
                ),
                PhysicsEvent::CollisionEnded(pair) => (
                    ContactEventKind::CollisionEnded,
                    pair.first(),
                    pair.second(),
                ),
                PhysicsEvent::TriggerEntered { sensor, other } => {
                    (ContactEventKind::TriggerEntered, sensor, other)
                }
                PhysicsEvent::TriggerExited { sensor, other } => {
                    (ContactEventKind::TriggerExited, sensor, other)
                }
            };
            PhysicsContactEvent {
                tick,
                kind,
                first: endpoint(a),
                second: endpoint(b),
            }
        })
        .collect();
    // Preserve transition families (ends before starts) and stable handle order.
    events.sort_by_key(|e| {
        let rank = match e.kind {
            ContactEventKind::CollisionEnded => 0,
            ContactEventKind::TriggerExited => 1,
            ContactEventKind::CollisionStarted => 2,
            ContactEventKind::TriggerEntered => 3,
        };
        (
            rank,
            e.first.body.index(),
            e.first.body.generation(),
            e.second.body.index(),
            e.second.body.generation(),
        )
    });
    events
}

pub(crate) fn step(world: &mut World) {
    let dt = world.resource::<EngineClocks>().fixed().delta_secs();
    let mut physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
    let mut solver = world.remove_resource::<PhysicsSolver>().unwrap();
    let mut diagnostics = world.remove_resource::<PhysicsDiagnostics>().unwrap();
    let mut bindings = world.remove_resource::<PhysicsBindings>().unwrap();
    let n = physics.config.default_substeps;
    let config = solver.0.config;
    let valid = dt.is_finite()
        && dt > 0.0
        && (1..=8).contains(&n)
        && physics.config.gravity.is_finite()
        && config.position_iterations > 0
        && config.contact_compliance.is_finite()
        && config.contact_compliance >= 0.0
        && config.restitution_threshold.is_finite()
        && config.restitution_threshold >= 0.0;
    let mut forces = Vec::new();
    for (&entity, binding) in &bindings.bodies {
        if !binding.active {
            continue;
        }
        if let Some(force) = world.get::<ExternalForce>(entity).copied() {
            if force.force.is_finite() && force.torque.is_finite() {
                forces.push((binding.handle, force));
            } else {
                issue(
                    &mut diagnostics,
                    Some(entity),
                    PhysicsIssue::InvalidAuthoring,
                );
            }
        }
    }
    let mut events = contacts(&mut physics, &mut bindings, diagnostics.tick);
    if valid {
        let h = dt / n as f32;
        for _ in 0..n {
            for &(handle, force) in &forces {
                if let Some(position) = physics.bodies.position(handle) {
                    if force.force != Vec3::ZERO {
                        physics.apply_force(handle, force.force, position);
                    }
                    if force.torque != Vec3::ZERO {
                        physics.apply_torque(handle, force.torque);
                    }
                }
            }
            solver.0.step(&mut physics, h, 1);
            events.extend(contacts(&mut physics, &mut bindings, diagnostics.tick));
        }
        diagnostics.simulated_dt = dt;
    } else {
        issue(&mut diagnostics, None, PhysicsIssue::InvalidConfiguration);
    }
    world.insert_resource(physics);
    world.insert_resource(solver);
    world.insert_resource(diagnostics);
    world
        .resource_mut::<Events<PhysicsContactEvent>>()
        .send_batch(events);
    bindings.retired.clear();
    world.insert_resource(bindings);
}

pub(crate) fn writeback(world: &mut World) {
    // Snapshot output before ECS mutation: all public resources remain present
    // while user Transform observers run.
    let physics = world.resource::<PhysicsWorldResource>();
    let bindings = world.resource::<PhysicsBindings>();
    let output: Vec<_> = bindings
        .bodies
        .iter()
        .filter_map(|(&entity, binding)| {
            if !binding.active || validated_pose(world, entity).is_err() {
                return None;
            }
            let handle = binding.handle;
            let q = physics.bodies.orientation(handle)?;
            Some((
                entity,
                Transform {
                    translation: native_vec(physics.bodies.position(handle)?),
                    rotation: prism_math::Quat::from_xyzw(q.x, q.y, q.z, q.w),
                    scale: prism_math::Vec3::ONE,
                },
                Velocity {
                    linear: physics.bodies.linear_velocity(handle)?,
                    angular: physics.bodies.angular_velocity(handle)?,
                },
            ))
        })
        .collect();
    for (entity, pose, velocity) in output {
        if let Some(mut current) = world.get_mut::<Transform>(entity) {
            *current = pose;
        }
        let has_velocity = world.has::<Velocity>(entity);
        if let Some(mut current) = world.get_mut::<Velocity>(entity) {
            *current = velocity;
        }
        if let Some(binding) = world
            .resource_mut::<PhysicsBindings>()
            .bodies
            .get_mut(&entity)
        {
            binding.pose = pose;
            binding.velocity = has_velocity.then_some(velocity);
        }
    }
    world.flush_changed_observers();
    if prism_transform::ecs::propagate_transforms(world).is_err() {
        issue(
            world.resource_mut::<PhysicsDiagnostics>(),
            None,
            PhysicsIssue::TransformPropagation,
        );
    }
    world.flush_changed_observers();
}
