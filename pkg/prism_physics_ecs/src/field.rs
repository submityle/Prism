//! ECS authoring and deterministic lifecycle bindings for core force fields
//! and aerodynamic functional elements.

use std::collections::{BTreeMap, BTreeSet};

use prism_physics_core::dynamics::field::{Field, FieldId};
use prism_physics_core::{AeroBodyConfig, AeroSurface, BodyHandle, BuoyancyVolume, Thruster};

use crate::{PhysicsBindings, PhysicsWorldResource, RigidBody, RigidBodyAero};
use prism_ecs::prelude::*;

/// Core-defined force-field authoring. Its core `FieldId` is assigned by the
/// binding layer and remains stable while this ECS entity remains registered.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ForceField(pub Field);

/// Additional elements on the target rigid-body entity itself. Composition is
/// additive: `RigidBodyAero` vectors first, then these vectors, then independent
/// components in source Entity order (including generation), separately per
/// family. Vector order is preserved; equal elements are not deduplicated.
/// Only `RigidBodyAero` supplies the model; otherwise the core default is used.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct AeroFunctionalComponents {
    /// Surface elements evaluated by the core aero model.
    pub surfaces: Vec<AeroSurface>,
    /// Propulsors evaluated by the core aero model.
    pub thrusters: Vec<Thruster>,
    /// Fluid displacement volumes evaluated by the core aero model.
    pub buoyancy: Vec<BuoyancyVolume>,
}

/// One independently editable surface targeting an ordinary ECS `RigidBody`.
/// All element coordinates are target body-local; the source entity's Transform
/// and hierarchy are intentionally ignored. Changes commit at fixed Sync.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct AeroSurfaceComponent {
    pub target: Entity,
    pub surface: AeroSurface,
}

/// One independently editable propulsor; coordinates follow
/// [`AeroSurfaceComponent`]. Multiple sources are additive, not overrides.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct AeroThrusterComponent {
    pub target: Entity,
    pub thruster: Thruster,
}

/// One independently editable displaced-fluid volume; coordinates follow
/// [`AeroSurfaceComponent`]. No aggregate component is required on the target.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct AeroBuoyancyComponent {
    pub target: Entity,
    pub volume: BuoyancyVolume,
}

/// Identifies the authoring source of a rejected target replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AeroSource {
    BodyConfig,
    FunctionalComponents,
    Surface,
    Thruster,
    Buoyancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AeroIssue {
    /// Target is despawned or its entity generation no longer exists.
    MissingTarget,
    /// Assembly aliases and other non-RigidBody entities are not targets.
    TargetNotRigidBody,
    /// Ordinary body is inactive, invalid, or has no live core binding.
    TargetUnavailable,
    /// Invalid authoring, never silently sanitized. Index is within the named
    /// source's family vector (zero for an independent element), or None for
    /// model parameters. The parameter path identifies the failing field.
    InvalidParameter {
        parameter: &'static str,
        index: Option<usize>,
    },
    /// A retarget connects this replacement to another rejected target. Both
    /// keep their last valid registrations so no moved source applies twice.
    ReplacementBlocked { rejected_target: Entity },
    /// Core refused registration; no partial element list was committed.
    RegistrationRejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AeroDiagnostic {
    pub entity: Entity,
    pub target: Entity,
    pub source: AeroSource,
    pub issue: AeroIssue,
}

/// Replaced every fixed Sync, ordered by target, source kind and source Entity.
/// A target replacement is atomic: any invalid source keeps the entire last
/// valid configuration on the SAME BodyHandle, even if other sources were
/// removed. Correct all reported sources to commit the pending replacement.
/// A rebuilt body never inherits this fallback. Retarget-connected targets
/// commit or roll back together; unrelated targets remain independent. Only
/// the first invalid parameter of each source is reported per fixed Sync.
/// While any aero authoring targets a body, ECS owns its core aero registration;
/// direct writes to that registration are reconciled at the next fixed Sync.
#[derive(Resource, Default)]
pub struct AeroDiagnostics {
    pub issues: Vec<AeroDiagnostic>,
}

struct AeroBinding {
    body: BodyHandle,
    config: AeroBodyConfig,
    sources: BTreeSet<(AeroSource, Entity)>,
}

struct PendingAero {
    body: Option<BodyHandle>,
    config: Option<AeroBodyConfig>,
    sources: BTreeSet<(AeroSource, Entity)>,
    invalid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FieldBinding {
    id: FieldId,
    authored: Field,
}

/// Deterministic ECS-to-core identity map for force fields.
#[derive(Resource, Default)]
pub struct FieldBindings {
    fields: BTreeMap<Entity, FieldBinding>,
    aero: BTreeMap<Entity, AeroBinding>,
}

impl FieldBindings {
    /// Returns the current core identity for an authored field entity.
    #[must_use]
    pub fn field(&self, entity: Entity) -> Option<FieldId> {
        self.fields.get(&entity).map(|binding| binding.id)
    }
}

pub(crate) fn reconcile(world: &mut World) {
    let mut physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
    let mut bindings = world.remove_resource::<FieldBindings>().unwrap();

    let query = world.query::<(Entity, &ForceField)>();
    let mut desired: Vec<_> = query
        .iter(world)
        .map(|(entity, field)| (entity, field.0))
        .collect();
    desired.sort_by_key(|(entity, _)| *entity);
    let desired_entities: std::collections::BTreeSet<_> =
        desired.iter().map(|(entity, _)| *entity).collect();

    let retired: Vec<_> = bindings
        .fields
        .keys()
        .filter(|entity| !desired_entities.contains(entity))
        .copied()
        .collect();
    for entity in retired {
        if let Some(binding) = bindings.fields.remove(&entity) {
            physics.fields.remove(binding.id);
        }
    }

    for (entity, mut authored) in desired {
        authored.id = FieldId(0);
        let previous = bindings.fields.get(&entity).copied();
        if let Some(binding) = previous {
            if binding.authored == authored && {
                // Core insert/remove preserve FieldId order, but manual
                // insertion may duplicate IDs. Match ANY equal-ID entry,
                // not just whichever one binary_search happens to return.
                let fields = physics.fields.fields();
                let start = fields.partition_point(|field| field.id < binding.id);
                let end = start + fields[start..].partition_point(|field| field.id == binding.id);
                let expected = authored_with_id(authored, binding.id);
                fields[start..end].iter().any(|field| *field == expected)
            } {
                continue;
            }
            physics.fields.remove(binding.id);
        }
        let mut registered = authored;
        registered.id = previous.map_or(FieldId(0), |binding| binding.id);
        let id = physics.fields.insert(registered);
        bindings
            .fields
            .insert(entity, FieldBinding { id, authored });
    }

    reconcile_aero(world, &mut physics, &mut bindings);

    world.insert_resource(physics);
    world.insert_resource(bindings);
}

fn authored_with_id(mut field: Field, id: FieldId) -> Field {
    field.id = id;
    field
}

fn reconcile_aero(
    world: &mut World,
    physics: &mut PhysicsWorldResource,
    bindings: &mut FieldBindings,
) {
    // Snapshot all authoring before touching core state. No source-to-body cache
    // can survive a retarget or an entity/body generation change.
    let mut desired: BTreeMap<Entity, Vec<(AeroSource, Entity, AeroBodyConfig)>> = BTreeMap::new();
    for (entity, base) in world.query::<(Entity, &RigidBodyAero)>().iter(world) {
        desired
            .entry(entity)
            .or_default()
            .push((AeroSource::BodyConfig, entity, base.0.clone()));
    }
    for (entity, elements) in world
        .query::<(Entity, &AeroFunctionalComponents)>()
        .iter(world)
    {
        desired.entry(entity).or_default().push((
            AeroSource::FunctionalComponents,
            entity,
            AeroBodyConfig {
                surfaces: elements.surfaces.clone(),
                thrusters: elements.thrusters.clone(),
                buoyancy: elements.buoyancy.clone(),
                ..AeroBodyConfig::default()
            },
        ));
    }
    for (entity, part) in world.query::<(Entity, &AeroSurfaceComponent)>().iter(world) {
        desired.entry(part.target).or_default().push((
            AeroSource::Surface,
            entity,
            AeroBodyConfig {
                surfaces: vec![part.surface],
                ..AeroBodyConfig::default()
            },
        ));
    }
    for (entity, part) in world
        .query::<(Entity, &AeroThrusterComponent)>()
        .iter(world)
    {
        desired.entry(part.target).or_default().push((
            AeroSource::Thruster,
            entity,
            AeroBodyConfig {
                thrusters: vec![part.thruster],
                ..AeroBodyConfig::default()
            },
        ));
    }
    for (entity, part) in world
        .query::<(Entity, &AeroBuoyancyComponent)>()
        .iter(world)
    {
        desired.entry(part.target).or_default().push((
            AeroSource::Buoyancy,
            entity,
            AeroBodyConfig {
                buoyancy: vec![part.volume],
                ..AeroBodyConfig::default()
            },
        ));
    }
    let mut diagnostics = AeroDiagnostics::default();
    // Include pending removals: a failed retarget must not remove the old
    // registration first, even if its last authoring source just moved away.
    let source_targets: BTreeMap<_, _> = desired
        .iter()
        .flat_map(|(target, sources)| {
            sources
                .iter()
                .map(move |(kind, entity, _)| ((*kind, *entity), *target))
        })
        .collect();
    let mut neighbors: BTreeMap<Entity, BTreeSet<Entity>> = BTreeMap::new();
    for (target, old) in &bindings.aero {
        desired.entry(*target).or_default();
        for source in &old.sources {
            if let Some(new_target) = source_targets.get(source).filter(|new| **new != *target) {
                neighbors.entry(*target).or_default().insert(*new_target);
                neighbors.entry(*new_target).or_default().insert(*target);
            }
        }
    }
    let mut pending = BTreeMap::new();
    for (target, mut sources) in desired {
        sources.sort_by_key(|(kind, entity, _)| (*kind, *entity));
        let body = world
            .resource::<PhysicsBindings>()
            .bodies
            .get(&target)
            .filter(|binding| binding.active && physics.bodies.contains(binding.handle))
            .filter(|_| world.has::<RigidBody>(target))
            .map(|binding| binding.handle);
        let target_issue = if !world.contains(target) {
            Some(AeroIssue::MissingTarget)
        } else if !world.has::<RigidBody>(target) {
            Some(AeroIssue::TargetNotRigidBody)
        } else if body.is_none() {
            Some(AeroIssue::TargetUnavailable)
        } else {
            None
        };
        let source_keys = sources
            .iter()
            .map(|(kind, entity, _)| (*kind, *entity))
            .collect();
        let has_authoring = !sources.is_empty();
        let issue_start = diagnostics.issues.len();
        let mut combined = AeroBodyConfig::default();
        for (source, entity, config) in sources {
            if let Some(issue) = target_issue {
                diagnostics.issues.push(AeroDiagnostic {
                    entity,
                    target,
                    source,
                    issue,
                });
            } else if let Err(issue) = validate_config(&config, source == AeroSource::BodyConfig) {
                diagnostics.issues.push(AeroDiagnostic {
                    entity,
                    target,
                    source,
                    issue,
                });
            }
            if source == AeroSource::BodyConfig {
                combined.model = config.model;
            }
            combined.surfaces.extend(config.surfaces);
            combined.thrusters.extend(config.thrusters);
            combined.buoyancy.extend(config.buoyancy);
        }
        pending.insert(
            target,
            PendingAero {
                body,
                config: has_authoring.then_some(combined),
                sources: source_keys,
                invalid: diagnostics.issues.len() != issue_start,
            },
        );
    }
    let mut visited = BTreeSet::new();
    let mut blocked = BTreeMap::new();
    for &target in pending.keys() {
        if !visited.insert(target) {
            continue;
        }
        let mut group = vec![target];
        let mut cursor = 0;
        while cursor < group.len() {
            if let Some(adjacent) = neighbors.get(&group[cursor]) {
                for &other in adjacent {
                    if visited.insert(other) {
                        group.push(other);
                    }
                }
            }
            cursor += 1;
        }
        if let Some(rejected) = group
            .iter()
            .copied()
            .filter(|target| pending[target].invalid)
            .min()
        {
            for target in group {
                blocked.insert(target, rejected);
            }
        }
    }
    for (target, candidate) in pending {
        if bindings
            .aero
            .get(&target)
            .is_some_and(|old| Some(old.body) != candidate.body)
        {
            let old = bindings.aero.remove(&target).unwrap();
            physics.unregister_aero_body(old.body);
        }
        if let Some(&rejected_target) = blocked.get(&target) {
            if !candidate.invalid {
                let sources = bindings
                    .aero
                    .get(&target)
                    .filter(|_| candidate.sources.is_empty())
                    .map_or(&candidate.sources, |old| &old.sources);
                for &(source, entity) in sources {
                    diagnostics.issues.push(AeroDiagnostic {
                        entity,
                        target,
                        source,
                        issue: AeroIssue::ReplacementBlocked { rejected_target },
                    });
                }
            }
            // Normal rigid-body Sync may already have overwritten the combined
            // config with the base, including invalid input. Restore only the
            // full previously accepted config on exactly the same live body.
            if let Some(old) = bindings.aero.get(&target) {
                if physics
                    .register_aero_body(old.body, old.config.clone())
                    .is_err()
                {
                    diagnostics.issues.push(AeroDiagnostic {
                        entity: target,
                        target,
                        source: AeroSource::BodyConfig,
                        issue: AeroIssue::RegistrationRejected,
                    });
                }
            } else if let Some(body) = candidate.body {
                physics.unregister_aero_body(body);
            }
            continue;
        }
        let (Some(body), Some(combined)) = (candidate.body, candidate.config) else {
            if let Some(old) = bindings.aero.remove(&target) {
                physics.unregister_aero_body(old.body);
            }
            continue;
        };
        if physics.aero_body(body) != Some(&combined)
            && physics.register_aero_body(body, combined.clone()).is_err()
        {
            diagnostics.issues.push(AeroDiagnostic {
                entity: target,
                target,
                source: AeroSource::BodyConfig,
                issue: AeroIssue::RegistrationRejected,
            });
            continue;
        }
        bindings.aero.insert(
            target,
            AeroBinding {
                body,
                config: combined,
                sources: candidate.sources,
            },
        );
    }

    // Base Sync can unregister/reinsert registrations. Canonicalize only the
    // occupied ECS-owned slots, preserving all foreign registrations and slots.
    let owners: BTreeMap<_, _> = bindings
        .aero
        .iter()
        .map(|(entity, binding)| ((binding.body.index(), binding.body.generation()), *entity))
        .collect();
    let mut slots = Vec::new();
    let mut entries = Vec::new();
    for (index, entry) in physics.aero_bodies.iter().enumerate() {
        if let Some(entity) = owners.get(&(entry.body.index(), entry.body.generation())) {
            let original = slots.len();
            slots.push(index);
            entries.push((*entity, original));
        }
    }
    if !entries.windows(2).all(|pair| pair[0].0 <= pair[1].0) {
        entries.sort_by_key(|(entity, _)| *entity);
        let mut destinations = vec![0; entries.len()];
        for (destination, (_, original)) in entries.into_iter().enumerate() {
            destinations[original] = destination;
        }
        // Each label is the destination of the entry currently at that managed
        // position. Swap whole registrations through every permutation cycle;
        // configs are moved, never cloned, and no foreign slot is touched.
        for current in 0..destinations.len() {
            while destinations[current] != current {
                let destination = destinations[current];
                physics.aero_bodies.swap(slots[current], slots[destination]);
                destinations.swap(current, destination);
            }
        }
    }
    diagnostics
        .issues
        .sort_by_key(|issue| (issue.target, issue.source, issue.entity));
    world.insert_resource(diagnostics);
}

fn nonnegative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}
fn direction(value: crate::Vec3) -> bool {
    value.is_finite() && value.length_squared().is_finite() && value.abs().max_element() > 1.0e-6
}
fn nonnegative_vec(value: crate::Vec3) -> bool {
    value.is_finite() && value.min_element() >= 0.0
}

/// Finite authoring only. Geometric directions must be nonzero, dimensions and
/// densities nonnegative, quadrature 1..=8 and control effectiveness 0..=1.
/// Signed thrust, spin, lift slopes, control deflection and Magnus are allowed.
/// Zero dimensions/coefficients intentionally disable individual effects.
fn validate_config(config: &AeroBodyConfig, model: bool) -> Result<(), AeroIssue> {
    macro_rules! require {
        ($valid:expr, $parameter:expr, $index:expr) => {
            if !$valid {
                return Err(AeroIssue::InvalidParameter {
                    parameter: $parameter,
                    index: $index,
                });
            }
        };
    }
    if model {
        let m = &config.model;
        require!(nonnegative(m.air_density), "model.air_density", None);
        require!(nonnegative(m.water_density), "model.water_density", None);
        require!(m.gravity.is_finite(), "model.gravity", None);
        if let Some(p) = m.water_plane {
            require!(p.point.is_finite(), "model.water_plane.point", None);
            require!(direction(p.normal), "model.water_plane.normal", None);
            require!(nonnegative(p.density), "model.water_plane.density", None);
        }
        if let Some(a) = m.added_mass {
            require!(nonnegative(a.volume), "model.added_mass.volume", None);
            require!(
                nonnegative_vec(a.linear_coefficients),
                "model.added_mass.linear_coefficients",
                None
            );
            require!(
                nonnegative_vec(a.angular_coefficients),
                "model.added_mass.angular_coefficients",
                None
            );
            require!(
                nonnegative(a.reference_radius),
                "model.added_mass.reference_radius",
                None
            );
        }
        if let Some(m) = m.magnus {
            require!(m.coefficient.is_finite(), "model.magnus.coefficient", None);
            require!(
                nonnegative(m.reference_area),
                "model.magnus.reference_area",
                None
            );
            require!(
                nonnegative(m.effective_radius),
                "model.magnus.effective_radius",
                None
            );
        }
    }
    for (index, s) in config.surfaces.iter().enumerate() {
        let i = Some(index);
        require!(s.local_position.is_finite(), "surface.local_position", i);
        require!(direction(s.local_normal), "surface.local_normal", i);
        require!(direction(s.local_chord), "surface.local_chord", i);
        require!(direction(s.local_span), "surface.local_span", i);
        require!(
            direction(s.local_normal.normalize().cross(s.local_chord.normalize())),
            "surface.normal_cross_chord",
            i
        );
        require!(nonnegative(s.area), "surface.area", i);
        require!(nonnegative(s.chord_length), "surface.chord_length", i);
        require!(nonnegative(s.drag_normal), "surface.drag_normal", i);
        require!(nonnegative(s.drag_chord), "surface.drag_chord", i);
        require!(nonnegative(s.drag_span), "surface.drag_span", i);
        require!(nonnegative(s.immersion_depth), "surface.immersion_depth", i);
        let l = s.lift_curve;
        require!(l.lift_slope.is_finite(), "surface.lift_curve.lift_slope", i);
        require!(
            nonnegative(l.stall_angle) && l.stall_angle <= std::f32::consts::FRAC_PI_2,
            "surface.lift_curve.stall_angle",
            i
        );
        require!(
            nonnegative(l.stall_blend),
            "surface.lift_curve.stall_blend",
            i
        );
        require!(
            l.post_stall_slope.is_finite(),
            "surface.lift_curve.post_stall_slope",
            i
        );
        require!(
            nonnegative(l.maximum_lift),
            "surface.lift_curve.maximum_lift",
            i
        );
        if let Some(c) = s.control {
            require!(c.deflection.is_finite(), "surface.control.deflection", i);
            require!(
                nonnegative(c.maximum_deflection),
                "surface.control.maximum_deflection",
                i
            );
            require!(
                c.effectiveness.is_finite() && (0.0..=1.0).contains(&c.effectiveness),
                "surface.control.effectiveness",
                i
            );
        }
    }
    for (index, t) in config.thrusters.iter().enumerate() {
        let i = Some(index);
        require!(t.local_position.is_finite(), "thruster.local_position", i);
        require!(direction(t.local_direction), "thruster.local_direction", i);
        require!(t.thrust.is_finite(), "thruster.thrust", i);
        require!(
            nonnegative(t.reaction_torque_coefficient),
            "thruster.reaction_torque_coefficient",
            i
        );
        require!(t.spin_direction.is_finite(), "thruster.spin_direction", i);
        if let Some(j) = t.slipstream {
            require!(j.origin.is_finite(), "thruster.slipstream.origin", i);
            require!(direction(j.direction), "thruster.slipstream.direction", i);
            require!(nonnegative(j.radius), "thruster.slipstream.radius", i);
            require!(nonnegative(j.length), "thruster.slipstream.length", i);
            require!(j.speed.is_finite(), "thruster.slipstream.speed", i);
            require!(
                nonnegative(j.edge_exponent),
                "thruster.slipstream.edge_exponent",
                i
            );
        }
    }
    for (index, b) in config.buoyancy.iter().enumerate() {
        let i = Some(index);
        require!(b.local_center.is_finite(), "buoyancy.local_center", i);
        require!(
            nonnegative_vec(b.local_half_extents),
            "buoyancy.local_half_extents",
            i
        );
        require!(nonnegative(b.volume), "buoyancy.volume", i);
        require!(
            (1..=8).contains(&b.samples_per_axis),
            "buoyancy.samples_per_axis",
            i
        );
    }
    Ok(())
}
