//! Lower Prism particle snapshots into device-independent Render Fabric data.

use prism_particle_app::{ParticleEmitterSnapshot, ParticleSnapshotEntry};
use prism_render_architecture::{
    motion::MotionSource,
    render_fabric::{
        builtin_resource_templates, DomainObjectHandle, DomainPayload, FrameRenderItem,
        PayloadRole, PayloadValue, RenderDomain, RenderObjectId, ResourceKey, ResourcePayload,
        ResourceTemplate, ValuePayload, PARTICLE_ALIVE_RESOURCE, PARTICLE_STREAM_RESOURCE,
    },
};
use thiserror::Error;

/// Stable app-world entity identity as copied into a render snapshot.
pub type ParticleInstanceKey = u64;

/// Immutable per-instance data extracted for one render frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleInstanceSnapshot {
    /// App-world entity identity, including its generation in the packed value.
    pub instance: ParticleInstanceKey,
    /// Runtime asset slot and generation, packed by the app adapter.
    pub system_asset: u64,
    /// Fixed simulation tick observed by this snapshot.
    pub fixed_tick: u64,
    /// Elapsed simulation time in seconds.
    pub elapsed_seconds: f64,
    /// User-controlled playback rate.
    pub playback_rate: f32,
    /// Stable per-instance random seed.
    pub seed: u64,
    /// Translation in world space.
    pub translation: [f32; 3],
}

/// A monotonically numbered extracted batch. Replacing a batch is atomic from
/// the render schedule's point of view; no partially extracted frame is queued.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleExtractBatch {
    /// Main-world frame sequence associated with this batch.
    pub frame: u64,
    /// Snapshots in deterministic entity iteration order.
    pub instances: Vec<ParticleInstanceSnapshot>,
}

/// Inline payload emitted for one extracted particle-system instance.
pub const PARTICLE_INSTANCE_PAYLOAD_ROLE: PayloadRole = PayloadRole(0x100);
/// Exact live-particle count for one runtime emitter, split into 16-bit lanes.
pub const PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE: PayloadRole = PayloadRole(0x101);
/// Axis-aligned bounds of the live particle positions for one runtime emitter.
pub const PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE: PayloadRole = PayloadRole(0x102);
/// Semantic GPU particle stream consumed by the sprite geometry path.
pub const PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE: PayloadRole = PayloadRole(0x103);
/// Semantic alive/visibility stream consumed by the sprite geometry path.
pub const PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE: PayloadRole = PayloadRole(0x104);

/// Particle items produced from one immutable extracted batch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleFabricFrame {
    frame: u64,
    items: Vec<FrameRenderItem>,
}

impl ParticleFabricFrame {
    /// Produces an empty frame with an explicit source frame number.
    #[must_use]
    pub const fn empty(frame: u64) -> Self {
        Self {
            frame,
            items: Vec::new(),
        }
    }

    /// Low-level producer entry point used by the render-world prepare phase.
    pub fn from_batch(batch: &ParticleExtractBatch) -> Result<Self, ParticleFabricError> {
        let mut items = Vec::with_capacity(batch.instances.len());
        for instance in &batch.instances {
            if !instance.translation.iter().all(|value| value.is_finite())
                || !instance.playback_rate.is_finite()
                || !instance.elapsed_seconds.is_finite()
            {
                return Err(ParticleFabricError::NonFiniteInstance {
                    instance: instance.instance,
                });
            }

            // This adapter publishes no temporal GPU resource. Keep the
            // non-temporal stream at its canonical key; system_asset remains
            // authored snapshot metadata, not an implicit history identity.
            let object = RenderObjectId::Particle(DomainObjectHandle::new(
                instance.instance as u32,
                (instance.instance >> 32) as u32,
            ));
            items.push(FrameRenderItem {
                object_id: object,
                domain: RenderDomain::Particle,
                history_key: 0,
                motion_source: MotionSource::Particle,
                payload: DomainPayload::Value(ValuePayload {
                    value: PayloadValue::Vector([
                        instance.translation[0],
                        instance.translation[1],
                        instance.translation[2],
                        instance.playback_rate,
                    ]),
                    role: PARTICLE_INSTANCE_PAYLOAD_ROLE,
                }),
            });
        }

        items.sort_unstable_by_key(particle_item_key);
        for pair in items.windows(2) {
            if particle_item_key(&pair[0]) == particle_item_key(&pair[1]) {
                return Err(ParticleFabricError::DuplicateInstance {
                    instance: pair[0].object_id,
                    history_key: pair[0].history_key,
                });
            }
        }

        Ok(Self {
            frame: batch.frame,
            items,
        })
    }

    /// Lowers the real packed runtime snapshot without manufacturing authored
    /// asset, simulation-clock or playback metadata that the snapshot does not
    /// contain. Every emitter gets an exact count payload; non-empty emitters
    /// also get bounds derived from their actual live particle positions.
    pub fn from_runtime_snapshot(
        frame: u64,
        emitters: &[ParticleEmitterSnapshot],
        particles: &[ParticleSnapshotEntry],
    ) -> Result<Self, ParticleFabricError> {
        let mut items = Vec::with_capacity(emitters.len().saturating_mul(4));
        let mut expected_first = 0_u32;

        for emitter in emitters {
            if emitter.first_particle != expected_first {
                return Err(ParticleFabricError::InvalidEmitterRange {
                    instance: emitter.emitter_entity.to_bits(),
                    first_particle: emitter.first_particle,
                    particle_count: emitter.particle_count,
                    total_particles: particles.len(),
                });
            }
            let Some(end) = emitter.first_particle.checked_add(emitter.particle_count) else {
                return Err(ParticleFabricError::InvalidEmitterRange {
                    instance: emitter.emitter_entity.to_bits(),
                    first_particle: emitter.first_particle,
                    particle_count: emitter.particle_count,
                    total_particles: particles.len(),
                });
            };
            if end as usize > particles.len() {
                return Err(ParticleFabricError::InvalidEmitterRange {
                    instance: emitter.emitter_entity.to_bits(),
                    first_particle: emitter.first_particle,
                    particle_count: emitter.particle_count,
                    total_particles: particles.len(),
                });
            }

            let instance = emitter.emitter_entity.to_bits();
            let object = particle_object(instance);
            let count = split_u32(emitter.particle_count);
            items.push(particle_value_item(
                object,
                PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE,
                PayloadValue::Vector([count[0], count[1], 0.0, 0.0]),
            ));
            items.push(particle_resource_item(
                object,
                PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                particle_resource_payload(
                    PARTICLE_STREAM_RESOURCE,
                    PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                )
                .expect("built-in particle stream template is declared"),
            ));
            items.push(particle_resource_item(
                object,
                PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                particle_resource_payload(
                    PARTICLE_ALIVE_RESOURCE,
                    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                )
                .expect("built-in particle alive template is declared"),
            ));

            let range = &particles[emitter.first_particle as usize..end as usize];
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for entry in range {
                if entry.emitter_entity != emitter.emitter_entity {
                    return Err(ParticleFabricError::EmitterRangeIdentityMismatch {
                        instance,
                        particle_instance: entry.emitter_entity.to_bits(),
                        slot: entry.slot,
                    });
                }
                let position = [
                    entry.particle.position.x,
                    entry.particle.position.y,
                    entry.particle.position.z,
                ];
                if !position.into_iter().all(f32::is_finite) {
                    return Err(ParticleFabricError::NonFiniteRuntimeParticle {
                        instance,
                        slot: entry.slot,
                    });
                }
                for axis in 0..3 {
                    min[axis] = min[axis].min(position[axis]);
                    max[axis] = max[axis].max(position[axis]);
                }
            }
            if !range.is_empty() {
                items.push(particle_value_item(
                    object,
                    PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
                    PayloadValue::Bounds { min, max },
                ));
            }
            expected_first = end;
        }

        if expected_first as usize != particles.len() {
            return Err(ParticleFabricError::UnclaimedRuntimeParticles {
                first_unclaimed: expected_first,
                total_particles: particles.len(),
            });
        }

        items.sort_unstable_by_key(particle_item_key);
        reject_duplicate_items(&items)?;
        Ok(Self { frame, items })
    }

    /// Frame sequence copied from the extracted particle batch.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Deterministically ordered particle Fabric items.
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }

    /// Returns whether this frame contains no particle payload items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Number of particle payload items in this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

fn particle_object(instance: ParticleInstanceKey) -> RenderObjectId {
    RenderObjectId::Particle(DomainObjectHandle::new(
        instance as u32,
        (instance >> 32) as u32,
    ))
}

fn particle_value_item(
    object_id: RenderObjectId,
    role: PayloadRole,
    value: PayloadValue,
) -> FrameRenderItem {
    FrameRenderItem {
        object_id,
        domain: RenderDomain::Particle,
        history_key: 0,
        motion_source: MotionSource::Particle,
        payload: DomainPayload::Value(ValuePayload { value, role }),
    }
}

fn particle_resource_item(
    object_id: RenderObjectId,
    role: PayloadRole,
    payload: ResourcePayload,
) -> FrameRenderItem {
    FrameRenderItem {
        object_id,
        domain: RenderDomain::Particle,
        history_key: 0,
        motion_source: MotionSource::Particle,
        payload: DomainPayload::Resource(ResourcePayload { role, ..payload }),
    }
}

/// Returns the canonical Fabric template for a particle runtime resource.
///
/// The particle bridge has no authority to declare or allocate GPU resources;
/// it may only reference the built-in semantic templates already owned by the
/// Render Fabric catalog.
#[must_use]
pub fn particle_resource_template(resource: ResourceKey) -> Option<ResourceTemplate> {
    matches!(resource, PARTICLE_STREAM_RESOURCE | PARTICLE_ALIVE_RESOURCE).then(|| {
        builtin_resource_templates()
            .iter()
            .copied()
            .find(|template| template.key == resource)
            .expect("built-in particle resource template is declared")
    })
}

/// Builds the canonical semantic resource descriptor for a particle payload
/// role. No physical GPU handle is created or encoded here.
#[must_use]
pub fn particle_resource_payload(
    resource: ResourceKey,
    role: PayloadRole,
) -> Option<ResourcePayload> {
    let template = particle_resource_template(resource)?;
    Some(ResourcePayload {
        resource: template.key,
        lifetime: template.lifetime,
        size: template
            .layout
            .size()
            .expect("built-in particle resource template has a valid layout"),
        alignment: template.alignment,
        layout: template.layout,
        role,
    })
}

fn split_u32(value: u32) -> [f32; 2] {
    [(value & 0xffff) as f32, (value >> 16) as f32]
}

fn reject_duplicate_items(items: &[FrameRenderItem]) -> Result<(), ParticleFabricError> {
    for pair in items.windows(2) {
        if particle_item_key(&pair[0]) == particle_item_key(&pair[1]) {
            return Err(ParticleFabricError::DuplicateInstance {
                instance: pair[0].object_id,
                history_key: pair[0].history_key,
            });
        }
    }
    Ok(())
}

fn particle_item_key(item: &FrameRenderItem) -> (RenderObjectId, u32, PayloadRole) {
    (item.object_id, item.history_key, item.payload_role())
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
/// Failure while lowering one immutable particle snapshot into Fabric items.
pub enum ParticleFabricError {
    /// A transform, elapsed time, or playback rate was not finite.
    #[error("particle instance {instance:#x} contains a non-finite transform or playback rate")]
    NonFiniteInstance {
        /// Offending particle instance identity.
        instance: u64,
    },
    /// Two snapshots claimed the same particle object and history stream.
    #[error(
        "particle instance {instance:?} appears more than once for history stream {history_key}"
    )]
    DuplicateInstance {
        /// Conflicting domain-local particle identity.
        instance: RenderObjectId,
        /// Conflicting temporal stream key.
        history_key: u32,
    },
    /// An emitter range was not contiguous or extended beyond the packed list.
    #[error(
        "particle emitter {instance:#x} has invalid packed range {first_particle}..+{particle_count} for {total_particles} particles"
    )]
    InvalidEmitterRange {
        /// Stable entity identity for the invalid emitter.
        instance: u64,
        /// First packed particle index claimed by the emitter.
        first_particle: u32,
        /// Number of packed particles claimed by the emitter.
        particle_count: u32,
        /// Total number of particles in the snapshot.
        total_particles: usize,
    },
    /// A particle inside an emitter range names a different source entity.
    #[error(
        "particle emitter {instance:#x} range contains slot {slot} from emitter {particle_instance:#x}"
    )]
    EmitterRangeIdentityMismatch {
        /// Stable entity identity for the range owner.
        instance: u64,
        /// Stable entity identity stored on the mismatched particle.
        particle_instance: u64,
        /// Stable emitter-pool slot stored on the particle.
        slot: u32,
    },
    /// A live runtime particle position was not finite.
    #[error("particle emitter {instance:#x} slot {slot} has a non-finite position")]
    NonFiniteRuntimeParticle {
        /// Stable entity identity for the emitter.
        instance: u64,
        /// Stable emitter-pool slot stored on the particle.
        slot: u32,
    },
    /// Packed particles remained after all emitter ranges were consumed.
    #[error(
        "particle snapshot leaves entries from {first_unclaimed} unclaimed out of {total_particles}"
    )]
    UnclaimedRuntimeParticles {
        /// First packed particle index not covered by an emitter range.
        first_unclaimed: u32,
        /// Total number of particles in the snapshot.
        total_particles: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::prelude::Entity;
    use prism_render_architecture::particle::runtime::ParticleRecord;
    use prism_render_architecture::particle::Vec3;

    fn snapshot(instance: u64, asset: u64, x: f32) -> ParticleInstanceSnapshot {
        ParticleInstanceSnapshot {
            instance,
            system_asset: asset,
            fixed_tick: 7,
            elapsed_seconds: 0.25,
            playback_rate: 1.0,
            seed: 11,
            translation: [x, 2.0, 3.0],
        }
    }

    #[test]
    fn particle_batch_lowers_to_typed_domain_items_with_stable_order() {
        let batch = ParticleExtractBatch {
            frame: 9,
            instances: vec![snapshot((3_u64 << 32) | 9, 12, 1.0), snapshot(2, 12, 0.0)],
        };
        let frame = ParticleFabricFrame::from_batch(&batch).unwrap();

        assert_eq!(frame.frame(), 9);
        assert_eq!(frame.items().len(), 2);
        assert_eq!(frame.items()[0].domain, RenderDomain::Particle);
        assert_eq!(frame.items()[0].motion_source, MotionSource::Particle);
        assert!(matches!(
            frame.items()[0].object_id,
            RenderObjectId::Particle(_)
        ));
        assert!(matches!(
            frame.items()[0].payload,
            DomainPayload::Value(ValuePayload {
                value: PayloadValue::Vector([_, _, _, 1.0]),
                role: PARTICLE_INSTANCE_PAYLOAD_ROLE,
            })
        ));
    }

    #[test]
    fn particle_generation_and_duplicate_identity_are_checked() {
        let first = snapshot((4_u64 << 32) | 5, 12, 0.0);
        let mut changed_generation = first;
        changed_generation.instance = (5_u64 << 32) | 5;
        let frame = ParticleFabricFrame::from_batch(&ParticleExtractBatch {
            frame: 1,
            instances: vec![first, changed_generation],
        })
        .unwrap();
        assert_ne!(frame.items()[0].object_id, frame.items()[1].object_id);

        let duplicate = ParticleFabricFrame::from_batch(&ParticleExtractBatch {
            frame: 1,
            instances: vec![first, first],
        });
        assert!(matches!(
            duplicate,
            Err(ParticleFabricError::DuplicateInstance { .. })
        ));
    }

    #[test]
    fn particle_input_rejects_non_finite_values() {
        let mut invalid = snapshot(1, 12, 0.0);
        invalid.translation[1] = f32::NAN;
        assert!(matches!(
            ParticleFabricFrame::from_batch(&ParticleExtractBatch {
                frame: 1,
                instances: vec![invalid],
            }),
            Err(ParticleFabricError::NonFiniteInstance { .. })
        ));
    }

    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64).unwrap()
    }

    fn runtime_particle(emitter: Entity, slot: u32, position: Vec3) -> ParticleSnapshotEntry {
        ParticleSnapshotEntry {
            emitter_entity: emitter,
            slot,
            particle: ParticleRecord {
                position,
                velocity: Vec3::ZERO,
                age: 0.0,
                lifetime: 1.0,
                particle_id: slot,
                birth_sequence: 1,
                size: 1.0,
            },
        }
    }

    #[test]
    fn runtime_snapshot_preserves_entity_generation_frame_and_stable_order() {
        let later = entity(9, 3);
        let earlier = entity(2, 1);
        let emitters = [
            ParticleEmitterSnapshot {
                emitter_entity: later,
                first_particle: 0,
                particle_count: 2,
            },
            ParticleEmitterSnapshot {
                emitter_entity: earlier,
                first_particle: 2,
                particle_count: 1,
            },
        ];
        let particles = [
            runtime_particle(later, 4, Vec3::new(2.0, -1.0, 5.0)),
            runtime_particle(later, 7, Vec3::new(-3.0, 4.0, 1.0)),
            runtime_particle(earlier, 1, Vec3::new(8.0, 9.0, 10.0)),
        ];

        let frame = ParticleFabricFrame::from_runtime_snapshot(41, &emitters, &particles).unwrap();

        assert_eq!(frame.frame(), 41);
        assert_eq!(frame.len(), 8);
        assert_eq!(
            frame.items()[0].object_id,
            RenderObjectId::Particle(DomainObjectHandle::new(2, 1))
        );
        assert!(frame.items().iter().any(|item| {
            item.object_id == RenderObjectId::Particle(DomainObjectHandle::new(9, 3))
                && matches!(
                    item.payload,
                    DomainPayload::Value(ValuePayload {
                        value: PayloadValue::Bounds {
                            min: [-3.0, -1.0, 1.0],
                            max: [2.0, 4.0, 5.0]
                        },
                        role: PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
                    })
                )
        }));
        for (resource, role) in [
            (
                PARTICLE_STREAM_RESOURCE,
                PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
            ),
            (
                PARTICLE_ALIVE_RESOURCE,
                PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
            ),
        ] {
            let payload = frame
                .items()
                .iter()
                .find_map(|item| match item.payload {
                    DomainPayload::Resource(payload)
                        if item.object_id
                            == RenderObjectId::Particle(DomainObjectHandle::new(9, 3))
                            && payload.resource == resource =>
                    {
                        Some(payload)
                    }
                    _ => None,
                })
                .expect("runtime emitter exposes its required semantic resource");
            assert_eq!(payload.role, role);
            assert_eq!(payload, particle_resource_payload(resource, role).unwrap());
            assert_eq!(
                particle_resource_template(resource).unwrap().layout,
                payload.layout
            );
        }
    }

    #[test]
    fn runtime_snapshot_rejects_invalid_ranges_identity_mismatch_and_non_finite_positions() {
        let emitter = entity(1, 2);
        let other = entity(2, 2);
        let invalid_range = [ParticleEmitterSnapshot {
            emitter_entity: emitter,
            first_particle: 1,
            particle_count: 1,
        }];
        let particles = [runtime_particle(emitter, 0, Vec3::ZERO)];
        assert!(matches!(
            ParticleFabricFrame::from_runtime_snapshot(1, &invalid_range, &particles),
            Err(ParticleFabricError::InvalidEmitterRange { .. })
        ));

        let range = [ParticleEmitterSnapshot {
            emitter_entity: emitter,
            first_particle: 0,
            particle_count: 1,
        }];
        let mismatch = [runtime_particle(other, 0, Vec3::ZERO)];
        assert!(matches!(
            ParticleFabricFrame::from_runtime_snapshot(1, &range, &mismatch),
            Err(ParticleFabricError::EmitterRangeIdentityMismatch { .. })
        ));

        let non_finite = [runtime_particle(emitter, 0, Vec3::new(f32::NAN, 0.0, 0.0))];
        assert!(matches!(
            ParticleFabricFrame::from_runtime_snapshot(1, &range, &non_finite),
            Err(ParticleFabricError::NonFiniteRuntimeParticle { .. })
        ));

        assert!(matches!(
            ParticleFabricFrame::from_runtime_snapshot(1, &[], &particles),
            Err(ParticleFabricError::UnclaimedRuntimeParticles { .. })
        ));
    }

    #[test]
    fn empty_runtime_snapshot_carries_the_frame_and_clears_items() {
        let frame = ParticleFabricFrame::from_runtime_snapshot(77, &[], &[]).unwrap();
        assert_eq!(frame.frame(), 77);
        assert!(frame.is_empty());
    }
}
