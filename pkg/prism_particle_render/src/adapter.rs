//! Lower Prism particle snapshots into device-independent Render Fabric data.

use prism_render_architecture::{
    motion::MotionSource,
    render_fabric::{
        DomainObjectHandle, DomainPayload, FrameRenderItem, PayloadRole, PayloadValue,
        RenderDomain, RenderObjectId, ValuePayload,
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

/// Particle items produced from one immutable extracted batch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleFabricFrame {
    frame: u64,
    items: Vec<FrameRenderItem>,
}

impl ParticleFabricFrame {
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

    /// Frame sequence copied from the extracted particle batch.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Deterministically ordered particle Fabric items.
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
