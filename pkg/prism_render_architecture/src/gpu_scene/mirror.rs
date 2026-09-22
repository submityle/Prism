use core::fmt;
use std::collections::BTreeMap;

use super::{InstanceRecord, SceneHandle, SceneOperation, SceneTransaction};

#[derive(Clone, Copy, Debug, Default)]
struct CpuSceneSlot {
    generation: u32,
    record: Option<InstanceRecord>,
    last_transform_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneApplyError {
    InvalidHandle(SceneHandle),
    SlotAlreadyOccupied(SceneHandle),
    MissingSlot(SceneHandle),
    StaleHandle {
        handle: SceneHandle,
        current_generation: u32,
    },
    NonMonotonicGeneration {
        handle: SceneHandle,
        retired_generation: u32,
    },
}

impl fmt::Display for SceneApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle(handle) => write!(formatter, "invalid scene handle {handle:?}"),
            Self::SlotAlreadyOccupied(handle) => {
                write!(formatter, "scene slot is already occupied by {handle:?}")
            }
            Self::MissingSlot(handle) => write!(formatter, "scene slot is missing for {handle:?}"),
            Self::StaleHandle {
                handle,
                current_generation,
            } => write!(
                formatter,
                "scene handle {handle:?} is stale; current generation is {current_generation}"
            ),
            Self::NonMonotonicGeneration {
                handle,
                retired_generation,
            } => write!(
                formatter,
                "scene handle {handle:?} does not advance retired generation {retired_generation}"
            ),
        }
    }
}

impl std::error::Error for SceneApplyError {}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneApplyReport {
    pub created: u32,
    pub destroyed: u32,
    pub updated: u32,
    pub errors: Vec<SceneApplyError>,
    pub scene_epoch: u64,
}

/// Authoritative CPU mirror used for validation, upload planning, capture, and
/// device-loss reconstruction.
#[derive(Default)]
pub struct CpuRenderScene {
    slots: Vec<CpuSceneSlot>,
    live_count: u32,
    scene_epoch: u64,
}

impl CpuRenderScene {
    pub fn apply(&mut self, transaction: &SceneTransaction) -> SceneApplyReport {
        let original_len = self.slots.len();
        let original_live_count = self.live_count;
        let mut original_slots = BTreeMap::new();
        let mut report = SceneApplyReport::default();
        for operation in transaction.operations.iter().copied() {
            let index = operation.handle().index as usize;
            original_slots.entry(index).or_insert_with(|| {
                self.slots
                    .get(index)
                    .copied()
                    .unwrap_or_else(CpuSceneSlot::default)
            });
            if let Err(error) =
                self.apply_operation(operation, transaction.frame_epoch, &mut report)
            {
                report.errors.push(error);
                break;
            }
        }
        if !report.errors.is_empty() {
            for (index, slot) in original_slots {
                if index < original_len {
                    self.slots[index] = slot;
                }
            }
            self.slots.truncate(original_len);
            self.live_count = original_live_count;
            report.created = 0;
            report.destroyed = 0;
            report.updated = 0;
            report.scene_epoch = self.scene_epoch;
            return report;
        }
        if report.created != 0 || report.destroyed != 0 || report.updated != 0 {
            self.scene_epoch += 1;
        }
        report.scene_epoch = self.scene_epoch;
        report
    }

    pub fn get(&self, handle: SceneHandle) -> Option<&InstanceRecord> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.record.as_ref())
            .flatten()
    }

    pub fn live_count(&self) -> u32 {
        self.live_count
    }

    pub fn scene_epoch(&self) -> u64 {
        self.scene_epoch
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn apply_operation(
        &mut self,
        operation: SceneOperation,
        frame_epoch: u64,
        report: &mut SceneApplyReport,
    ) -> Result<(), SceneApplyError> {
        let handle = operation.handle();
        if !handle.is_valid() || handle.index == 0 {
            return Err(SceneApplyError::InvalidHandle(handle));
        }
        if self.slots.len() <= handle.index as usize {
            self.slots
                .resize(handle.index as usize + 1, CpuSceneSlot::default());
        }

        match operation {
            SceneOperation::Create { record, .. } => {
                let slot = &self.slots[handle.index as usize];
                if slot.record.is_some() {
                    return if slot.generation == handle.generation {
                        Err(SceneApplyError::SlotAlreadyOccupied(handle))
                    } else {
                        Err(SceneApplyError::StaleHandle {
                            handle,
                            current_generation: slot.generation,
                        })
                    };
                }
                if handle.generation <= slot.generation {
                    return Err(SceneApplyError::NonMonotonicGeneration {
                        handle,
                        retired_generation: slot.generation,
                    });
                }
                self.slots[handle.index as usize] = CpuSceneSlot {
                    generation: handle.generation,
                    record: Some(record),
                    last_transform_frame: frame_epoch,
                };
                self.live_count += 1;
                report.created += 1;
            }
            SceneOperation::Destroy { .. } => {
                self.validate_slot(handle)?;
                self.slots[handle.index as usize].record = None;
                self.live_count -= 1;
                report.destroyed += 1;
            }
            SceneOperation::SetTransform { current, .. } => {
                let slot = self.validate_slot_mut(handle)?;
                if slot.last_transform_frame != frame_epoch {
                    let record = slot.record.as_mut().expect("validated live scene slot");
                    record.previous_transform = record.current_transform;
                    slot.last_transform_frame = frame_epoch;
                }
                slot.record
                    .as_mut()
                    .expect("validated live scene slot")
                    .current_transform = current;
                report.updated += 1;
            }
            SceneOperation::SetBounds { bounds, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .bounds = bounds;
                report.updated += 1;
            }
            SceneOperation::SetGeometry { geometry, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .geometry = geometry;
                report.updated += 1;
            }
            SceneOperation::SetMaterial { material, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .material = material;
                report.updated += 1;
            }
            SceneOperation::SetFlags { mask, value, .. } => {
                let record = self
                    .validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot");
                record.flags = (record.flags & !mask) | (value & mask);
                report.updated += 1;
            }
            SceneOperation::SetRenderLayers { render_layers, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .render_layers = render_layers;
                report.updated += 1;
            }
        }
        Ok(())
    }

    fn validate_slot(&self, handle: SceneHandle) -> Result<&CpuSceneSlot, SceneApplyError> {
        let slot = &self.slots[handle.index as usize];
        if slot.generation != handle.generation {
            return Err(SceneApplyError::StaleHandle {
                handle,
                current_generation: slot.generation,
            });
        }
        if slot.record.is_none() {
            return Err(SceneApplyError::MissingSlot(handle));
        }
        Ok(slot)
    }

    fn validate_slot_mut(
        &mut self,
        handle: SceneHandle,
    ) -> Result<&mut CpuSceneSlot, SceneApplyError> {
        let slot = &mut self.slots[handle.index as usize];
        if slot.generation != handle.generation {
            return Err(SceneApplyError::StaleHandle {
                handle,
                current_generation: slot.generation,
            });
        }
        if slot.record.is_none() {
            return Err(SceneApplyError::MissingSlot(handle));
        }
        Ok(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::{SceneBounds, SceneTransactionBuilder, SceneTransform};

    fn handle(generation: u32) -> SceneHandle {
        SceneHandle {
            index: 1,
            generation,
        }
    }

    fn translated(x: f32) -> SceneTransform {
        let mut value = SceneTransform::IDENTITY;
        value.rows[0][3] = x;
        value
    }

    #[test]
    fn previous_transform_is_captured_only_on_first_change_per_frame() {
        let mut scene = CpuRenderScene::default();
        let mut create = SceneTransactionBuilder::new(1, 1);
        create.push(SceneOperation::Create {
            handle: handle(1),
            record: InstanceRecord::default(),
        });
        assert!(scene.apply(&create.finish()).errors.is_empty());

        let mut same_frame = SceneTransactionBuilder::new(2, 2);
        same_frame
            .push(SceneOperation::SetTransform {
                handle: handle(1),
                current: translated(1.0),
            })
            .push(SceneOperation::SetTransform {
                handle: handle(1),
                current: translated(2.0),
            });
        scene.apply(&same_frame.finish());
        let record = scene.get(handle(1)).unwrap();
        assert_eq!(record.previous_transform, SceneTransform::IDENTITY);
        assert_eq!(record.current_transform, translated(2.0));

        let mut next_frame = SceneTransactionBuilder::new(3, 3);
        next_frame.push(SceneOperation::SetTransform {
            handle: handle(1),
            current: translated(3.0),
        });
        scene.apply(&next_frame.finish());
        let record = scene.get(handle(1)).unwrap();
        assert_eq!(record.previous_transform, translated(2.0));
        assert_eq!(record.current_transform, translated(3.0));
    }

    #[test]
    fn stale_updates_do_not_touch_reused_slot() {
        let mut scene = CpuRenderScene::default();
        let create = SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: handle(2),
                record: InstanceRecord::default(),
            }],
        };
        scene.apply(&create);
        let update = SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: vec![SceneOperation::SetTransform {
                handle: handle(1),
                current: translated(5.0),
            }],
        };
        let report = scene.apply(&update);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            scene.get(handle(2)).unwrap().current_transform,
            SceneTransform::IDENTITY
        );
    }

    #[test]
    fn transaction_is_atomic_when_any_operation_fails() {
        let mut scene = CpuRenderScene::default();
        let create = SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: handle(1),
                record: InstanceRecord::default(),
            }],
        };
        scene.apply(&create);

        let transaction = SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: vec![
                SceneOperation::SetTransform {
                    handle: handle(1),
                    current: translated(9.0),
                },
                SceneOperation::SetBounds {
                    handle: handle(2),
                    bounds: SceneBounds::default(),
                },
            ],
        };
        let report = scene.apply(&transaction);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            scene.get(handle(1)).unwrap().current_transform,
            SceneTransform::IDENTITY
        );
        assert_eq!(scene.scene_epoch(), 1);
    }

    #[test]
    fn destroyed_generation_is_retained_as_a_tombstone() {
        let mut scene = CpuRenderScene::default();
        scene.apply(&SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: handle(3),
                record: InstanceRecord::default(),
            }],
        });
        scene.apply(&SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: vec![SceneOperation::Destroy { handle: handle(3) }],
        });
        let stale_create = scene.apply(&SceneTransaction {
            frame_epoch: 3,
            sequence: 3,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: handle(3),
                record: InstanceRecord::default(),
            }],
        });
        assert_eq!(stale_create.errors.len(), 1);
        assert!(matches!(
            stale_create.errors[0],
            SceneApplyError::NonMonotonicGeneration { .. }
        ));
        let fresh_create = scene.apply(&SceneTransaction {
            frame_epoch: 4,
            sequence: 4,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: handle(4),
                record: InstanceRecord::default(),
            }],
        });
        assert!(fresh_create.errors.is_empty());
    }
}
