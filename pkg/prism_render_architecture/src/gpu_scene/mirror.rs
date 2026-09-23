use alloc::collections::BTreeMap;
use core::fmt;

use super::{InstanceRecord, SceneHandle, SceneOperation, SceneTransaction};

/// Fields in the GPU scene that changed atomically for one slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SceneFieldMask(pub u16);

impl SceneFieldMask {
    pub const INSTANCE: Self = Self(1 << 0);
    pub const CURRENT_TRANSFORM: Self = Self(1 << 1);
    pub const PREVIOUS_TRANSFORM: Self = Self(1 << 2);
    pub const BOUNDS: Self = Self(1 << 3);
    pub const GENERATION: Self = Self(1 << 4);
    pub const ALL: Self = Self(
        Self::INSTANCE.0
            | Self::CURRENT_TRANSFORM.0
            | Self::PREVIOUS_TRANSFORM.0
            | Self::BOUNDS.0
            | Self::GENERATION.0,
    );

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirtySceneSlot {
    pub handle: SceneHandle,
    pub fields: SceneFieldMask,
}

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
    pub dirty_slots: Vec<DirtySceneSlot>,
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
            report.dirty_slots.clear();
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

    pub fn live_handles(&self) -> Vec<SceneHandle> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                slot.record.map(|_| SceneHandle {
                    index: index as u32,
                    generation: slot.generation,
                })
            })
            .collect()
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
                mark_dirty(report, handle, SceneFieldMask::ALL);
            }
            SceneOperation::Destroy { .. } => {
                self.validate_slot(handle)?;
                self.slots[handle.index as usize].record = None;
                self.live_count -= 1;
                report.destroyed += 1;
                mark_dirty(
                    report,
                    handle,
                    SceneFieldMask(
                        SceneFieldMask::INSTANCE.0
                            | SceneFieldMask::BOUNDS.0
                            | SceneFieldMask::GENERATION.0,
                    ),
                );
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
                mark_dirty(
                    report,
                    handle,
                    SceneFieldMask(
                        SceneFieldMask::CURRENT_TRANSFORM.0 | SceneFieldMask::PREVIOUS_TRANSFORM.0,
                    ),
                );
            }
            SceneOperation::SetBounds { bounds, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .bounds = bounds;
                report.updated += 1;
                mark_dirty(report, handle, SceneFieldMask::BOUNDS);
            }
            SceneOperation::SetGeometry { geometry, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .geometry = geometry;
                report.updated += 1;
                mark_dirty(report, handle, SceneFieldMask::INSTANCE);
            }
            SceneOperation::SetMaterial { material, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .material = material;
                report.updated += 1;
                mark_dirty(report, handle, SceneFieldMask::INSTANCE);
            }
            SceneOperation::SetFlags { mask, value, .. } => {
                let record = self
                    .validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot");
                record.flags = (record.flags & !mask) | (value & mask);
                report.updated += 1;
                mark_dirty(report, handle, SceneFieldMask::INSTANCE);
            }
            SceneOperation::SetRenderLayers { render_layers, .. } => {
                self.validate_slot_mut(handle)?
                    .record
                    .as_mut()
                    .expect("validated live scene slot")
                    .render_layers = render_layers;
                report.updated += 1;
                mark_dirty(report, handle, SceneFieldMask::INSTANCE);
            }
        }
        Ok(())
    }

    pub fn record_at(&self, index: u32) -> Option<(u32, Option<&InstanceRecord>)> {
        let slot = self.slots.get(index as usize)?;
        Some((slot.generation, slot.record.as_ref()))
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

fn mark_dirty(report: &mut SceneApplyReport, handle: SceneHandle, fields: SceneFieldMask) {
    if let Some(slot) = report
        .dirty_slots
        .iter_mut()
        .find(|slot| slot.handle == handle)
    {
        slot.fields.insert(fields);
    } else {
        report.dirty_slots.push(DirtySceneSlot { handle, fields });
    }
}
