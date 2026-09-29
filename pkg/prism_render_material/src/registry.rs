use crate::{
    GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters, MaterialCapacityError,
    MaterialHandleAllocator, MaterialHandleError, MaterialRecord,
};
use alloc::{collections::BTreeSet, vec::Vec};
use prism_render_architecture::{abi::GenerationalHandle, gpu_scene::GpuCompletionValue};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaterialSnapshot {
    pub epoch: u64,
    pub active_materials: u32,
    pub buffer_version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialRegistryError {
    CapacityExceeded,
    InvalidHandle(GenerationalHandle),
    StaleRevision { current: u32, incoming: u32 },
    TooManyTextures { count: usize, maximum: usize },
}

#[derive(Clone)]
struct Slot {
    revision: u32,
    record: MaterialRecord,
}

pub struct MaterialRegistry {
    allocator: MaterialHandleAllocator,
    slots: Vec<Option<Slot>>,
    epoch: u64,
    buffer_version: u32,
    dirty: BTreeSet<u32>,
    generations: Vec<u32>,
}

impl MaterialRegistry {
    pub fn new(max_materials: u32) -> Self {
        Self {
            allocator: MaterialHandleAllocator::new(max_materials),
            slots: vec![None],
            epoch: 0,
            buffer_version: 1,
            dirty: BTreeSet::from([0]),
            generations: vec![0],
        }
    }
    pub fn allocate(&mut self) -> Result<GenerationalHandle, MaterialCapacityError> {
        self.allocator.allocate()
    }
    pub fn publish(&mut self, record: MaterialRecord) -> Result<(), MaterialRegistryError> {
        if !self.allocator.validate(record.handle) {
            return Err(MaterialRegistryError::InvalidHandle(record.handle));
        }
        if record.textures.len() > crate::MAX_MATERIAL_TEXTURES {
            return Err(MaterialRegistryError::TooManyTextures {
                count: record.textures.len(),
                maximum: crate::MAX_MATERIAL_TEXTURES,
            });
        }
        let index = record.handle.index as usize;
        self.generations
            .resize(self.generations.len().max(index + 1), 0);
        self.generations[index] = record.handle.generation;
        self.slots
            .resize_with(self.slots.len().max(index + 1), || None);
        if let Some(slot) = &self.slots[index]
            && record.revision <= slot.revision
        {
            return Err(MaterialRegistryError::StaleRevision {
                current: slot.revision,
                incoming: record.revision,
            });
        }
        self.slots[index] = Some(Slot {
            revision: record.revision,
            record,
        });
        self.dirty.insert(index as u32);
        self.epoch += 1;
        Ok(())
    }
    pub fn get(&self, handle: GenerationalHandle) -> Option<&MaterialRecord> {
        self.slots
            .get(handle.index as usize)?
            .as_ref()
            .map(|slot| &slot.record)
            .filter(|record| record.handle == handle)
    }
    pub fn retire(
        &mut self,
        handle: GenerationalHandle,
        completion: GpuCompletionValue,
    ) -> Result<(), MaterialHandleError> {
        self.allocator.retire(handle, completion)?;
        if let Some(slot) = self.slots.get_mut(handle.index as usize) {
            *slot = None;
        }
        self.dirty.insert(handle.index);
        self.epoch += 1;
        Ok(())
    }
    pub fn reclaim_completed(&mut self, completed: GpuCompletionValue) -> u32 {
        self.allocator.reclaim_completed(completed)
    }
    pub fn cancel_allocation(
        &mut self,
        handle: GenerationalHandle,
    ) -> Result<(), MaterialHandleError> {
        self.allocator.cancel(handle)
    }
    pub fn snapshot(&self) -> MaterialSnapshot {
        MaterialSnapshot {
            epoch: self.epoch,
            active_materials: self.slots.iter().flatten().count() as u32,
            buffer_version: self.buffer_version,
        }
    }
    pub fn take_dirty(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.dirty).into_iter().collect()
    }
    pub fn mark_all_dirty(&mut self) {
        self.dirty.extend(0..self.slots.len() as u32);
        self.buffer_version = self.buffer_version.wrapping_add(1).max(1);
    }
    pub fn record_at(&self, index: u32) -> Option<&MaterialRecord> {
        self.slots
            .get(index as usize)?
            .as_ref()
            .map(|slot| &slot.record)
    }
    pub fn record_or_fallback(&self, handle: GenerationalHandle) -> MaterialRecord {
        self.get(handle)
            .cloned()
            .unwrap_or_else(|| crate::fallback_material_record(handle, self.epoch))
    }
    pub fn generation_at(&self, index: u32) -> Option<u32> {
        self.generations.get(index as usize).copied()
    }
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }
    pub fn gpu_tables(
        &self,
    ) -> (
        Vec<GpuMaterialHeader>,
        Vec<GpuSurfaceParameters>,
        Vec<GpuMaterialTexture>,
    ) {
        let mut headers = vec![GpuMaterialHeader::default(); self.slots.len()];
        let mut parameters = vec![GpuSurfaceParameters::default(); self.slots.len()];
        let mut textures = Vec::new();
        headers[0] = crate::fallback_material_header(self.epoch);
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(slot) = slot {
                headers[index] =
                    slot.record
                        .header(index as u32, textures.len() as u32, 0, self.epoch);
                parameters[index] = slot.record.surface;
                textures.extend_from_slice(&slot.record.textures);
            }
        }
        (headers, parameters, textures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuSurfaceParameters, Illumination, MaterialDomain, MaterialFeatureFlags,
        MaterialRenderClass,
    };

    fn record(handle: GenerationalHandle, revision: u32) -> MaterialRecord {
        MaterialRecord {
            handle,
            revision,
            domain: MaterialDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            illumination: Illumination::Lit,
            features: MaterialFeatureFlags::default(),
            closure_mask: 1,
            surface: GpuSurfaceParameters::default(),
            textures: Vec::new(),
            custom_program: None,
        }
    }

    #[test]
    fn publish_rejects_unallocated_handle() {
        let mut registry = MaterialRegistry::new(8);
        let rogue = GenerationalHandle::new(4, 1);
        assert_eq!(
            registry.publish(record(rogue, 1)),
            Err(MaterialRegistryError::InvalidHandle(rogue))
        );
    }

    #[test]
    fn publish_rejects_stale_revision() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 5)).unwrap();
        // A revision less than or equal to the live one is a stale write.
        assert_eq!(
            registry.publish(record(handle, 5)),
            Err(MaterialRegistryError::StaleRevision {
                current: 5,
                incoming: 5,
            })
        );
        // A strictly newer revision supersedes the slot.
        registry.publish(record(handle, 6)).unwrap();
        assert_eq!(registry.get(handle).unwrap().revision, 6);
    }

    #[test]
    fn get_is_filtered_by_generation() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        let stale = handle.with_generation(handle.generation.wrapping_add(1));
        assert!(registry.get(stale).is_none());
        assert!(registry.get(handle).is_some());
    }

    #[test]
    fn retire_clears_the_slot() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        registry.retire(handle, GpuCompletionValue(1)).unwrap();
        assert!(registry.get(handle).is_none());
        assert!(registry.record_at(handle.index).is_none());
    }

    #[test]
    fn record_or_fallback_returns_fallback_for_unknown_handle() {
        let registry = MaterialRegistry::new(8);
        let unknown = GenerationalHandle::new(5, 2);
        let value = registry.record_or_fallback(unknown);
        assert_eq!(value.handle, unknown);
        assert_eq!(value.illumination, Illumination::Lit);
        assert_eq!(value.closure_mask, 1);
    }

    #[test]
    fn generation_and_epoch_track_publishes() {
        let mut registry = MaterialRegistry::new(8);
        let before = registry.snapshot().epoch;
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        assert_eq!(
            registry.generation_at(handle.index),
            Some(handle.generation)
        );
        let after = registry.snapshot();
        assert!(after.epoch > before);
        assert_eq!(after.active_materials, 1);
    }

    #[test]
    fn mark_all_dirty_bumps_buffer_version_and_redirties_rows() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        let version_before = registry.snapshot().buffer_version;
        let _ = registry.take_dirty();
        assert!(registry.take_dirty().is_empty());
        registry.mark_all_dirty();
        assert!(registry.snapshot().buffer_version > version_before);
        assert!(registry.take_dirty().contains(&handle.index));
    }
}
