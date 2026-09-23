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
                        .header(index as u32, textures.len() as u32, self.epoch);
                parameters[index] = slot.record.surface;
                textures.extend_from_slice(&slot.record.textures);
            }
        }
        (headers, parameters, textures)
    }
}
