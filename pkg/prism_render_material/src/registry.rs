use crate::{
    DynamicChannels, GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters,
    MaterialCapacityError, MaterialHandleAllocator, MaterialHandleError, MaterialRecord,
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
    /// Per-slot runtime dynamic channels (§17.1). Kept beside the records rather
    /// than inside them: they change every frame, while a record is the *baked*
    /// contract, and a state change must be re-uploadable without touching the
    /// published revision or the compiled permutation.
    dynamic: Vec<DynamicChannels>,
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
            dynamic: vec![DynamicChannels::NONE],
        }
    }
    pub fn allocate(&mut self) -> Result<GenerationalHandle, MaterialCapacityError> {
        self.allocator.allocate()
    }
    pub fn publish(&mut self, record: MaterialRecord) -> Result<(), MaterialRegistryError> {
        if !self.allocator.validate(record.handle) {
            return Err(MaterialRegistryError::InvalidHandle(record.handle));
        }
        let index = record.handle.index as usize;
        self.generations
            .resize(self.generations.len().max(index + 1), 0);
        self.generations[index] = record.handle.generation;
        // A (re)published record starts from the inert dynamic state: the new
        // bake is a different material, so the previous wetness/snow/damage/
        // dissolve must not leak onto it.
        self.dynamic
            .resize(self.dynamic.len().max(index + 1), DynamicChannels::NONE);
        self.dynamic[index] = DynamicChannels::NONE;
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
    /// Set a material's runtime dynamic channels (§17.1).
    ///
    /// This is the *only* mutation that does not go through [`Self::publish`]: a
    /// wetness/snow/damage/dissolve change is not a new bake, so it must not bump
    /// the revision (which would invalidate the visibility buffer's generation
    /// check) or the compiled permutation — it only rewrites the four dynamic
    /// header words and marks the row dirty for the next upload.
    ///
    /// # Errors
    ///
    /// Returns [`MaterialRegistryError::InvalidHandle`] for an unknown or stale
    /// handle.
    pub fn set_dynamic_channels(
        &mut self,
        handle: GenerationalHandle,
        channels: DynamicChannels,
    ) -> Result<(), MaterialRegistryError> {
        if !self.allocator.validate(handle) {
            return Err(MaterialRegistryError::InvalidHandle(handle));
        }
        let index = handle.index as usize;
        if self.slots.get(index).and_then(Option::as_ref).is_none() {
            return Err(MaterialRegistryError::InvalidHandle(handle));
        }
        self.dynamic
            .resize(self.dynamic.len().max(index + 1), DynamicChannels::NONE);
        if self.dynamic[index] == channels {
            // Idempotent: an unchanged state must not dirty the row, or a system
            // that re-asserts its state every frame would re-upload every frame.
            return Ok(());
        }
        self.dynamic[index] = channels;
        self.dirty.insert(index as u32);
        Ok(())
    }

    /// The runtime dynamic channels of row `index` (inert when unset).
    pub fn dynamic_channels_at(&self, index: u32) -> DynamicChannels {
        self.dynamic
            .get(index as usize)
            .copied()
            .unwrap_or(DynamicChannels::NONE)
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
                let mut header =
                    slot.record
                        .header(index as u32, textures.len() as u32, self.epoch);
                let [wetness, snow, damage, dissolve] =
                    self.dynamic_channels_at(index as u32).to_words();
                header.dynamic_wetness = wetness;
                header.dynamic_snow = snow;
                header.dynamic_damage = damage;
                header.dynamic_dissolve = dissolve;
                headers[index] = header;
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
    fn dynamic_channels_reach_the_uploaded_header_without_a_republish() {
        let mut registry = MaterialRegistry::new(4);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        let revision_before = registry.get(handle).unwrap().revision;
        let generation_before = registry.generation_at(handle.index).unwrap();
        let _ = registry.take_dirty();

        // A runtime state change rewrites the four dynamic words and dirties the
        // row — nothing else: no revision bump (which would invalidate the
        // visibility buffer's generation check) and no epoch churn per frame.
        let channels = DynamicChannels {
            wetness: 0.75,
            snow: 0.25,
            damage: 0.5,
            dissolve: 1.0,
        };
        registry.set_dynamic_channels(handle, channels).unwrap();
        assert_eq!(registry.take_dirty(), vec![handle.index]);
        assert_eq!(registry.get(handle).unwrap().revision, revision_before);
        assert_eq!(registry.generation_at(handle.index), Some(generation_before));

        let (headers, _, _) = registry.gpu_tables();
        let header = &headers[handle.index as usize];
        assert_eq!(
            DynamicChannels::from_words([
                header.dynamic_wetness,
                header.dynamic_snow,
                header.dynamic_damage,
                header.dynamic_dissolve,
            ]),
            channels
        );

        // Re-asserting the same state is idempotent, so a system that writes its
        // state every frame does not re-upload every frame.
        registry.set_dynamic_channels(handle, channels).unwrap();
        assert!(registry.take_dirty().is_empty());

        // A republish is a different bake, so the previous state must not leak.
        registry.publish(record(handle, 2)).unwrap();
        let (headers, _, _) = registry.gpu_tables();
        assert_eq!(
            DynamicChannels::from_words([
                headers[handle.index as usize].dynamic_wetness,
                headers[handle.index as usize].dynamic_snow,
                headers[handle.index as usize].dynamic_damage,
                headers[handle.index as usize].dynamic_dissolve,
            ]),
            DynamicChannels::NONE
        );

        // Unknown handles are rejected rather than silently ignored.
        let stale = prism_render_architecture::abi::GenerationalHandle {
            index: handle.index,
            generation: handle.generation + 1,
        };
        assert!(matches!(
            registry.set_dynamic_channels(stale, channels),
            Err(MaterialRegistryError::InvalidHandle(_))
        ));
    }

    #[test]
    fn dynamic_channel_words_sanitize_runtime_state() {
        // Runtime state is outside the schema validators, so the pack step is the
        // contract: out-of-range clamps, non-finite goes inert.
        let words = DynamicChannels {
            wetness: 4.0,
            snow: -3.0,
            damage: f32::NAN,
            dissolve: f32::INFINITY,
        }
        .to_words();
        let unpacked = DynamicChannels::from_words(words);
        assert_eq!(unpacked.wetness, 1.0);
        assert_eq!(unpacked.snow, 0.0);
        assert_eq!(unpacked.damage, 0.0);
        assert_eq!(unpacked.dissolve, 0.0);
        assert!(DynamicChannels::NONE.is_inert());
        assert!(!unpacked.is_inert());
        // The identity round-trips exactly (bit-for-bit through the words).
        let exact = DynamicChannels {
            wetness: 0.1,
            snow: 0.2,
            damage: 0.3,
            dissolve: 0.4,
        };
        assert_eq!(DynamicChannels::from_words(exact.to_words()), exact);
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
