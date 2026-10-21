use crate::{
    DynamicChannels, GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters,
    MaterialCapacityError, MaterialExecutionBinding, MaterialExecutionTables,
    MaterialHandleAllocator, MaterialHandleError, MaterialRecord, EXECUTION_PLAN_ROW_WORDS,
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
    StaleRevision {
        current: u32,
        incoming: u32,
    },
    InvalidExecutionBinding,
    InvalidTextureResidencyUpdate {
        handle: GenerationalHandle,
        texture_index: usize,
    },
}

/// A compare-and-update patch for one texture's runtime residency metadata.
/// `expected` must match the entire current texture row. `replacement` may only
/// change its four virtual-texture fields, never the authored slot identity,
/// semantic or sampler. This also guards against stale same-revision residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialTextureResidencyUpdate {
    pub handle: GenerationalHandle,
    pub expected_revision: u32,
    pub texture_index: usize,
    pub expected: GpuMaterialTexture,
    pub replacement: GpuMaterialTexture,
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
    /// Verified dense execution identities live beside the baked record. They
    /// are runtime table bindings and must not be reconstructed from a hash.
    execution_bindings: Vec<Option<MaterialExecutionBinding>>,
    execution_plan_rows: Vec<[u32; EXECUTION_PLAN_ROW_WORDS]>,
    execution_program_rows: Vec<[u32; 5]>,
    execution_package_count: u32,
    execution_table_version: u32,
    execution_tables: Option<MaterialExecutionTables>,
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
            execution_bindings: vec![None],
            execution_plan_rows: Vec::new(),
            execution_program_rows: Vec::new(),
            execution_package_count: 0,
            execution_table_version: 1,
            execution_tables: None,
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
        // Validate before touching any per-slot state. A rejected publication
        // must preserve the live record, dynamic channels, epoch and dirty set.
        if let Some(Some(slot)) = self.slots.get(index)
            && record.revision <= slot.revision
        {
            return Err(MaterialRegistryError::StaleRevision {
                current: slot.revision,
                incoming: record.revision,
            });
        }
        self.generations
            .resize(self.generations.len().max(index + 1), 0);
        self.generations[index] = record.handle.generation;
        // A (re)published record starts from the inert dynamic state: the new
        // bake is a different material, so the previous wetness/snow/damage/
        // dissolve must not leak onto it.
        self.dynamic
            .resize(self.dynamic.len().max(index + 1), DynamicChannels::NONE);
        self.dynamic[index] = DynamicChannels::NONE;
        self.execution_bindings
            .resize(self.execution_bindings.len().max(index + 1), None);
        self.execution_bindings[index] = None;
        self.slots
            .resize_with(self.slots.len().max(index + 1), || None);
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

    /// Atomically update texture residency without republishing authored data.
    ///
    /// Every handle, exact authored revision and expected texture row is checked
    /// before any mutation. Missing textures, duplicate targets, stale rows and
    /// attempts to change slot/generation/semantic/sampler reject the whole batch
    /// with `InvalidTextureResidencyUpdate`. No material record is cloned.
    ///
    /// Returns the number of changed material rows. Changed rows are marked
    /// dirty and the epoch advances once per nonempty transaction; authored
    /// revisions (including `u32::MAX`), execution bindings and dynamic channels
    /// are untouched. Empty/unchanged batches do not write any registry state.
    pub fn update_texture_residency(
        &mut self,
        updates: &[MaterialTextureResidencyUpdate],
    ) -> Result<u32, MaterialRegistryError> {
        let mut targets = BTreeSet::new();
        let mut changed_rows = BTreeSet::new();
        for update in updates {
            if !self.allocator.validate(update.handle) {
                return Err(MaterialRegistryError::InvalidHandle(update.handle));
            }
            let record = self
                .get(update.handle)
                .ok_or(MaterialRegistryError::InvalidHandle(update.handle))?;
            if record.revision != update.expected_revision {
                return Err(MaterialRegistryError::StaleRevision {
                    current: record.revision,
                    incoming: update.expected_revision,
                });
            }
            let expected = update.expected;
            let replacement = update.replacement;
            if !targets.insert((update.handle.index, update.texture_index))
                || record.textures.get(update.texture_index) != Some(&expected)
                || (
                    expected.index,
                    expected.generation,
                    expected.semantic,
                    expected.sampler_index,
                ) != (
                    replacement.index,
                    replacement.generation,
                    replacement.semantic,
                    replacement.sampler_index,
                )
            {
                return Err(MaterialRegistryError::InvalidTextureResidencyUpdate {
                    handle: update.handle,
                    texture_index: update.texture_index,
                });
            }
            if expected != replacement {
                changed_rows.insert(update.handle.index);
            }
        }
        if changed_rows.is_empty() {
            return Ok(0);
        }
        // The exclusive borrow keeps the fully preflighted rows valid through
        // commit. Only the four residency words are ever assigned.
        for update in updates {
            if update.expected == update.replacement {
                continue;
            }
            let slot = self.slots[update.handle.index as usize]
                .as_mut()
                .expect("preflighted texture residency handle");
            let texture = &mut slot.record.textures[update.texture_index];
            texture.virtual_texture_id = update.replacement.virtual_texture_id;
            texture.virtual_texture_width = update.replacement.virtual_texture_width;
            texture.virtual_texture_height = update.replacement.virtual_texture_height;
            texture.virtual_texture_page_size = update.replacement.virtual_texture_page_size;
        }
        let count = changed_rows.len() as u32;
        self.dirty.extend(changed_rows);
        self.epoch += 1;
        Ok(count)
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
        if let Some(binding) = self.execution_bindings.get_mut(handle.index as usize) {
            *binding = None;
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
    /// Like texture residency and execution bindings, this runtime mutation
    /// does not go through [`Self::publish`]: a
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
        // Compare the GPU-visible value, not an unsanitized float: repeated
        // NaN/out-of-range inputs must not trigger an upload every frame.
        let channels = DynamicChannels::from_words(channels.to_words());
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

    /// Attach a binding that has already been checked against the dense plan,
    /// program, package and permutation tables.
    pub fn set_execution_binding(
        &mut self,
        handle: GenerationalHandle,
        binding: MaterialExecutionBinding,
    ) -> Result<(), MaterialRegistryError> {
        if !self.allocator.validate(handle)
            || self
                .slots
                .get(handle.index as usize)
                .and_then(Option::as_ref)
                .is_none()
        {
            return Err(MaterialRegistryError::InvalidHandle(handle));
        }
        let plan = self
            .execution_plan_rows
            .get(binding.shading_plan_index.get() as usize)
            .ok_or(MaterialRegistryError::InvalidExecutionBinding)?;
        let program = self
            .execution_program_rows
            .get(binding.response_program_index.get() as usize)
            .ok_or(MaterialRegistryError::InvalidExecutionBinding)?;
        if plan[0] != binding.response_program_index.get()
            && plan[1] != binding.response_program_index.get()
        {
            return Err(MaterialRegistryError::InvalidExecutionBinding);
        }
        if program[1] != binding.package_index.get()
            || binding.package_index.get() >= self.execution_package_count
        {
            return Err(MaterialRegistryError::InvalidExecutionBinding);
        }
        let permutation = binding.permutation_index.get();
        let limit = u64::from(program[2]) | (u64::from(program[3]) << 32);
        if permutation >= limit {
            return Err(MaterialRegistryError::InvalidExecutionBinding);
        }
        let index = handle.index as usize;
        self.execution_bindings
            .resize(self.execution_bindings.len().max(index + 1), None);
        if self.execution_bindings[index] == Some(binding) {
            return Ok(());
        }
        self.execution_bindings[index] = Some(binding);
        self.dirty.insert(handle.index);
        Ok(())
    }

    /// Remove the execution binding and restore the five-word ABI sentinel.
    pub fn clear_execution_binding(
        &mut self,
        handle: GenerationalHandle,
    ) -> Result<(), MaterialRegistryError> {
        if !self.allocator.validate(handle)
            || self
                .slots
                .get(handle.index as usize)
                .and_then(Option::as_ref)
                .is_none()
        {
            return Err(MaterialRegistryError::InvalidHandle(handle));
        }
        let index = handle.index as usize;
        self.execution_bindings
            .resize(self.execution_bindings.len().max(index + 1), None);
        if self.execution_bindings[index].take().is_some() {
            self.dirty.insert(handle.index);
        }
        Ok(())
    }

    /// Return the verified execution identity attached to row `index`.
    #[must_use]
    pub fn execution_binding_at(&self, index: u32) -> Option<MaterialExecutionBinding> {
        self.execution_bindings
            .get(index as usize)
            .copied()
            .flatten()
    }

    /// Installs the verified dense execution tables used by both CPU
    /// classification and the GPU material bind group. Existing material
    /// bindings are revalidated before the new rows become visible.
    pub fn install_execution_tables(
        &mut self,
        tables: &MaterialExecutionTables,
    ) -> Result<(), MaterialRegistryError> {
        let plan_rows = tables.gpu_plan_rows();
        let program_rows = tables.gpu_program_rows();
        let package_count = tables.gpu_package_count();
        for binding in self.execution_bindings.iter().flatten().copied() {
            let Some(plan) = plan_rows.get(binding.shading_plan_index.get() as usize) else {
                return Err(MaterialRegistryError::InvalidExecutionBinding);
            };
            let Some(program) = program_rows.get(binding.response_program_index.get() as usize)
            else {
                return Err(MaterialRegistryError::InvalidExecutionBinding);
            };
            if plan[0] != binding.response_program_index.get()
                && plan[1] != binding.response_program_index.get()
                || program[1] != binding.package_index.get()
                || binding.package_index.get() >= package_count
            {
                return Err(MaterialRegistryError::InvalidExecutionBinding);
            }
            let limit = u64::from(program[2]) | (u64::from(program[3]) << 32);
            if binding.permutation_index.get() >= limit {
                return Err(MaterialRegistryError::InvalidExecutionBinding);
            }
        }
        self.execution_plan_rows = plan_rows;
        self.execution_program_rows = program_rows;
        self.execution_package_count = package_count;
        self.execution_tables = Some(tables.clone());
        self.execution_table_version = self.execution_table_version.wrapping_add(1).max(1);
        Ok(())
    }

    #[must_use]
    pub fn execution_tables(&self) -> Option<&MaterialExecutionTables> {
        self.execution_tables.as_ref()
    }

    #[must_use]
    pub fn execution_plan_rows(&self) -> &[[u32; EXECUTION_PLAN_ROW_WORDS]] {
        &self.execution_plan_rows
    }

    #[must_use]
    pub fn execution_program_rows(&self) -> &[[u32; 5]] {
        &self.execution_program_rows
    }

    #[must_use]
    pub const fn execution_package_count(&self) -> u32 {
        self.execution_package_count
    }

    #[must_use]
    pub const fn execution_table_version(&self) -> u32 {
        self.execution_table_version
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
                let mut header = slot.record.header_with_execution(
                    index as u32,
                    textures.len() as u32,
                    self.epoch,
                    self.execution_binding_at(index as u32),
                );
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
    use crate::{GpuSurfaceParameters, MaterialDomain, MaterialFeatureFlags, MaterialRenderClass};
    use prism_render_architecture::render_fabric::RenderDomain;

    fn record(handle: GenerationalHandle, revision: u32) -> MaterialRecord {
        MaterialRecord {
            handle,
            revision,
            domain: MaterialDomain::Surface,
            render_domain: RenderDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            response_plan: crate::ShadingPlan::default(),
            features: MaterialFeatureFlags::default(),
            closure_mask: 1,
            surface: GpuSurfaceParameters::default(),
            textures: Vec::new(),
        }
    }

    fn texture_record(handle: GenerationalHandle, revision: u32) -> MaterialRecord {
        let mut record = record(handle, revision);
        record.textures = vec![
            GpuMaterialTexture::non_virtual(7, 3, 11, 13),
            GpuMaterialTexture::non_virtual(8, 4, 17, 19),
        ];
        record
    }

    fn residency_update(
        record: &MaterialRecord,
        texture_index: usize,
    ) -> MaterialTextureResidencyUpdate {
        let expected = record.textures[texture_index];
        MaterialTextureResidencyUpdate {
            handle: record.handle,
            expected_revision: record.revision,
            texture_index,
            expected,
            replacement: GpuMaterialTexture {
                virtual_texture_id: 23,
                virtual_texture_width: 256,
                virtual_texture_height: 128,
                virtual_texture_page_size: 64,
                ..expected
            },
        }
    }

    #[test]
    fn texture_residency_at_revision_max_preserves_authored_data_and_storage() {
        let mut registry = MaterialRegistry::new(4);
        let handle = registry.allocate().unwrap();
        let mut expected = texture_record(handle, u32::MAX);
        registry.publish(expected.clone()).unwrap();
        let channels = DynamicChannels {
            wetness: 0.5,
            ..DynamicChannels::NONE
        };
        registry.set_dynamic_channels(handle, channels).unwrap();
        registry.take_dirty();
        let before = registry.snapshot();
        let textures_ptr = registry.get(handle).unwrap().textures.as_ptr();
        let update = residency_update(&expected, 0);
        assert_eq!(registry.update_texture_residency(&[update]), Ok(1));
        expected.textures[0] = update.replacement;
        assert_eq!(registry.get(handle), Some(&expected));
        assert_eq!(
            registry.get(handle).unwrap().textures.as_ptr(),
            textures_ptr
        );
        assert_eq!(
            registry.slots[handle.index as usize]
                .as_ref()
                .unwrap()
                .revision,
            u32::MAX
        );
        assert_eq!(registry.dynamic_channels_at(handle.index), channels);
        assert_eq!(
            registry.generation_at(handle.index),
            Some(handle.generation)
        );
        assert_eq!(
            registry.snapshot(),
            MaterialSnapshot {
                epoch: before.epoch + 1,
                ..before
            }
        );
        assert_eq!(registry.take_dirty(), vec![handle.index]);

        // Revocation must remain possible without an authored revision left.
        let revoke = MaterialTextureResidencyUpdate {
            expected: update.replacement,
            replacement: update.expected,
            ..update
        };
        assert_eq!(registry.update_texture_residency(&[revoke]), Ok(1));
        expected.textures[0] = update.expected;
        assert_eq!(registry.get(handle), Some(&expected));
        assert_eq!(registry.dynamic_channels_at(handle.index), channels);
        assert_eq!(registry.take_dirty(), vec![handle.index]);
    }

    #[test]
    fn texture_residency_empty_and_unchanged_batches_are_noops() {
        let mut registry = MaterialRegistry::new(4);
        let handle = registry.allocate().unwrap();
        let record = texture_record(handle, 5);
        let update = residency_update(&record, 0);
        registry.publish(record.clone()).unwrap();
        registry.take_dirty();
        let before = registry.snapshot();
        let noop = MaterialTextureResidencyUpdate {
            replacement: update.expected,
            ..update
        };
        for _ in 0..2 {
            assert_eq!(registry.update_texture_residency(&[]), Ok(0));
            assert_eq!(registry.update_texture_residency(&[noop]), Ok(0));
            assert_eq!(registry.snapshot(), before);
            assert_eq!(registry.get(handle), Some(&record));
            assert!(registry.take_dirty().is_empty());
        }
        // A no-op may not consume an already pending dirty row either.
        registry.mark_all_dirty();
        let before = registry.snapshot();
        assert_eq!(registry.update_texture_residency(&[noop]), Ok(0));
        assert_eq!(registry.snapshot(), before);
        assert_eq!(registry.take_dirty(), vec![0, handle.index]);
    }

    #[test]
    fn texture_residency_batch_preflight_rolls_back_every_row_and_runtime_state() {
        let mut registry = MaterialRegistry::new(4);
        let first = registry.allocate().unwrap();
        let second = registry.allocate().unwrap();
        let first_record = texture_record(first, 5);
        let second_record = texture_record(second, 7);
        let valid = residency_update(&first_record, 0);
        let last = residency_update(&second_record, 0);
        registry.publish(first_record.clone()).unwrap();
        registry.publish(second_record.clone()).unwrap();
        let channels = DynamicChannels {
            snow: 0.5,
            ..DynamicChannels::NONE
        };
        registry.set_dynamic_channels(first, channels).unwrap();
        registry.set_dynamic_channels(second, channels).unwrap();
        registry.take_dirty();
        let before = registry.snapshot();
        let invalid_texture =
            |texture_index| MaterialRegistryError::InvalidTextureResidencyUpdate {
                handle: second,
                texture_index,
            };
        let stale = second.with_generation(second.generation + 1);
        let mut rejected = vec![
            (
                MaterialTextureResidencyUpdate {
                    handle: stale,
                    ..last
                },
                MaterialRegistryError::InvalidHandle(stale),
            ),
            (
                MaterialTextureResidencyUpdate {
                    expected_revision: 6,
                    ..last
                },
                MaterialRegistryError::StaleRevision {
                    current: 7,
                    incoming: 6,
                },
            ),
            (
                MaterialTextureResidencyUpdate {
                    expected_revision: 8,
                    ..last
                },
                MaterialRegistryError::StaleRevision {
                    current: 7,
                    incoming: 8,
                },
            ),
            (
                MaterialTextureResidencyUpdate {
                    texture_index: usize::MAX,
                    ..last
                },
                invalid_texture(usize::MAX),
            ),
            (
                MaterialTextureResidencyUpdate {
                    expected: last.replacement,
                    ..last
                },
                invalid_texture(0),
            ),
            (
                valid,
                MaterialRegistryError::InvalidTextureResidencyUpdate {
                    handle: first,
                    texture_index: 0,
                },
            ),
        ];
        // Each authored identity word is independently protected.
        for word in 0..4 {
            let mut replacement = last.replacement;
            match word {
                0 => replacement.index += 1,
                1 => replacement.generation += 1,
                2 => replacement.semantic += 1,
                _ => replacement.sampler_index += 1,
            }
            rejected.push((
                MaterialTextureResidencyUpdate {
                    replacement,
                    ..last
                },
                invalid_texture(0),
            ));
        }
        for (invalid, error) in rejected {
            assert_eq!(
                registry.update_texture_residency(&[valid, invalid]),
                Err(error)
            );
            assert_eq!(registry.get(first), Some(&first_record));
            assert_eq!(registry.get(second), Some(&second_record));
            assert_eq!(registry.dynamic_channels_at(first.index), channels);
            assert_eq!(registry.dynamic_channels_at(second.index), channels);
            assert_eq!(registry.snapshot(), before);
            assert!(registry.take_dirty().is_empty());
        }
        // Preserve pending dirtiness on rejection, not only an empty dirty set.
        registry.mark_all_dirty();
        let before = registry.snapshot();
        assert!(registry.update_texture_residency(&[valid, valid]).is_err());
        assert_eq!(registry.snapshot(), before);
        assert_eq!(registry.take_dirty(), vec![0, first.index, second.index]);
    }

    #[test]
    fn texture_residency_rejects_unpublished_retired_and_reused_handles() {
        let mut registry = MaterialRegistry::new(1);
        let handle = registry.allocate().unwrap();
        let record = texture_record(handle, 5);
        let update = residency_update(&record, 0);
        assert_eq!(
            registry.update_texture_residency(&[update]),
            Err(MaterialRegistryError::InvalidHandle(handle))
        );
        registry.publish(record).unwrap();
        registry.retire(handle, GpuCompletionValue(1)).unwrap();
        assert_eq!(
            registry.update_texture_residency(&[update]),
            Err(MaterialRegistryError::InvalidHandle(handle))
        );
        registry.reclaim_completed(GpuCompletionValue(1));
        let replacement = registry.allocate().unwrap();
        assert_eq!(replacement.index, handle.index);
        assert_ne!(replacement.generation, handle.generation);
        let record = texture_record(replacement, 5);
        registry.publish(record.clone()).unwrap();
        registry.take_dirty();
        let before = registry.snapshot();
        assert_eq!(
            registry.update_texture_residency(&[update]),
            Err(MaterialRegistryError::InvalidHandle(handle))
        );
        assert_eq!(registry.get(replacement), Some(&record));
        assert_eq!(registry.snapshot(), before);
        assert!(registry.take_dirty().is_empty());
    }

    #[test]
    fn texture_residency_counts_materials_once_and_rejects_same_revision_stale_rows() {
        let mut registry = MaterialRegistry::new(4);
        let first = registry.allocate().unwrap();
        let second = registry.allocate().unwrap();
        let first_record = texture_record(first, 5);
        let second_record = texture_record(second, 7);
        let updates = [
            residency_update(&first_record, 0),
            residency_update(&first_record, 1),
            residency_update(&second_record, 0),
        ];
        registry.publish(first_record).unwrap();
        registry.publish(second_record).unwrap();
        registry.take_dirty();
        let before = registry.snapshot();
        assert_eq!(registry.update_texture_residency(&updates), Ok(2));
        assert_eq!(registry.snapshot().epoch, before.epoch + 1);
        assert_eq!(registry.take_dirty(), vec![first.index, second.index]);
        let before = registry.snapshot();
        // This is not an idempotent retry: its expected old residency is stale.
        assert_eq!(
            registry.update_texture_residency(&updates),
            Err(MaterialRegistryError::InvalidTextureResidencyUpdate {
                handle: first,
                texture_index: 0
            })
        );
        assert_eq!(registry.snapshot(), before);
        assert_eq!(
            registry.get(first).unwrap().textures[0],
            updates[0].replacement
        );
        assert!(registry.take_dirty().is_empty());
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
    fn rejected_publish_preserves_all_live_state() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 5)).unwrap();
        let channels = DynamicChannels {
            wetness: 0.7,
            snow: 0.2,
            damage: 0.4,
            dissolve: 0.1,
        };
        registry.set_dynamic_channels(handle, channels).unwrap();
        let snapshot = registry.snapshot();
        let live = registry.get(handle).unwrap().clone();
        registry.take_dirty();
        for revision in [0, 4, 5] {
            assert!(matches!(
                registry.publish(record(handle, revision)),
                Err(MaterialRegistryError::StaleRevision { .. })
            ));
            assert_eq!(registry.snapshot(), snapshot);
            assert_eq!(registry.get(handle), Some(&live));
            assert_eq!(registry.dynamic_channels_at(handle.index), channels);
            assert_eq!(
                registry.generation_at(handle.index),
                Some(handle.generation)
            );
            assert!(registry.take_dirty().is_empty());
        }
    }

    #[test]
    fn sanitized_dynamic_inputs_are_upload_idempotent() {
        let mut registry = MaterialRegistry::new(8);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        registry.take_dirty();
        let channels = DynamicChannels {
            wetness: 4.0,
            snow: -3.0,
            damage: f32::NAN,
            dissolve: f32::INFINITY,
        };
        registry.set_dynamic_channels(handle, channels).unwrap();
        assert_eq!(registry.take_dirty(), vec![handle.index]);
        registry.set_dynamic_channels(handle, channels).unwrap();
        assert!(registry.take_dirty().is_empty());
        assert_eq!(
            registry.dynamic_channels_at(handle.index),
            DynamicChannels {
                wetness: 1.0,
                ..DynamicChannels::NONE
            }
        );
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
        assert_eq!(value.response_plan, crate::ShadingPlan::default());
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
        assert_eq!(
            registry.generation_at(handle.index),
            Some(generation_before)
        );

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
        let stale = GenerationalHandle {
            index: handle.index,
            generation: handle.generation + 1,
        };
        assert!(matches!(
            registry.set_dynamic_channels(stale, channels),
            Err(MaterialRegistryError::InvalidHandle(_))
        ));
    }

    #[test]
    fn gpu_upload_preserves_the_unbound_execution_sentinel() {
        let mut registry = MaterialRegistry::new(4);
        let handle = registry.allocate().unwrap();
        registry.publish(record(handle, 1)).unwrap();
        let (headers, _, _) = registry.gpu_tables();

        assert_eq!(
            headers[handle.index as usize].execution_binding_words(),
            [GpuMaterialHeader::EXECUTION_BINDING_SENTINEL; 5]
        );
        assert!(!headers[handle.index as usize].has_execution_binding());
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
