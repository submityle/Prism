use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::{HashMap, HashSet};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{GpuMaterialTexture, MaterialRegistry, MaterialRegistryError};
use std::collections::VecDeque;

use super::portable_atlas::{portable_texture_fallback, PORTABLE_TEXTURE_PAGE_SIZE};
use super::standard_adapter::{
    import_standard_material, material_surface_context, MaterialSurfaceContext,
    StandardMaterialTextureResolver, TextureSemantic,
};
use super::texture_heap::{BindlessHeapStats, BindlessTextureHeap};
use super::vt_source::{
    material_texture_ids, MaterialVtActivePublication, MaterialVtSourceLookup,
    MaterialVtSourceRegistry,
};

#[derive(Resource)]
pub(crate) struct RenderMaterialRegistry {
    pub registry: MaterialRegistry,
    pub assets: HashMap<AssetId<StandardMaterial>, GenerationalHandle>,
    pub revisions: HashMap<AssetId<StandardMaterial>, u32>,
    pub dirty_assets: HashSet<AssetId<StandardMaterial>>,
    /// Ordered retry queues. Sequence tokens invalidate superseded lifecycle
    /// actions in O(1), avoiding a linear queue scan for every published row.
    pending_publish: VecDeque<(AssetId<StandardMaterial>, u64)>,
    pending_publish_tokens: HashMap<AssetId<StandardMaterial>, u64>,
    pending_retire: VecDeque<(AssetId<StandardMaterial>, u64)>,
    pending_retire_tokens: HashMap<AssetId<StandardMaterial>, u64>,
    next_pending_token: u64,
    /// Bounded bindless slot allocator backing every material texture.
    textures: BindlessTextureHeap,
    /// Set of images each published material currently holds a reference to, so
    /// republishing or retiring a material releases exactly the slots it owned.
    material_textures: HashMap<AssetId<StandardMaterial>, Vec<AssetId<Image>>>,
    /// Reverse image-to-material references used to invalidate only materials
    /// that actually consume a changed image.
    image_materials: HashMap<AssetId<Image>, HashSet<AssetId<StandardMaterial>>>,
    /// Scratch list the active [`TextureResolver`] pushes acquired images into
    /// while lowering a single material. Drained by `publish_asset`.
    pending_textures: Vec<AssetId<Image>>,
    /// The last successfully lowered material value for each live row. The VT
    /// producer uses this render-world snapshot to republish rows in the same
    /// frame that a completed page-table slice becomes active; waiting for the
    /// next main-world extraction would expose one frame of mixed source
    /// metadata and residency.
    material_snapshots: HashMap<AssetId<StandardMaterial>, StandardMaterial>,
    /// Stable render-world UV/channel context paired with each material row.
    /// This is kept outside the canonical material ABI because it is scene
    /// adapter state, not part of the format-independent surface record.
    surface_contexts: Vec<MaterialSurfaceContext>,
    /// Whether the render world has enumerated the complete main-world material
    /// asset set at least once. Keeping this state with the registry guarantees
    /// that replacing/rebuilding the registry also re-runs the bootstrap.
    initial_asset_scan_started: bool,
    initial_asset_scan_complete: bool,
    initial_assets_pending: HashSet<AssetId<StandardMaterial>>,
    /// Non-bindless devices use the shared Material-VT atlas for authored
    /// normalized images that fit the portable row contract.
    portable_atlas: bool,
}

impl Default for RenderMaterialRegistry {
    fn default() -> Self {
        Self {
            registry: MaterialRegistry::new(1 << 20),
            assets: HashMap::default(),
            revisions: HashMap::default(),
            dirty_assets: HashSet::default(),
            pending_publish: VecDeque::new(),
            pending_publish_tokens: HashMap::default(),
            pending_retire: VecDeque::new(),
            pending_retire_tokens: HashMap::default(),
            next_pending_token: 0,
            textures: BindlessTextureHeap::default(),
            material_textures: HashMap::default(),
            image_materials: HashMap::default(),
            pending_textures: Vec::new(),
            material_snapshots: HashMap::default(),
            surface_contexts: vec![MaterialSurfaceContext::default()],
            initial_asset_scan_started: false,
            initial_asset_scan_complete: false,
            initial_assets_pending: HashSet::default(),
            portable_atlas: false,
        }
    }
}

impl RenderMaterialRegistry {
    fn allocate_pending_token(&mut self) -> u64 {
        let token = self.next_pending_token;
        self.next_pending_token = self.next_pending_token.wrapping_add(1);
        token
    }

    pub(crate) fn queue_publish(&mut self, id: AssetId<StandardMaterial>) {
        self.pending_retire_tokens.remove(&id);
        if !self.initial_asset_scan_complete {
            self.initial_assets_pending.insert(id);
        }
        if !self.pending_publish_tokens.contains_key(&id) {
            let token = self.allocate_pending_token();
            self.pending_publish_tokens.insert(id, token);
            self.pending_publish.push_back((id, token));
        }
    }

    pub(crate) fn queue_retire(&mut self, id: AssetId<StandardMaterial>) {
        self.pending_publish_tokens.remove(&id);
        self.initial_assets_pending.remove(&id);
        if !self.pending_retire_tokens.contains_key(&id) {
            let token = self.allocate_pending_token();
            self.pending_retire_tokens.insert(id, token);
            self.pending_retire.push_back((id, token));
        }
    }

    pub(crate) fn pending_publish_count(&self) -> usize {
        self.pending_publish.len()
    }

    pub(crate) fn pending_retire_count(&self) -> usize {
        self.pending_retire.len()
    }

    pub(crate) fn is_pending_retire(&self, id: AssetId<StandardMaterial>) -> bool {
        self.pending_retire_tokens.contains_key(&id)
    }

    pub(crate) fn has_pending_publish(&self, id: AssetId<StandardMaterial>) -> bool {
        self.pending_publish_tokens.contains_key(&id)
    }

    pub(crate) fn pop_pending_publish(&mut self) -> Option<(AssetId<StandardMaterial>, u64)> {
        self.pending_publish.pop_front()
    }

    pub(crate) fn pop_pending_retire(&mut self) -> Option<(AssetId<StandardMaterial>, u64)> {
        self.pending_retire.pop_front()
    }

    pub(crate) fn pending_publish_token_is_current(
        &self,
        id: AssetId<StandardMaterial>,
        token: u64,
    ) -> bool {
        self.pending_publish_tokens.get(&id) == Some(&token)
    }

    pub(crate) fn pending_retire_token_is_current(
        &self,
        id: AssetId<StandardMaterial>,
        token: u64,
    ) -> bool {
        self.pending_retire_tokens.get(&id) == Some(&token)
    }

    pub(crate) fn retry_pending_publish(&mut self, id: AssetId<StandardMaterial>, token: u64) {
        if self.pending_publish_token_is_current(id, token) {
            self.pending_publish.push_back((id, token));
        }
    }

    pub(crate) fn retry_pending_retire(&mut self, id: AssetId<StandardMaterial>, token: u64) {
        if self.pending_retire_token_is_current(id, token) {
            self.pending_retire.push_back((id, token));
        }
    }

    pub(crate) fn finish_pending_publish(&mut self, id: AssetId<StandardMaterial>, token: u64) {
        if self.pending_publish_token_is_current(id, token) {
            self.pending_publish_tokens.remove(&id);
            self.initial_assets_pending.remove(&id);
        }
    }

    pub(crate) fn cancel_pending_publish(&mut self, id: AssetId<StandardMaterial>) {
        self.pending_publish_tokens.remove(&id);
    }

    pub(crate) fn finish_pending_retire(&mut self, id: AssetId<StandardMaterial>, token: u64) {
        if self.pending_retire_token_is_current(id, token) {
            self.pending_retire_tokens.remove(&id);
        }
    }

    pub(crate) const fn initial_asset_scan_complete(&self) -> bool {
        self.initial_asset_scan_complete
    }

    pub(crate) const fn initial_asset_scan_started(&self) -> bool {
        self.initial_asset_scan_started
    }

    pub(crate) fn begin_initial_asset_scan(
        &mut self,
        ids: impl IntoIterator<Item = AssetId<StandardMaterial>>,
    ) {
        self.initial_asset_scan_started = true;
        for id in ids {
            if !self.revisions.contains_key(&id) && !self.is_pending_retire(id) {
                self.queue_publish(id);
            }
        }
        self.finish_initial_asset_scan_if_ready();
    }

    pub(crate) fn finish_initial_asset_scan_if_ready(&mut self) {
        if self.initial_asset_scan_started && self.initial_assets_pending.is_empty() {
            self.initial_asset_scan_complete = true;
        }
    }

    fn replace_material_texture_references(
        &mut self,
        id: AssetId<StandardMaterial>,
        new_images: &[AssetId<Image>],
    ) {
        if let Some(previous) = self.material_snapshots.get(&id) {
            for image in material_texture_ids(previous) {
                if new_images.contains(&image) {
                    continue;
                }
                let remove_entry = if let Some(materials) = self.image_materials.get_mut(&image) {
                    materials.remove(&id);
                    materials.is_empty()
                } else {
                    false
                };
                if remove_entry {
                    self.image_materials.remove(&image);
                }
            }
        }
        for image in new_images {
            self.image_materials.entry(*image).or_default().insert(id);
        }
    }

    pub fn material_handle(&self, asset: AssetId<StandardMaterial>) -> Option<GenerationalHandle> {
        self.assets.get(&asset).copied()
    }
    pub fn publish_asset(
        &mut self,
        id: AssetId<StandardMaterial>,
        material: &StandardMaterial,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<GenerationalHandle, MaterialRegistryError> {
        self.publish_asset_with_sources(id, material, completion, None, None)
    }

    pub fn publish_asset_with_sources(
        &mut self,
        id: AssetId<StandardMaterial>,
        material: &StandardMaterial,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
        sources: Option<&MaterialVtSourceRegistry>,
        active_vt: Option<&MaterialVtActivePublication>,
    ) -> Result<GenerationalHandle, MaterialRegistryError> {
        let current = self.revisions.get(&id).copied().unwrap_or(0);
        let revision = current
            .checked_add(1)
            .ok_or(MaterialRegistryError::StaleRevision {
                current,
                incoming: current,
            })?;
        let (handle, allocated) = match self.assets.get(&id).copied() {
            Some(handle) => (handle, false),
            None => {
                let handle = self
                    .registry
                    .allocate()
                    .map_err(|_| MaterialRegistryError::CapacityExceeded)?;
                self.assets.insert(id, handle);
                (handle, true)
            }
        };
        // Each `resolve` call during the OpenPBR import acquires a bindless slot and
        // records the image here so we can reconcile references afterwards.
        self.pending_textures.clear();
        let record = {
            let mut resolver = self.texture_resolver_with_sources(sources, active_vt);
            import_standard_material(handle, revision, material, &mut resolver)
        };
        if let Err(error) = self.registry.publish(record) {
            // Roll back the slots this attempt acquired so a failed publish
            // never leaks references.
            for image in core::mem::take(&mut self.pending_textures) {
                self.textures.release(image);
            }
            if allocated {
                self.assets.remove(&id);
                let _ = self.registry.cancel_allocation(handle);
            }
            return Err(error);
        }
        // Success: adopt the freshly acquired texture set and release the set the
        // previous revision held. Acquiring before releasing keeps shared
        // textures resident without a transient generation bump.
        let new_set = core::mem::take(&mut self.pending_textures);
        // Authored dependencies include overflow fallbacks, but ownership must
        // include only successful acquires. A fallback must never retire a
        // reference acquired later by a different material for the same image.
        let dependencies: Vec<_> = material_texture_ids(material).collect();
        self.replace_material_texture_references(id, &dependencies);
        let previous = self
            .material_textures
            .insert(id, new_set)
            .unwrap_or_default();
        for image in previous {
            // The previous material row may still be referenced by the GPU.
            // Keep its slot quarantined until the submission that could read
            // that row has completed; immediate release would allow a new
            // image to reuse the slot and create a use-after-republish.
            self.textures.retire(image, completion);
        }
        self.material_snapshots.insert(id, material.clone());
        let index = handle.index as usize;
        if self.surface_contexts.len() <= index {
            self.surface_contexts
                .resize(index + 1, MaterialSurfaceContext::default());
        }
        self.surface_contexts[index] = material_surface_context(material);
        self.revisions.insert(id, revision);
        self.dirty_assets.insert(id);
        Ok(handle)
    }
    pub fn retire_asset(
        &mut self,
        id: AssetId<StandardMaterial>,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<Option<GenerationalHandle>, prism_render_material::MaterialHandleError> {
        let Some(handle) = self.assets.get(&id).copied() else {
            return Ok(None);
        };
        self.registry.retire(handle, completion)?;
        self.assets.remove(&id);
        self.revisions.remove(&id);
        self.replace_material_texture_references(id, &[]);
        self.material_snapshots.remove(&id);
        if let Some(context) = self.surface_contexts.get_mut(handle.index as usize) {
            *context = MaterialSurfaceContext::default();
        }
        self.cancel_pending_publish(id);
        if let Some(images) = self.material_textures.remove(&id) {
            for image in images {
                self.textures.retire(image, completion);
            }
        }
        self.dirty_assets.insert(id);
        Ok(Some(handle))
    }
    pub fn texture_resolver(&mut self) -> TextureResolver<'_> {
        self.texture_resolver_with_sources(None, None)
    }

    fn texture_resolver_with_sources<'a>(
        &'a mut self,
        sources: Option<&'a MaterialVtSourceRegistry>,
        active_vt: Option<&'a MaterialVtActivePublication>,
    ) -> TextureResolver<'a> {
        TextureResolver {
            registry: self,
            sources,
            active_vt,
        }
    }

    /// Current bindless texture-heap occupancy, surfaced for diagnostics and
    /// consumed by the bindless upload system when it sizes its slot arrays.
    pub fn texture_heap_stats(&self) -> BindlessHeapStats {
        self.textures.stats()
    }

    pub(crate) fn texture_generation_table(&self) -> &[u32] {
        self.textures.generation_table()
    }

    /// Reclaims texture slots whose GPU completion proof has arrived.
    pub fn reclaim_completed_textures(
        &mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> u32 {
        self.textures.reclaim_completed(completed)
    }

    /// High-water bound of live bindless slots. The upload path only needs to
    /// materialise `TextureView`s for slots below this value.
    pub fn texture_high_water(&self) -> u32 {
        self.textures.high_water()
    }

    /// Iterates the resident `(slot_index, image)` pairs so the upload system
    /// can place each live image at its assigned bindless slot.
    pub fn texture_slots(&self) -> impl Iterator<Item = (AssetId<Image>, u32)> + '_ {
        self.textures
            .iter_slots()
            .map(|(slot, image)| (image, slot))
    }

    pub(crate) fn texture_slot(
        &self,
        image: AssetId<Image>,
    ) -> Option<super::texture_heap::BindlessSlot> {
        self.textures.slot_of(image)
    }

    /// Returns material assets that currently retain a reference to `image`.
    pub(crate) fn materials_referencing_image(
        &self,
        image: AssetId<Image>,
    ) -> impl Iterator<Item = AssetId<StandardMaterial>> + '_ {
        self.image_materials
            .get(&image)
            .into_iter()
            .flat_map(|materials| materials.iter().copied())
    }

    /// Returns the last successfully published material value for a live row.
    /// This is render-world state, not a second source of truth: extraction
    /// replaces it whenever the main-world asset changes.
    pub(crate) fn material_snapshot(
        &self,
        material: AssetId<StandardMaterial>,
    ) -> Option<&StandardMaterial> {
        self.material_snapshots.get(&material)
    }

    pub(crate) fn surface_context_at(&self, index: u32) -> MaterialSurfaceContext {
        self.surface_contexts
            .get(index as usize)
            .copied()
            .unwrap_or_default()
    }

    /// Preflight the only fallible row operations before a VT slice can flip.
    /// Source-only republishing reuses live handles and snapshots, so it cannot
    /// allocate a new material. With exclusive access to this registry, these
    /// checks remain valid through the subsequent calls to the normal row API.
    pub(crate) fn can_republish_source_row(&self, id: AssetId<StandardMaterial>) -> bool {
        let Some(handle) = self.assets.get(&id).copied() else {
            return false;
        };
        let Some(record) = self.registry.get(handle) else {
            return false;
        };
        self.material_snapshots.contains_key(&id)
            && self
                .revisions
                .get(&id)
                .and_then(|revision| revision.checked_add(1))
                .is_some_and(|revision| revision > record.revision)
    }

    /// Clamps the bindless slot space to `capacity` before any material is
    /// published, so `acquire` can never hand out a slot the bound
    /// `binding_array` cannot address. Only valid while the heap is empty.
    pub fn set_texture_capacity(&mut self, capacity: u32) {
        self.textures.reset_with_capacity(capacity);
    }

    pub(crate) fn set_portable_atlas(&mut self, enabled: bool) {
        self.portable_atlas = enabled;
    }
}

pub(crate) struct TextureResolver<'a> {
    registry: &'a mut RenderMaterialRegistry,
    sources: Option<&'a MaterialVtSourceRegistry>,
    active_vt: Option<&'a MaterialVtActivePublication>,
}
impl StandardMaterialTextureResolver for TextureResolver<'_> {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture {
        // Acquire (or reference) the bindless slot and remember the image so the
        // owning material releases exactly what it took on republish/retire.
        let slot = self.registry.textures.acquire(image);
        if self.registry.textures.slot_of(image) != Some(slot) {
            // An overflow fallback does not own an authored atlas tile. In
            // particular it must never advertise portable slot zero as ready.
            let fallback = BindlessTextureHeap::default_slot(semantic);
            return GpuMaterialTexture::non_virtual(
                fallback.index,
                fallback.generation,
                semantic as u32,
                fallback.index,
            );
        }
        self.registry.pending_textures.push(image);
        // The sampler heap is bound in parallel with the texture heap, so a
        // texture's sampler lives at the same bindless slot as the texture
        // itself. Reserved slots 0/1/2 carry the default filtering sampler.
        let Some(MaterialVtSourceLookup::Available { identity, snapshot }) =
            self.sources
                .and_then(|sources| match sources.lookup(image) {
                    MaterialVtSourceLookup::Available { identity, snapshot } => {
                        Some(MaterialVtSourceLookup::Available { identity, snapshot })
                    }
                    _ => None,
                })
        else {
            return GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic as u32,
                slot.index,
            );
        };
        let published = self
            .active_vt
            .is_some_and(|publication| publication.contains(identity));
        if self.registry.portable_atlas {
            if let Some(fallback) = portable_texture_fallback(snapshot)
                && self
                    .active_vt
                    .is_some_and(|publication| publication.contains_portable(identity, slot))
            {
                return GpuMaterialTexture::with_portable_atlas(
                    slot.index,
                    slot.generation,
                    semantic as u32,
                    slot.index,
                    fallback.width,
                    fallback.height,
                    PORTABLE_TEXTURE_PAGE_SIZE,
                );
            }
            // A regular VT marker has no non-bindless consumer. Keep the
            // authored identity, but do not claim a tile until its exact slot
            // generation has a completed portable publication.
            return GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic as u32,
                slot.index,
            );
        }
        if self
            .sources
            .is_none_or(|sources| !sources.eligible_for_default_producer(image))
        {
            return GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic as u32,
                slot.index,
            );
        }
        if !published {
            return GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic as u32,
                slot.index,
            );
        }
        let descriptor = snapshot.descriptor();
        GpuMaterialTexture::with_virtual_texture(
            slot.index,
            // `generation` guards the bindless heap slot in the shader. The
            // source revision identifies the CPU snapshot and may advance
            // while the image remains resident in the same heap slot; using
            // that revision here would make an otherwise live VT row stale.
            slot.generation,
            semantic as u32,
            slot.index,
            descriptor.texture_id(),
            descriptor.width(),
            descriptor.height(),
            descriptor.page_size(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, Handle};
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot, NO_VIRTUAL_TEXTURE,
        PORTABLE_TEXTURE_ATLAS,
    };

    fn source(extent: u32, page_size: u32) -> MaterialVtSourceSnapshot {
        let mip_count = extent.ilog2() as u8 + 1;
        let descriptor = MaterialVtSourceDescriptor::new(
            17,
            extent,
            extent,
            1,
            page_size,
            mip_count,
            MaterialVtFormat::Rgba8Unorm,
        )
        .unwrap();
        let bytes: Vec<u8> = (0..u32::from(mip_count))
            .flat_map(|mip| std::iter::repeat_n(17, ((extent >> mip).max(1).pow(2) * 4) as usize))
            .collect();
        MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap()
    }

    fn publish_row(
        runtime: &mut RenderMaterialRegistry,
        material: &StandardMaterial,
        sources: &MaterialVtSourceRegistry,
        active: &MaterialVtActivePublication,
    ) -> GpuMaterialTexture {
        let id = Handle::<StandardMaterial>::from(Uuid::from_u128(0xE1)).id();
        let handle = runtime
            .publish_asset_with_sources(
                id,
                material,
                GpuCompletionValue(1),
                Some(sources),
                Some(active),
            )
            .unwrap();
        runtime.registry.get(handle).unwrap().textures[0]
    }

    #[test]
    fn portable_rows_require_completed_exact_slot_and_source_revision() {
        let image = Handle::<Image>::from(Uuid::from_u128(0xE2));
        let material = StandardMaterial {
            base_color_texture: Some(image.clone()),
            ..Default::default()
        };
        let mut sources = MaterialVtSourceRegistry::default();
        let identity = sources.insert_snapshot(image.id(), source(4, 128));
        let mut active = MaterialVtActivePublication::default();
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        let fallback = publish_row(&mut runtime, &material, &sources, &active);
        let slot = runtime.texture_slot(image.id()).unwrap();
        assert_eq!(fallback.virtual_texture_id, NO_VIRTUAL_TEXTURE);
        assert_eq!(fallback.index, slot.index);
        assert_eq!(fallback.sampler_index, slot.index);
        active.replace([identity]);
        assert_eq!(
            publish_row(&mut runtime, &material, &sources, &active).virtual_texture_id,
            NO_VIRTUAL_TEXTURE
        );
        let wrong_slot = super::super::texture_heap::BindlessSlot {
            generation: slot.generation + 1,
            ..slot
        };
        active.replace_with_portable_slots([identity], [(image.id(), wrong_slot)]);
        assert_eq!(
            publish_row(&mut runtime, &material, &sources, &active).virtual_texture_id,
            NO_VIRTUAL_TEXTURE
        );
        active.replace_with_portable_slots([identity], [(image.id(), slot)]);
        let ready = publish_row(&mut runtime, &material, &sources, &active);
        assert_eq!(ready.virtual_texture_id, PORTABLE_TEXTURE_ATLAS);
        assert_eq!(
            (ready.index, ready.generation, ready.sampler_index),
            (slot.index, slot.generation, slot.index)
        );
        assert_eq!(
            (
                ready.virtual_texture_width,
                ready.virtual_texture_height,
                ready.virtual_texture_page_size
            ),
            (4, 4, 128)
        );
        let next = sources.insert_snapshot(image.id(), source(8, 128));
        assert_ne!(identity, next);
        assert_eq!(
            publish_row(&mut runtime, &material, &sources, &active).virtual_texture_id,
            NO_VIRTUAL_TEXTURE
        );
    }

    #[test]
    fn portable_rows_use_authored_fallback_mip_for_large_sources() {
        let image = Handle::<Image>::from(Uuid::from_u128(0xE3));
        let material = StandardMaterial {
            base_color_texture: Some(image.clone()),
            ..Default::default()
        };
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        let mut sources = MaterialVtSourceRegistry::default();
        let identity = sources.insert_snapshot(image.id(), source(256, 128));
        let mut active = MaterialVtActivePublication::default();
        publish_row(&mut runtime, &material, &sources, &active);
        let slot = runtime.texture_slot(image.id()).unwrap();
        active.replace_with_portable_slots([identity], [(image.id(), slot)]);
        let row = publish_row(&mut runtime, &material, &sources, &active);
        assert_eq!(row.virtual_texture_id, PORTABLE_TEXTURE_ATLAS);
        assert_eq!(row.index, slot.index);
        assert_eq!(
            (
                row.virtual_texture_width,
                row.virtual_texture_height,
                row.virtual_texture_page_size
            ),
            (128, 128, 128)
        );
    }

    #[test]
    fn portable_incompatible_page_geometry_uses_source_backed_tile() {
        let image = Handle::<Image>::from(Uuid::from_u128(0xE5));
        let material = StandardMaterial {
            base_color_texture: Some(image.clone()),
            ..Default::default()
        };
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        let mut sources = MaterialVtSourceRegistry::default();
        let identity = sources.insert_snapshot(image.id(), source(4, 4));
        let mut active = MaterialVtActivePublication::default();
        publish_row(&mut runtime, &material, &sources, &active);
        let slot = runtime.texture_slot(image.id()).unwrap();
        active.replace_with_portable_slots([identity], [(image.id(), slot)]);
        let row = publish_row(&mut runtime, &material, &sources, &active);
        assert_eq!(row.virtual_texture_id, PORTABLE_TEXTURE_ATLAS);
        assert_eq!(row.index, slot.index);
        assert_eq!(
            (
                row.virtual_texture_width,
                row.virtual_texture_height,
                row.virtual_texture_page_size
            ),
            (4, 4, 128)
        );
    }

    #[test]
    fn portable_overflow_preserves_semantic_fallback_without_claiming_a_tile() {
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_texture_capacity(4);
        runtime.set_portable_atlas(true);
        runtime
            .textures
            .acquire(Handle::<Image>::from(Uuid::from_u128(0xE0)).id());
        let image = Handle::<Image>::from(Uuid::from_u128(0xE4)).id();
        for semantic in [
            TextureSemantic::BaseColor,
            TextureSemantic::Normal,
            TextureSemantic::Emissive,
        ] {
            let row = runtime.texture_resolver().resolve(image, semantic);
            let expected = BindlessTextureHeap::default_slot(semantic);
            assert_eq!(
                (row.index, row.generation, row.sampler_index),
                (expected.index, expected.generation, expected.index)
            );
            assert_eq!(row.virtual_texture_id, NO_VIRTUAL_TEXTURE);
            assert!(runtime.texture_slot(image).is_none());
        }
        assert_eq!(runtime.texture_heap_stats().overflow, 3);
        assert!(runtime.pending_textures.is_empty());
    }

    #[test]
    fn overflow_row_retirement_does_not_release_a_later_owner() {
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_texture_capacity(4);
        runtime.set_portable_atlas(true);
        let blocker = Handle::<StandardMaterial>::from(Uuid::from_u128(0xF1)).id();
        let fallback = Handle::<StandardMaterial>::from(Uuid::from_u128(0xF2)).id();
        let owner = Handle::<StandardMaterial>::from(Uuid::from_u128(0xF3)).id();
        let image = Handle::<Image>::from(Uuid::from_u128(0xF4));
        let material = StandardMaterial {
            base_color_texture: Some(image.clone()),
            ..Default::default()
        };
        runtime
            .publish_asset(
                blocker,
                &StandardMaterial {
                    base_color_texture: Some(Handle::<Image>::from(Uuid::from_u128(0xF5))),
                    ..Default::default()
                },
                GpuCompletionValue(1),
            )
            .unwrap();
        runtime
            .publish_asset(fallback, &material, GpuCompletionValue(1))
            .unwrap();
        assert!(runtime.material_textures[&fallback].is_empty());
        assert_eq!(
            runtime
                .materials_referencing_image(image.id())
                .collect::<Vec<_>>(),
            [fallback]
        );
        runtime
            .retire_asset(blocker, GpuCompletionValue(1))
            .unwrap();
        runtime.reclaim_completed_textures(GpuCompletionValue(1));
        runtime
            .publish_asset(owner, &material, GpuCompletionValue(2))
            .unwrap();
        let slot = runtime.texture_slot(image.id()).unwrap();
        runtime
            .retire_asset(fallback, GpuCompletionValue(2))
            .unwrap();
        runtime.reclaim_completed_textures(GpuCompletionValue(2));
        assert_eq!(runtime.texture_slot(image.id()), Some(slot));
        assert_eq!(
            runtime
                .materials_referencing_image(image.id())
                .collect::<Vec<_>>(),
            [owner]
        );
        runtime.retire_asset(owner, GpuCompletionValue(3)).unwrap();
        runtime.reclaim_completed_textures(GpuCompletionValue(3));
        assert!(runtime.texture_slot(image.id()).is_none());
        assert!(runtime
            .materials_referencing_image(image.id())
            .next()
            .is_none());
    }

    #[test]
    fn revision_overflow_does_not_allocate_or_mutate_a_material() {
        let mut runtime = RenderMaterialRegistry::default();
        let id = Handle::<StandardMaterial>::from(Uuid::from_u128(0xE5)).id();
        runtime.revisions.insert(id, u32::MAX);
        let before = runtime.registry.snapshot();
        assert!(matches!(
            runtime.publish_asset(id, &StandardMaterial::default(), GpuCompletionValue(1)),
            Err(MaterialRegistryError::StaleRevision {
                current: u32::MAX,
                incoming: u32::MAX
            })
        ));
        assert!(runtime.assets.is_empty());
        assert!(runtime.material_snapshots.is_empty());
        assert!(runtime.dirty_assets.is_empty());
        assert_eq!(runtime.registry.snapshot(), before);
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct PrismMaterialDiagnostics {
    pub active: u32,
    pub created: u32,
    pub updated: u32,
    pub retired: u32,
    pub reclaimed: u32,
    pub errors: u32,
    pub uploaded_rows: u32,
    pub uploaded_bytes: u64,
    pub epoch: u64,
    pub buffer_version: u32,
    pub buffer_rebuilds: u32,
    /// High-water mark of live bindless texture slots (0 when non-bindless).
    pub texture_slots_live: u32,
    /// Total addressable bindless texture slots for the active device.
    pub texture_slots_capacity: u32,
    /// Distinct images currently resident in the bindless heap.
    pub texture_images_resident: u32,
    /// Cumulative bindless `acquire` calls that hit the capacity ceiling and
    /// fell back to the white slot.
    pub texture_overflow: u32,
}
