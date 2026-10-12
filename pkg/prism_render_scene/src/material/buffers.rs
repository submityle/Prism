use super::parameter_heap::ParameterHeap;
use super::runtime::RenderMaterialRegistry;
use alloc::sync::Arc;
use alloc::vec::Vec;
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferUsages};
use prism_render_material::{
    fallback_material_header, inactive_material_header, GpuMaterialHeader, GpuMaterialTexture,
    GpuSurfaceParameters, LobeMask, SurfaceParameterBlock,
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialHeaderRow(pub GpuMaterialHeader);
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialTextureRow(pub GpuMaterialTexture);
bevy_render::impl_atomic_pod!(MaterialHeaderRow, MaterialHeaderRowBlob);
bevy_render::impl_atomic_pod!(MaterialTextureRow, MaterialTextureRowBlob);

/// GPU-visible material tables (unified Material ABI v5).
///
/// * `headers` — one fixed-stride [`GpuMaterialHeader`] per material index.
/// * `parameters` — a flat `u32` **word heap** holding each material's packed
///   surface block (über-BSDF core + only the live lobes, see
///   [`SurfaceParameterBlock`]). A material's header carries the word
///   `parameter_offset` and byte `parameter_size` into this heap. Variable-
///   length blocks are sub-allocated by [`ParameterHeap`]; `allocs` remembers
///   each index's current `(word_offset, word_len)` so a re-published material
///   can free its old run before packing a new one (its lobe set may change).
/// * `textures` — a flat **row heap** of [`GpuMaterialTexture`] records, one
///   row per live texture reference. ABI v5 deleted the old
///   `MAX_MATERIAL_TEXTURES` fixed-stride window (§8.2): rows are sub-allocated
///   by a second [`ParameterHeap`] exactly like parameter words, and the header
///   carries the run's `texture_offset` / `texture_count`. A material may
///   therefore reference an unbounded number of bindless texture slots; the
///   shaders read `material_textures[texture_offset .. +texture_count]`.
#[derive(Resource)]
pub(crate) struct MaterialGpuBuffers {
    pub headers: AtomicSparseBufferVec<MaterialHeaderRow>,
    pub parameters: AtomicSparseBufferVec<u32>,
    pub textures: AtomicSparseBufferVec<MaterialTextureRow>,
    /// Address-space allocator over the `parameters` word heap.
    heap: ParameterHeap,
    /// Address-space allocator over the `textures` row heap (ABI v5, §8.2).
    texture_heap: ParameterHeap,
    /// Per-material-index current heap allocation `(word_offset, word_len)`, or
    /// `None` for indices with no live parameter block (retired/inactive).
    allocs: Vec<Option<(u32, u32)>>,
    /// Per-material-index current texture-row run `(row_offset, row_count)`, or
    /// `None` for indices with no live texture rows (retired/inactive).
    texture_allocs: Vec<Option<(u32, u32)>>,
}

impl FromWorld for MaterialGpuBuffers {
    fn from_world(_: &mut World) -> Self {
        Self {
            headers: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material headers"),
            ),
            parameters: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material parameters"),
            ),
            textures: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material textures"),
            ),
            heap: ParameterHeap::new(),
            texture_heap: ParameterHeap::new(),
            allocs: Vec::new(),
            texture_allocs: Vec::new(),
        }
    }
}

impl MaterialGpuBuffers {
    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.headers.buffer()?,
            self.parameters.buffer()?,
            self.textures.buffer()?,
        ))
    }

    /// Free the heap run currently recorded for `index`, if any, and clear its
    /// slot. Returns the freed word count so callers can keep buffer stats.
    fn release_parameters(&mut self, index: u32) {
        if let Some((offset, len)) = self.allocs.get(index as usize).copied().flatten() {
            self.heap.free(offset, len);
            self.allocs[index as usize] = None;
        }
    }

    /// Pack `block` into a freshly allocated heap run for `index`, write the
    /// words into the storage vec, record the allocation, and return the word
    /// offset the header must point at. The caller must have released any prior
    /// allocation for `index` first.
    /// Free the texture-row run currently recorded for `index`, if any, and
    /// clear its slot.
    fn release_textures(&mut self, index: u32) {
        if let Some((offset, len)) = self.texture_allocs.get(index as usize).copied().flatten() {
            self.texture_heap.free(offset, len);
            self.texture_allocs[index as usize] = None;
        }
    }

    /// Pack a material's texture rows into a freshly allocated run of the row
    /// heap and return the row offset the header must point at. The caller must
    /// have released any prior allocation for `index` first.
    fn store_textures(&mut self, index: u32, textures: &[GpuMaterialTexture]) -> u32 {
        let len = textures.len() as u32;
        let offset = self.texture_heap.alloc(len);
        for (i, texture) in textures.iter().enumerate() {
            self.textures
                .grow_and_set(offset + i as u32, MaterialTextureRow(*texture));
        }
        if self.texture_allocs.len() <= index as usize {
            self.texture_allocs.resize(index as usize + 1, None);
        }
        self.texture_allocs[index as usize] = Some((offset, len));
        offset
    }

    fn store_parameters(&mut self, index: u32, block: &SurfaceParameterBlock) -> u32 {
        let words = block.pack();
        let len = words.len() as u32;
        let offset = self.heap.alloc(len);
        for (i, &word) in words.iter().enumerate() {
            self.parameters.grow_and_set(offset + i as u32, word);
        }
        if self.allocs.len() <= index as usize {
            self.allocs.resize(index as usize + 1, None);
        }
        self.allocs[index as usize] = Some((offset, len));
        offset
    }

    pub fn apply_dirty(&mut self, runtime: &mut RenderMaterialRegistry) -> (u32, u64) {
        let dirty = runtime.registry.take_dirty();
        let epoch = runtime.registry.snapshot().epoch;
        let mut parameter_bytes: u64 = 0;
        let mut texture_bytes: u64 = 0;
        for &index in &dirty {
            let generation = runtime.registry.generation_at(index).unwrap_or(0);
            // Every path first reclaims the index's previous heap runs: a
            // re-published material may have grown/shrunk its lobe or texture
            // set, and a retired one must return its words/rows to the free list.
            self.release_parameters(index);
            self.release_textures(index);
            if let Some(record) = runtime.registry.record_at(index) {
                let block = record.packed_parameters();
                let offset = self.store_parameters(index, &block);
                let texture_offset = self.store_textures(index, &record.textures);
                parameter_bytes += block.packed_size_bytes() as u64;
                texture_bytes += size_of_val(record.textures.as_slice()) as u64;
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(record.header(offset, texture_offset, epoch)),
                );
            } else if index == 0 {
                // Slot zero is the permanent principled fallback; it always
                // carries a live (core-only) parameter block.
                let block = SurfaceParameterBlock::from_full(
                    &GpuSurfaceParameters::default(),
                    LobeMask::default(),
                );
                let offset = self.store_parameters(index, &block);
                parameter_bytes += block.packed_size_bytes() as u64;
                let mut header = fallback_material_header(epoch);
                header.parameter_offset = offset;
                self.headers.grow_and_set(index, MaterialHeaderRow(header));
            } else {
                // Retired/inactive: `is_active == 0`, so shaders bail before
                // touching parameters. We keep no heap run for it (already freed
                // above) and leave `parameter_offset` at its default.
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(inactive_material_header(generation)),
                );
            }
        }
        let header_bytes = dirty.len() as u64 * size_of::<GpuMaterialHeader>() as u64;
        (
            dirty.len() as u32,
            header_bytes + parameter_bytes + texture_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::world::{FromWorld, World};
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{Illumination, MaterialRenderClass};

    use super::*;

    #[test]
    fn dirty_rows_publish_fallback_live_and_retired_generations() {
        let mut world = World::new();
        let mut buffers = MaterialGpuBuffers::from_world(&mut world);
        let mut runtime = RenderMaterialRegistry::default();
        let handle = runtime.registry.allocate().unwrap();
        runtime
            .registry
            .publish(prism_render_material::MaterialRecord {
                handle,
                revision: 1,
                domain: prism_render_material::MaterialDomain::Surface,
                render_class: MaterialRenderClass::Opaque,
                illumination: Illumination::Lit,
                features: Default::default(),
                closure_mask: 1,
                surface: GpuSurfaceParameters::default(),
                textures: vec![],
                custom_program: None,
            })
            .unwrap();

        assert_eq!(buffers.apply_dirty(&mut runtime).0, 2);
        runtime
            .registry
            .retire(handle, GpuCompletionValue(1))
            .unwrap();
        assert_eq!(buffers.apply_dirty(&mut runtime).0, 1);
        assert_eq!(
            runtime.registry.generation_at(handle.index),
            Some(handle.generation)
        );
    }

    #[test]
    fn retiring_a_material_returns_its_words_to_the_heap() {
        let mut world = World::new();
        let mut buffers = MaterialGpuBuffers::from_world(&mut world);
        let mut runtime = RenderMaterialRegistry::default();
        let handle = runtime.registry.allocate().unwrap();
        runtime
            .registry
            .publish(prism_render_material::MaterialRecord {
                handle,
                revision: 1,
                domain: prism_render_material::MaterialDomain::Surface,
                render_class: MaterialRenderClass::Opaque,
                illumination: Illumination::Lit,
                features: Default::default(),
                closure_mask: 1,
                surface: GpuSurfaceParameters::default(),
                textures: vec![],
                custom_program: None,
            })
            .unwrap();
        buffers.apply_dirty(&mut runtime);
        // Slot 0 fallback + slot 1 material each packed a core-only block whose
        // word count is the current `SURFACE_CORE_WORDS` (20 since the base,
        // specular, coat and fuzz groups all grew the core).
        assert_eq!(buffers.allocs[0], Some((0, 20)));
        assert_eq!(buffers.allocs[1], Some((20, 20)));

        runtime
            .registry
            .retire(handle, GpuCompletionValue(1))
            .unwrap();
        buffers.apply_dirty(&mut runtime);
        // The retired material's run is freed; the fallback keeps its block.
        assert_eq!(buffers.allocs[1], None);
        assert_eq!(buffers.allocs[0], Some((0, 20)));
    }
}
