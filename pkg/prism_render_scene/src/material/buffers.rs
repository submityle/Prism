use super::parameter_heap::ParameterHeap;
use super::runtime::RenderMaterialRegistry;
use alloc::sync::Arc;
use alloc::vec::Vec;
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferUsages};
use prism_render_material::{
    fallback_material_header, inactive_material_header, GpuMaterialHeader, GpuMaterialTexture,
    GpuSurfaceParameters, LobeMask, SurfaceParameterBlock, MAX_MATERIAL_TEXTURES,
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialHeaderRow(pub GpuMaterialHeader);
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialTextureRow(pub GpuMaterialTexture);
bevy_render::impl_atomic_pod!(MaterialHeaderRow, MaterialHeaderRowBlob);
bevy_render::impl_atomic_pod!(MaterialTextureRow, MaterialTextureRowBlob);

/// GPU-visible material tables (unified Material ABI v4).
///
/// * `headers` — one fixed-stride [`GpuMaterialHeader`] per material index.
/// * `parameters` — a flat `u32` **word heap** holding each material's packed
///   surface block (über-BSDF core + only the live lobes, see
///   [`SurfaceParameterBlock`]). A material's header carries the word
///   `parameter_offset` and byte `parameter_size` into this heap. Variable-
///   length blocks are sub-allocated by [`ParameterHeap`]; `allocs` remembers
///   each index's current `(word_offset, word_len)` so a re-published material
///   can free its old run before packing a new one (its lobe set may change).
/// * `textures` — `MAX_MATERIAL_TEXTURES` fixed-stride texture rows per index.
#[derive(Resource)]
pub(crate) struct MaterialGpuBuffers {
    pub headers: AtomicSparseBufferVec<MaterialHeaderRow>,
    pub parameters: AtomicSparseBufferVec<u32>,
    pub textures: AtomicSparseBufferVec<MaterialTextureRow>,
    /// Address-space allocator over the `parameters` word heap.
    heap: ParameterHeap,
    /// Per-material-index current heap allocation `(word_offset, word_len)`, or
    /// `None` for indices with no live parameter block (retired/inactive).
    allocs: Vec<Option<(u32, u32)>>,
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
            allocs: Vec::new(),
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
        for &index in &dirty {
            let generation = runtime.registry.generation_at(index).unwrap_or(0);
            // Every path first reclaims the index's previous heap run: a
            // re-published material may have grown/shrunk its lobe set, and a
            // retired one must return its words to the free list.
            self.release_parameters(index);
            if let Some(record) = runtime.registry.record_at(index) {
                let block = record.packed_parameters();
                let offset = self.store_parameters(index, &block);
                parameter_bytes += block.packed_size_bytes() as u64;
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(record.header(
                        offset,
                        index * MAX_MATERIAL_TEXTURES as u32,
                        // No serialized closure-graph blob buffer yet; RT/deferred
                        // consumers fall back to the packed axes until it lands.
                        0,
                        epoch,
                    )),
                );
                for (slot, texture) in record.fixed_texture_rows().into_iter().enumerate() {
                    self.textures.grow_and_set(
                        index * MAX_MATERIAL_TEXTURES as u32 + slot as u32,
                        MaterialTextureRow(texture),
                    );
                }
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
                for slot in 0..MAX_MATERIAL_TEXTURES as u32 {
                    self.textures
                        .grow_and_set(slot, MaterialTextureRow::default());
                }
            } else {
                // Retired/inactive: `is_active == 0`, so shaders bail before
                // touching parameters. We keep no heap run for it (already freed
                // above) and leave `parameter_offset` at its default.
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(inactive_material_header(generation)),
                );
                for slot in 0..MAX_MATERIAL_TEXTURES as u32 {
                    self.textures.grow_and_set(
                        index * MAX_MATERIAL_TEXTURES as u32 + slot,
                        MaterialTextureRow::default(),
                    );
                }
            }
        }
        let fixed_bytes = dirty.len() as u64
            * (size_of::<GpuMaterialHeader>()
                + MAX_MATERIAL_TEXTURES * size_of::<GpuMaterialTexture>()) as u64;
        (dirty.len() as u32, fixed_bytes + parameter_bytes)
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
        // Slot 0 fallback + slot 1 material each packed a 12-word core block.
        assert_eq!(buffers.allocs[0], Some((0, 12)));
        assert_eq!(buffers.allocs[1], Some((12, 12)));

        runtime
            .registry
            .retire(handle, GpuCompletionValue(1))
            .unwrap();
        buffers.apply_dirty(&mut runtime);
        // The retired material's run is freed; the fallback keeps its block.
        assert_eq!(buffers.allocs[1], None);
        assert_eq!(buffers.allocs[0], Some((0, 12)));
    }
}
