//! Mesh pipeline specialization.
//!
//! Bevy's `SpecializedMeshPipelines` turns a mesh + variant key into a concrete
//! pipeline, building (and caching) one per distinct vertex layout and variant.
//! Prism's equivalent is [`MeshPipelineSpecializer`]: the caller supplies the
//! variant part of the key ([`MeshPipelineKey`]) and the mesh's vertex layouts,
//! and the specializer composes the full [`PipelineKey`] — including the vertex
//! layout hash the base variant key omits — and specializes through the shared
//! [`crate::PipelineCache`], so a layout/variant pair compiles once and is
//! reused.

use prism_render_driver::{
    PrimitiveTopology, RenderDevice, RenderPipelineDescriptor, RenderPipelineId, TextureFormat,
    VertexBufferLayout,
};
use prism_render_material::{LobeMask, SpecializationId};

use crate::pipeline_cache::{
    hash_vertex_layouts, PipelineCache, PipelineCompiler, PipelineKey, PipelineResolution,
};

/// The variant part of a mesh pipeline key: everything but the vertex layout.
///
/// The vertex layout is supplied separately to
/// [`MeshPipelineSpecializer::specialize`] because it comes from the mesh, while
/// the variant (material class, target formats, samples, topology, shader) is
/// the renderer's. Two meshes with the same variant but different layouts
/// specialize to different pipelines.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshPipelineKey {
    /// The material permutation / bin class.
    pub class: u32,
    /// The single color-target format, or `None` for a depth-only pipeline.
    pub color_format: Option<TextureFormat>,
    /// The depth/stencil format, or `None` when there is no depth state.
    pub depth_format: Option<TextureFormat>,
    /// The MSAA sample count (`1` for no multisampling).
    pub sample_count: u32,
    /// The primitive topology.
    pub topology: PrimitiveTopology,
    /// A stable identity for the shader module and its entry points.
    pub shader_id: u64,
}

impl MeshPipelineKey {
    /// The full pipeline key for a mesh with `layout`.
    #[must_use]
    pub fn with_layout(self, layout: &[VertexBufferLayout]) -> PipelineKey {
        PipelineKey::new(
            self.class,
            self.color_format,
            self.depth_format,
            self.sample_count,
            hash_vertex_layouts(layout),
            self.topology,
            self.shader_id,
        )
    }
}

/// The full-lobe über fallback for a cold specialization.
///
/// The fallback keeps the illumination and render-class axes but expands the
/// live lobe set to every schema lobe. A provider can therefore select the
/// all-lobe über shader from the fallback key while the exact specialization
/// compiles in the background. `None` means the compact class did not decode to
/// a legal `SpecializationId`, in which case the caller keeps the synchronous
/// path.
#[must_use]
pub fn full_lobe_fallback(variant: MeshPipelineKey) -> Option<MeshPipelineKey> {
    let specialization = SpecializationId::from_bin_class(variant.class)?;
    let illumination = specialization.illumination()?;
    let full = SpecializationId::new(
        illumination,
        LobeMask::all().0,
        specialization.render_class_bits(),
    );
    Some(MeshPipelineKey {
        class: full.bin_class(),
        shader_id: full.0,
        ..variant
    })
}

/// Specializes and caches a pipeline per `(variant, vertex layout)` pair.
#[derive(Default, prism_ecs::prelude::Resource)]
pub struct MeshPipelineSpecializer {
    cache: PipelineCache,
}

/// Supplies the pipeline descriptor for a mesh variant.
///
/// The queue owns *which* variant to specialize; the descriptor (shader module,
/// entry points, target state) is the renderer's, so it is injected through this
/// seam. A real renderer implements it with its mesh/PBR pipelines; a test
/// implements it with a canned descriptor.
pub trait MeshPipelineProvider {
    /// The pipeline descriptor for `variant`.
    ///
    /// The descriptor's *vertex stage* is not the provider's to decide: the
    /// specializer fills it from the mesh layout it keyed the pipeline on (see
    /// [`MeshPipelineSpecializer::specialize`]), so a provider cannot
    /// accidentally build a pipeline whose vertex stage disagrees with its own
    /// cache key.
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor;
}

/// Fills `descriptor`'s vertex stage from the mesh's layout.
///
/// A specialized pipeline is cached under `(variant, layout)`, so the pipeline
/// that is built has to *be* that pair: whatever buffers a provider put in the
/// descriptor are replaced by the layout the key hashed. Without this the cache
/// would claim one thing and the GPU would be configured with another.
#[must_use]
pub fn with_mesh_layout(
    mut descriptor: RenderPipelineDescriptor,
    layout: &[VertexBufferLayout],
) -> RenderPipelineDescriptor {
    descriptor.vertex.buffers = layout.to_vec();
    descriptor
}

impl MeshPipelineSpecializer {
    /// An empty specializer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: PipelineCache::new(),
        }
    }

    /// A specializer whose cache compiles cold permutations in the background.
    #[must_use]
    pub fn with_worker<D>(device: alloc::sync::Arc<D>) -> Self
    where
        D: RenderDevice + Send + Sync + 'static,
    {
        Self {
            cache: PipelineCache::with_worker(device),
        }
    }

    /// A specializer with an injected background compiler.
    #[must_use]
    pub fn with_compiler(compiler: alloc::sync::Arc<dyn PipelineCompiler>) -> Self {
        Self {
            cache: PipelineCache::with_compiler(compiler),
        }
    }

    /// Drains completed background compiles. Runs once per frame before the
    /// queue specializes.
    pub fn pump<D: RenderDevice>(&mut self, device: &D) {
        self.cache.pump(device);
    }

    /// The number of distinct specialized pipelines.
    #[must_use]
    pub fn pipeline_count(&self) -> usize {
        self.cache.len()
    }

    /// The number of specializations currently compiling in the background.
    #[must_use]
    pub fn pending_pipeline_count(&self) -> usize {
        self.cache.pending_len()
    }

    /// Returns the pipeline for `(variant, layout)`, compiling `descriptor` on
    /// first use.
    ///
    /// `descriptor` supplies the shader, target state, and fixed-function state;
    /// its vertex stage is replaced by [`with_mesh_layout`], because the mesh
    /// owns the vertex layout and the cache key hashes it.
    pub fn specialize<D: RenderDevice>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
        descriptor: RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        let descriptor = with_mesh_layout(descriptor, layout);
        self.cache
            .get_or_create(device, variant.with_layout(layout), &descriptor)
    }

    /// Specializes `variant`, returning a ready pipeline or the full-lobe über
    /// fallback while the exact permutation compiles in the background.
    ///
    /// `build` supplies both the requested and fallback descriptors; providers
    /// can therefore return different shader entry points for the fallback.
    pub fn specialize_with<D, F>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
        build: F,
    ) -> PipelineResolution
    where
        D: RenderDevice,
        F: Fn(&MeshPipelineKey) -> RenderPipelineDescriptor,
    {
        let key = variant.with_layout(layout);
        let Some(fallback_variant) = full_lobe_fallback(variant) else {
            let descriptor = with_mesh_layout(build(&variant), layout);
            return PipelineResolution::Ready(self.cache.get_or_create(device, key, &descriptor));
        };
        let descriptor = with_mesh_layout(build(&variant), layout);
        let fallback_descriptor = with_mesh_layout(build(&fallback_variant), layout);
        self.cache.resolve_or_request(
            device,
            key,
            descriptor,
            fallback_variant.with_layout(layout),
            fallback_descriptor,
        )
    }

    /// Specializes `variant` through a [`MeshPipelineProvider`], with the
    /// full-lobe über fallback path enabled.
    pub fn specialize_with_provider<D, P>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
        provider: &P,
    ) -> PipelineResolution
    where
        D: RenderDevice,
        P: MeshPipelineProvider,
    {
        self.specialize_with(device, variant, layout, |variant| {
            provider.descriptor(variant)
        })
    }

    /// Queues a batch of hot mesh specializations for background compilation.
    ///
    /// Archetype/CI warmup supplies the variants and their layouts; `build`
    /// supplies the provider descriptors. Invalid compact classes stay on the
    /// synchronous path because they have no legal full-lobe fallback.
    pub fn prewarm<D, I, F>(&mut self, device: &D, requests: I, build: F)
    where
        D: RenderDevice,
        I: IntoIterator<Item = (MeshPipelineKey, Vec<VertexBufferLayout>)>,
        F: Fn(&MeshPipelineKey) -> RenderPipelineDescriptor,
    {
        let mut warm = Vec::new();
        for (variant, layout) in requests {
            let Some(fallback_variant) = full_lobe_fallback(variant) else {
                let descriptor = with_mesh_layout(build(&variant), &layout);
                let _ = self
                    .cache
                    .get_or_create(device, variant.with_layout(&layout), &descriptor);
                continue;
            };
            warm.push((
                variant.with_layout(&layout),
                with_mesh_layout(build(&variant), &layout),
                fallback_variant.with_layout(&layout),
                with_mesh_layout(build(&fallback_variant), &layout),
            ));
        }
        self.cache.prewarm(device, warm);
    }

    /// Drops one reference to `(variant, layout)`, destroying the pipeline at
    /// zero. Returns whether it was evicted.
    pub fn release<D: RenderDevice>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
    ) -> bool {
        self.cache.release(device, &variant.with_layout(layout))
    }

    /// Destroys every specialized pipeline.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        self.cache.clear(device);
    }
}
