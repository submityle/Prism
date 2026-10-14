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

use prism_render_architecture::shader_package::{
    ExecutionFamily, ModuleId, PermutationIndex, PermutationSpace,
};
use prism_render_driver::{
    PrimitiveTopology, RenderDevice, RenderPipelineDescriptor, RenderPipelineId, TextureFormat,
    VertexBufferLayout,
};
use std::collections::{BTreeMap, HashMap};

use crate::pipeline_cache::{PipelineCache, PipelineCompiler, PipelineKey, PipelineResolution};

/// The variant part of a mesh pipeline key: everything but the vertex layout.
///
/// The vertex layout is supplied separately to
/// [`MeshPipelineSpecializer::specialize`] because it comes from the mesh, while
/// the variant (material class, target formats, samples, topology, shader) is
/// the renderer's. Two meshes with the same variant but different layouts
/// specialize to different pipelines.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshPipelineKey {
    /// Shared execution path, not a packed material-style axis.
    pub execution_family: ExecutionFamily,
    /// Driver-local code, entry point and binding ABI identity.
    pub response_module_id: ModuleId,
    /// Checked dense index in the module's declared permutation space.
    pub permutation_index: PermutationIndex,
    /// Explicit module-declared fallback. Never inferred from a lobe bitmask.
    pub fallback_permutation: Option<PermutationIndex>,
    /// The single color-target format, or `None` for a depth-only pipeline.
    pub color_format: Option<TextureFormat>,
    /// The depth/stencil format, or `None` when there is no depth state.
    pub depth_format: Option<TextureFormat>,
    /// The MSAA sample count (`1` for no multisampling).
    pub sample_count: u32,
    /// The primitive topology.
    pub topology: PrimitiveTopology,
}

impl MeshPipelineKey {
    fn request_id(self, layout: &[VertexBufferLayout]) -> MeshSpecializationId {
        MeshSpecializationId {
            variant: self,
            layout: MeshLayoutId::from_layouts(layout),
        }
    }

    fn key_from_descriptor(self, descriptor: &RenderPipelineDescriptor) -> PipelineKey {
        PipelineKey::from_descriptor(
            self.execution_family,
            self.response_module_id,
            self.permutation_index,
            descriptor,
        )
    }
}

/// Exact in-memory identity of a mesh specialization request.
///
/// The index maps the public `(variant, layout)` release input back to the
/// complete descriptor-derived cache key, including provider-owned state.
#[derive(Clone, PartialEq, Eq, Hash)]
struct MeshSpecializationId {
    variant: MeshPipelineKey,
    layout: MeshLayoutId,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MeshLayoutId(Vec<MeshBufferId>);

#[derive(Clone, PartialEq, Eq, Hash)]
struct MeshBufferId {
    array_stride: u64,
    step_mode: u32,
    attributes: Vec<MeshAttributeId>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MeshAttributeId {
    format: u32,
    offset: u64,
    shader_location: u32,
}

impl MeshLayoutId {
    fn from_layouts(layouts: &[VertexBufferLayout]) -> Self {
        Self(
            layouts
                .iter()
                .map(|layout| MeshBufferId {
                    array_stride: layout.array_stride,
                    step_mode: match layout.step_mode {
                        prism_render_driver::VertexStepMode::Vertex => 0,
                        prism_render_driver::VertexStepMode::Instance => 1,
                    },
                    attributes: layout
                        .attributes
                        .iter()
                        .map(|attribute| MeshAttributeId {
                            format: attribute.format as u32,
                            offset: attribute.offset,
                            shader_location: attribute.shader_location,
                        })
                        .collect(),
                })
                .collect(),
        )
    }
}

/// The full-lobe über fallback for a cold specialization.
///
/// The module declaration supplies a checked dense fallback index. A mask is
/// data/layout information and cannot be decoded from a permutation index.
#[must_use]
pub fn full_lobe_fallback(variant: MeshPipelineKey) -> Option<MeshPipelineKey> {
    Some(MeshPipelineKey {
        permutation_index: variant.fallback_permutation?,
        ..variant
    })
}

/// Specializes and caches a pipeline per `(variant, vertex layout)` pair.
#[derive(Default, prism_ecs::prelude::Resource)]
pub struct MeshPipelineSpecializer {
    cache: PipelineCache,
    key_index: HashMap<MeshSpecializationId, PipelineKey>,
    module_spaces: BTreeMap<ModuleId, (PermutationSpace, Option<PermutationIndex>)>,
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
            key_index: HashMap::new(),
            module_spaces: BTreeMap::new(),
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
            key_index: HashMap::new(),
            module_spaces: BTreeMap::new(),
        }
    }

    /// A specializer with an injected background compiler.
    #[must_use]
    pub fn with_compiler(compiler: alloc::sync::Arc<dyn PipelineCompiler>) -> Self {
        Self {
            cache: PipelineCache::with_compiler(compiler),
            key_index: HashMap::new(),
            module_spaces: BTreeMap::new(),
        }
    }

    /// Registers the declaration that owns a module's dense permutation space.
    ///
    /// The declaration is optional for the legacy spine/test provider, but once
    /// present every exact and fallback request is checked before it reaches
    /// the cache. A raw `u64` can therefore never silently become a different
    /// module permutation.
    pub fn register_response_module(
        &mut self,
        module: ModuleId,
        permutations: PermutationSpace,
        fallback: Option<PermutationIndex>,
    ) {
        permutations
            .checked_total()
            .expect("response module permutation declaration overflows");
        if let Some(index) = fallback {
            permutations
                .check_index(index.get())
                .expect("response module fallback is outside its permutation space");
        }
        if let Some((existing, existing_fallback)) = self.module_spaces.get(&module) {
            assert_eq!(
                existing, &permutations,
                "response module declaration changed"
            );
            assert_eq!(
                existing_fallback, &fallback,
                "response module fallback changed"
            );
            return;
        }
        self.module_spaces.insert(module, (permutations, fallback));
    }

    /// Registers all module declarations from an app-owned response cache.
    pub fn sync_response_modules(&mut self, modules: &crate::mesh_pipelines::ResponseModuleCache) {
        for module in modules.iter() {
            self.register_response_module(
                module.module_id,
                module.permutations.clone(),
                module.full_lobe_fallback,
            );
        }
    }

    fn assert_valid_variant(&self, variant: &MeshPipelineKey) {
        let Some((space, declared_fallback)) = self.module_spaces.get(&variant.response_module_id)
        else {
            return;
        };
        space
            .check_index(variant.permutation_index.get())
            .unwrap_or_else(|error| {
                panic!(
                    "invalid dense permutation for module {:?}: {error:?}",
                    variant.response_module_id
                )
            });
        assert_eq!(
            variant.fallback_permutation, *declared_fallback,
            "mesh variant fallback does not match the response module declaration"
        );
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
        self.assert_valid_variant(&variant);
        let descriptor = with_mesh_layout(descriptor, layout);
        let key = variant.key_from_descriptor(&descriptor);
        self.key_index
            .insert(variant.request_id(layout), key.clone());
        self.cache.get_or_create(device, key, &descriptor)
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
        self.assert_valid_variant(&variant);
        let Some(fallback_variant) = full_lobe_fallback(variant) else {
            let descriptor = with_mesh_layout(build(&variant), layout);
            let key = variant.key_from_descriptor(&descriptor);
            self.key_index
                .insert(variant.request_id(layout), key.clone());
            return PipelineResolution::Ready(self.cache.get_or_create(device, key, &descriptor));
        };
        let descriptor = with_mesh_layout(build(&variant), layout);
        let fallback_descriptor = with_mesh_layout(build(&fallback_variant), layout);
        let key = variant.key_from_descriptor(&descriptor);
        let fallback_key = fallback_variant.key_from_descriptor(&fallback_descriptor);
        self.key_index
            .insert(variant.request_id(layout), key.clone());
        self.key_index
            .insert(fallback_variant.request_id(layout), fallback_key.clone());
        self.cache
            .resolve_or_request(device, key, descriptor, fallback_key, fallback_descriptor)
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
            self.assert_valid_variant(&variant);
            let Some(fallback_variant) = full_lobe_fallback(variant) else {
                let descriptor = with_mesh_layout(build(&variant), &layout);
                let key = variant.key_from_descriptor(&descriptor);
                self.key_index
                    .insert(variant.request_id(&layout), key.clone());
                let _ = self.cache.get_or_create(device, key, &descriptor);
                continue;
            };
            let descriptor = with_mesh_layout(build(&variant), &layout);
            let fallback_descriptor = with_mesh_layout(build(&fallback_variant), &layout);
            let key = variant.key_from_descriptor(&descriptor);
            let fallback_key = fallback_variant.key_from_descriptor(&fallback_descriptor);
            self.key_index
                .insert(variant.request_id(&layout), key.clone());
            self.key_index
                .insert(fallback_variant.request_id(&layout), fallback_key.clone());
            warm.push((key, descriptor, fallback_key, fallback_descriptor));
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
        let request = variant.request_id(layout);
        let Some(key) = self.key_index.get(&request) else {
            return false;
        };
        let released = self.cache.release(device, key);
        if released {
            self.key_index.remove(&request);
        }
        released
    }

    /// Destroys every specialized pipeline.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        self.cache.clear(device);
        self.key_index.clear();
    }
}
