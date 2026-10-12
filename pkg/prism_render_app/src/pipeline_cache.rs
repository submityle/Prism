//! The render world's pipeline cache.
//!
//! Bevy's render world carries a `PipelineCache` (and `SpecializedMeshPipelines`
//! on top of it) so a mesh/variant that was compiled once is reused. Prism's
//! equivalent is a thin renderer-side cache over the RHI's
//! [`ContentCache`]: the key is the pipeline's *content* (the material
//! permutation class plus the fixed-function state that changes the PSO), and
//! the value is the RHI pipeline id. Identical keys return the same id, so a PSO
//! is built once and shared; the cache is reference-counted so a variant that
//! stops being used can be evicted (and its GPU object destroyed) without
//! disturbing a variant that is still live.
//!
//! The key must fully determine the pipeline: the caller owns that contract.
//! `get_or_create` deliberately ignores a later descriptor for an already-cached
//! key, which is what makes dedup stable.

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    CacheStats, ContentCache, RenderDevice, RenderPipelineDescriptor, RenderPipelineId,
    PrimitiveTopology, TextureFormat, VertexBufferLayout,
};

/// The content key of one render pipeline.
///
/// `class` is the material permutation / bin class (the material layer's
/// `SpecializationId::bin_class()`); the rest is the fixed-function state that
/// changes the compiled pipeline. The fields are plain integers so the key is
/// `Ord` and the cache is a deterministic `BTreeMap`.
///
/// # Coverage
///
/// Covered: the material permutation class (which folds every material-driven
/// state — blend, depth test/write, culling — that the material layer already
/// keys), the color/depth attachment formats, the sample count, the full vertex
/// layout (a hash of the strides, step modes, and attributes), the primitive
/// topology, and the shader identity.
///
/// Not covered: per-pipeline fixed-function overrides that no material class
/// keys (e.g. an explicit depth-bias). A pipeline that needs those must fold
/// them into `class` until the key grows a field for them — `get_or_create`
/// trusts the key, so an under-specified key would silently alias two
/// pipelines.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PipelineKey {
    /// The material permutation / bin class.
    pub class: u32,
    /// The single color-target format encoded as `format + 1`, or `0` for a
    /// depth-only pipeline (the RHI format enum is not `Ord`, so the key keeps
    /// an integer).
    color_format: u32,
    /// The depth/stencil format encoded as `format + 1`, or `0` for no depth.
    depth_format: u32,
    /// The MSAA sample count (`1` for no multisampling).
    pub sample_count: u32,
    /// A deterministic hash of the **full** vertex layout set the PSO bakes in.
    ///
    /// A single stride is not enough: two pipelines whose buffers have the same
    /// total stride but different attribute formats, offsets, step modes, or
    /// buffer counts compile to different vertex stages. Use
    /// [`hash_vertex_layouts`] to derive this.
    vertex_layout_hash: u64,
    /// The primitive topology, encoded as the RHI enum's discriminant.
    topology: u32,
    /// A stable identity for the shader module **and its entry points**.
    ///
    /// The RHI's `ShaderModuleId` is backend-assigned, so it is not a portable
    /// identity; the caller supplies a content id (e.g. from `prism_shader`'s
    /// permutation) that is equal exactly when two pipelines compile the same
    /// shader program.
    shader_id: u64,
}

impl PipelineKey {
    /// A key for a single-color-target pipeline.
    #[must_use]
    pub const fn new(
        class: u32,
        color_format: Option<TextureFormat>,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        vertex_layout_hash: u64,
        topology: PrimitiveTopology,
        shader_id: u64,
    ) -> Self {
        Self {
            class,
            color_format: encode_format(color_format),
            depth_format: encode_format(depth_format),
            sample_count,
            vertex_layout_hash,
            topology: topology as u32,
            shader_id,
        }
    }

}

/// A deterministic 64-bit hash of a vertex layout set (FNV-1a).
///
/// Encodes the buffer count, then each buffer's stride, step mode, and every
/// attribute's format, offset, and shader location, so two layout sets hash
/// equal exactly when they describe the same vertex stage. Used to fold the
/// layout into a [`PipelineKey`].
#[must_use]
pub fn hash_vertex_layouts(layouts: &[VertexBufferLayout]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn write(hash: &mut u64, bytes: u64) {
        for byte in bytes.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    let mut hash = FNV_OFFSET;
    write(&mut hash, layouts.len() as u64);
    for layout in layouts {
        write(&mut hash, layout.array_stride);
        write(
            &mut hash,
            match layout.step_mode {
                prism_render_driver::VertexStepMode::Vertex => 0,
                prism_render_driver::VertexStepMode::Instance => 1,
            },
        );
        write(&mut hash, layout.attributes.len() as u64);
        for attribute in &layout.attributes {
            write(&mut hash, attribute.format as u64);
            write(&mut hash, attribute.offset);
            write(&mut hash, u64::from(attribute.shader_location));
        }
    }
    hash
}


/// Encodes an optional format as `format + 1` (`0` is the "none" sentinel), so
/// the key is an `Ord` integer even though `TextureFormat` is not.
const fn encode_format(format: Option<TextureFormat>) -> u32 {
    match format {
        None => 0,
        Some(format) => format as u32 + 1,
    }
}

/// A content-addressed, reference-counted cache of compiled render pipelines.
#[derive(Resource, Default)]
pub struct PipelineCache {
    cache: ContentCache<PipelineKey, RenderPipelineId>,
}

impl PipelineCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: ContentCache::new(),
        }
    }

    /// The number of distinct cached pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache holds no pipelines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The hit/miss counters and live entry count.
    #[must_use]
    pub fn stats(&mut self) -> CacheStats {
        self.cache.stats()
    }

    /// Returns the id for `key`, compiling `descriptor` on first use.
    ///
    /// A later call with the same key returns the cached id and ignores the new
    /// descriptor: the key is the pipeline's identity, so it must determine the
    /// descriptor exactly.
    pub fn get_or_create<D: RenderDevice>(
        &mut self,
        device: &D,
        key: PipelineKey,
        descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        self.cache
            .get_or_create(key, || device.create_render_pipeline(descriptor))
    }

    /// Drops one reference to `key`, destroying the pipeline when it reaches
    /// zero. Returns whether the pipeline was evicted.
    pub fn release<D: RenderDevice>(&mut self, device: &D, key: &PipelineKey) -> bool {
        match self.cache.release(key) {
            Some(id) => {
                device.destroy_render_pipeline(id);
                true
            }
            None => false,
        }
    }

    /// Destroys every cached pipeline regardless of reference count, for device
    /// loss or teardown.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        self.cache
            .drain_all(|id| device.destroy_render_pipeline(id));
    }
}
