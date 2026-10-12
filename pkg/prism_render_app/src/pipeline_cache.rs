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
    TextureFormat,
};

/// The content key of one render pipeline.
///
/// `class` is the material permutation / bin class (the material layer's
/// `SpecializationId::bin_class()`); the rest is the fixed-function state that
/// changes the compiled pipeline. The fields are plain integers so the key is
/// `Ord` and the cache is a deterministic `BTreeMap`.
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
    /// The total vertex stride, which fixes the vertex layout the PSO bakes in.
    pub vertex_stride: u32,
}

impl PipelineKey {
    /// A key for a single-color-target pipeline.
    #[must_use]
    pub const fn new(
        class: u32,
        color_format: Option<TextureFormat>,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        vertex_stride: u32,
    ) -> Self {
        Self {
            class,
            color_format: encode_format(color_format),
            depth_format: encode_format(depth_format),
            sample_count,
            vertex_stride,
        }
    }

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
