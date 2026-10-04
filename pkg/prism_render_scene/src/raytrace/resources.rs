//! Device buffer resources, host-side packing and hit decoding for the
//! production ray-traversal service.
//!
//! The packed acceleration-structure words come straight from the golden,
//! float-audited `prism_render_architecture::ray_scene` layout
//! (`GpuBvhBuffers` / `GpuTlasBuffers` / `GpuBlasPool`); this module only uploads
//! those `Vec<u32>` buffers to the `GPU` and lays out the per-dispatch ray,
//! hit, footprint and `UNIFORM` buffers against the frozen [`super::abi`]
//! strides. Nothing here reinterprets the layout — the strides, the ray/hit
//! word packing and the `+inf` / `u32::MAX` miss sentinels are all pinned to the
//! `ABI` so a drift fails the shared contract tests rather than corrupting a
//! dispatch.

use bevy_render::render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages};
use bevy_render::renderer::RenderDevice;

use prism_render_architecture::ray_scene::{GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, Ray};

use super::abi::{
    GpuFootprintParams, GpuRayTraverseParams, FOOTPRINT_RESULT_WORDS, FOOTPRINT_WORDS, HIT_WORDS,
    RAY_WORDS, TLAS_HIT_WORDS,
};

/// A single-`BLAS` hit decoded from the `ray_traverse` kernel's
/// [`HIT_WORDS`]-stride output buffer.
///
/// A miss carries `primitive == u32::MAX` ([`super::abi::MISS_PRIMITIVE`]) and a
/// `+inf` `t`, exactly as the `CPU` golden `GpuBvhBuffers::closest_hit` reports
/// `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuRayHit {
    /// Ray parameter `t` at the intersection (`+inf` on a miss).
    pub(crate) t: f32,
    /// First barycentric coordinate `u` of the hit point.
    pub(crate) u: f32,
    /// Second barycentric coordinate `v` of the hit point.
    pub(crate) v: f32,
    /// Stable primitive id of the hit triangle (`u32::MAX` on a miss).
    pub(crate) primitive: u32,
}

/// A top-level hit decoded from the `tlas_traverse` kernel's
/// [`TLAS_HIT_WORDS`]-stride output buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuTlasHit {
    /// Ray parameter `t` at the intersection (`+inf` on a miss).
    pub(crate) t: f32,
    /// First barycentric coordinate `u` of the hit point.
    pub(crate) u: f32,
    /// Second barycentric coordinate `v` of the hit point.
    pub(crate) v: f32,
    /// Stable primitive id of the hit triangle inside its `BLAS` (`u32::MAX` on
    /// a miss).
    pub(crate) primitive: u32,
    /// Stable instance id carried by the hit `TLAS` instance.
    pub(crate) instance_id: u32,
    /// Packed-order index of the hit instance in the `TLAS` instance buffer.
    pub(crate) instance_index: u32,
}

/// One ray-cone footprint request the `ray_footprint` kernel consumes, in the
/// kernel's native field order.
///
/// The three cone slopes mirror the golden
/// `prism_render_architecture::ray_scene::RayFootprint` inputs and the texel
/// size is the per-surface world size the mip math divides by; all four are
/// uploaded as their raw `f32` bit patterns and sanitized on read exactly as
/// the `CPU` golden `RayFootprint` sanitizes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FootprintRequest {
    /// Ray-cone width at the primary hit.
    pub(crate) cone_width: f32,
    /// Ray-cone spread angle, stored as the per-unit-distance slope.
    pub(crate) cone_spread_angle: f32,
    /// Distance along the ray to the shaded hit.
    pub(crate) hit_distance: f32,
    /// World-space size of one texel on the shaded surface.
    pub(crate) texel_world_size: f32,
}

/// A footprint result decoded from the `ray_footprint` kernel's
/// [`FOOTPRINT_RESULT_WORDS`]-stride output buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuFootprintResult {
    /// Projected footprint width in world units at the hit.
    pub(crate) projected_width: f32,
    /// Footprint width expressed in texels (`projected_width / texel`).
    pub(crate) texel_span: f32,
    /// Continuous mip level (`log2` of the texel span, clamped to `max_mip`).
    pub(crate) mip_level: f32,
    /// Discrete mip bucket (the floor of the continuous `mip_level`).
    pub(crate) mip_floor: u32,
}

/// Packs a ray batch into the `ray_traverse` / `tlas_traverse` kernels'
/// [`RAY_WORDS`]-stride `u32` buffer.
///
/// Each ray contributes `origin.xyz`, the clamped `t_min`, `dir.xyz` and the
/// clamped `t_max`, every scalar stored as its `to_bits` pattern so the device
/// recovers the identical `float32` the `CPU` golden walks see. The direction is
/// left un-normalized exactly as [`Ray::new`] stores it; the kernel derives
/// `inv_dir = 1.0 / dir` to match the golden reciprocal slab test.
pub(crate) fn pack_rays(rays: &[Ray]) -> Vec<u32> {
    let mut words = vec![0u32; rays.len() * RAY_WORDS];
    for (i, ray) in rays.iter().enumerate() {
        let base = i * RAY_WORDS;
        let o = ray.origin();
        let d = ray.direction();
        words[base] = o[0].to_bits();
        words[base + 1] = o[1].to_bits();
        words[base + 2] = o[2].to_bits();
        words[base + 3] = ray.t_min().to_bits();
        words[base + 4] = d[0].to_bits();
        words[base + 5] = d[1].to_bits();
        words[base + 6] = d[2].to_bits();
        words[base + 7] = ray.t_max().to_bits();
    }
    words
}

/// Packs a footprint batch into the `ray_footprint` kernel's
/// [`FOOTPRINT_WORDS`]-stride `u32` buffer.
pub(crate) fn pack_footprints(requests: &[FootprintRequest]) -> Vec<u32> {
    let mut words = vec![0u32; requests.len() * FOOTPRINT_WORDS];
    for (i, req) in requests.iter().enumerate() {
        let base = i * FOOTPRINT_WORDS;
        words[base] = req.cone_width.to_bits();
        words[base + 1] = req.cone_spread_angle.to_bits();
        words[base + 2] = req.hit_distance.to_bits();
        words[base + 3] = req.texel_world_size.to_bits();
    }
    words
}

/// Decodes the `ray_traverse` kernel's packed hit buffer into [`GpuRayHit`]s.
pub(crate) fn decode_bvh_hits(raw: &[u32], ray_count: usize) -> Vec<GpuRayHit> {
    let mut hits = Vec::with_capacity(ray_count);
    for i in 0..ray_count {
        let base = i * HIT_WORDS;
        hits.push(GpuRayHit {
            t: f32::from_bits(raw[base]),
            u: f32::from_bits(raw[base + 1]),
            v: f32::from_bits(raw[base + 2]),
            primitive: raw[base + 3],
        });
    }
    hits
}

/// Decodes the `tlas_traverse` kernel's packed hit buffer into [`GpuTlasHit`]s.
pub(crate) fn decode_tlas_hits(raw: &[u32], ray_count: usize) -> Vec<GpuTlasHit> {
    let mut hits = Vec::with_capacity(ray_count);
    for i in 0..ray_count {
        let base = i * TLAS_HIT_WORDS;
        hits.push(GpuTlasHit {
            t: f32::from_bits(raw[base]),
            u: f32::from_bits(raw[base + 1]),
            v: f32::from_bits(raw[base + 2]),
            primitive: raw[base + 3],
            instance_id: raw[base + 4],
            instance_index: raw[base + 5],
        });
    }
    hits
}

/// Decodes the `ray_footprint` kernel's packed result buffer into
/// [`GpuFootprintResult`]s.
pub(crate) fn decode_footprint_results(raw: &[u32], count: usize) -> Vec<GpuFootprintResult> {
    let mut results = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * FOOTPRINT_RESULT_WORDS;
        results.push(GpuFootprintResult {
            projected_width: f32::from_bits(raw[base]),
            texel_span: f32::from_bits(raw[base + 1]),
            mip_level: f32::from_bits(raw[base + 2]),
            mip_floor: raw[base + 3],
        });
    }
    results
}

/// Uploads a `u32` word slice as an immutable `STORAGE` buffer.
fn storage_buffer(device: &RenderDevice, label: &str, words: &[u32]) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(words),
        usage: BufferUsages::STORAGE,
    })
}

/// Allocates a readable `STORAGE | COPY_SRC` output buffer plus its paired
/// `MAP_READ | COPY_DST` staging buffer, both sized for `word_count` `u32`s.
fn output_and_stage(
    device: &RenderDevice,
    label: &str,
    word_count: usize,
) -> (Buffer, Buffer, u64) {
    let bytes = (word_count * size_of::<u32>()) as u64;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("rt_stage"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (output, stage, bytes)
}

/// Uploads a [`GpuRayTraverseParams`] `UNIFORM` buffer.
fn traverse_params(device: &RenderDevice, ray_count: u32, mode: u32) -> Buffer {
    let params = GpuRayTraverseParams {
        ray_count,
        mode,
        pad0: 0,
        pad1: 0,
    };
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("rt_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    })
}

/// The `GPU` buffers for one single-`BLAS` `ray_traverse` dispatch.
pub(crate) struct BvhTraversalResources {
    /// Packed `BVH` nodes bound read-only at `@binding(0)`.
    pub(crate) nodes: Buffer,
    /// Packed leaf triangles bound read-only at `@binding(1)`.
    pub(crate) triangles: Buffer,
    /// Packed rays bound read-only at `@binding(2)`.
    pub(crate) rays: Buffer,
    /// Read-write hit output bound at `@binding(3)`.
    pub(crate) hits: Buffer,
    /// Traversal parameter `UNIFORM` bound at `@binding(4)`.
    pub(crate) params: Buffer,
    /// Host-readable staging copy of [`Self::hits`].
    pub(crate) stage: Buffer,
    /// Number of rays in the batch.
    pub(crate) ray_count: u32,
    /// Byte length of the hit / staging buffers.
    pub(crate) hit_bytes: u64,
}

impl BvhTraversalResources {
    /// Uploads the packed `BVH`, the ray batch and the traversal parameters for
    /// a `mode` dispatch over `ray_count` rays.
    pub(crate) fn new(
        device: &RenderDevice,
        buffers: &GpuBvhBuffers,
        ray_words: &[u32],
        ray_count: u32,
        mode: u32,
    ) -> Self {
        let (hits, stage, hit_bytes) =
            output_and_stage(device, "rt_hits", ray_count as usize * HIT_WORDS);
        Self {
            nodes: storage_buffer(device, "rt_nodes", &buffers.nodes),
            triangles: storage_buffer(device, "rt_triangles", &buffers.triangles),
            rays: storage_buffer(device, "rt_rays", ray_words),
            hits,
            params: traverse_params(device, ray_count, mode),
            stage,
            ray_count,
            hit_bytes,
        }
    }
}

/// The `GPU` buffers for one two-level `tlas_traverse` dispatch.
pub(crate) struct TlasTraversalResources {
    /// Packed top-level `TLAS` nodes bound read-only at `@binding(0)`.
    pub(crate) tlas_nodes: Buffer,
    /// Packed instance records bound read-only at `@binding(1)`.
    pub(crate) instances: Buffer,
    /// Pooled `BLAS` nodes bound read-only at `@binding(2)`.
    pub(crate) pool_nodes: Buffer,
    /// Pooled `BLAS` triangles bound read-only at `@binding(3)`.
    pub(crate) pool_triangles: Buffer,
    /// Per-`BLAS` offset records bound read-only at `@binding(4)`.
    pub(crate) pool_offsets: Buffer,
    /// Packed rays bound read-only at `@binding(5)`.
    pub(crate) rays: Buffer,
    /// Read-write hit output bound at `@binding(6)`.
    pub(crate) hits: Buffer,
    /// Traversal parameter `UNIFORM` bound at `@binding(7)`.
    pub(crate) params: Buffer,
    /// Host-readable staging copy of [`Self::hits`].
    pub(crate) stage: Buffer,
    /// Number of rays in the batch.
    pub(crate) ray_count: u32,
    /// Byte length of the hit / staging buffers.
    pub(crate) hit_bytes: u64,
}

impl TlasTraversalResources {
    /// Uploads the packed `TLAS`, the shared `BLAS` pool, the ray batch and the
    /// traversal parameters for a `mode` dispatch over `ray_count` rays.
    pub(crate) fn new(
        device: &RenderDevice,
        tlas: &GpuTlasBuffers,
        pool: &GpuBlasPool,
        ray_words: &[u32],
        ray_count: u32,
        mode: u32,
    ) -> Self {
        let (hits, stage, hit_bytes) =
            output_and_stage(device, "tlas_hits", ray_count as usize * TLAS_HIT_WORDS);
        Self {
            tlas_nodes: storage_buffer(device, "tlas_nodes", &tlas.nodes),
            instances: storage_buffer(device, "tlas_instances", &tlas.instances),
            pool_nodes: storage_buffer(device, "tlas_pool_nodes", &pool.nodes),
            pool_triangles: storage_buffer(device, "tlas_pool_triangles", &pool.triangles),
            pool_offsets: storage_buffer(device, "tlas_pool_offsets", &pool.offsets),
            rays: storage_buffer(device, "tlas_rays", ray_words),
            hits,
            params: traverse_params(device, ray_count, mode),
            stage,
            ray_count,
            hit_bytes,
        }
    }
}

/// The `GPU` buffers for one `ray_footprint` dispatch.
pub(crate) struct FootprintResources {
    /// Packed footprint requests bound read-only at `@binding(0)`.
    pub(crate) footprints: Buffer,
    /// Read-write result output bound at `@binding(1)`.
    pub(crate) results: Buffer,
    /// Footprint parameter `UNIFORM` bound at `@binding(2)`.
    pub(crate) params: Buffer,
    /// Host-readable staging copy of [`Self::results`].
    pub(crate) stage: Buffer,
    /// Number of footprints in the batch.
    pub(crate) count: u32,
    /// Byte length of the result / staging buffers.
    pub(crate) result_bytes: u64,
}

impl FootprintResources {
    /// Uploads the packed footprint batch and its parameters for a dispatch
    /// clamped to `max_mip`.
    pub(crate) fn new(
        device: &RenderDevice,
        footprint_words: &[u32],
        count: u32,
        max_mip: u32,
    ) -> Self {
        let (results, stage, result_bytes) = output_and_stage(
            device,
            "footprint_results",
            count as usize * FOOTPRINT_RESULT_WORDS,
        );
        let params = GpuFootprintParams {
            footprint_count: count,
            max_mip,
            pad0: 0,
            pad1: 0,
        };
        let params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("footprint_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        Self {
            footprints: storage_buffer(device, "footprints", footprint_words),
            results,
            params,
            stage,
            count,
            result_bytes,
        }
    }
}
