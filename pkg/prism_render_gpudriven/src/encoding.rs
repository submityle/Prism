//! Little-endian byte layouts shared with the WGSL shaders.
//!
//! Every struct the shaders read or write is laid out here, in one place, so
//! the host uploads and the GPU reads can never drift apart: the padded AABB
//! storage stride, the frustum uniform, the cull parameters and the packed
//! indirect-draw argument record.

use alloc::vec::Vec;

use crate::frustum::{Aabb, Frustum};

/// The storage stride of one instance AABB: `min.xyz` + pad, `max.xyz` + pad.
pub const AABB_STRIDE: u64 = 32;
/// The size of the frustum uniform: six `vec4<f32>` planes.
pub const FRUSTUM_UNIFORM_BYTES: u64 = 6 * 16;
/// The size of the cull parameter uniform (four 32-bit fields).
pub const PARAMS_BYTES: u64 = 16;

/// One packed indirect-draw argument record — the exact five 32-bit fields
/// wgpu's `DrawIndexedIndirectArgs` and the RHI's
/// `DRAW_INDEXED_INDIRECT_ARGS_SIZE` describe.
///
/// The cull pass writes it on the GPU; the draw pass reads it back as the
/// indirect arguments of an indexed draw, which is what makes the draw
/// GPU-driven: the CPU uploads only the constant fields and never the count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndirectArgs {
    /// How many indices the draw covers (the geometry's index count).
    pub index_count: u32,
    /// How many instances the draw covers; the cull pass computes this.
    pub instance_count: u32,
    /// The first index to fetch.
    pub first_index: u32,
    /// A value added to every index before fetching.
    pub base_vertex: i32,
    /// The first instance index, used as the base for per-instance data.
    pub first_instance: u32,
}

impl IndirectArgs {
    /// The packed record size in bytes.
    pub const BYTES: u64 = 20;

    /// A record from its five fields.
    #[must_use]
    pub const fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    /// The record packed little-endian, ready for an upload or an `INDIRECT`
    /// buffer.
    #[must_use]
    pub fn to_bytes(self) -> [u8; Self::BYTES as usize] {
        let mut bytes = [0u8; Self::BYTES as usize];
        bytes[0..4].copy_from_slice(&self.index_count.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.instance_count.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.first_index.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.base_vertex.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.first_instance.to_le_bytes());
        bytes
    }

    /// Decodes a record from at least its first [`Self::BYTES`] bytes.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` is shorter than a packed record.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= Self::BYTES as usize,
            "an indirect record needs {} bytes, got {}",
            Self::BYTES,
            bytes.len()
        );
        let u32_at = |offset: usize| {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        Self {
            index_count: u32_at(0),
            instance_count: u32_at(4),
            first_index: u32_at(8),
            base_vertex: u32_at(12) as i32,
            first_instance: u32_at(16),
        }
    }
}

/// The six planes packed as `vec4<f32>` uniform data: `(normal.xyz, distance)`.
#[must_use]
pub fn frustum_uniform_bytes(frustum: &Frustum) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FRUSTUM_UNIFORM_BYTES as usize);
    for plane in frustum.planes() {
        for component in plane.normal {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
        bytes.extend_from_slice(&plane.distance.to_le_bytes());
    }
    bytes
}

/// The instance AABBs packed at [`AABB_STRIDE`] bytes each.
#[must_use]
pub fn instance_storage_bytes(instances: &[Aabb]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instances.len() * AABB_STRIDE as usize);
    for aabb in instances {
        for component in aabb.min {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        for component in aabb.max {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
    }
    bytes
}

/// The cull parameter uniform: `{ count, index_count, first_index, base_vertex }`.
#[must_use]
pub fn params_uniform_bytes(
    count: u32,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(PARAMS_BYTES as usize);
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&index_count.to_le_bytes());
    bytes.extend_from_slice(&first_index.to_le_bytes());
    bytes.extend_from_slice(&base_vertex.to_le_bytes());
    bytes
}
