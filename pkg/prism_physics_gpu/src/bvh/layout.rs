//! Small `wgpu` bind-group layout helpers for the `LBVH` pipelines.
//!
//! A deliberate local copy of the identical helpers in the scan, radix, grid,
//! and other modules: every `LBVH` binding is a compute-visible buffer with no
//! dynamic offset, and copying the two tiny constructors keeps this feature
//! from coupling to another module's private tree.
//!
//! # Provenance
//!
//! Plain `wgpu` descriptor construction; no Unreal Engine source or derived
//! code.

use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ShaderStages,
};

/// Builds a compute-visible buffer binding layout entry for `binding`.
#[must_use]
pub fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Builds a bind-group entry binding all of `buffer` to `binding`.
#[must_use]
pub fn entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
