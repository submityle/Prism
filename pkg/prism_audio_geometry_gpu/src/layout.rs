//! Small `wgpu` bind-group layout helpers shared by the acoustics kernels.
//!
//! Every binding both kernels use is a compute-visible buffer with no dynamic
//! offset, so the two tiny constructors here keep the direct and reflection
//! pipelines from repeating the same descriptor boilerplate.
//!
//! # Provenance
//!
//! Original work; plain `wgpu` descriptor construction; no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio
//! source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Used by [`crate::direct`] and [`crate::reflection`] to build their bind
//! group layouts and bind groups.

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
