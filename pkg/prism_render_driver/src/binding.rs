//! Bind group layout description: how shader resources are grouped and typed.

use crate::flags::ShaderStages;
use crate::format::{TextureFormat, TextureSampleType};
use crate::resource::{BindGroupLayoutId, BufferId, SamplerId, TextureViewId};
use crate::texture::TextureViewDimension;
use alloc::string::String;
use alloc::vec::Vec;

/// How a bound buffer is interpreted by a shader.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BufferBindingType {
    /// A read-only uniform buffer.
    Uniform,
    /// A storage buffer; `read_only` distinguishes SRV-like from UAV-like use.
    Storage {
        /// Whether the shader only reads the buffer.
        read_only: bool,
    },
}

/// How a bound sampler behaves.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SamplerBindingType {
    /// A sampler that can linearly filter.
    Filtering,
    /// A sampler restricted to nearest filtering.
    NonFiltering,
    /// A comparison sampler for shadow mapping.
    Comparison,
}

/// Read/write access for a storage texture binding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StorageTextureAccess {
    /// Shader may only write.
    WriteOnly,
    /// Shader may only read.
    ReadOnly,
    /// Shader may read and write.
    ReadWrite,
}

/// The type of a single binding slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BindingType {
    /// A uniform or storage buffer.
    Buffer {
        /// How the buffer is interpreted.
        ty: BufferBindingType,
        /// Whether the binding has a dynamic offset supplied at bind time.
        has_dynamic_offset: bool,
        /// The minimum guaranteed binding size in bytes, if validated.
        min_binding_size: Option<u64>,
    },
    /// A sampler.
    Sampler(SamplerBindingType),
    /// A sampled texture.
    Texture {
        /// How the texture's samples are interpreted.
        sample_type: TextureSampleType,
        /// The view dimensionality.
        view_dimension: TextureViewDimension,
        /// Whether the texture is multisampled.
        multisampled: bool,
    },
    /// A read/write storage texture.
    StorageTexture {
        /// The access the shader has.
        access: StorageTextureAccess,
        /// The storage format.
        format: TextureFormat,
        /// The view dimensionality.
        view_dimension: TextureViewDimension,
    },
}

/// One entry in a bind group layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindGroupLayoutEntry {
    /// The `@binding(n)` slot within the group.
    pub binding: u32,
    /// Which shader stages can see this binding.
    pub visibility: ShaderStages,
    /// The resource type expected at this slot.
    pub ty: BindingType,
    /// For arrayed bindings, the fixed array length (`None` for a single
    /// resource).
    pub count: Option<u32>,
}

impl BindGroupLayoutEntry {
    /// A uniform buffer binding visible to the given stages.
    #[must_use]
    pub const fn uniform(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// A filterable sampled 2D texture binding.
    #[must_use]
    pub const fn texture_2d(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    /// A filtering sampler binding.
    #[must_use]
    pub const fn sampler(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        }
    }
}

/// A bind group layout: an ordered set of binding entries.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BindGroupLayoutDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The entries, which should have unique `binding` slots.
    pub entries: Vec<BindGroupLayoutEntry>,
}

impl BindGroupLayoutDescriptor {
    /// Whether every entry has a distinct binding slot. Backends must reject a
    /// layout with duplicate slots.
    #[must_use]
    pub fn has_unique_bindings(&self) -> bool {
        for (i, entry) in self.entries.iter().enumerate() {
            if self.entries[..i].iter().any(|e| e.binding == entry.binding) {
                return false;
            }
        }
        true
    }
}

/// A concrete GPU resource bound to a slot when assembling a bind group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BindingResource {
    /// A buffer range binding.
    Buffer {
        /// The buffer to bind.
        buffer: BufferId,
        /// The byte offset into the buffer.
        offset: u64,
        /// The bound size in bytes, or `None` to bind to the end.
        size: Option<u64>,
    },
    /// A sampler binding.
    Sampler(SamplerId),
    /// A sampled or storage texture view binding.
    TextureView(TextureViewId),
}

/// One concrete binding in a bind group: a slot paired with the resource placed
/// there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindGroupEntry {
    /// The `@binding(n)` slot, which must match a slot in the layout.
    pub binding: u32,
    /// The resource bound at this slot.
    pub resource: BindingResource,
}

/// A request to create a bind group: a layout plus the concrete resources that
/// fill each of its slots.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BindGroupDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The layout this group conforms to.
    pub layout: Option<BindGroupLayoutId>,
    /// The concrete bindings, which should cover the layout's slots exactly
    /// once each.
    pub entries: Vec<BindGroupEntry>,
}

impl BindGroupDescriptor {
    /// Whether every entry targets a distinct binding slot. Backends must
    /// reject a group with duplicate slots.
    #[must_use]
    pub fn has_unique_bindings(&self) -> bool {
        for (i, entry) in self.entries.iter().enumerate() {
            if self.entries[..i].iter().any(|e| e.binding == entry.binding) {
                return false;
            }
        }
        true
    }
}
