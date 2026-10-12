//! Bind group layout description: how shader resources are grouped and typed.
//!
//! A layout entry names either a single resource or a *descriptor array*: the
//! shader-side `binding_array<T, N>` form that binds `N` resources of one class
//! into a single `@binding` slot and lets the shader index them dynamically.
//! Descriptor arrays are the RHI's bindless primitive.
//!
//! ## The array invariant
//!
//! Array-ness is declared twice, on purpose, and the two halves must agree:
//!
//! - [`BindingType::BufferArray`], [`BindingType::SamplerArray`],
//!   [`BindingType::TextureArray`] and [`BindingType::StorageTextureArray`]
//!   describe *what an element is*, mirroring the resource class so a shader
//!   reflection pass can see array-ness from the type alone; and
//! - [`BindGroupLayoutEntry::count`] carries *how many elements* the array has.
//!
//! [`BindGroupLayoutEntry::validate`] enforces the pairing: an array binding
//! must carry `count = Some(n)` with `n >= 1`, and a non-array binding must
//! carry `count = None`. Backends must reject an entry that fails validation
//! rather than silently binding a single resource where the shader expects an
//! array (or vice versa). The corresponding [`BindingResource`] array variants
//! must then supply exactly `n` elements.
//!
//! ## The `Features::BINDLESS` gate
//!
//! Every array binding is a [`Features::BINDLESS`] capability: a backend must
//! only translate an array entry when the device advertises the matching
//! [`Features`] bit (`BINDLESS_BUFFER_ARRAY`, `BINDLESS_TEXTURE_ARRAY`,
//! `BINDLESS_SAMPLER_ARRAY`, `BINDLESS_STORAGE_TEXTURE_ARRAY`), and a shader
//! that indexes an array with a non-uniform value additionally needs
//! `BINDLESS_NON_UNIFORM_INDEXING`.
//!
//! [`Features::BINDLESS`]: crate::Features::BINDLESS
//! [`Features`]: crate::Features

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
///
/// The `*Array` variants declare a descriptor array (`binding_array<T, N>` in
/// WGSL): a single `@binding` slot holding `N` resources of the element class
/// described by the variant's fields. The array length is not carried here — it
/// lives in [`BindGroupLayoutEntry::count`], and
/// [`BindGroupLayoutEntry::validate`] requires the two to agree. See the module
/// docs for the invariant and the [`Features::BINDLESS`] gate.
///
/// [`Features::BINDLESS`]: crate::Features::BINDLESS
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
    /// An array of uniform or storage buffers.
    ///
    /// Elements are described exactly as for [`BindingType::Buffer`]; the
    /// element count comes from [`BindGroupLayoutEntry::count`], and the bound
    /// resources come from [`BindingResource::BufferArray`], whose length must
    /// match. Requires `BINDLESS_BUFFER_ARRAY` (storage elements additionally
    /// require the device to expose storage-resource arrays, reported through
    /// the same [`Features::BINDLESS`] umbrella).
    ///
    /// [`Features::BINDLESS`]: crate::Features::BINDLESS
    BufferArray {
        /// How each element buffer is interpreted.
        ty: BufferBindingType,
        /// Whether each element has a dynamic offset supplied at bind time.
        ///
        /// Backends that mirror WebGPU reject dynamic offsets on a group that
        /// contains any descriptor array; prefer `false` unless the target API
        /// is known to allow it.
        has_dynamic_offset: bool,
        /// The minimum guaranteed binding size of each element, in bytes.
        min_binding_size: Option<u64>,
    },
    /// A sampler.
    Sampler(SamplerBindingType),
    /// An array of samplers.
    ///
    /// Elements are described exactly as for [`BindingType::Sampler`]; the
    /// element count comes from [`BindGroupLayoutEntry::count`], and the bound
    /// resources come from [`BindingResource::SamplerArray`], whose length must
    /// match. Requires `BINDLESS_SAMPLER_ARRAY`.
    SamplerArray(SamplerBindingType),
    /// A sampled texture.
    Texture {
        /// How the texture's samples are interpreted.
        sample_type: TextureSampleType,
        /// The view dimensionality.
        view_dimension: TextureViewDimension,
        /// Whether the texture is multisampled.
        multisampled: bool,
    },
    /// An array of sampled textures.
    ///
    /// Elements are described exactly as for [`BindingType::Texture`]; the
    /// element count comes from [`BindGroupLayoutEntry::count`], and the bound
    /// resources come from [`BindingResource::TextureViewArray`], whose length
    /// must match. Requires `BINDLESS_TEXTURE_ARRAY`.
    TextureArray {
        /// How each element texture's samples are interpreted.
        sample_type: TextureSampleType,
        /// The element view dimensionality.
        view_dimension: TextureViewDimension,
        /// Whether each element texture is multisampled.
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
    /// An array of read/write storage textures.
    ///
    /// Elements are described exactly as for [`BindingType::StorageTexture`];
    /// the element count comes from [`BindGroupLayoutEntry::count`], and the
    /// bound resources come from [`BindingResource::TextureViewArray`], whose
    /// length must match. Requires `BINDLESS_STORAGE_TEXTURE_ARRAY`.
    StorageTextureArray {
        /// The access the shader has to each element.
        access: StorageTextureAccess,
        /// The storage format of each element.
        format: TextureFormat,
        /// The element view dimensionality.
        view_dimension: TextureViewDimension,
    },
}

impl BindingType {
    /// Whether this binding declares a descriptor array.
    ///
    /// An array binding must carry a
    /// [`count`](BindGroupLayoutEntry::count), and a non-array binding must
    /// not; see [`BindGroupLayoutEntry::validate`].
    #[must_use]
    pub const fn is_array(self) -> bool {
        matches!(
            self,
            Self::BufferArray { .. }
                | Self::SamplerArray(_)
                | Self::TextureArray { .. }
                | Self::StorageTextureArray { .. }
        )
    }
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
    /// The descriptor-array length: `Some(n)` (with `n >= 1`) when
    /// [`ty`](Self::ty) is an array binding type, `None` for a single resource.
    ///
    /// [`validate`](Self::validate) requires this to agree with `ty`; both the
    /// layout and the matching [`BindingResource`] array must agree on `n`.
    pub count: Option<u32>,
}

/// Why a [`BindGroupLayoutEntry`] is internally inconsistent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum BindGroupLayoutEntryError {
    /// An array binding type (`BufferArray`, `SamplerArray`, `TextureArray`,
    /// `StorageTextureArray`) was declared without an array length, or with a
    /// length of zero.
    ArrayWithoutCount,
    /// A single-resource binding type carried a descriptor-array length.
    CountWithoutArray,
}

impl core::fmt::Display for BindGroupLayoutEntryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ArrayWithoutCount => {
                f.write_str("an array binding type requires `count = Some(n)` with n >= 1")
            }
            Self::CountWithoutArray => {
                f.write_str("a non-array binding type must use `count = None`")
            }
        }
    }
}

impl BindGroupLayoutEntry {
    /// Checks the array/`count` invariant documented on [`BindingType`].
    ///
    /// Backends must reject (not translate) an entry that returns an error, so
    /// a shader's `binding_array<T, N>` can never be bound a different number
    /// of resources than it declares.
    pub fn validate(&self) -> Result<(), BindGroupLayoutEntryError> {
        match (self.ty.is_array(), self.count) {
            (true, None) | (true, Some(0)) => Err(BindGroupLayoutEntryError::ArrayWithoutCount),
            (false, Some(_)) => Err(BindGroupLayoutEntryError::CountWithoutArray),
            (true, Some(_)) | (false, None) => Ok(()),
        }
    }

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

/// One buffer range inside a [`BindingResource::BufferArray`].
///
/// Mirrors the scalar [`BindingResource::Buffer`] fields so an array element is
/// described exactly like a single buffer binding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferBinding {
    /// The buffer to bind.
    pub buffer: BufferId,
    /// The byte offset into the buffer.
    pub offset: u64,
    /// The bound size in bytes, or `None` to bind to the end.
    pub size: Option<u64>,
}

/// A concrete GPU resource bound to a slot when assembling a bind group.
///
/// The array variants supply the elements of a descriptor array declared by a
/// [`BindingType`] array variant; the number of elements must equal the
/// layout entry's [`count`](BindGroupLayoutEntry::count). This type is `Clone`
/// but not `Copy`, because the array variants own a variable-length element
/// list.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
    /// The elements of a [`BindingType::BufferArray`].
    ///
    /// The length must equal the layout entry's `count`.
    BufferArray(Vec<BufferBinding>),
    /// A sampler binding.
    Sampler(SamplerId),
    /// The elements of a [`BindingType::SamplerArray`].
    ///
    /// The length must equal the layout entry's `count`.
    SamplerArray(Vec<SamplerId>),
    /// A sampled or storage texture view binding.
    TextureView(TextureViewId),
    /// The elements of a [`BindingType::TextureArray`] or
    /// [`BindingType::StorageTextureArray`].
    ///
    /// The length must equal the layout entry's `count`.
    TextureViewArray(Vec<TextureViewId>),
}

/// One concrete binding in a bind group: a slot paired with the resource placed
/// there.
///
/// `Clone` but not `Copy`, because [`BindingResource`] can own an element list.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceId;

    fn buffer_array_entry(count: Option<u32>) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::BufferArray {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count,
        }
    }

    #[test]
    fn array_types_report_array_ness() {
        let singletons = [
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::Sampler(SamplerBindingType::Filtering),
            BindingType::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
        ];
        for ty in singletons {
            assert!(!ty.is_array(), "{ty:?} is a single-resource binding");
        }

        assert!(buffer_array_entry(Some(4)).ty.is_array());
        assert!(BindingType::SamplerArray(SamplerBindingType::NonFiltering).is_array());
        assert!(BindingType::TextureArray {
            sample_type: TextureSampleType::Float,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }
        .is_array());
        assert!(BindingType::StorageTextureArray {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        }
        .is_array());
    }

    #[test]
    fn validate_pairs_array_types_with_counts() {
        assert_eq!(buffer_array_entry(Some(1)).validate(), Ok(()));
        assert_eq!(buffer_array_entry(Some(8)).validate(), Ok(()));
        assert_eq!(
            buffer_array_entry(None).validate(),
            Err(BindGroupLayoutEntryError::ArrayWithoutCount)
        );
        assert_eq!(
            buffer_array_entry(Some(0)).validate(),
            Err(BindGroupLayoutEntryError::ArrayWithoutCount)
        );

        // A single-resource binding is only valid without a count.
        let singleton = BindGroupLayoutEntry::sampler(3, ShaderStages::FRAGMENT);
        assert_eq!(singleton.validate(), Ok(()));
        assert_eq!(
            BindGroupLayoutEntry {
                count: Some(2),
                ..singleton
            }
            .validate(),
            Err(BindGroupLayoutEntryError::CountWithoutArray)
        );
    }

    #[test]
    fn array_resources_own_their_elements() {
        let element = BufferBinding {
            buffer: ResourceId::from_parts(7, 1),
            offset: 256,
            size: Some(64),
        };
        let resource = BindingResource::BufferArray(alloc::vec![element]);
        // `BindingResource` must remain cloneable and comparable even though it
        // is no longer `Copy` (the array variants own a `Vec`).
        assert_eq!(resource.clone(), resource);
        let entry = BindGroupEntry {
            binding: 0,
            resource,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.resource, entry.resource);

        let views = BindingResource::TextureViewArray(alloc::vec![
            ResourceId::from_parts(1, 0),
            ResourceId::from_parts(2, 0),
        ]);
        assert_ne!(views, BindingResource::TextureViewArray(alloc::vec![]));
        let samplers = BindingResource::SamplerArray(alloc::vec![ResourceId::from_parts(9, 0)]);
        assert_ne!(
            samplers,
            BindingResource::Sampler(ResourceId::from_parts(9, 0))
        );
    }

    #[test]
    fn duplicate_binding_slots_are_detected() {
        let layout = BindGroupLayoutDescriptor {
            label: None,
            entries: alloc::vec![
                BindGroupLayoutEntry::sampler(0, ShaderStages::FRAGMENT),
                BindGroupLayoutEntry::texture_2d(0, ShaderStages::FRAGMENT),
            ],
        };
        assert!(!layout.has_unique_bindings());
    }
}
