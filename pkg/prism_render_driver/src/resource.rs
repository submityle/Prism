//! Generational, type-tagged handles to GPU resources owned by a device.
//!
//! Backends allocate resources (buffers, textures, pipelines, …) and hand back
//! lightweight `Copy` ids. The generation guards against use-after-free: when a
//! slot is recycled its generation is bumped, so a stale id resolves to nothing
//! rather than silently aliasing a new resource. The marker types below make
//! the ids type-safe — a [`BufferId`] can never be passed where a [`TextureId`]
//! is expected.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// A generational `(index, generation)` slot id for a GPU resource of kind `K`.
pub struct ResourceId<K: ?Sized> {
    index: u32,
    generation: u32,
    marker: PhantomData<fn() -> K>,
}

impl<K: ?Sized> ResourceId<K> {
    /// Builds an id from raw parts. Normally minted by a backend's resource
    /// table; exposed for backends and serialization.
    #[must_use]
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            marker: PhantomData,
        }
    }

    /// The slot position.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The reuse counter.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

// Manual impls: the `fn() -> K` tag means `K` need not satisfy these bounds.
impl<K: ?Sized> Clone for ResourceId<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: ?Sized> Copy for ResourceId<K> {}
impl<K: ?Sized> PartialEq for ResourceId<K> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<K: ?Sized> Eq for ResourceId<K> {}
impl<K: ?Sized> PartialOrd for ResourceId<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: ?Sized> Ord for ResourceId<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}
impl<K: ?Sized> Hash for ResourceId<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<K: ?Sized> fmt::Debug for ResourceId<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceId")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

/// Declares an empty marker type plus a `*Id` alias over [`ResourceId`].
macro_rules! resource_kinds {
    ($(
        $(#[$meta:meta])*
        $kind:ident => $alias:ident
    );* $(;)?) => {
        $(
            $(#[$meta])*
            #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
            pub enum $kind {}

            $(#[$meta])*
            pub type $alias = ResourceId<$kind>;
        )*
    };
}

resource_kinds! {
    /// Marker for GPU buffers.
    BufferKind => BufferId;
    /// Marker for GPU textures.
    TextureKind => TextureId;
    /// Marker for texture views.
    TextureViewKind => TextureViewId;
    /// Marker for samplers.
    SamplerKind => SamplerId;
    /// Marker for shader modules.
    ShaderModuleKind => ShaderModuleId;
    /// Marker for bind group layouts.
    BindGroupLayoutKind => BindGroupLayoutId;
    /// Marker for bind groups.
    BindGroupKind => BindGroupId;
    /// Marker for pipeline layouts.
    PipelineLayoutKind => PipelineLayoutId;
    /// Marker for render pipelines.
    RenderPipelineKind => RenderPipelineId;
    /// Marker for compute pipelines.
    ComputePipelineKind => ComputePipelineId;
}
