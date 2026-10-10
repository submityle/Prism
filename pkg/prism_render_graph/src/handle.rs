//! Virtual-resource identity and SSA-versioned handles.
//!
//! The graph never touches real GPU memory while it is being built. Instead
//! every texture or buffer is a *virtual resource* identified by a small
//! integer index into the graph's resource table, and every reference a pass
//! holds is a **handle**: that index paired with a **write version**.
//!
//! The version is what makes dependency derivation exact (see the graph's SSA
//! write-versioning). Creating or writing a resource mints a *new* version;
//! reads capture the version visible to them. Two handles to the same resource
//! at different versions therefore denote two distinct points in the resource's
//! timeline, and an edge "reader of version `v` depends on the writer that
//! produced `v`" falls straight out of the handle values with no extra
//! bookkeeping.

use core::fmt;
use core::marker::PhantomData;

/// A position in the graph's virtual-resource table.
///
/// Stable for the lifetime of one graph build. Not a GPU handle: it addresses
/// the *description* of a resource, not any realized memory.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceIndex(pub(crate) u32);

impl ResourceIndex {
    /// The underlying table position.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for ResourceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Marker for texture-kind handles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureMarker {}

/// Marker for buffer-kind handles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BufferMarker {}

/// A strongly-typed, SSA-versioned reference to a virtual resource of kind `K`.
///
/// `resource` names *which* virtual resource; `version` names *which write* of
/// it this handle observes. Version `0` is the resource's declared/imported
/// initial contents; each write bumps the version by one and yields a fresh
/// handle. The type tag `K` keeps texture and buffer handles from being
/// confused at compile time.
pub struct ResourceHandle<K> {
    pub(crate) resource: ResourceIndex,
    pub(crate) version: u32,
    marker: PhantomData<fn() -> K>,
}

impl<K> ResourceHandle<K> {
    pub(crate) const fn new(resource: ResourceIndex, version: u32) -> Self {
        Self {
            resource,
            version,
            marker: PhantomData,
        }
    }

    /// The virtual resource this handle refers to.
    #[must_use]
    pub const fn resource(self) -> ResourceIndex {
        self.resource
    }

    /// The write version this handle observes.
    #[must_use]
    pub const fn version(self) -> u32 {
        self.version
    }
}

// Manual trait impls: the `fn() -> K` tag means `K` itself need not be `Clone`
// etc., exactly as the driver's `ResourceId` does.
impl<K> Clone for ResourceHandle<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for ResourceHandle<K> {}
impl<K> PartialEq for ResourceHandle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource && self.version == other.version
    }
}
impl<K> Eq for ResourceHandle<K> {}
impl<K> fmt::Debug for ResourceHandle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}@v{}", self.resource, self.version)
    }
}

/// An SSA-versioned handle to a virtual texture.
pub type TextureHandle = ResourceHandle<TextureMarker>;
/// An SSA-versioned handle to a virtual buffer.
pub type BufferHandle = ResourceHandle<BufferMarker>;
