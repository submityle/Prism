//! Platform capabilities and the set that describes what a device offers
//! (§5.3).

use crate::Namespace;
use core::fmt;

/// A platform capability an extension may need (§5.3).
///
/// The names mirror the Vulkan habit of declaring requirements rather than
/// probing: an extension states what it needs, the platform broadcasts what it
/// has, and the negotiation runs once at bake/load time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Capability {
    /// A `binding_array<texture_2d<f32>>` sampler heap (bindless textures).
    BindlessTextures,
    /// Hardware virtual texturing (sparse/indirection page tables).
    VirtualTexture,
    /// An object-space shading cache that can feed step 4(a) of §8.5.
    ObjectSpaceCache,
    /// `f16` storage buffers.
    StorageF16,
    /// Compute scatter with atomics for the per-tile classification.
    ComputeScatter,
    /// Branchy, multi-sample lobes (glint, measured BRDF, hero-wavelength).
    HeavySampling,
}

impl Capability {
    /// Every capability, in bit order.
    pub const ALL: [Self; 6] = [
        Self::BindlessTextures,
        Self::VirtualTexture,
        Self::ObjectSpaceCache,
        Self::StorageF16,
        Self::ComputeScatter,
        Self::HeavySampling,
    ];

    /// Bit index inside [`CapabilitySet`].
    pub const fn index(self) -> u32 {
        self as u32
    }

    /// Single-bit mask for this capability.
    pub const fn bit(self) -> u32 {
        1 << self.index()
    }

    /// The canonical spelling used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BindlessTextures => "bindless_textures",
            Self::VirtualTexture => "virtual_texture",
            Self::ObjectSpaceCache => "object_space_cache",
            Self::StorageF16 => "storage_f16",
            Self::ComputeScatter => "compute_scatter",
            Self::HeavySampling => "heavy_sampling",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A set of platform capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct CapabilitySet(u32);

impl CapabilitySet {
    /// The empty set (a platform offering nothing optional).
    pub const EMPTY: Self = Self(0);

    /// The set containing only `capability`.
    pub const fn of(capability: Capability) -> Self {
        Self(capability.bit())
    }

    /// The set containing every known capability — a fully-featured target.
    pub const fn full() -> Self {
        Self((1 << Capability::ALL.len()) - 1)
    }

    /// Build from a raw bitmask, ignoring bits with no known capability.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The raw bitmask, for hashing into a permutation key or a manifest.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether `capability` is offered.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// Add `capability` to the set.
    pub fn insert(&mut self, capability: Capability) {
        self.0 |= capability.bit();
    }

    /// The union of two sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The intersection of two sets.
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Whether every capability in `self` is also in `other`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Iterate the capabilities present, lowest bit first.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        Capability::ALL
            .into_iter()
            .filter(move |capability| self.contains(*capability))
    }
}

impl FromIterator<Capability> for CapabilitySet {
    fn from_iter<T: IntoIterator<Item = Capability>>(iter: T) -> Self {
        let mut set = Self::EMPTY;
        for capability in iter {
            set.insert(capability);
        }
        set
    }
}

/// The capability a namespace's conformance contract implies (§5.3).
///
/// Every namespace except the specification core is allowed to require a
/// platform capability; a `Khr` item must not, because the specification core
/// has to run everywhere (`§4.5` maintenance obligation: the core is the
/// portable subset).
pub const fn namespace_may_require_caps(namespace: Namespace) -> bool {
    !matches!(namespace, Namespace::Khr)
}
