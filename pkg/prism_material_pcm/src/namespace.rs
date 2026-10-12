//! The PCM vocabulary: namespaces, bit regions, stability and extension kinds.

use core::fmt;
use core::ops::Range;

/// Governance namespace of an extension (§5.4).
///
/// The namespace is *attribution*, not identity: promotion
/// (`Studio → Ext → Khr`) rewrites it, while the [`uid`](super::ExtensionManifest::uid)
/// and [`registry_slot`](super::ExtensionManifest::registry_slot) never move.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Namespace {
    /// The `OpenPBR` specification core: a `Khr` item's response is the
    /// conformance reference the §4.5 gate compares against.
    Khr,
    /// Prism-private, first-party extensions shipped with the engine.
    Prism,
    /// Cross-vendor extensions promoted out of a studio namespace.
    Ext,
    /// A studio-private extension, not yet promoted.
    Studio,
}

impl Namespace {
    /// Every namespace, in promotion order (`Studio` is the origin of the
    /// promotion chain, `Khr` its terminus).
    pub const ALL: [Self; 4] = [Self::Khr, Self::Prism, Self::Ext, Self::Studio];

    /// The canonical uppercase spelling used in `id`s and manifests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Khr => "KHR",
            Self::Prism => "PRISM",
            Self::Ext => "EXT",
            Self::Studio => "X_STUDIO",
        }
    }

    /// Parse the canonical spelling (case-insensitive).
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|namespace| namespace.as_str().eq_ignore_ascii_case(text))
    }

    /// The bit region new slots are allocated from for this namespace
    /// (§4.5 constraint 2: the bit/region partition key is the namespace).
    pub const fn region(self) -> Region {
        match self {
            Self::Khr => Region::Specification,
            Self::Prism => Region::Prism,
            Self::Ext => Region::CrossVendor,
            Self::Studio => Region::Studio,
        }
    }

    /// Whether an extension may be promoted from `self` to `to` (§5.4).
    ///
    /// Promotion is a one-way state machine, `Studio → Ext → Khr`. A `Khr`
    /// item never demotes, and `Prism` sits beside the chain: first-party
    /// extensions promote directly to `Khr` once they are absorbed by the
    /// specification, but never *become* `Ext` (that namespace means
    /// "cross-vendor", which a first-party item is not).
    pub const fn can_promote_to(self, to: Self) -> bool {
        matches!(
            (self, to),
            (Self::Studio, Self::Ext)
                | (Self::Studio, Self::Khr)
                | (Self::Ext, Self::Khr)
                | (Self::Prism, Self::Khr)
        )
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A dense, never-recycled slot region (§4.5 constraint 2).
///
/// Regions are disjoint and static, so a private extension can never alias a
/// specification field and an upstream specification addition can never
/// collide with a private bit. The region is a property of where a slot was
/// *allocated*, which is immutable, so promotion never moves a bit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum Region {
    /// `KHR` specification core, slots `0..16`.
    Specification = 0,
    /// Prism-private, slots `16..32`.
    Prism = 1,
    /// Cross-vendor, slots `32..48`.
    CrossVendor = 2,
    /// Studio-private, slots `48..64`.
    Studio = 3,
}

impl Region {
    /// Every region, lowest slot first.
    pub const ALL: [Self; 4] = [
        Self::Specification,
        Self::Prism,
        Self::CrossVendor,
        Self::Studio,
    ];

    /// Slots per region. Four regions tile [`HEADER_DIGEST_SLOTS`].
    ///
    /// [`HEADER_DIGEST_SLOTS`]: super::HEADER_DIGEST_SLOTS
    pub const SLOTS_PER_REGION: u32 = 16;

    /// The half-open slot range this region owns.
    pub const fn range(self) -> Range<u32> {
        let base = self as u32 * Self::SLOTS_PER_REGION;
        base..base + Self::SLOTS_PER_REGION
    }

    /// First slot in the region.
    pub const fn base(self) -> u32 {
        self as u32 * Self::SLOTS_PER_REGION
    }

    /// One past the last slot in the region.
    pub const fn end(self) -> u32 {
        self.base() + Self::SLOTS_PER_REGION
    }

    /// The region owning `slot`, or `None` for a slot outside the digest.
    pub const fn of(slot: u32) -> Option<Self> {
        let region = slot / Self::SLOTS_PER_REGION;
        match region {
            0 => Some(Self::Specification),
            1 => Some(Self::Prism),
            2 => Some(Self::CrossVendor),
            3 => Some(Self::Studio),
            _ => None,
        }
    }

    /// Whether `slot` lies inside this region.
    pub const fn contains(self, slot: u32) -> bool {
        slot >= self.base() && slot < self.end()
    }

    /// The namespace that owns this region and allocates from it.
    pub const fn origin_namespace(self) -> Namespace {
        match self {
            Self::Specification => Namespace::Khr,
            Self::Prism => Namespace::Prism,
            Self::CrossVendor => Namespace::Ext,
            Self::Studio => Namespace::Studio,
        }
    }

    /// The canonical lowercase spelling used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Specification => "specification",
            Self::Prism => "prism",
            Self::CrossVendor => "cross-vendor",
            Self::Studio => "studio",
        }
    }
}

/// Maturity of an extension (§5.4).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Stability {
    /// Under active design; may change layout or response.
    Experimental,
    /// Frozen layout, response still settling; a promotion candidate.
    Candidate,
    /// Frozen layout and response; may be relied on across releases.
    Stable,
}

impl Stability {
    /// Every level, least to most mature.
    pub const ALL: [Self; 3] = [Self::Experimental, Self::Candidate, Self::Stable];

    /// The canonical lowercase spelling used in manifests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Experimental => "experimental",
            Self::Candidate => "candidate",
            Self::Stable => "stable",
        }
    }

    /// Parse the canonical spelling (case-insensitive).
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|level| level.as_str().eq_ignore_ascii_case(text))
    }

    /// Promotion may only *raise* stability, never lower it.
    pub const fn can_promote_to(self, to: Self) -> bool {
        (to as u32) >= (self as u32)
    }
}

impl fmt::Display for Stability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The nine extension-point kinds (§5), each with its own typed hook ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Kind {
    /// An optional BSDF lobe — `unpack`/`classify`/`sample`/`respond` hooks.
    Lobe,
    /// A dynamic parameter channel fed into the unpack step (§17.1).
    Channel,
    /// A shading response mode; the only kind that touches the step-2 class
    /// table and therefore the only hot-path structural change (§5.2).
    ResponseMode,
    /// A transfer remap applied between response and composition.
    Transfer,
    /// View-independent cache auxiliary output (§14.3).
    CacheAux,
    /// A data provider that produces a page/atlas (SDF, LUT, measured BTF).
    Provider,
    /// A distance/quality downgrade policy hook (§7.2).
    Downgrade,
    /// An author-node type in the authoring layer (§4).
    AuthorNode,
    /// A composite pass recorded into step 6 (e.g. outline).
    Composite,
}

impl Kind {
    /// Every kind, in the order the design lists them.
    pub const ALL: [Self; 9] = [
        Self::Lobe,
        Self::Channel,
        Self::ResponseMode,
        Self::Transfer,
        Self::CacheAux,
        Self::Provider,
        Self::Downgrade,
        Self::AuthorNode,
        Self::Composite,
    ];

    /// The canonical lowercase spelling used in `id`s and manifests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lobe => "lobe",
            Self::Channel => "channel",
            Self::ResponseMode => "response_mode",
            Self::Transfer => "transfer",
            Self::CacheAux => "cache_aux",
            Self::Provider => "provider",
            Self::Downgrade => "downgrade",
            Self::AuthorNode => "author_node",
            Self::Composite => "composite",
        }
    }

    /// Parse the canonical spelling (case-insensitive).
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str().eq_ignore_ascii_case(text))
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
