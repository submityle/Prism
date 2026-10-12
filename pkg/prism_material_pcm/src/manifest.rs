//! One extension point's declaration (§5.4).
//!
//! A manifest splits three groups of fields that the early drafts conflated:
//!
//! | group | fields | mutability |
//! |---|---|---|
//! | identity | [`uid`](ExtensionManifest::uid), [`registry_slot`](ExtensionManifest::registry_slot), [`origin`](ExtensionManifest::origin) | permanent |
//! | attribution | [`namespace`](ExtensionManifest::namespace), [`vendor`](ExtensionManifest::vendor), [`stability`](ExtensionManifest::stability) | promoted |
//! | capability | [`kind`](ExtensionManifest::kind), `version`, `requires_caps`, `view_dependent`, `replaces_core` | per release |
//!
//! The display [`id`](ExtensionManifest::id) is **derived** from the namespace
//! and the semantic name and is *not* an identity anchor — which is what lets
//! promotion (`PRISM → KHR`) flip attribution without breaking a single
//! cross-reference, the failure mode the early "namespace as `id` prefix"
//! draft had.

use crate::namespace::{Kind, Namespace, Region, Stability};
use crate::CapabilitySet;
use alloc::string::String;
use core::fmt;

/// A manifest that violates the §5.4 identity/namespace contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    /// The `uid` is empty, or contains whitespace / non-ASCII.
    EmptyOrMalformedUid(String),
    /// A slot outside the fixed-width header digest was requested.
    SlotOutOfDigest {
        /// The requested slot.
        slot: u32,
        /// The digest width.
        width: u32,
    },
    /// The slot does not lie in the region its origin namespace owns.
    SlotOutsideOriginRegion {
        /// The requested slot.
        slot: u32,
        /// The region the origin namespace owns.
        expected: Region,
        /// The region the slot actually lies in.
        actual: Region,
    },
    /// A `KHR` (specification core) item declared a platform requirement.
    SpecificationCoreRequiresCapability {
        /// The capability it asked for.
        capability: crate::Capability,
    },
    /// A zero `version`.
    ZeroVersion,
    /// The requested promotion is not on the `Studio → Ext → KHR` chain.
    IllegalPromotion {
        /// Current namespace.
        from: Namespace,
        /// Requested namespace.
        to: Namespace,
    },
    /// The promotion would lower stability.
    StabilityRegression {
        /// Current stability.
        from: Stability,
        /// Requested stability.
        to: Stability,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOrMalformedUid(uid) => write!(f, "malformed extension uid {uid:?}"),
            Self::SlotOutOfDigest { slot, width } => {
                write!(f, "registry slot {slot} outside the {width}-slot header digest")
            }
            Self::SlotOutsideOriginRegion {
                slot,
                expected,
                actual,
            } => write!(
                f,
                "registry slot {slot} lives in {actual:?} but its origin owns {expected:?}"
            ),
            Self::SpecificationCoreRequiresCapability { capability } => write!(
                f,
                "a KHR specification-core item may not require the {capability} capability"
            ),
            Self::ZeroVersion => write!(f, "an extension version starts at 1"),
            Self::IllegalPromotion { from, to } => {
                write!(f, "illegal extension promotion {from} -> {to}")
            }
            Self::StabilityRegression { from, to } => {
                write!(f, "promotion may not lower stability {from} -> {to}")
            }
        }
    }
}

impl core::error::Error for ManifestError {}

/// A derived, human-facing extension display name (`KHR_lobe_face`).
///
/// Interchange formats (glTF / `MaterialX`) read this; nothing binds on it.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExtensionId(String);

impl ExtensionId {
    /// The derived name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExtensionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One registered extension point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensionManifest {
    uid: &'static str,
    registry_slot: u32,
    origin: Namespace,
    namespace: Namespace,
    vendor: &'static str,
    stability: Stability,
    kind: Kind,
    version: u32,
    requires_caps: CapabilitySet,
    view_dependent: bool,
    replaces_core: bool,
}

impl ExtensionManifest {
    /// Begin a manifest for the given immutable identity and kind.
    pub const fn new(uid: &'static str, registry_slot: u32, origin: Namespace, kind: Kind) -> Self {
        Self {
            uid,
            registry_slot,
            origin,
            namespace: origin,
            vendor: "",
            stability: Stability::Experimental,
            kind,
            version: 1,
            requires_caps: CapabilitySet::EMPTY,
            view_dependent: false,
            replaces_core: false,
        }
    }

    /// Set the current attribution namespace (may differ from `origin` after a
    /// promotion).
    #[must_use]
    pub const fn with_namespace(mut self, namespace: Namespace) -> Self {
        self.namespace = namespace;
        self
    }

    /// Set the publishing vendor.
    #[must_use]
    pub const fn with_vendor(mut self, vendor: &'static str) -> Self {
        self.vendor = vendor;
        self
    }

    /// Set the stability level.
    #[must_use]
    pub const fn with_stability(mut self, stability: Stability) -> Self {
        self.stability = stability;
        self
    }

    /// Set the extension's own version (independent of the `OpenPBR` axis).
    #[must_use]
    pub const fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Declare the platform capabilities this extension needs.
    #[must_use]
    pub const fn with_requires_caps(mut self, requires_caps: CapabilitySet) -> Self {
        self.requires_caps = requires_caps;
        self
    }

    /// Mark the extension's response as view dependent (it cannot be cached in
    /// the object-space shading cache of §14.3).
    #[must_use]
    pub const fn with_view_dependent(mut self, view_dependent: bool) -> Self {
        self.view_dependent = view_dependent;
        self
    }

    /// Mark the extension as replacing a core response (§5.4).
    #[must_use]
    pub const fn with_replaces_core(mut self, replaces_core: bool) -> Self {
        self.replaces_core = replaces_core;
        self
    }

    /// Permanently-pinned global identity. Never rewritten, not even by a
    /// promotion.
    pub const fn uid(&self) -> &'static str {
        self.uid
    }

    /// Permanently-pinned registry slot / bit index. Never recycled.
    pub const fn registry_slot(&self) -> u32 {
        self.registry_slot
    }

    /// The namespace this extension was *allocated* under. Together with the
    /// slot this fixes the bit region, so promotion never moves a bit.
    pub const fn origin(&self) -> Namespace {
        self.origin
    }

    /// The bit region this extension's slot lives in.
    pub const fn region(&self) -> Region {
        self.origin.region()
    }

    /// Current attribution namespace (mutable across promotions).
    pub const fn namespace(&self) -> Namespace {
        self.namespace
    }

    /// Publishing vendor (mutable across promotions).
    pub const fn vendor(&self) -> &'static str {
        self.vendor
    }

    /// Current stability level (mutable across promotions).
    pub const fn stability(&self) -> Stability {
        self.stability
    }

    /// Which of the nine extension-point kinds this is.
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// The extension's own version.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// The platform capabilities this extension requires.
    pub const fn requires_caps(&self) -> CapabilitySet {
        self.requires_caps
    }

    /// Whether the response depends on the view direction.
    pub const fn view_dependent(&self) -> bool {
        self.view_dependent
    }

    /// Whether the extension replaces rather than adds to a core response.
    pub const fn replaces_core(&self) -> bool {
        self.replaces_core
    }

    /// The derived display name (`{NAMESPACE}_{kind}_{semantic}`).
    ///
    /// The semantic name is the last segment of the `uid`, so the name tracks
    /// the namespace automatically and never has to be kept in sync by hand.
    pub fn id(&self) -> ExtensionId {
        let semantic = self
            .uid
            .rsplit('.')
            .next()
            .filter(|segment| !segment.is_empty())
            .unwrap_or(self.uid);
        ExtensionId(alloc::format!(
            "{}_{}_{}",
            self.namespace.as_str(),
            self.kind.as_str(),
            semantic
        ))
    }

    /// Validate every §5.4 identity/namespace predicate.
    ///
    /// # Errors
    ///
    /// Returns the first violated predicate: a malformed `uid`, a slot outside
    /// the header digest or outside the origin namespace's region, a
    /// specification-core item that requires a platform capability, or a zero
    /// version.
    pub fn validate(&self) -> Result<(), ManifestError> {
        let uid_ok = !self.uid.is_empty()
            && self.uid.is_ascii()
            && !self.uid.chars().any(char::is_whitespace)
            && self.uid.contains('.');
        if !uid_ok {
            return Err(ManifestError::EmptyOrMalformedUid(
                String::from(self.uid),
            ));
        }
        if self.version == 0 {
            return Err(ManifestError::ZeroVersion);
        }
        let Some(actual) = Region::of(self.registry_slot) else {
            return Err(ManifestError::SlotOutOfDigest {
                slot: self.registry_slot,
                width: crate::HEADER_DIGEST_SLOTS,
            });
        };
        let expected = self.origin.region();
        if actual != expected {
            return Err(ManifestError::SlotOutsideOriginRegion {
                slot: self.registry_slot,
                expected,
                actual,
            });
        }
        if self.namespace == Namespace::Khr && !self.requires_caps.is_empty() {
            let capability = self
                .requires_caps
                .iter()
                .next()
                .expect("a non-empty capability set has a first element");
            return Err(ManifestError::SpecificationCoreRequiresCapability { capability });
        }
        Ok(())
    }

    /// Promote the manifest to a new attribution namespace (§5.4).
    ///
    /// Promotion flips `namespace` / `vendor` / `stability` **only**. The
    /// `uid`, `registry_slot` and `origin` are carried across unchanged, so
    /// every cross-reference, dependency and bit binding survives.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError::IllegalPromotion`] when the transition is not
    /// on the `Studio → Ext → KHR` chain, or
    /// [`ManifestError::StabilityRegression`] when it would lower stability.
    pub fn promote(
        &self,
        to: Namespace,
        vendor: &'static str,
        stability: Stability,
    ) -> Result<Self, ManifestError> {
        if !self.namespace.can_promote_to(to) {
            return Err(ManifestError::IllegalPromotion {
                from: self.namespace,
                to,
            });
        }
        if !self.stability.can_promote_to(stability) {
            return Err(ManifestError::StabilityRegression {
                from: self.stability,
                to: stability,
            });
        }
        let promoted = Self {
            namespace: to,
            vendor,
            stability,
            ..*self
        };
        promoted.validate()?;
        Ok(promoted)
    }
}
