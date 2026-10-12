//! One extension point's declaration (§5.4).
//!
//! A manifest splits three groups of fields that the early drafts conflated:
//!
//! | group | fields | mutability |
//! |---|---|---|
//! | identity | [`uid`](ExtensionManifest::uid), [`registry_slot`](ExtensionManifest::registry_slot), [`origin`](ExtensionManifest::origin) | permanent |
//! | attribution | [`namespace`](ExtensionManifest::namespace), [`vendor`](ExtensionManifest::vendor), [`stability`](ExtensionManifest::stability) | promoted |
//! | capability | [`kind`](ExtensionManifest::kind), `version`, `requires_caps`, [`energy_policy`](ExtensionManifest::energy_policy), `view_dependent`, `replaces_core` | per release |
//!
//! The display [`id`](ExtensionManifest::id) is **derived** from the namespace
//! and the semantic name and is *not* an identity anchor — which is what lets
//! promotion (`PRISM → KHR`) flip attribution without breaking a single
//! cross-reference, the failure mode the early "namespace as `id` prefix"
//! draft had.
//!
//! The identity group has two shapes, and the difference is a §5.2 fact rather
//! than a bookkeeping one: a manifest either pins a
//! [`registry_slot`](ExtensionManifest::registry_slot) in the material header
//! digest ([`ExtensionManifest::new`]) because the runtime reads a per-material
//! fact about it, or it is *identity-only*
//! ([`ExtensionManifest::new_identity_only`]) because it resolves at bake/pass
//! time and a per-material bit would have no reader. Both are registered,
//! promoted and negotiated the same way; see [`Kind::header_resident`].

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
    /// A kind that resolves at bake/pass time was given a header-digest slot.
    SlotForIdentityOnlyKind {
        /// The offending kind.
        kind: Kind,
    },
    /// A kind the runtime reads per material was registered without a slot.
    MissingSlotForHeaderResidentKind {
        /// The offending kind.
        kind: Kind,
    },
    /// A non-lobe kind declared an energy policy.
    EnergyPolicyOutsideLobe {
        /// The offending kind.
        kind: Kind,
        /// The policy it asked for.
        policy: EnergyPolicy,
    },
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
            Self::SlotForIdentityOnlyKind { kind } => write!(
                f,
                "a {kind} extension resolves at bake/pass time and must not consume \
                 a header-digest slot"
            ),
            Self::MissingSlotForHeaderResidentKind { kind } => write!(
                f,
                "a {kind} extension is read per material and must pin a registry slot"
            ),
            Self::EnergyPolicyOutsideLobe { kind, policy } => write!(
                f,
                "a {kind} extension cannot declare the {policy} energy policy: only a \
                 lobe carries energy (§17.2)"
            ),
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

/// How an extension participates in the layered energy budget (§17.2).
///
/// This is the manifest-level rollup of the schema's *field*-level
/// `energy_policy`: a lobe that owns at least one `coupled` field shares a
/// budget with the lobes it layers over (the `OpenPBR` coat/base coupling, an
/// alternative to a plain sum of lobes), an `independent` one adds its own
/// energy (emission), and `None` marks a lobe that carries no energy parameter
/// at all (the stylized face terminator). [`Self::roll_up`] is the exact rule,
/// and it is order independent — the schema may declare fields in any order.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EnergyPolicy {
    /// Shares a coupled energy budget with the lobes it layers over (§17.2).
    Coupled,
    /// Contributes energy on its own, independent of the other lobes.
    Independent,
    /// Carries no energy parameter (the default for every non-lobe kind).
    #[default]
    None,
}

impl EnergyPolicy {
    /// Every policy, most coupled first.
    pub const ALL: [Self; 3] = [Self::Coupled, Self::Independent, Self::None];

    /// The canonical lowercase spelling used in manifests and the schema.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coupled => "coupled",
            Self::Independent => "independent",
            Self::None => "none",
        }
    }

    /// Parse the canonical spelling (case-insensitive).
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|policy| policy.as_str().eq_ignore_ascii_case(text))
    }

    /// Roll a lobe's per-field policies up into the lobe-level policy:
    /// `coupled` dominates `independent` dominates `none`.
    ///
    /// The rollup is a maximum over a priority order rather than a fold, so it
    /// does not depend on the order the schema declares the fields in.
    pub fn roll_up(policies: impl IntoIterator<Item = Self>) -> Self {
        let mut rolled = Self::None;
        for policy in policies {
            if policy.rank() > rolled.rank() {
                rolled = policy;
            }
        }
        rolled
    }

    /// Priority inside [`Self::roll_up`] (higher couples harder).
    const fn rank(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Independent => 1,
            Self::Coupled => 2,
        }
    }

    /// Whether the policy carries energy at all.
    pub const fn is_energy_carrying(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Display for EnergyPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One registered extension point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensionManifest {
    uid: &'static str,
    registry_slot: Option<u32>,
    origin: Namespace,
    namespace: Namespace,
    vendor: &'static str,
    stability: Stability,
    kind: Kind,
    version: u32,
    requires_caps: CapabilitySet,
    energy_policy: EnergyPolicy,
    view_dependent: bool,
    replaces_core: bool,
}

impl ExtensionManifest {
    /// Begin a manifest for a **header-resident** extension: one the runtime
    /// reads a per-material fact about, so its `registry_slot` is pinned in the
    /// material header digest.
    pub const fn new(uid: &'static str, registry_slot: u32, origin: Namespace, kind: Kind) -> Self {
        Self {
            uid,
            registry_slot: Some(registry_slot),
            origin,
            namespace: origin,
            vendor: "",
            stability: Stability::Experimental,
            kind,
            version: 1,
            requires_caps: CapabilitySet::EMPTY,
            energy_policy: EnergyPolicy::None,
            view_dependent: false,
            replaces_core: false,
        }
    }

    /// Begin a manifest for an **identity-only** extension: one that resolves at
    /// bake/pass time and therefore consumes no header-digest address (§5.2).
    ///
    /// It keeps the same identity, attribution and capability contract as any
    /// other manifest; only the per-material slot is absent, because the runtime
    /// would never read it.
    pub const fn new_identity_only(uid: &'static str, origin: Namespace, kind: Kind) -> Self {
        Self {
            uid,
            registry_slot: None,
            origin,
            namespace: origin,
            vendor: "",
            stability: Stability::Experimental,
            kind,
            version: 1,
            requires_caps: CapabilitySet::EMPTY,
            energy_policy: EnergyPolicy::None,
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

    /// Declare how the extension participates in the layered energy budget
    /// (§17.2). Only a lobe may carry energy; every other kind stays
    /// [`EnergyPolicy::None`], which [`Self::validate`] enforces.
    #[must_use]
    pub const fn with_energy_policy(mut self, energy_policy: EnergyPolicy) -> Self {
        self.energy_policy = energy_policy;
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

    /// Permanently-pinned registry slot / bit index, or `None` for an
    /// identity-only extension that consumes no header-digest address.
    ///
    /// Never recycled when present, and never reassigned by a promotion.
    pub const fn registry_slot(&self) -> Option<u32> {
        self.registry_slot
    }

    /// Whether this extension is header resident (has a registry slot) or
    /// resolves entirely at bake/pass time (identity-only, §5.2).
    pub const fn is_header_resident(&self) -> bool {
        self.registry_slot.is_some()
    }

    /// How the extension participates in the layered energy budget (§17.2).
    pub const fn energy_policy(&self) -> EnergyPolicy {
        self.energy_policy
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
        // §5.2: a slot is per-material runtime state, so only a kind the runtime
        // actually reads may hold one — and every such kind must hold one.
        match (self.kind.header_resident(), self.registry_slot) {
            (false, Some(_)) => {
                return Err(ManifestError::SlotForIdentityOnlyKind { kind: self.kind });
            }
            (true, None) => {
                return Err(ManifestError::MissingSlotForHeaderResidentKind {
                    kind: self.kind,
                });
            }
            _ => {}
        }
        if let Some(slot) = self.registry_slot {
            let Some(actual) = Region::of(slot) else {
                return Err(ManifestError::SlotOutOfDigest {
                    slot,
                    width: crate::HEADER_DIGEST_SLOTS,
                });
            };
            let expected = self.origin.region();
            if actual != expected {
                return Err(ManifestError::SlotOutsideOriginRegion {
                    slot,
                    expected,
                    actual,
                });
            }
        }
        if self.energy_policy.is_energy_carrying() && self.kind != Kind::Lobe {
            return Err(ManifestError::EnergyPolicyOutsideLobe {
                kind: self.kind,
                policy: self.energy_policy,
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
