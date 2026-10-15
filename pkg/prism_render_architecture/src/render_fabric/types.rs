use core::num::NonZeroU32;

pub use crate::backend::capability::CapabilitySet;
pub use crate::frame_graph::{AccessKind, QueueClass};
use crate::shader_package::{PermutationKey, PermutationSpace};

/// Author-supplied, persistent semantic identity; never assigned by a catalog.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FeatureId(pub u32);

/// Globally unique, explicitly authored implementation identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariantId(pub u32);

/// Globally stable pass-template identity, independent of frame-local pass IDs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PassKey(pub u32);

/// Explicit semantic service identity. No built-in or engine-provider exceptions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ServiceId(pub u32);

/// One reusable static set representation for both hard and soft requirements.
/// Repeated entries are rejected by the catalog; declaration order is irrelevant.
#[derive(Clone, Copy, Debug)]
pub struct ServiceSet(&'static [ServiceId]);

impl PartialEq for ServiceSet {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().all(|service| other.0.contains(service))
            && other.0.iter().all(|service| self.0.contains(service))
    }
}

impl Eq for ServiceSet {}

impl ServiceSet {
    pub const EMPTY: Self = Self(&[]);

    pub const fn new(services: &'static [ServiceId]) -> Self {
        Self(services)
    }

    pub const fn as_slice(self) -> &'static [ServiceId] {
        self.0
    }

    pub fn contains(self, service: ServiceId) -> bool {
        self.0.contains(&service)
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn is_superset(self, other: Self) -> bool {
        other.0.iter().all(|&service| self.contains(service))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResourceOwner {
    Engine,
    Feature(FeatureId),
}

/// Namespaced semantic key. Declarations and references are separate tables.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceKey {
    owner: ResourceOwner,
    slot: u16,
}

impl ResourceKey {
    pub const fn engine(slot: u16) -> Self {
        Self {
            owner: ResourceOwner::Engine,
            slot,
        }
    }

    pub const fn of(owner: FeatureId, slot: u16) -> Self {
        Self {
            owner: ResourceOwner::Feature(owner),
            slot,
        }
    }

    pub const fn owner(self) -> ResourceOwner {
        self.owner
    }
    pub const fn slot(self) -> u16 {
        self.slot
    }
}

/// Whole-resource access only until the authoritative graph supports subresources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceScope {
    Whole,
}

/// Positive history distance. The parent integration must map this to persistent
/// frame-versioned reads, never to same-frame dependency edges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameLag(NonZeroU32);

impl FrameLag {
    pub const fn new(frames: u32) -> Option<Self> {
        match NonZeroU32::new(frames) {
            Some(frames) => Some(Self(frames)),
            None => None,
        }
    }

    pub const fn frames(self) -> u32 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceUse {
    pub key: ResourceKey,
    pub access: AccessKind,
    pub scope: ResourceScope,
    pub frame_lag: Option<FrameLag>,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderDomain {
    Surface,
    Particle,
    Volume,
    Hair,
    Decal,
}

/// Response axis shared by domain renderers. It is orthogonal to
/// [`RenderDomain`] and to blend/raster state; a domain chooses where it is
/// drawn, while this value chooses how its lighting response is evaluated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResponseProgram {
    Pbr,
    Npr,
    Custom,
}

/// Device/configuration quality, not a per-frame budget or measurement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RenderQualityTier {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceOutput {
    pub service: ServiceId,
    pub resources: &'static [ResourceKey],
}

/// Only services can publish a semantic interface, enforced by enum shape.
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::FeatureKind;
/// let _ = FeatureKind::Subsystem { provides: &[] };
/// ```
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::FeatureKind;
/// let _ = FeatureKind::Response { provides: &[] };
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureKind {
    Subsystem,
    Service { provides: &'static [ServiceOutput] },
    Response,
}

impl FeatureKind {
    pub const fn provides(self) -> &'static [ServiceOutput] {
        match self {
            Self::Service { provides } => provides,
            Self::Subsystem | Self::Response => &[],
        }
    }

    pub const fn is_service(self) -> bool {
        matches!(self, Self::Service { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassTemplate {
    pub id: PassKey,
    pub accesses: &'static [ResourceUse],
    pub queue: QueueClass,
}

/// Bind an existing shader permutation system, without another variant enum.
/// `None` on a feature variant means it requires no shader permutation binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderPermutationBinding {
    pub space: &'static PermutationSpace,
    pub key: &'static PermutationKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureVariant {
    pub id: VariantId,
    pub rank: u16,
    pub min_quality: RenderQualityTier,
    pub required_caps: CapabilitySet,
    pub requires_services_extra: ServiceSet,
    pub passes: &'static [PassTemplate],
    pub shader_permutation: Option<ShaderPermutationBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureDefinition {
    pub id: FeatureId,
    pub kind: FeatureKind,
    pub domain: Option<RenderDomain>,
    pub requires_services: ServiceSet,
    pub variants: &'static [FeatureVariant],
}

impl FeatureDefinition {
    pub const fn subsystem(
        id: FeatureId,
        domain: RenderDomain,
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Subsystem,
            domain: Some(domain),
            requires_services,
            variants,
        }
    }

    pub const fn service(
        id: FeatureId,
        provides: &'static [ServiceOutput],
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Service { provides },
            domain: None,
            requires_services,
            variants,
        }
    }

    pub const fn response(
        id: FeatureId,
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Response,
            domain: None,
            requires_services,
            variants,
        }
    }
}
