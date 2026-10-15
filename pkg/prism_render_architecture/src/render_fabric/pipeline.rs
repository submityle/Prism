use core::ops::{BitOr, BitOrAssign};

use super::RenderDomain;

/// The seven-value blend/raster axis. Object domains are deliberately not
/// represented here; they are carried by [`RenderDomain`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialRenderClass {
    Opaque = 0,
    OpaqueTwoSided = 1,
    Masked = 2,
    MaskedTwoSided = 3,
    Transmissive = 4,
    Transparent = 5,
    Additive = 6,
}

impl MaterialRenderClass {
    /// Every class in its stable dense order.
    pub const ALL: [Self; 7] = [
        Self::Opaque,
        Self::OpaqueTwoSided,
        Self::Masked,
        Self::MaskedTwoSided,
        Self::Transmissive,
        Self::Transparent,
        Self::Additive,
    ];

    pub const fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Opaque,
            1 => Self::OpaqueTwoSided,
            2 => Self::Masked,
            3 => Self::MaskedTwoSided,
            4 => Self::Transmissive,
            5 => Self::Transparent,
            6 => Self::Additive,
            _ => return None,
        })
    }

    pub const fn from_u32(value: u32) -> Option<Self> {
        if value > u8::MAX as u32 {
            None
        } else {
            Self::from_u8(value as u8)
        }
    }

    pub const fn index(self) -> u8 {
        self as u8
    }

    const fn is_two_sided(self) -> bool {
        matches!(self, Self::OpaqueTwoSided | Self::MaskedTwoSided)
    }
}

impl Default for MaterialRenderClass {
    fn default() -> Self {
        Self::Opaque
    }
}

impl RenderDomain {
    /// Every runtime object domain in its stable dense order.
    pub const ALL: [Self; 5] = [
        Self::Surface,
        Self::Particle,
        Self::Volume,
        Self::Hair,
        Self::Decal,
    ];

    pub const fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Surface,
            1 => Self::Particle,
            2 => Self::Volume,
            3 => Self::Hair,
            4 => Self::Decal,
            _ => return None,
        })
    }

    pub const fn index(self) -> u8 {
        match self {
            Self::Surface => 0,
            Self::Particle => 1,
            Self::Volume => 2,
            Self::Hair => 3,
            Self::Decal => 4,
        }
    }
}

/// Features that can change the fixed-function pipeline state. Response
/// programs and service dependencies remain separate render-fabric axes.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PipelineFeatureSet(u16);

impl PipelineFeatureSet {
    pub const NONE: Self = Self(0);
    pub const DOUBLE_SIDED: Self = Self(1 << 0);
    pub const WATER: Self = Self(1 << 1);
    pub const KNOWN_BITS: u16 = Self::DOUBLE_SIDED.0 | Self::WATER.0;

    pub const fn from_bits(bits: u16) -> Option<Self> {
        if bits & !Self::KNOWN_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Preserve raw input so [`PipelineState::resolve`] can report an
    /// explicit unknown-feature error instead of silently dropping bits.
    pub const fn from_bits_retain(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn contains(self, feature: Self) -> bool {
        self.0 & feature.0 == feature.0
    }

    /// Combine feature bits in const and runtime contexts.
    ///
    /// The `BitOr` trait implementation remains the ergonomic runtime
    /// spelling, while this named operation is available to compile-time
    /// tables because trait operators cannot currently be called from stable
    /// `const` expressions.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Compatibility spelling for callers that describe these bits as material
/// feature flags. The render-fabric contract remains independent of the
/// material crate and therefore does not depend on its record types.
pub type MaterialFeatureFlags = PipelineFeatureSet;
pub type MaterialFeatureSet = PipelineFeatureSet;

impl BitOr for PipelineFeatureSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitOrAssign for PipelineFeatureSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// The queue phase selected by the typed render-state mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelinePhase {
    Opaque,
    Transparent,
    Custom,
}

/// The color target family selected by the typed render-state mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineTarget {
    Opaque,
    HdrTransparent,
}

/// The source blend family selected by the typed render-state mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineBlend {
    Opaque,
    Alpha,
    PremultipliedAlpha,
    Additive,
}

/// The logical depth comparison. Backend reverse-Z translation happens after
/// this semantic contract is resolved.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineDepthTest {
    LessEqual,
    Always,
}

/// Face-culling policy selected by the typed render-state mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineCullMode {
    Back,
    None,
}

/// Dense, reversible bin identity for `(RenderDomain, MaterialRenderClass)`.
/// It is not a hand-written bit field and does not depend on declaration order.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PipelineBinClass(u8);

impl PipelineBinClass {
    pub const DOMAIN_COUNT: u8 = 5;
    pub const CLASS_COUNT: u8 = 7;
    pub const COUNT: u8 = Self::DOMAIN_COUNT * Self::CLASS_COUNT;

    pub const fn from_parts(domain: RenderDomain, render_class: MaterialRenderClass) -> Self {
        Self(domain.index() * Self::CLASS_COUNT + render_class.index())
    }

    pub const fn from_u8(value: u8) -> Option<Self> {
        if value < Self::COUNT {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn decode(self) -> (RenderDomain, MaterialRenderClass) {
        // The constructor and from_u8 enforce this range. Keeping decode
        // total makes it usable in const code after a validated lookup.
        let domain = match RenderDomain::from_u8(self.0 / Self::CLASS_COUNT) {
            Some(domain) => domain,
            None => RenderDomain::Surface,
        };
        let render_class = match MaterialRenderClass::from_u8(self.0 % Self::CLASS_COUNT) {
            Some(render_class) => render_class,
            None => MaterialRenderClass::Opaque,
        };
        (domain, render_class)
    }

    pub const fn domain(self) -> RenderDomain {
        self.decode().0
    }

    pub const fn render_class(self) -> MaterialRenderClass {
        self.decode().1
    }
}

/// Errors are explicit because invalid domain/feature combinations must not
/// silently fall back to an unrelated pipeline family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineStateError {
    UnknownFeatureBits(u16),
    WaterRequiresSurface { domain: RenderDomain },
}

/// Complete semantic pipeline identity derived from domain, blend/raster class
/// and fixed-function features. This is the render-fabric contract consumed by
/// queueing and PSO lowering; it is not a backend descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PipelineState {
    pub domain: RenderDomain,
    pub render_class: MaterialRenderClass,
    pub features: PipelineFeatureSet,
    pub bin_class: PipelineBinClass,
    pub phase: PipelinePhase,
    pub target: PipelineTarget,
    pub blend: PipelineBlend,
    pub depth_test: PipelineDepthTest,
    pub depth_write: bool,
    pub cull_mode: PipelineCullMode,
}

impl PipelineState {
    pub const fn resolve(
        domain: RenderDomain,
        render_class: MaterialRenderClass,
        features: PipelineFeatureSet,
    ) -> Result<Self, PipelineStateError> {
        if features.bits() & !PipelineFeatureSet::KNOWN_BITS != 0 {
            return Err(PipelineStateError::UnknownFeatureBits(features.bits()));
        }
        if features.contains(PipelineFeatureSet::WATER) {
            // Keep this exhaustive match const-evaluable: derived PartialEq
            // for RenderDomain cannot be called from this const function.
            match domain {
                RenderDomain::Surface => {}
                RenderDomain::Particle
                | RenderDomain::Volume
                | RenderDomain::Hair
                | RenderDomain::Decal => {
                    return Err(PipelineStateError::WaterRequiresSurface { domain });
                }
            }
        }

        let bin_class = PipelineBinClass::from_parts(domain, render_class);
        if features.contains(PipelineFeatureSet::WATER) {
            return Ok(Self {
                domain,
                render_class,
                features,
                bin_class,
                phase: PipelinePhase::Transparent,
                target: PipelineTarget::HdrTransparent,
                blend: PipelineBlend::PremultipliedAlpha,
                depth_test: PipelineDepthTest::LessEqual,
                depth_write: false,
                cull_mode: PipelineCullMode::None,
            });
        }

        let state = match domain {
            RenderDomain::Volume => Self {
                domain,
                render_class,
                features,
                bin_class,
                phase: PipelinePhase::Custom,
                target: PipelineTarget::HdrTransparent,
                blend: PipelineBlend::PremultipliedAlpha,
                depth_test: PipelineDepthTest::LessEqual,
                depth_write: false,
                cull_mode: PipelineCullMode::None,
            },
            RenderDomain::Decal => Self {
                domain,
                render_class,
                features,
                bin_class,
                phase: PipelinePhase::Transparent,
                target: PipelineTarget::HdrTransparent,
                blend: PipelineBlend::Alpha,
                depth_test: PipelineDepthTest::LessEqual,
                depth_write: false,
                cull_mode: if features.contains(PipelineFeatureSet::DOUBLE_SIDED) {
                    PipelineCullMode::None
                } else {
                    PipelineCullMode::Back
                },
            },
            RenderDomain::Surface | RenderDomain::Particle | RenderDomain::Hair => {
                let (phase, target, blend, depth_write) = match render_class {
                    MaterialRenderClass::Opaque
                    | MaterialRenderClass::OpaqueTwoSided
                    | MaterialRenderClass::Masked
                    | MaterialRenderClass::MaskedTwoSided => (
                        PipelinePhase::Opaque,
                        PipelineTarget::Opaque,
                        PipelineBlend::Opaque,
                        true,
                    ),
                    MaterialRenderClass::Transmissive | MaterialRenderClass::Transparent => (
                        PipelinePhase::Transparent,
                        PipelineTarget::HdrTransparent,
                        PipelineBlend::PremultipliedAlpha,
                        false,
                    ),
                    MaterialRenderClass::Additive => (
                        PipelinePhase::Transparent,
                        PipelineTarget::HdrTransparent,
                        PipelineBlend::Additive,
                        false,
                    ),
                };
                let two_sided = render_class.is_two_sided()
                    || features.contains(PipelineFeatureSet::DOUBLE_SIDED);
                Self {
                    domain,
                    render_class,
                    features,
                    bin_class,
                    phase,
                    target,
                    blend,
                    depth_test: PipelineDepthTest::LessEqual,
                    depth_write,
                    cull_mode: if two_sided {
                        PipelineCullMode::None
                    } else {
                        PipelineCullMode::Back
                    },
                }
            }
        };
        Ok(state)
    }

    pub const fn is_transparent(self) -> bool {
        !matches!(self.blend, PipelineBlend::Opaque)
    }

    pub const fn decode_bin_class(self) -> (RenderDomain, MaterialRenderClass) {
        self.bin_class.decode()
    }
}

/// Resolve the complete semantic state for a material axis tuple.
pub const fn pipeline_state_for_material(
    domain: RenderDomain,
    render_class: MaterialRenderClass,
    features: PipelineFeatureSet,
) -> Result<PipelineState, PipelineStateError> {
    PipelineState::resolve(domain, render_class, features)
}
