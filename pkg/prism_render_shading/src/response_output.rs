//! Structured CPU response composition (02 §4.3–4.5).
//!
//! This is result-channel arithmetic, not a new BSDF, shader dispatcher, GPU
//! buffer ABI or conservation claim. Lighting is evaluated before either arm.

use crate::DirectLightSample;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
};
pub use prism_render_material::{BlendDomain, ChannelMask, ResponseChannel};
use prism_render_material::{MaterialRenderClass, ShadingPlan};

/// Typed CPU auxiliary payload. Field names are caller-defined semantics, not
/// a second schema catalogue. Each present field needs an explicit merge rule.
#[derive(Clone, Debug, PartialEq)]
pub enum AuxiliaryValue {
    Scalar(f32),
    Vector3([f32; 3]),
    Unsigned(u32),
    Flags(u32),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Auxiliary {
    pub fields: BTreeMap<String, AuxiliaryValue>,
}

/// No implicit rules: IDs cannot accidentally interpolate and normals are not
/// silently treated as colours. `Linear` is an explicit caller decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxiliaryMergeRule {
    Shared,
    Left,
    Right,
    RequireEqual,
    Linear,
    Min,
    Max,
    BitOr,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AuxiliaryMergeRules {
    pub fields: BTreeMap<String, AuxiliaryMergeRule>,
    pub shared: Auxiliary,
}

/// Linear, unpremultiplied result channels. Emission is one channel, not a
/// term added again after composition. No GPU layout is implied by this type.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResponseOutput {
    pub direct: [f32; 3],
    pub indirect_diffuse: [f32; 3],
    pub reflection: [f32; 3],
    pub transmission: [f32; 3],
    pub emission: [f32; 3],
    pub alpha: f32,
    pub auxiliary: Auxiliary,
}

impl ResponseOutput {
    /// Adds the five radiance channels exactly once, without alpha weighting.
    pub fn radiance(&self) -> [f32; 3] {
        core::array::from_fn(|i| {
            self.direct[i]
                + self.indirect_diffuse[i]
                + self.reflection[i]
                + self.transmission[i]
                + self.emission[i]
        })
    }

    pub fn validate(&self) -> Result<(), CompositionError> {
        for (channel, value) in [
            (ResponseChannel::Direct, self.direct),
            (ResponseChannel::IndirectDiffuse, self.indirect_diffuse),
            (ResponseChannel::Reflection, self.reflection),
            (ResponseChannel::Transmission, self.transmission),
            (ResponseChannel::Emission, self.emission),
        ] {
            if !finite3(value) {
                return Err(CompositionError::NonFiniteRadiance(channel));
            }
        }
        check_alpha(self.alpha)?;
        validate_auxiliary(&self.auxiliary)
    }
}

/// Alpha already resolved by the shared surface stage using geometry opacity
/// and render class. This layer does not invent coverage/raster policies for
/// hair, water, decals, volumes or masked surfaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedSurfaceAlpha {
    alpha: f32,
    render_class: MaterialRenderClass,
}

impl SharedSurfaceAlpha {
    pub fn new(alpha: f32, render_class: MaterialRenderClass) -> Result<Self, CompositionError> {
        check_alpha(alpha)?;
        Ok(Self {
            alpha,
            render_class,
        })
    }
    pub fn alpha(self) -> f32 {
        self.alpha
    }
    pub fn render_class(self) -> MaterialRenderClass {
        self.render_class
    }
}

/// Shared is the default. Overrides are deliberate and never mask lerps.
/// An override changes only the output alpha, not the material's render class.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AlphaMergeRule {
    #[default]
    Shared,
    Left,
    Right,
    Min,
    Max,
}

/// Already sampled, immutable lighting inputs shared by both responses.
/// Reuses the existing direct-light/visibility contract. Optional environment
/// values are sampled light terms, NOT BRDF-weighted final response colours.
/// `shared_terms` borrows caller-owned LTC/history/etc. terms without imposing
/// another representation or pretending material-dependent LTC is precomputed.
#[derive(Debug)]
pub struct LightingContext<'a, T: ?Sized = ()> {
    pub direct: &'a [DirectLightSample],
    pub indirect_diffuse_irradiance: Option<[f32; 3]>,
    pub reflection_radiance: Option<[f32; 3]>,
    pub transmission_radiance: Option<[f32; 3]>,
    pub shared_terms: &'a T,
}

impl<'terms, T: ?Sized> LightingContext<'terms, T> {
    pub fn validate(&self) -> Result<(), CompositionError> {
        for light in self.direct {
            if !finite3(light.direction)
                || !finite3(light.illuminance)
                || !light.visibility.is_finite()
                || !(0.0..=1.0).contains(&light.visibility)
            {
                return Err(CompositionError::InvalidLighting);
            }
        }
        for term in [
            self.indirect_diffuse_irradiance,
            self.reflection_radiance,
            self.transmission_radiance,
        ]
        .into_iter()
        .flatten()
        {
            if !finite3(term) {
                return Err(CompositionError::InvalidLighting);
            }
        }
        Ok(())
    }

    /// Binds a computed output to this immutable input snapshot. The caller
    /// must actually evaluate against it; this is not shader provenance proof.
    pub fn bind(
        &self,
        output: ResponseOutput,
    ) -> Result<EvaluatedResponse<'_, 'terms, T>, CompositionError> {
        self.validate()?;
        output.validate()?;
        Ok(EvaluatedResponse {
            lighting: self,
            output,
        })
    }
}

/// Independent split-arm outputs can be retained while borrowing one context.
#[derive(Debug)]
pub struct EvaluatedResponse<'a, 'terms, T: ?Sized = ()> {
    lighting: &'a LightingContext<'terms, T>,
    output: ResponseOutput,
}

impl<'terms, T: ?Sized> EvaluatedResponse<'_, 'terms, T> {
    pub fn output(&self) -> &ResponseOutput {
        &self.output
    }
    pub fn lighting(&self) -> &LightingContext<'terms, T> {
        self.lighting
    }
}

/// On success, evaluates both arms exactly once against the same context;
/// validation failures return early. No shadow or environment sampler is
/// invoked here. The functions supply actual responses.
pub fn evaluate_response_pair<'a, 'terms, T: ?Sized>(
    lighting: &'a LightingContext<'terms, T>,
    left: impl FnOnce(&LightingContext<'terms, T>) -> ResponseOutput,
    right: impl FnOnce(&LightingContext<'terms, T>) -> ResponseOutput,
) -> Result<
    (
        EvaluatedResponse<'a, 'terms, T>,
        EvaluatedResponse<'a, 'terms, T>,
    ),
    CompositionError,
> {
    lighting.validate()?;
    let left = lighting.bind(left(lighting))?;
    let right = lighting.bind(right(lighting))?;
    Ok((left, right))
}

#[derive(Clone, Copy, Debug)]
pub struct CompositionOptions<'a> {
    pub mask: f32,
    pub domain: BlendDomain,
    pub shared_alpha: SharedSurfaceAlpha,
    pub alpha_rule: AlphaMergeRule,
    pub auxiliary_rules: &'a AuxiliaryMergeRules,
}

/// Authoritative input validation and auxiliary merge for CPU/GPU composition.
/// Checks both complete outputs (including unselected channels), the mask and
/// domain, and every auxiliary rule. Does not bind lighting, clone outputs,
/// blend or sum radiance, or prove that the arms used shared lighting. The
/// returned auxiliary payload owns its fields; auxiliary merging may allocate.
pub fn validate_response_composition_inputs(
    left: &ResponseOutput,
    right: &ResponseOutput,
    options: CompositionOptions<'_>,
) -> Result<Auxiliary, CompositionError> {
    check_mask(options.mask)?;
    composition_channels(options.domain)?;
    left.validate()?;
    right.validate()?;
    let auxiliary = merge_auxiliary(
        &left.auxiliary,
        &right.auxiliary,
        options.auxiliary_rules,
        options.mask,
    )?;
    validate_auxiliary(&auxiliary)?;
    Ok(auxiliary)
}

/// Selected radiance channels lerp; all unselected channels remain LEFT. This
/// fixed baseline makes DirectOnly/custom masks fully defined. No addition,
/// premultiplication, alpha lerp or implicit auxiliary merge occurs.
pub fn blend_responses<T: ?Sized>(
    left: &EvaluatedResponse<'_, '_, T>,
    right: &EvaluatedResponse<'_, '_, T>,
    options: CompositionOptions<'_>,
) -> Result<ResponseOutput, CompositionError> {
    if !core::ptr::eq(left.lighting, right.lighting) {
        return Err(CompositionError::DifferentLightingContexts);
    }
    let left = &left.output;
    let right = &right.output;
    let auxiliary = validate_response_composition_inputs(left, right, options)?;
    let channels = composition_channels(options.domain)?;
    let channel = |id, a, b| {
        if channels.contains(ChannelMask::one(id)) {
            linear3(a, b, options.mask)
        } else {
            a
        }
    };
    let result = ResponseOutput {
        direct: channel(ResponseChannel::Direct, left.direct, right.direct),
        indirect_diffuse: channel(
            ResponseChannel::IndirectDiffuse,
            left.indirect_diffuse,
            right.indirect_diffuse,
        ),
        reflection: channel(
            ResponseChannel::Reflection,
            left.reflection,
            right.reflection,
        ),
        transmission: channel(
            ResponseChannel::Transmission,
            left.transmission,
            right.transmission,
        ),
        emission: channel(ResponseChannel::Emission, left.emission, right.emission),
        alpha: match options.alpha_rule {
            AlphaMergeRule::Shared => options.shared_alpha.alpha(),
            AlphaMergeRule::Left => left.alpha,
            AlphaMergeRule::Right => right.alpha,
            AlphaMergeRule::Min => left.alpha.min(right.alpha),
            AlphaMergeRule::Max => left.alpha.max(right.alpha),
        },
        auxiliary,
    };
    result.validate()?;
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionError {
    InvalidMask,
    InvalidChannelMask,
    InvalidAlpha,
    InvalidLighting,
    DifferentLightingContexts,
    NonFiniteRadiance(ResponseChannel),
    InvalidAuxiliary(String),
    MissingAuxiliaryRule(String),
    UnusedAuxiliaryRule(String),
    MissingAuxiliaryValue { field: String, source: &'static str },
    AuxiliaryTypeMismatch(String),
    InvalidAuxiliaryOperation(String),
    UnequalAuxiliary(String),
    CounterOverflow,
}

fn finite3(value: [f32; 3]) -> bool {
    value.into_iter().all(f32::is_finite)
}
fn composition_channels(domain: BlendDomain) -> Result<ChannelMask, CompositionError> {
    match domain {
        BlendDomain::FullRadiance => Ok(ChannelMask::radiance()),
        BlendDomain::DirectOnly => Ok(ChannelMask::one(ResponseChannel::Direct)),
        BlendDomain::CustomChannelMask(channels) => {
            if channels.0 == 0 || !ChannelMask::radiance().contains(channels) {
                Err(CompositionError::InvalidChannelMask)
            } else {
                Ok(channels)
            }
        }
    }
}
fn check_alpha(alpha: f32) -> Result<(), CompositionError> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        Err(CompositionError::InvalidAlpha)
    } else {
        Ok(())
    }
}
fn check_mask(mask: f32) -> Result<(), CompositionError> {
    if !mask.is_finite() || !(0.0..=1.0).contains(&mask) {
        Err(CompositionError::InvalidMask)
    } else {
        Ok(())
    }
}
fn linear(a: f32, b: f32, mask: f32) -> f32 {
    // Exact endpoints (including signed zero); avoid overflow of b - a.
    if mask == 0.0 {
        a
    } else if mask == 1.0 {
        b
    } else {
        (1.0 - mask) * a + mask * b
    }
}
fn linear3(a: [f32; 3], b: [f32; 3], mask: f32) -> [f32; 3] {
    core::array::from_fn(|i| linear(a[i], b[i], mask))
}
fn validate_auxiliary(aux: &Auxiliary) -> Result<(), CompositionError> {
    for (name, value) in &aux.fields {
        let finite = match value {
            AuxiliaryValue::Scalar(value) => value.is_finite(),
            AuxiliaryValue::Vector3(value) => finite3(*value),
            _ => true,
        };
        if name.trim().is_empty() || !finite {
            return Err(CompositionError::InvalidAuxiliary(name.clone()));
        }
    }
    Ok(())
}

fn require_auxiliary_value<'a>(
    value: Option<&'a AuxiliaryValue>,
    field: &str,
    source: &'static str,
) -> Result<&'a AuxiliaryValue, CompositionError> {
    value.ok_or_else(|| CompositionError::MissingAuxiliaryValue {
        field: field.to_string(),
        source,
    })
}

fn merge_auxiliary(
    left: &Auxiliary,
    right: &Auxiliary,
    rules: &AuxiliaryMergeRules,
    mask: f32,
) -> Result<Auxiliary, CompositionError> {
    validate_auxiliary(&rules.shared)?;
    let names: BTreeSet<_> = left
        .fields
        .keys()
        .chain(right.fields.keys())
        .chain(rules.shared.fields.keys())
        .collect();
    for name in rules.fields.keys() {
        if !names.contains(name) {
            return Err(CompositionError::UnusedAuxiliaryRule(name.clone()));
        }
    }
    let mut merged = Auxiliary::default();
    for name in names {
        let rule = rules
            .fields
            .get(name)
            .ok_or_else(|| CompositionError::MissingAuxiliaryRule(name.clone()))?;
        let a = left.fields.get(name);
        let b = right.fields.get(name);
        let shared = rules.shared.fields.get(name);
        // A semantic must have one type even when one source is not selected.
        let mut values = [a, b, shared].into_iter().flatten();
        if let Some(first) = values.next() {
            if values.any(|value| core::mem::discriminant(first) != core::mem::discriminant(value))
            {
                return Err(CompositionError::AuxiliaryTypeMismatch(name.clone()));
            }
        }
        let value = match rule {
            AuxiliaryMergeRule::Shared => require_auxiliary_value(shared, name, "shared")?.clone(),
            AuxiliaryMergeRule::Left => require_auxiliary_value(a, name, "left")?.clone(),
            AuxiliaryMergeRule::Right => require_auxiliary_value(b, name, "right")?.clone(),
            AuxiliaryMergeRule::RequireEqual => {
                let a = require_auxiliary_value(a, name, "left")?;
                if a != require_auxiliary_value(b, name, "right")? {
                    return Err(CompositionError::UnequalAuxiliary(name.clone()));
                }
                a.clone()
            }
            operation => {
                let a = require_auxiliary_value(a, name, "left")?;
                let b = require_auxiliary_value(b, name, "right")?;
                let scalar = |a: f32, b: f32| match operation {
                    AuxiliaryMergeRule::Linear => linear(a, b, mask),
                    AuxiliaryMergeRule::Min => a.min(b),
                    AuxiliaryMergeRule::Max => a.max(b),
                    _ => unreachable!(),
                };
                match (a, b, operation) {
                    (
                        AuxiliaryValue::Scalar(a),
                        AuxiliaryValue::Scalar(b),
                        AuxiliaryMergeRule::Linear
                        | AuxiliaryMergeRule::Min
                        | AuxiliaryMergeRule::Max,
                    ) => AuxiliaryValue::Scalar(scalar(*a, *b)),
                    (
                        AuxiliaryValue::Vector3(a),
                        AuxiliaryValue::Vector3(b),
                        AuxiliaryMergeRule::Linear
                        | AuxiliaryMergeRule::Min
                        | AuxiliaryMergeRule::Max,
                    ) => AuxiliaryValue::Vector3(core::array::from_fn(|i| scalar(a[i], b[i]))),
                    (
                        AuxiliaryValue::Unsigned(a),
                        AuxiliaryValue::Unsigned(b),
                        AuxiliaryMergeRule::Min,
                    ) => AuxiliaryValue::Unsigned((*a).min(*b)),
                    (
                        AuxiliaryValue::Unsigned(a),
                        AuxiliaryValue::Unsigned(b),
                        AuxiliaryMergeRule::Max,
                    ) => AuxiliaryValue::Unsigned((*a).max(*b)),
                    (
                        AuxiliaryValue::Flags(a),
                        AuxiliaryValue::Flags(b),
                        AuxiliaryMergeRule::BitOr,
                    ) => AuxiliaryValue::Flags(a | b),
                    _ => return Err(CompositionError::InvalidAuxiliaryOperation(name.clone())),
                }
            }
        };
        merged.fields.insert(name.to_string(), value);
    }
    Ok(merged)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConservationSuite {
    WhiteFurnace,
    SpecificationComparison,
}

/// Counts admission, not successful conservation results. Failed assertions in
/// Single cases are returned unchanged and must still fail the actual suite.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConservationSuiteCounts {
    pub admitted: u64,
    pub skipped_mask_blend: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConservationCaseResult<T> {
    Tested(T),
    SkippedMaskBlend,
}

/// Mandatory gate seam for suites that accept material ShadingPlans. Existing
/// pure-BSDF tests remain unchanged; scene/material harnesses must adopt this.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConservationGate {
    white_furnace: ConservationSuiteCounts,
    specification_comparison: ConservationSuiteCounts,
}

impl ConservationGate {
    pub fn counts(&self, suite: ConservationSuite) -> ConservationSuiteCounts {
        match suite {
            ConservationSuite::WhiteFurnace => self.white_furnace,
            ConservationSuite::SpecificationComparison => self.specification_comparison,
        }
    }
    pub fn run_case<T>(
        &mut self,
        suite: ConservationSuite,
        plan: &ShadingPlan,
        test: impl FnOnce() -> T,
    ) -> Result<ConservationCaseResult<T>, CompositionError> {
        let counts = match suite {
            ConservationSuite::WhiteFurnace => &mut self.white_furnace,
            ConservationSuite::SpecificationComparison => &mut self.specification_comparison,
        };
        if matches!(plan, ShadingPlan::MaskBlend { .. }) {
            counts.skipped_mask_blend = counts
                .skipped_mask_blend
                .checked_add(1)
                .ok_or(CompositionError::CounterOverflow)?;
            Ok(ConservationCaseResult::SkippedMaskBlend)
        } else {
            counts.admitted = counts
                .admitted
                .checked_add(1)
                .ok_or(CompositionError::CounterOverflow)?;
            Ok(ConservationCaseResult::Tested(test()))
        }
    }
}
