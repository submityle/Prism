//! Backend-neutral CPU contract for Native material response composition.
//!
//! This module deliberately does not know about RHI buffers, shaders, GBuffer
//! layouts, or fallback resources. It validates the metadata that a Native
//! material consumer must provide and delegates response-channel arithmetic to
//! `prism_render_shading`'s canonical `ResponseOutput` contract.

use prism_render_material::{BlendDomain, ChannelMask, ProgramRef, ResponseChannel, ShadingPlan};
use prism_render_shading::{
    blend_responses, evaluate_response_pair, AlphaMergeRule, AuxiliaryMergeRules, CompositionError,
    CompositionOptions, LightingContext, ResponseOutput, SharedSurfaceAlpha,
};

/// Backend-neutral metadata required by a Native material composition consumer.
///
/// The fields mirror the semantic contents of an execution-plan row without
/// inventing a second GPU layout. A producer must provide the same program
/// identities and exact blended-channel mask as the authored [`ShadingPlan`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeMaterialCompositionMetadata {
    pub(crate) plan: ShadingPlan,
    pub(crate) left_program: ProgramRef,
    pub(crate) right_program: Option<ProgramRef>,
    pub(crate) blended_channels: ChannelMask,
}

impl NativeMaterialCompositionMetadata {
    pub(crate) fn from_plan(plan: ShadingPlan) -> Self {
        let (left_program, right_program, blended_channels) = match &plan {
            ShadingPlan::Single { program } => (*program, None, ChannelMask::default()),
            ShadingPlan::MaskBlend {
                left,
                right,
                domain,
                ..
            } => (*left, Some(*right), expected_channels(*domain)),
        };
        Self {
            plan,
            left_program,
            right_program,
            blended_channels,
        }
    }

    fn validate(&self) -> Result<(), NativeMaterialCompositionError> {
        match &self.plan {
            ShadingPlan::Single { program } => {
                if self.left_program != *program
                    || self.right_program.is_some()
                    || self.blended_channels != ChannelMask::default()
                {
                    return Err(NativeMaterialCompositionError::InvalidMetadata);
                }
            }
            ShadingPlan::MaskBlend {
                left,
                right,
                domain,
                ..
            } => {
                if self.left_program != *left
                    || self.right_program != Some(*right)
                    || self.blended_channels != expected_channels(*domain)
                {
                    return Err(NativeMaterialCompositionError::InvalidMetadata);
                }
            }
        }
        Ok(())
    }
}

/// Errors are intentionally explicit: Native must reject an incomplete or
/// contradictory response contract instead of silently choosing an arm or a
/// default channel set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeMaterialCompositionError {
    InvalidMetadata,
    MissingRightResponse,
    UnexpectedRightResponse,
    Composition(CompositionError),
}

/// CPU oracle/contract used by Native material tests and future CPU admission.
///
/// The adapter only composes already-evaluated response arms. It does not fake
/// a registered evaluator, synthesize albedo lighting, or provide a fallback
/// response for missing inputs.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NativeMaterialCompositionAdapter;

impl NativeMaterialCompositionAdapter {
    pub(crate) fn compose(
        &self,
        metadata: &NativeMaterialCompositionMetadata,
        left: ResponseOutput,
        right: Option<ResponseOutput>,
        mask: f32,
        shared_alpha: SharedSurfaceAlpha,
        alpha_rule: AlphaMergeRule,
        auxiliary_rules: &AuxiliaryMergeRules,
    ) -> Result<ResponseOutput, NativeMaterialCompositionError> {
        metadata.validate()?;

        match &metadata.plan {
            ShadingPlan::Single { .. } => {
                if right.is_some() {
                    return Err(NativeMaterialCompositionError::UnexpectedRightResponse);
                }
                if !mask.is_finite() || !(0.0..=1.0).contains(&mask) {
                    return Err(NativeMaterialCompositionError::Composition(
                        CompositionError::InvalidMask,
                    ));
                }
                left.validate()
                    .map_err(NativeMaterialCompositionError::Composition)?;
                Ok(left)
            }
            ShadingPlan::MaskBlend { domain, .. } => {
                let right = right.ok_or(NativeMaterialCompositionError::MissingRightResponse)?;
                let direct = [];
                let shared_terms = ();
                let lighting = LightingContext {
                    direct: &direct,
                    indirect_diffuse_irradiance: None,
                    reflection_radiance: None,
                    transmission_radiance: None,
                    shared_terms: &shared_terms,
                };
                let (left, right) = evaluate_response_pair(&lighting, |_| left, |_| right)
                    .map_err(NativeMaterialCompositionError::Composition)?;
                blend_responses(
                    &left,
                    &right,
                    CompositionOptions {
                        mask,
                        domain: *domain,
                        shared_alpha,
                        alpha_rule,
                        auxiliary_rules,
                    },
                )
                .map_err(NativeMaterialCompositionError::Composition)
            }
        }
    }
}

fn expected_channels(domain: BlendDomain) -> ChannelMask {
    match domain {
        BlendDomain::FullRadiance => ChannelMask::radiance(),
        BlendDomain::DirectOnly => ChannelMask::one(ResponseChannel::Direct),
        BlendDomain::CustomChannelMask(channels) => channels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(seed: f32) -> ResponseOutput {
        ResponseOutput {
            direct: [seed, seed + 1.0, seed + 2.0],
            indirect_diffuse: [seed + 3.0, seed + 4.0, seed + 5.0],
            reflection: [seed + 6.0, seed + 7.0, seed + 8.0],
            transmission: [seed + 9.0, seed + 10.0, seed + 11.0],
            emission: [seed + 12.0, seed + 13.0, seed + 14.0],
            alpha: 0.8,
            auxiliary: Default::default(),
        }
    }

    fn shared_alpha() -> SharedSurfaceAlpha {
        SharedSurfaceAlpha::new(0.65, prism_render_material::MaterialRenderClass::Opaque)
            .expect("fixture alpha is valid")
    }

    fn options() -> AuxiliaryMergeRules {
        AuxiliaryMergeRules::default()
    }

    #[test]
    fn single_returns_one_verified_response_without_a_second_arm() {
        let adapter = NativeMaterialCompositionAdapter;
        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::single_builtin(
            prism_render_material::BuiltinProgram::Lit,
        ));
        let left = response(1.0);
        let expected = left.clone();
        let actual = adapter
            .compose(
                &metadata,
                left,
                None,
                0.25,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            )
            .expect("valid single response");
        assert_eq!(actual, expected);
    }

    #[test]
    fn mask_blend_full_radiance_blends_all_radiance_channels() {
        let adapter = NativeMaterialCompositionAdapter;
        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: BlendDomain::FullRadiance,
        });
        let left = response(0.0);
        let right = response(10.0);
        let actual = adapter
            .compose(
                &metadata,
                left.clone(),
                Some(right.clone()),
                0.25,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            )
            .expect("valid full-radiance blend");

        assert_eq!(actual.direct, [2.5, 3.5, 4.5]);
        assert_eq!(actual.indirect_diffuse, [5.5, 6.5, 7.5]);
        assert_eq!(actual.reflection, [8.5, 9.5, 10.5]);
        assert_eq!(actual.transmission, [11.5, 12.5, 13.5]);
        assert_eq!(actual.emission, [14.5, 15.5, 16.5]);
        assert_eq!(actual.alpha, 0.65);
    }

    #[test]
    fn mask_blend_direct_only_preserves_non_direct_channels() {
        let adapter = NativeMaterialCompositionAdapter;
        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Unlit),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: BlendDomain::DirectOnly,
        });
        let left = response(0.0);
        let right = response(10.0);
        let actual = adapter
            .compose(
                &metadata,
                left.clone(),
                Some(right),
                0.5,
                shared_alpha(),
                AlphaMergeRule::Left,
                &options(),
            )
            .expect("valid direct-only blend");

        assert_eq!(actual.direct, [5.0, 6.0, 7.0]);
        assert_eq!(actual.indirect_diffuse, left.indirect_diffuse);
        assert_eq!(actual.reflection, left.reflection);
        assert_eq!(actual.transmission, left.transmission);
        assert_eq!(actual.emission, left.emission);
        assert_eq!(actual.alpha, left.alpha);
    }

    #[test]
    fn custom_channel_mask_blends_only_the_declared_dynamic_channels() {
        let adapter = NativeMaterialCompositionAdapter;
        let channels = ChannelMask::one(ResponseChannel::Reflection)
            .union(ChannelMask::one(ResponseChannel::Emission));
        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::Channel("wetness".into()),
            domain: BlendDomain::CustomChannelMask(channels),
        });
        let left = response(0.0);
        let right = response(20.0);
        let actual = adapter
            .compose(
                &metadata,
                left.clone(),
                Some(right),
                0.25,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            )
            .expect("valid custom-channel blend");

        assert_eq!(actual.direct, left.direct);
        assert_eq!(actual.indirect_diffuse, left.indirect_diffuse);
        assert_eq!(actual.reflection, [11.0, 12.0, 13.0]);
        assert_eq!(actual.transmission, left.transmission);
        assert_eq!(actual.emission, [17.0, 18.0, 19.0]);
    }

    #[test]
    fn contradictory_metadata_and_missing_arms_fail_closed() {
        let adapter = NativeMaterialCompositionAdapter;
        let valid_plan = ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: BlendDomain::DirectOnly,
        };
        let mut metadata = NativeMaterialCompositionMetadata::from_plan(valid_plan);
        metadata.blended_channels = ChannelMask::radiance();
        assert_eq!(
            adapter.compose(
                &metadata,
                response(0.0),
                Some(response(1.0)),
                0.5,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            ),
            Err(NativeMaterialCompositionError::InvalidMetadata)
        );

        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: BlendDomain::DirectOnly,
        });
        assert_eq!(
            adapter.compose(
                &metadata,
                response(0.0),
                None,
                0.5,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            ),
            Err(NativeMaterialCompositionError::MissingRightResponse)
        );

        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::single_builtin(
            prism_render_material::BuiltinProgram::Lit,
        ));
        assert_eq!(
            adapter.compose(
                &metadata,
                response(0.0),
                Some(response(1.0)),
                0.5,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            ),
            Err(NativeMaterialCompositionError::UnexpectedRightResponse)
        );
    }

    #[test]
    fn invalid_channel_mask_and_non_finite_response_fail_closed() {
        let adapter = NativeMaterialCompositionAdapter;
        let invalid_channels = ChannelMask::one(ResponseChannel::Alpha);
        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Lit),
            right: ProgramRef::Builtin(prism_render_material::BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: BlendDomain::CustomChannelMask(invalid_channels),
        });
        assert_eq!(
            adapter.compose(
                &metadata,
                response(0.0),
                Some(response(1.0)),
                0.5,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            ),
            Err(NativeMaterialCompositionError::Composition(
                CompositionError::InvalidChannelMask
            ))
        );

        let metadata = NativeMaterialCompositionMetadata::from_plan(ShadingPlan::single_builtin(
            prism_render_material::BuiltinProgram::Lit,
        ));
        let mut invalid = response(0.0);
        invalid.direct[0] = f32::NAN;
        assert_eq!(
            adapter.compose(
                &metadata,
                invalid,
                None,
                0.0,
                shared_alpha(),
                AlphaMergeRule::Shared,
                &options(),
            ),
            Err(NativeMaterialCompositionError::Composition(
                CompositionError::NonFiniteRadiance(ResponseChannel::Direct)
            ))
        );
    }
}
