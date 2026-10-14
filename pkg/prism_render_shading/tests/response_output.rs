//! CPU response composition contracts, arithmetic, and conservation admission.

use prism_render_material::{
    BuiltinProgram, MaterialRenderClass, ProgramRef, ResponseMaskSource, ShadingPlan,
};
use prism_render_shading::response_output::*;
use prism_render_shading::{
    evaluate_principled_direct, evaluate_stylized_direct,
    validate_response_composition_inputs as validate_composition_for_gpu, DirectLightSample,
    ShadingFrame, StylizedParams, SurfaceSample,
};
use std::{cell::Cell, collections::BTreeMap};

fn lighting(direct: &[DirectLightSample]) -> LightingContext<'_> {
    LightingContext {
        direct,
        indirect_diffuse_irradiance: Some([1.0; 3]),
        reflection_radiance: Some([2.0; 3]),
        transmission_radiance: Some([3.0; 3]),
        shared_terms: &(),
    }
}
fn output(base: f32) -> ResponseOutput {
    ResponseOutput {
        direct: [base, base + 1.0, base + 2.0],
        indirect_diffuse: [base + 3.0; 3],
        reflection: [base + 4.0; 3],
        transmission: [base + 5.0; 3],
        emission: [base + 6.0; 3],
        alpha: base / 100.0,
        auxiliary: Auxiliary::default(),
    }
}
fn options(rules: &AuxiliaryMergeRules, mask: f32, domain: BlendDomain) -> CompositionOptions<'_> {
    CompositionOptions {
        mask,
        domain,
        shared_alpha: SharedSurfaceAlpha::new(0.75, MaterialRenderClass::Transparent).unwrap(),
        alpha_rule: AlphaMergeRule::Shared,
        auxiliary_rules: rules,
    }
}
fn compose(
    left: ResponseOutput,
    right: ResponseOutput,
    options: CompositionOptions<'_>,
) -> Result<ResponseOutput, CompositionError> {
    let ctx = lighting(&[]);
    blend_responses(&ctx.bind(left)?, &ctx.bind(right)?, options)
}
fn channels(value: &ResponseOutput) -> [[f32; 3]; 5] {
    [
        value.direct,
        value.indirect_diffuse,
        value.reflection,
        value.transmission,
        value.emission,
    ]
}
fn aux(entries: &[(&str, AuxiliaryValue)]) -> Auxiliary {
    Auxiliary {
        fields: entries
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    }
}
fn rules(entries: &[(&str, AuxiliaryMergeRule)]) -> AuxiliaryMergeRules {
    AuxiliaryMergeRules {
        fields: entries.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
        shared: Auxiliary::default(),
    }
}

#[test]
fn raw_validation_returns_same_auxiliary_without_lighting_or_input_mutation() {
    let mut left = output(10.0);
    let mut right = output(30.0);
    left.auxiliary = aux(&[
        ("weight", AuxiliaryValue::Scalar(2.0)),
        ("flags", AuxiliaryValue::Flags(0b0011)),
        ("id", AuxiliaryValue::Unsigned(7)),
    ]);
    right.auxiliary = aux(&[
        ("weight", AuxiliaryValue::Scalar(6.0)),
        ("flags", AuxiliaryValue::Flags(0b0101)),
        ("id", AuxiliaryValue::Unsigned(7)),
    ]);
    let mut rules = rules(&[
        ("weight", AuxiliaryMergeRule::Linear),
        ("flags", AuxiliaryMergeRule::BitOr),
        ("id", AuxiliaryMergeRule::RequireEqual),
        ("shared", AuxiliaryMergeRule::Shared),
    ]);
    rules.shared = aux(&[("shared", AuxiliaryValue::Vector3([4.0; 3]))]);
    let before = (left.clone(), right.clone());
    let opts = options(&rules, 0.25, BlendDomain::DirectOnly);
    // Crate-root reexport is the intended GPU build/upload seam.
    let merged = validate_composition_for_gpu(&left, &right, opts).unwrap();
    assert_eq!(
        merged,
        aux(&[
            ("weight", AuxiliaryValue::Scalar(3.0)),
            ("flags", AuxiliaryValue::Flags(0b0111)),
            ("id", AuxiliaryValue::Unsigned(7)),
            ("shared", AuxiliaryValue::Vector3([4.0; 3])),
        ])
    );
    assert_eq!((&left, &right), (&before.0, &before.1));
    assert_eq!(compose(left, right, opts).unwrap().auxiliary, merged);
}

#[test]
fn raw_validation_checks_domain_and_mask_without_lighting() {
    let left = output(10.0);
    let right = output(30.0);
    let rules = rules(&[]);
    for mask in 1..32 {
        assert_eq!(
            validate_response_composition_inputs(
                &left,
                &right,
                options(
                    &rules,
                    0.5,
                    BlendDomain::CustomChannelMask(ChannelMask(mask))
                ),
            ),
            Ok(Auxiliary::default())
        );
    }
    for mask in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            validate_response_composition_inputs(
                &left,
                &right,
                options(&rules, mask, BlendDomain::FullRadiance),
            ),
            Err(CompositionError::InvalidMask)
        );
    }
    for mask in [0, 1 << 5, 1 << 6, 1 << 7, u32::MAX, 1 | 1 << 5] {
        assert_eq!(
            validate_response_composition_inputs(
                &left,
                &right,
                options(
                    &rules,
                    0.5,
                    BlendDomain::CustomChannelMask(ChannelMask(mask))
                ),
            ),
            Err(CompositionError::InvalidChannelMask)
        );
    }
}

#[test]
fn raw_validation_rejects_bad_unselected_channels_alpha_and_auxiliary_in_either_arm() {
    let good = output(10.0);
    let rules = rules(&[("ignored", AuxiliaryMergeRule::Left)]);
    let mut good = good;
    good.auxiliary = aux(&[("ignored", AuxiliaryValue::Scalar(1.0))]);
    let mut bad_channel = good.clone();
    bad_channel.emission[1] = f32::NAN;
    let mut bad_alpha = good.clone();
    bad_alpha.alpha = f32::INFINITY;
    let mut bad_auxiliary = good.clone();
    bad_auxiliary.auxiliary = aux(&[("ignored", AuxiliaryValue::Scalar(f32::NAN))]);
    for (bad, expected) in [
        (
            bad_channel,
            CompositionError::NonFiniteRadiance(ResponseChannel::Emission),
        ),
        (bad_alpha, CompositionError::InvalidAlpha),
        (
            bad_auxiliary,
            CompositionError::InvalidAuxiliary("ignored".into()),
        ),
    ] {
        for (left, right) in [(&bad, &good), (&good, &bad)] {
            assert_eq!(
                validate_response_composition_inputs(
                    left,
                    right,
                    options(&rules, 0.0, BlendDomain::DirectOnly),
                ),
                Err(expected.clone())
            );
        }
    }
}

#[test]
fn raw_validation_uses_authoritative_auxiliary_errors() {
    let mut left = output(10.0);
    let mut right = output(30.0);
    left.auxiliary = aux(&[("value", AuxiliaryValue::Scalar(1.0))]);
    right.auxiliary = aux(&[("value", AuxiliaryValue::Scalar(2.0))]);
    for (rules, expected) in [
        (
            rules(&[]),
            CompositionError::MissingAuxiliaryRule("value".into()),
        ),
        (
            rules(&[("value", AuxiliaryMergeRule::Shared)]),
            CompositionError::MissingAuxiliaryValue {
                field: "value".into(),
                source: "shared",
            },
        ),
        (
            rules(&[("value", AuxiliaryMergeRule::RequireEqual)]),
            CompositionError::UnequalAuxiliary("value".into()),
        ),
        (
            rules(&[("value", AuxiliaryMergeRule::BitOr)]),
            CompositionError::InvalidAuxiliaryOperation("value".into()),
        ),
        (
            rules(&[
                ("value", AuxiliaryMergeRule::Left),
                ("unused", AuxiliaryMergeRule::Left),
            ]),
            CompositionError::UnusedAuxiliaryRule("unused".into()),
        ),
    ] {
        let opts = options(&rules, 0.5, BlendDomain::FullRadiance);
        assert_eq!(
            validate_response_composition_inputs(&left, &right, opts),
            Err(expected.clone())
        );
        assert_eq!(compose(left.clone(), right.clone(), opts), Err(expected));
    }
    let mut rules = rules(&[("value", AuxiliaryMergeRule::Left)]);
    right.auxiliary = aux(&[("value", AuxiliaryValue::Unsigned(2))]);
    assert_eq!(
        validate_response_composition_inputs(
            &left,
            &right,
            options(&rules, 0.5, BlendDomain::FullRadiance)
        ),
        Err(CompositionError::AuxiliaryTypeMismatch("value".into()))
    );
    right.auxiliary = left.auxiliary.clone();
    rules.shared = aux(&[("value", AuxiliaryValue::Scalar(f32::INFINITY))]);
    assert_eq!(
        validate_response_composition_inputs(
            &left,
            &right,
            options(&rules, 0.5, BlendDomain::FullRadiance)
        ),
        Err(CompositionError::InvalidAuxiliary("value".into()))
    );
}

#[test]
fn raw_validation_accepts_finite_channels_even_if_their_sum_would_overflow() {
    let value = ResponseOutput {
        direct: [f32::MAX; 3],
        indirect_diffuse: [f32::MAX; 3],
        reflection: [f32::MAX; 3],
        transmission: [f32::MAX; 3],
        emission: [f32::MAX; 3],
        alpha: 1.0,
        auxiliary: Auxiliary::default(),
    };
    assert!(value.radiance().into_iter().all(f32::is_infinite));
    assert_eq!(
        validate_response_composition_inputs(
            &value,
            &value,
            options(&rules(&[]), 0.5, BlendDomain::FullRadiance),
        ),
        Ok(Auxiliary::default())
    );
}

#[test]
fn full_radiance_lerps_all_five_channels_without_readding_emission() {
    let left = output(10.0);
    let right = output(30.0);
    let result = compose(
        left.clone(),
        right.clone(),
        options(&rules(&[]), 0.25, BlendDomain::FullRadiance),
    )
    .unwrap();
    for ((a, b), mixed) in channels(&left)
        .into_iter()
        .zip(channels(&right))
        .zip(channels(&result))
    {
        assert_eq!(mixed, core::array::from_fn(|i| 0.75 * a[i] + 0.25 * b[i]));
    }
    assert_eq!(result.emission, [21.0; 3]);
    assert_eq!(result.radiance(), [93.0, 94.0, 95.0]);
}

#[test]
fn direct_only_preserves_all_other_left_channels() {
    let a = output(10.0);
    let b = output(30.0);
    let result = compose(
        a.clone(),
        b,
        options(&rules(&[]), 1.0, BlendDomain::DirectOnly),
    )
    .unwrap();
    assert_eq!(result.direct, [30.0, 31.0, 32.0]);
    assert_eq!(&channels(&result)[1..], &channels(&a)[1..]);
}

#[test]
fn every_nonempty_radiance_mask_has_exact_selected_and_left_baseline_channels() {
    let a = output(10.0);
    let b = output(30.0);
    for mask in 1..32 {
        let result = compose(
            a.clone(),
            b.clone(),
            options(
                &rules(&[]),
                0.5,
                BlendDomain::CustomChannelMask(ChannelMask(mask)),
            ),
        )
        .unwrap();
        for (i, value) in channels(&result).into_iter().enumerate() {
            let expected = if mask & (1 << i) != 0 {
                channels(&output(20.0))[i]
            } else {
                channels(&a)[i]
            };
            assert_eq!(value, expected, "mask {mask}, channel {i}");
        }
    }
}

#[test]
fn empty_unknown_alpha_and_auxiliary_channel_masks_fail() {
    for mask in [0, 1 << 5, 1 << 6, 1 << 7, u32::MAX, 1 | 1 << 5] {
        assert_eq!(
            compose(
                output(10.0),
                output(30.0),
                options(
                    &rules(&[]),
                    0.5,
                    BlendDomain::CustomChannelMask(ChannelMask(mask))
                )
            ),
            Err(CompositionError::InvalidChannelMask)
        );
    }
}

#[test]
fn endpoints_are_bit_exact_for_selected_channels_including_signed_zero() {
    let mut a = output(10.0);
    a.direct = [-0.0, f32::MAX, -f32::MAX];
    let mut b = output(30.0);
    b.direct = [0.0, -f32::MAX, f32::MAX];
    for (mask, expected) in [(0.0, &a), (1.0, &b)] {
        let result = compose(
            a.clone(),
            b.clone(),
            options(&rules(&[]), mask, BlendDomain::FullRadiance),
        )
        .unwrap();
        for (actual, expected) in channels(&result)
            .into_iter()
            .flatten()
            .zip(channels(expected).into_iter().flatten())
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    let result = compose(a, b, options(&rules(&[]), 0.5, BlendDomain::FullRadiance)).unwrap();
    assert_eq!(result.direct, [0.0; 3]);
}

#[test]
fn invalid_mask_is_not_silently_clamped() {
    for mask in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            compose(
                output(10.0),
                output(30.0),
                options(&rules(&[]), mask, BlendDomain::FullRadiance)
            ),
            Err(CompositionError::InvalidMask)
        );
    }
}

#[test]
fn shared_alpha_ignores_arm_alpha_and_mask_for_all_render_classes() {
    let classes = [
        MaterialRenderClass::Opaque,
        MaterialRenderClass::OpaqueTwoSided,
        MaterialRenderClass::Masked,
        MaterialRenderClass::MaskedTwoSided,
        MaterialRenderClass::Transmissive,
        MaterialRenderClass::Transparent,
        MaterialRenderClass::Additive,
    ];
    for class in classes {
        let shared = SharedSurfaceAlpha::new(0.75, class).unwrap();
        assert_eq!(shared.render_class(), class);
        for mask in [0.0, 0.5, 1.0] {
            let rules = rules(&[]);
            let mut opts = options(&rules, mask, BlendDomain::FullRadiance);
            opts.shared_alpha = shared;
            assert_eq!(
                compose(output(10.0), output(30.0), opts).unwrap().alpha,
                0.75
            );
        }
    }
}

#[test]
fn every_alpha_override_is_explicit_and_not_lerped() {
    for (rule, alpha) in [
        (AlphaMergeRule::Shared, 0.75),
        (AlphaMergeRule::Left, 0.1),
        (AlphaMergeRule::Right, 0.3),
        (AlphaMergeRule::Min, 0.1),
        (AlphaMergeRule::Max, 0.3),
    ] {
        let rules = rules(&[]);
        let mut opts = options(&rules, 0.5, BlendDomain::FullRadiance);
        opts.alpha_rule = rule;
        assert_eq!(
            compose(output(10.0), output(30.0), opts).unwrap().alpha,
            alpha
        );
    }
}

#[test]
fn invalid_alpha_and_nonfinite_radiance_fail_even_for_unselected_channels() {
    for alpha in [-1.0, 1.1, f32::NAN, f32::INFINITY] {
        assert_eq!(
            SharedSurfaceAlpha::new(alpha, MaterialRenderClass::Opaque),
            Err(CompositionError::InvalidAlpha)
        );
        let mut a = output(10.0);
        a.alpha = alpha;
        assert_eq!(a.validate(), Err(CompositionError::InvalidAlpha));
    }
    for i in 0..5 {
        let mut a = output(10.0);
        let value = match i {
            0 => &mut a.direct,
            1 => &mut a.indirect_diffuse,
            2 => &mut a.reflection,
            3 => &mut a.transmission,
            _ => &mut a.emission,
        };
        value[1] = f32::NAN;
        assert!(matches!(
            compose(
                output(10.0),
                a,
                options(&rules(&[]), 0.0, BlendDomain::DirectOnly)
            ),
            Err(CompositionError::NonFiniteRadiance(_))
        ));
    }
}

#[test]
fn each_auxiliary_field_uses_its_own_typed_rule() {
    use AuxiliaryMergeRule as R;
    use AuxiliaryValue::*;
    let mut a = output(10.0);
    let mut b = output(30.0);
    a.auxiliary = aux(&[
        ("id", Unsigned(7)),
        ("normal", Vector3([1.0, 0.0, 0.0])),
        ("confidence", Scalar(0.25)),
        ("width", Scalar(2.0)),
        ("vector", Vector3([1.0, 9.0, 4.0])),
        ("flags", Flags(1)),
        ("equal", Unsigned(11)),
    ]);
    b.auxiliary = aux(&[
        ("id", Unsigned(9)),
        ("normal", Vector3([0.0, 1.0, 0.0])),
        ("confidence", Scalar(0.75)),
        ("width", Scalar(4.0)),
        ("vector", Vector3([3.0, 5.0, 8.0])),
        ("flags", Flags(4)),
        ("equal", Unsigned(11)),
    ]);
    let mut r = rules(&[
        ("id", R::Left),
        ("normal", R::Right),
        ("confidence", R::Min),
        ("width", R::Linear),
        ("vector", R::Max),
        ("flags", R::BitOr),
        ("equal", R::RequireEqual),
        ("shared", R::Shared),
    ]);
    r.shared = aux(&[("shared", Unsigned(42))]);
    let result = compose(a, b, options(&r, 0.25, BlendDomain::FullRadiance)).unwrap();
    assert_eq!(
        result.auxiliary,
        aux(&[
            ("id", Unsigned(7)),
            ("normal", Vector3([0.0, 1.0, 0.0])),
            ("confidence", Scalar(0.25)),
            ("width", Scalar(2.5)),
            ("vector", Vector3([3.0, 9.0, 8.0])),
            ("flags", Flags(5)),
            ("equal", Unsigned(11)),
            ("shared", Unsigned(42))
        ])
    );
}

#[test]
fn auxiliary_numeric_operations_cover_scalar_vector_and_unsigned() {
    use AuxiliaryMergeRule as R;
    use AuxiliaryValue::*;
    for (a, b, rule, expected) in [
        (Scalar(2.0), Scalar(6.0), R::Max, Scalar(6.0)),
        (
            Vector3([2.0, 8.0, 4.0]),
            Vector3([6.0, 4.0, 8.0]),
            R::Linear,
            Vector3([3.0, 7.0, 5.0]),
        ),
        (
            Vector3([2.0, 8.0, 4.0]),
            Vector3([6.0, 4.0, 8.0]),
            R::Min,
            Vector3([2.0, 4.0, 4.0]),
        ),
        (Unsigned(2), Unsigned(6), R::Min, Unsigned(2)),
        (Unsigned(2), Unsigned(6), R::Max, Unsigned(6)),
    ] {
        let mut left = output(10.0);
        let mut right = output(30.0);
        left.auxiliary = aux(&[("value", a)]);
        right.auxiliary = aux(&[("value", b)]);
        assert_eq!(
            compose(
                left,
                right,
                options(&rules(&[("value", rule)]), 0.25, BlendDomain::FullRadiance)
            )
            .unwrap()
            .auxiliary,
            aux(&[("value", expected)])
        );
    }
}

#[test]
fn missing_or_unused_auxiliary_rules_are_loud_errors() {
    let mut a = output(10.0);
    a.auxiliary = aux(&[("id", AuxiliaryValue::Unsigned(7))]);
    assert_eq!(
        compose(
            a.clone(),
            output(30.0),
            options(&rules(&[]), 0.0, BlendDomain::FullRadiance)
        ),
        Err(CompositionError::MissingAuxiliaryRule("id".into()))
    );
    assert_eq!(
        compose(
            output(10.0),
            output(30.0),
            options(
                &rules(&[("ghost", AuxiliaryMergeRule::Left)]),
                0.0,
                BlendDomain::FullRadiance
            )
        ),
        Err(CompositionError::UnusedAuxiliaryRule("ghost".into()))
    );
    let result = compose(
        a,
        output(30.0),
        options(
            &rules(&[("id", AuxiliaryMergeRule::Left)]),
            0.5,
            BlendDomain::FullRadiance,
        ),
    )
    .unwrap();
    assert_eq!(result.auxiliary.fields["id"], AuxiliaryValue::Unsigned(7));
}

#[test]
fn auxiliary_rules_never_silently_substitute_a_missing_source() {
    for (rule, source) in [
        (AuxiliaryMergeRule::Shared, "shared"),
        (AuxiliaryMergeRule::Right, "right"),
        (AuxiliaryMergeRule::Linear, "right"),
        (AuxiliaryMergeRule::RequireEqual, "right"),
    ] {
        let mut a = output(10.0);
        a.auxiliary = aux(&[("field", AuxiliaryValue::Scalar(1.0))]);
        assert_eq!(
            compose(
                a,
                output(30.0),
                options(&rules(&[("field", rule)]), 0.0, BlendDomain::FullRadiance)
            ),
            Err(CompositionError::MissingAuxiliaryValue {
                field: "field".into(),
                source
            })
        );
    }
}

#[test]
fn auxiliary_rejects_type_conflicts_id_lerps_flags_min_and_unequal_identity() {
    use AuxiliaryMergeRule as R;
    use AuxiliaryValue::*;
    for (a, b, rule, error) in [
        (
            Scalar(1.0),
            Unsigned(1),
            R::Left,
            CompositionError::AuxiliaryTypeMismatch("f".into()),
        ),
        (
            Unsigned(1),
            Unsigned(2),
            R::Linear,
            CompositionError::InvalidAuxiliaryOperation("f".into()),
        ),
        (
            Flags(1),
            Flags(2),
            R::Min,
            CompositionError::InvalidAuxiliaryOperation("f".into()),
        ),
        (
            Scalar(1.0),
            Scalar(2.0),
            R::BitOr,
            CompositionError::InvalidAuxiliaryOperation("f".into()),
        ),
        (
            Unsigned(1),
            Unsigned(2),
            R::RequireEqual,
            CompositionError::UnequalAuxiliary("f".into()),
        ),
    ] {
        let mut left = output(10.0);
        let mut right = output(30.0);
        left.auxiliary = aux(&[("f", a)]);
        right.auxiliary = aux(&[("f", b)]);
        assert_eq!(
            compose(
                left,
                right,
                options(&rules(&[("f", rule)]), 0.5, BlendDomain::FullRadiance)
            ),
            Err(error)
        );
    }
}

#[test]
fn auxiliary_nonfinite_empty_name_and_invalid_shared_values_fail() {
    for (name, value) in [
        ("", AuxiliaryValue::Unsigned(1)),
        ("bad", AuxiliaryValue::Scalar(f32::NAN)),
        ("bad", AuxiliaryValue::Vector3([1.0, f32::INFINITY, 1.0])),
    ] {
        let mut a = output(10.0);
        a.auxiliary = aux(&[(name, value)]);
        assert_eq!(
            a.validate(),
            Err(CompositionError::InvalidAuxiliary(name.into()))
        );
    }
    let mut r = rules(&[("bad", AuxiliaryMergeRule::Shared)]);
    r.shared = aux(&[("bad", AuxiliaryValue::Scalar(f32::NAN))]);
    assert_eq!(
        compose(
            output(10.0),
            output(30.0),
            options(&r, 0.5, BlendDomain::FullRadiance)
        ),
        Err(CompositionError::InvalidAuxiliary("bad".into()))
    );
}

#[test]
fn split_outputs_must_borrow_identical_lighting_context_not_equal_copies() {
    let a = lighting(&[]);
    let b = lighting(&[]);
    let r = rules(&[]);
    assert_eq!(
        blend_responses(
            &a.bind(output(10.0)).unwrap(),
            &b.bind(output(30.0)).unwrap(),
            options(&r, 0.5, BlendDomain::FullRadiance)
        ),
        Err(CompositionError::DifferentLightingContexts)
    );
    let left = a.bind(output(10.0)).unwrap();
    let right = a.bind(output(30.0)).unwrap();
    assert!(blend_responses(&left, &right, options(&r, 0.5, BlendDomain::FullRadiance)).is_ok());
}

#[test]
fn pair_reads_one_visibility_sample_and_shared_terms_with_two_response_calls() {
    let shadow_samples = Cell::new(0);
    let response_calls = Cell::new(0);
    let lights = [{
        shadow_samples.set(shadow_samples.get() + 1);
        DirectLightSample {
            direction: [0.0, 0.0, 1.0],
            illuminance: [4.0; 3],
            visibility: 0.25,
        }
    }];
    let terms = [7u32, 11];
    let ctx = LightingContext {
        direct: &lights,
        indirect_diffuse_irradiance: Some([2.0; 3]),
        reflection_radiance: None,
        transmission_radiance: None,
        shared_terms: &terms,
    };
    let arm = |lighting: &LightingContext<'_, [u32; 2]>| {
        response_calls.set(response_calls.get() + 1);
        assert!(std::ptr::eq(lighting, &ctx));
        assert!(std::ptr::eq(lighting.shared_terms, &terms));
        let mut result = output(10.0);
        result.direct = [lighting.direct[0].visibility; 3];
        result
    };
    let (a, b) = evaluate_response_pair(&ctx, arm, arm).unwrap();
    assert_eq!(shadow_samples.get(), 1);
    assert_eq!(response_calls.get(), 2);
    assert!(std::ptr::eq(a.lighting(), b.lighting()));
    assert_eq!(
        blend_responses(&a, &b, options(&rules(&[]), 0.5, BlendDomain::FullRadiance))
            .unwrap()
            .direct,
        [0.25; 3]
    );
}

#[test]
fn existing_principled_and_stylized_math_can_feed_the_shared_context() {
    let surface = SurfaceSample::default();
    let frame = ShadingFrame {
        normal: [0.0, 0.0, 1.0],
        view: [0.0, 0.0, 1.0],
        tangent: [1.0, 0.0, 0.0],
        bitangent: [0.0, 1.0, 0.0],
    };
    let samples = [DirectLightSample {
        direction: [0.0, 0.0, 1.0],
        illuminance: [2.0; 3],
        visibility: 0.4,
    }];
    let ctx = lighting(&samples);
    let params = StylizedParams::with_bands(4);
    // Zero emission fixture: these legacy evaluators return aggregate colours,
    // so this test is direct-only integration, NOT full response decomposition.
    let (a, b) = evaluate_response_pair(
        &ctx,
        |l| ResponseOutput {
            direct: evaluate_principled_direct(surface, frame, l.direct[0]),
            alpha: 1.0,
            ..Default::default()
        },
        |l| ResponseOutput {
            direct: evaluate_stylized_direct(surface, frame, l.direct[0], &params),
            alpha: 1.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_ne!(a.output().direct, b.output().direct);
    let result =
        blend_responses(&a, &b, options(&rules(&[]), 0.5, BlendDomain::DirectOnly)).unwrap();
    assert_eq!(
        result.direct,
        core::array::from_fn(|i| 0.5 * a.output().direct[i] + 0.5 * b.output().direct[i])
    );
}

#[test]
fn malformed_lighting_rejected_before_evaluating_either_arm() {
    for visibility in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        let samples = [DirectLightSample {
            direction: [0.0, 0.0, 1.0],
            illuminance: [2.0; 3],
            visibility,
        }];
        let ctx = lighting(&samples);
        let calls = Cell::new(0);
        let arm = |_: &LightingContext<'_>| {
            calls.set(calls.get() + 1);
            output(10.0)
        };
        assert!(matches!(
            evaluate_response_pair(&ctx, arm, arm),
            Err(CompositionError::InvalidLighting)
        ));
        assert_eq!(calls.get(), 0);
    }
    let mut ctx = lighting(&[]);
    ctx.reflection_radiance = Some([f32::NAN; 3]);
    assert_eq!(ctx.validate(), Err(CompositionError::InvalidLighting));
}

fn single() -> ShadingPlan {
    ShadingPlan::Single {
        program: ProgramRef::Builtin(BuiltinProgram::Lit),
    }
}
fn hybrid() -> ShadingPlan {
    ShadingPlan::MaskBlend {
        left: ProgramRef::Builtin(BuiltinProgram::Lit),
        right: ProgramRef::Builtin(BuiltinProgram::Npr),
        mask: ResponseMaskSource::RegionMask,
        domain: BlendDomain::FullRadiance,
    }
}

#[test]
fn mask_blend_materials_are_excluded_from_conservation_suites_and_counted() {
    let mut gate = ConservationGate::default();
    for suite in [
        ConservationSuite::WhiteFurnace,
        ConservationSuite::SpecificationComparison,
    ] {
        for domain in [
            BlendDomain::FullRadiance,
            BlendDomain::DirectOnly,
            BlendDomain::CustomChannelMask(ChannelMask::one(ResponseChannel::Emission)),
        ] {
            let mut plan = hybrid();
            if let ShadingPlan::MaskBlend { domain: d, .. } = &mut plan {
                *d = domain;
            }
            assert_eq!(
                gate.run_case(suite, &plan, || panic!(
                    "MaskBlend must not enter conservation test"
                ))
                .unwrap(),
                ConservationCaseResult::<()>::SkippedMaskBlend
            );
        }
        // Propagate an actual test failure unchanged; never report it as green.
        assert_eq!(
            gate.run_case(suite, &single(), || Err::<(), _>("failed white furnace"))
                .unwrap(),
            ConservationCaseResult::Tested(Err("failed white furnace"))
        );
        assert_eq!(
            gate.counts(suite),
            ConservationSuiteCounts {
                admitted: 1,
                skipped_mask_blend: 3
            }
        );
    }
}

#[test]
fn conservation_suite_counts_are_independent_and_single_callbacks_run_once() {
    let mut gate = ConservationGate::default();
    let calls = Cell::new(0);
    assert_eq!(
        gate.run_case(ConservationSuite::WhiteFurnace, &single(), || {
            calls.set(calls.get() + 1);
            42
        })
        .unwrap(),
        ConservationCaseResult::Tested(42)
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(
        gate.counts(ConservationSuite::WhiteFurnace),
        ConservationSuiteCounts {
            admitted: 1,
            skipped_mask_blend: 0
        }
    );
    assert_eq!(
        gate.counts(ConservationSuite::SpecificationComparison),
        ConservationSuiteCounts::default()
    );
}

#[test]
fn auxiliary_order_is_deterministic() {
    let left = aux(&[
        ("z", AuxiliaryValue::Unsigned(1)),
        ("a", AuxiliaryValue::Unsigned(2)),
    ]);
    let mut a = output(10.0);
    a.auxiliary = left;
    let result = compose(
        a,
        output(30.0),
        options(
            &rules(&[
                ("z", AuxiliaryMergeRule::Left),
                ("a", AuxiliaryMergeRule::Left),
            ]),
            0.5,
            BlendDomain::FullRadiance,
        ),
    )
    .unwrap();
    assert_eq!(
        result
            .auxiliary
            .fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["a", "z"]
    );
    assert_eq!(
        result.auxiliary.fields.len(),
        BTreeMap::from([("a", 1), ("z", 2)]).len()
    );
}
