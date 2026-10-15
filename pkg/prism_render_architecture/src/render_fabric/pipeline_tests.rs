use super::{
    pipeline_state_for_material, MaterialFeatureFlags, MaterialRenderClass, PipelineBinClass,
    PipelineBlend, PipelineCullMode, PipelineDepthTest, PipelineFeatureSet, PipelinePhase,
    PipelineState, PipelineStateError, PipelineTarget, RenderDomain,
};

#[test]
fn pipeline_mapping_is_const_evaluable_for_every_domain_class_and_feature_set() {
    const FEATURES: [PipelineFeatureSet; 4] = [
        PipelineFeatureSet::NONE,
        PipelineFeatureSet::DOUBLE_SIDED,
        PipelineFeatureSet::WATER,
        PipelineFeatureSet::DOUBLE_SIDED.union(PipelineFeatureSet::WATER),
    ];
    // Build the whole table at compile time so a non-const operation in either
    // public resolver fails compilation, even if runtime tests still pass.
    const STATES: [[[Result<PipelineState, PipelineStateError>; FEATURES.len()];
        MaterialRenderClass::ALL.len()]; RenderDomain::ALL.len()] = {
        let mut states = [[[PipelineState::resolve(
            RenderDomain::Surface,
            MaterialRenderClass::Opaque,
            PipelineFeatureSet::NONE,
        ); FEATURES.len()]; MaterialRenderClass::ALL.len()];
            RenderDomain::ALL.len()];
        let mut domain = 0;
        while domain < RenderDomain::ALL.len() {
            let mut class = 0;
            while class < MaterialRenderClass::ALL.len() {
                let mut feature = 0;
                while feature < FEATURES.len() {
                    states[domain][class][feature] = pipeline_state_for_material(
                        RenderDomain::ALL[domain],
                        MaterialRenderClass::ALL[class],
                        FEATURES[feature],
                    );
                    feature += 1;
                }
                class += 1;
            }
            domain += 1;
        }
        states
    };

    for (domain_index, domain) in RenderDomain::ALL.into_iter().enumerate() {
        for (class_index, render_class) in MaterialRenderClass::ALL.into_iter().enumerate() {
            for (feature_index, features) in FEATURES.into_iter().enumerate() {
                let result = STATES[domain_index][class_index][feature_index];
                assert_eq!(
                    result,
                    PipelineState::resolve(domain, render_class, features),
                    "const/runtime mismatch for {domain:?}/{render_class:?}/{features:?}",
                );
                if features.contains(PipelineFeatureSet::WATER) && domain != RenderDomain::Surface {
                    assert_eq!(
                        result,
                        Err(PipelineStateError::WaterRequiresSurface { domain })
                    );
                } else {
                    let state = result.expect("valid const pipeline state");
                    assert_eq!(state.decode_bin_class(), (domain, render_class));
                    if features.contains(PipelineFeatureSet::WATER) {
                        assert_eq!(state.phase, PipelinePhase::Transparent);
                        assert_eq!(state.target, PipelineTarget::HdrTransparent);
                        assert_eq!(state.blend, PipelineBlend::PremultipliedAlpha);
                        assert_eq!(state.depth_test, PipelineDepthTest::LessEqual);
                        assert!(!state.depth_write);
                        assert_eq!(state.cull_mode, PipelineCullMode::None);
                    }
                }
            }
        }
    }
}

#[test]
fn const_pipeline_validation_preserves_unknown_bits_and_error_precedence() {
    const RAW_BITS: u16 = (1 << 15) | PipelineFeatureSet::WATER.bits();
    // Both inputs violate the water-domain rule as well. Unknown feature bits
    // must be reported first, with the original mask available for diagnostics.
    const INVALID: [Result<PipelineState, PipelineStateError>; 2] = [
        PipelineState::resolve(
            RenderDomain::Volume,
            MaterialRenderClass::Opaque,
            PipelineFeatureSet::from_bits_retain(RAW_BITS),
        ),
        pipeline_state_for_material(
            RenderDomain::Particle,
            MaterialRenderClass::Transparent,
            PipelineFeatureSet::from_bits_retain(RAW_BITS),
        ),
    ];
    for result in INVALID {
        assert_eq!(
            result,
            Err(PipelineStateError::UnknownFeatureBits(RAW_BITS))
        );
    }
}

#[test]
fn every_domain_and_render_class_has_a_complete_typed_state() {
    for domain in RenderDomain::ALL {
        for render_class in MaterialRenderClass::ALL {
            let state = PipelineState::resolve(domain, render_class, PipelineFeatureSet::NONE)
                .expect("every domain/class pair has a state");
            assert_eq!(state.domain, domain);
            assert_eq!(state.render_class, render_class);
            assert_eq!(state.bin_class.domain(), domain);
            assert_eq!(state.bin_class.render_class(), render_class);
            assert_eq!(state.decode_bin_class(), (domain, render_class));
            assert_eq!(
                state.bin_class,
                PipelineBinClass::from_parts(domain, render_class)
            );
            assert!(state.bin_class.index() < PipelineBinClass::COUNT);
            assert!(matches!(
                (state.phase, state.target, state.blend, state.depth_test),
                (
                    PipelinePhase::Opaque,
                    PipelineTarget::Opaque,
                    PipelineBlend::Opaque,
                    PipelineDepthTest::LessEqual
                ) | (
                    PipelinePhase::Transparent,
                    PipelineTarget::HdrTransparent,
                    PipelineBlend::PremultipliedAlpha,
                    PipelineDepthTest::LessEqual
                ) | (
                    PipelinePhase::Transparent,
                    PipelineTarget::HdrTransparent,
                    PipelineBlend::Additive,
                    PipelineDepthTest::LessEqual
                ) | (
                    PipelinePhase::Transparent,
                    PipelineTarget::HdrTransparent,
                    PipelineBlend::Alpha,
                    PipelineDepthTest::LessEqual
                ) | (
                    PipelinePhase::Custom,
                    PipelineTarget::HdrTransparent,
                    PipelineBlend::PremultipliedAlpha,
                    PipelineDepthTest::LessEqual
                )
            ));
            assert_eq!(
                state.is_transparent(),
                !state.depth_write || state.phase != PipelinePhase::Opaque
            );
        }
    }
}

#[test]
fn every_domain_class_and_feature_combination_has_an_explicit_result() {
    let feature_sets = [
        PipelineFeatureSet::NONE,
        PipelineFeatureSet::DOUBLE_SIDED,
        PipelineFeatureSet::WATER,
        PipelineFeatureSet::DOUBLE_SIDED | PipelineFeatureSet::WATER,
    ];
    for domain in RenderDomain::ALL {
        for render_class in MaterialRenderClass::ALL {
            for features in feature_sets {
                let result = PipelineState::resolve(domain, render_class, features);
                if features.contains(PipelineFeatureSet::WATER) && domain != RenderDomain::Surface {
                    assert_eq!(
                        result,
                        Err(PipelineStateError::WaterRequiresSurface { domain })
                    );
                } else {
                    let state = result.expect("valid feature combination");
                    assert_eq!(state.domain, domain);
                    assert_eq!(state.render_class, render_class);
                    assert_eq!(state.features, features);
                }
            }
        }
    }
}

#[test]
fn surface_class_mapping_is_explicit_and_double_sided_is_monotonic() {
    for render_class in MaterialRenderClass::ALL {
        let baseline = pipeline_state_for_material(
            RenderDomain::Surface,
            render_class,
            MaterialFeatureFlags::NONE,
        )
        .unwrap();
        let double_sided = pipeline_state_for_material(
            RenderDomain::Surface,
            render_class,
            MaterialFeatureFlags::DOUBLE_SIDED,
        )
        .unwrap();
        assert_eq!(double_sided.phase, baseline.phase);
        assert_eq!(double_sided.target, baseline.target);
        assert_eq!(double_sided.blend, baseline.blend);
        assert_eq!(double_sided.depth_test, baseline.depth_test);
        assert_eq!(double_sided.depth_write, baseline.depth_write);
        assert_eq!(double_sided.cull_mode, PipelineCullMode::None);
        assert_eq!(double_sided.features, MaterialFeatureFlags::DOUBLE_SIDED);
    }
}

#[test]
fn water_is_a_surface_feature_and_uses_the_declared_transparent_contract() {
    for render_class in MaterialRenderClass::ALL {
        let state = PipelineState::resolve(
            RenderDomain::Surface,
            render_class,
            MaterialFeatureFlags::WATER,
        )
        .unwrap();
        assert_eq!(state.phase, PipelinePhase::Transparent);
        assert_eq!(state.target, PipelineTarget::HdrTransparent);
        assert_eq!(state.blend, PipelineBlend::PremultipliedAlpha);
        assert_eq!(state.depth_test, PipelineDepthTest::LessEqual);
        assert!(!state.depth_write);
        assert_eq!(state.cull_mode, PipelineCullMode::None);
        assert_eq!(state.domain, RenderDomain::Surface);
        assert_eq!(state.render_class, render_class);
    }
    assert_eq!(
        PipelineState::resolve(
            RenderDomain::Volume,
            MaterialRenderClass::Opaque,
            MaterialFeatureFlags::WATER,
        ),
        Err(PipelineStateError::WaterRequiresSurface {
            domain: RenderDomain::Volume,
        })
    );
}

#[test]
fn domain_owned_pipeline_families_do_not_depend_on_render_class_names() {
    for render_class in MaterialRenderClass::ALL {
        let volume =
            PipelineState::resolve(RenderDomain::Volume, render_class, PipelineFeatureSet::NONE)
                .unwrap();
        assert_eq!(volume.phase, PipelinePhase::Custom);
        assert_eq!(volume.target, PipelineTarget::HdrTransparent);
        assert_eq!(volume.blend, PipelineBlend::PremultipliedAlpha);
        assert!(!volume.depth_write);
        assert_eq!(volume.cull_mode, PipelineCullMode::None);

        let decal =
            PipelineState::resolve(RenderDomain::Decal, render_class, PipelineFeatureSet::NONE)
                .unwrap();
        assert_eq!(decal.phase, PipelinePhase::Transparent);
        assert_eq!(decal.target, PipelineTarget::HdrTransparent);
        assert_eq!(decal.blend, PipelineBlend::Alpha);
        assert!(!decal.depth_write);
        assert_eq!(decal.cull_mode, PipelineCullMode::Back);
    }
}

#[test]
fn bin_class_is_dense_reversible_and_not_declaration_order_dependent() {
    assert_eq!(PipelineBinClass::COUNT, 35);
    for index in 0..PipelineBinClass::COUNT {
        let key = PipelineBinClass::from_u8(index).unwrap();
        assert_eq!(key.index(), index);
        assert_eq!(PipelineBinClass::from_u8(key.index()), Some(key));
    }
    assert_eq!(PipelineBinClass::from_u8(35), None);
    assert_ne!(
        PipelineBinClass::from_parts(RenderDomain::Surface, MaterialRenderClass::Opaque),
        PipelineBinClass::from_parts(RenderDomain::Particle, MaterialRenderClass::Opaque)
    );
}

#[test]
fn unknown_features_have_an_explicit_error_path() {
    let features = PipelineFeatureSet::from_bits_retain(1 << 15);
    assert_eq!(
        PipelineState::resolve(RenderDomain::Surface, MaterialRenderClass::Opaque, features),
        Err(PipelineStateError::UnknownFeatureBits(1 << 15))
    );
    assert!(PipelineFeatureSet::from_bits(1 << 15).is_none());
}
