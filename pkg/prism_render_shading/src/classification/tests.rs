use super::*;
use crate::VisibilityPixel;
use prism_render_architecture::{
    abi::{AbiHash, AbiVersion},
    shader_package::{
        ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
        ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
    },
};
use prism_render_material::{
    fallback_material_header, BlendDomain, ChannelMask, ClosureKind, CompiledEntry,
    MaterialFeatureFlags, ResponseChannel, ResponseMaskSource, ShadingPlan,
};

/// A verified executable fixture, not a fabricated dense execution tuple.
pub(crate) fn execution_fixture(
    programs: &[(ProgramRef, ExecutionFamily)],
) -> (MaterialExecutionTables, Vec<GpuMaterialHeader>) {
    let manifest = ShaderPackageManifest {
        id: ShaderPackageId::new("shading-test"),
        source_revision: "rev-1".into(),
        entry_point: "main".into(),
        abi_version: AbiVersion::from_parts(1, 0),
        abi_hash: AbiHash::from_bytes(b"shading-test"),
        variants: ShaderPackageVariantManifest::from_digests(
            &PermutationSpace::new([2]),
            [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
        )
        .unwrap(),
    };
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        (2 << 16) | 17,
        1,
        (3 << 16) | 14,
        0,
        1,
        (5 << 16) | 15,
        5,
        3,
        0x6e69616d,
        0,
        (6 << 16) | 16,
        3,
        17,
        1,
        1,
        1,
        (2 << 16) | 19,
        1,
        (3 << 16) | 33,
        2,
        1,
        (5 << 16) | 54,
        1,
        3,
        0,
        2,
        (2 << 16) | 248,
        4,
        (1 << 16) | 253,
        (1 << 16) | 56,
    ];
    let entry = CompiledEntry::from_spirv(manifest.clone(), binary).unwrap();
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(entry.artifact().clone())
        .unwrap();
    packages
        .validate(
            &manifest.id,
            &ManifestExpectation::new(manifest.abi_version, manifest.abi_hash),
        )
        .unwrap();
    let mut tables = MaterialExecutionTables::new();
    let package = tables.register_package(&entry, &packages).unwrap();
    let mut headers = Vec::new();
    for &(program, family) in programs {
        let index = tables
            .register_program(
                program,
                &entry,
                package,
                ModuleId(7),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let plan = tables
            .register_plan(
                ShadingPlan::Single { program },
                index,
                None,
                family,
                ChannelMask::default(),
            )
            .unwrap();
        let words = tables
            .bind_raw(plan.get(), index.get(), package.get(), 1)
            .unwrap()
            .to_words();
        headers.push(header_with_words(words));
    }
    (tables, headers)
}

pub(crate) fn header_with_words(words: [u32; 5]) -> GpuMaterialHeader {
    let mut header = fallback_material_header(1);
    header.execution_plan_index = words[0];
    header.execution_program_index = words[1];
    header.execution_package_index = words[2];
    header.execution_permutation_low = words[3];
    header.execution_permutation_high = words[4];
    header
}

fn pixel(material_index: u32, generation: u32) -> VisibilityPixel {
    VisibilityPixel {
        scene_index: 1,
        scene_generation: 1,
        primitive_id: 0,
        geometry_lod_or_cluster: 0,
        material_index,
        material_generation: generation,
        barycentrics_unorm16: 0,
        coverage_and_flags: 255,
    }
}

#[test]
fn response_buckets_are_variable_length_and_deterministic() {
    let keys: Vec<_> = (0..37)
        .map(|id| {
            ResponseBucketKey::new(
                ExecutionFamily::SplitResponse,
                ProgramId(id),
                RenderPhase::Custom,
            )
        })
        .collect();
    let table = ResponseBucketTable::from_keys(keys.iter().copied().chain(keys.iter().copied()));
    assert_eq!(
        table,
        ResponseBucketTable::from_keys(keys.into_iter().rev())
    );
    assert_eq!(table.len(), 37);
    for (id, bucket) in table.buckets().iter().enumerate() {
        assert_eq!(bucket.bucket_id, id as u32);
        assert_eq!(bucket.reserved, 0);
        assert_eq!(table.bucket_id(bucket.key()), Some(id as u32));
    }
    assert_eq!(
        table.validate_compiled_entries(|key| key.program_id != ProgramId(27)),
        Err(ResponseBucketValidationError::MissingCompiledEntry {
            bucket: table.buckets()[27]
        })
    );
    table.validate_compiled_entries(|_| true).unwrap();
    assert!(ResponseBucketTable::from_keys([]).is_empty());
}

#[test]
fn builtin_ids_and_response_kinds_match_execution_upload() {
    let families = [
        ExecutionFamily::FusedBuiltin,
        ExecutionFamily::SplitResponse,
        ExecutionFamily::SplitResponse,
    ];
    let (tables, _) = execution_fixture(&BUILTINS.into_iter().zip(families).collect::<Vec<_>>());
    for ((program, family), row) in BUILTINS
        .into_iter()
        .zip(families)
        .zip(tables.gpu_program_rows())
    {
        let key = ResponseBucketKey::from_program(family, program, RenderPhase::Opaque);
        assert_eq!(key.program_id.0, response_kind_for_program(program));
        assert_eq!(row[0], response_kind_for_program(program));
        assert_eq!(row[4], key.program_id.0);
    }
    assert_eq!(
        response_kind_for_program(ProgramRef::Registered(ProgramId(0))),
        3
    );
    assert_eq!(size_of::<ShadingWorkItem>(), 32);
    assert_eq!(core::mem::offset_of!(ShadingWorkItem, response_kind), 12);
    assert_eq!(
        core::mem::offset_of!(ShadingWorkItem, right_shading_class),
        20
    );
    assert_eq!(core::mem::offset_of!(ShadingWorkItem, response_meta), 28);
}

#[test]
fn binary_selector_changes_both_real_builtin_entry_and_phase() {
    let registered = ProgramRef::Registered(ProgramId(3));
    let (tables, mut headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (BUILTINS[1], ExecutionFamily::SplitResponse),
        (BUILTINS[2], ExecutionFamily::SplitResponse),
        (registered, ExecutionFamily::SplitResponse),
    ]);
    for header in &mut headers {
        header.feature_flags |= MaterialFeatureFlags::REGION_ILLUMINATION.0;
    }
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    for (original, header) in headers.iter().enumerate() {
        for selector in [0, 1, 2, u32::MAX] {
            let expected = if original < 2 && selector < 2 {
                BUILTINS[selector as usize]
            } else if original < 3 {
                BUILTINS[original]
            } else {
                registered
            };
            assert_eq!(
                classify_material_header_with_context(header, Some(&tables), selector),
                Ok(expected)
            );
            let key = response_bucket_key_for_header(header, Some(&tables), selector).unwrap();
            assert_eq!(
                key.program_id,
                ResponseBucketKey::from_program(key.execution_family, expected, key.render_phase)
                    .program_id
            );
            assert_eq!(key.render_phase, render_phase_for_header(header, expected));
            assert!(buckets.bucket_id(key).is_some());
            let mut transparent = *header;
            transparent.render_class = MaterialRenderClass::Transparent as u32;
            assert_eq!(
                response_bucket_key_for_header(&transparent, Some(&tables), selector),
                Err(ClassificationError::NonSurfaceClass)
            );
        }
    }
    // In particular a fused Lit-only execution table must contain Npr/Stylized.
    let (lit_only, mut headers) =
        execution_fixture(&[(BUILTINS[0], ExecutionFamily::FusedBuiltin)]);
    headers[0].feature_flags |= MaterialFeatureFlags::REGION_ILLUMINATION.0;
    let key = response_bucket_key_for_header(&headers[0], Some(&lit_only), 1).unwrap();
    assert_eq!(key.program_id, ProgramId(1));
    assert_eq!(key.render_phase, RenderPhase::Stylized);
    assert!(ResponseBucketTable::from_execution_tables(&lit_only)
        .unwrap()
        .bucket_id(key)
        .is_some());
}

#[test]
fn opaque_eligibility_matches_gpu_render_classes_and_domains() {
    // RenderDomain::Hair is domain 3, not an extra response/raster class.
    for domain in 0..=7 {
        for render_class in (0..=7).chain([u32::MAX]) {
            let mut header = fallback_material_header(1);
            header.render_domain = domain;
            header.render_class = render_class;
            let eligible = matches!(domain, 0 | 3) && render_class <= 3;
            let expected = if eligible {
                Ok(BUILTINS[0])
            } else {
                Err(ClassificationError::NonSurfaceClass)
            };
            assert_eq!(
                classify_material_header(&header),
                expected,
                "domain={domain}, render_class={render_class}"
            );
            assert_eq!(
                classify_material_header_with_program(&header, BUILTINS[0], u32::MAX),
                expected
            );
            let result = ShadingWorkPlan::build([(0, pixel(0, header.generation))], &[header], 1);
            if eligible {
                assert_eq!(result.unwrap().work.len(), 1);
            } else {
                assert_eq!(
                    result,
                    Err(ShadingWorkPlanError::Classification {
                        pixel_index: 0,
                        error: ClassificationError::NonSurfaceClass,
                    })
                );
            }
        }
    }
}

#[test]
fn composition_table_contains_both_verified_endpoints_and_selected_phases() {
    let (mut tables, headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (BUILTINS[1], ExecutionFamily::SplitResponse),
    ]);
    let left = ResponseProgramIndex::from_raw(0, tables.programs()).unwrap();
    let right = ResponseProgramIndex::from_raw(1, tables.programs()).unwrap();
    let plan = tables
        .register_plan(
            ShadingPlan::MaskBlend {
                left: BUILTINS[0],
                right: BUILTINS[1],
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::DirectOnly,
            },
            left,
            Some(right),
            ExecutionFamily::Composition,
            ChannelMask::one(ResponseChannel::Direct),
        )
        .unwrap();
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    for original in headers {
        let mut header = original;
        header.execution_plan_index = plan.get();
        for region in [0, 1, u32::MAX] {
            let key = response_bucket_key_for_header(&header, Some(&tables), region).unwrap();
            assert_eq!(key.execution_family, ExecutionFamily::Composition);
            assert!(buckets.bucket_id(key).is_some());
        }
    }
}

#[test]
fn mixed_closures_hair_domain_and_water_feature_never_change_identity() {
    let base = fallback_material_header(1);
    let key = response_bucket_key_for_header(&base, None, u32::MAX).unwrap();
    let lobes = [
        ClosureKind::Subsurface,
        ClosureKind::ClearCoat,
        ClosureKind::Sheen,
        ClosureKind::Hair,
    ];
    for bits in 0..16 {
        let mut header = base;
        header.closure_mask = lobes.iter().enumerate().fold(0, |mask, (i, kind)| {
            mask | if bits & (1 << i) != 0 {
                1 << *kind as u32
            } else {
                0
            }
        });
        for domain in [RenderDomain::Surface, RenderDomain::Hair] {
            header.render_domain = domain as u32;
            for features in [0, MaterialFeatureFlags::WATER.0] {
                header.feature_flags = features;
                assert_eq!(classify_material_header(&header), Ok(BUILTINS[0]));
                assert_eq!(
                    response_bucket_key_for_header(&header, None, u32::MAX),
                    Ok(key)
                );
                for program in BUILTINS
                    .into_iter()
                    .chain([ProgramRef::Registered(ProgramId(0))])
                {
                    assert_eq!(
                        classify_material_header_with_program(&header, program, u32::MAX),
                        Ok(program)
                    );
                }
            }
        }
    }
}

#[test]
fn unknown_stale_partial_and_wrong_owner_execution_are_hard_errors() {
    let (tables, headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (BUILTINS[2], ExecutionFamily::SplitResponse),
    ]);
    assert_eq!(
        classify_material_header(&headers[0]),
        Err(ClassificationError::ExecutionContextRequired)
    );
    for words in [
        [99, 0, 0, 0, 0],
        [0, 99, 0, 0, 0],
        [0, 0, 99, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, u32::MAX, 0, 0],
    ] {
        let header = header_with_words(words);
        assert_eq!(
            classify_material_header_with_execution(&header, &tables, 0),
            Err(ClassificationError::InvalidExecutionBinding),
            "{words:?}"
        );
        assert_eq!(
            response_bucket_key_for_header(&header, Some(&tables), 0),
            Err(ClassificationError::InvalidExecutionBinding)
        );
    }
    let partial = header_with_words([0, 0, u32::MAX, 0, 0]);
    assert_eq!(
        classify_material_header_with_program(&partial, BUILTINS[0], 0),
        Err(ClassificationError::InvalidExecutionBinding)
    );
    assert_eq!(
        classify_material_header_with_program(
            &fallback_material_header(1),
            ProgramRef::Registered(ProgramId(u32::MAX)),
            0
        ),
        Err(ClassificationError::InvalidProgramId)
    );
    assert_eq!(
        classify_material_header_with_execution(&headers[0], &MaterialExecutionTables::new(), 0),
        Err(ClassificationError::InvalidExecutionBinding)
    );
}

#[test]
fn registered_zero_is_valid_but_ambiguous_raw_namespaces_are_rejected() {
    let registered = ProgramRef::Registered(ProgramId(0));
    let (tables, headers) = execution_fixture(&[(registered, ExecutionFamily::SplitResponse)]);
    assert_eq!(
        classify_material_header_with_execution(&headers[0], &tables, 1),
        Ok(registered)
    );
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    assert!(buckets
        .bucket_id(ResponseBucketKey::from_program(
            ExecutionFamily::SplitResponse,
            registered,
            RenderPhase::Custom
        ))
        .is_some());
    // FusedBuiltin registered plans are rejected earlier by material tables;
    // SplitResponse allows both kinds, so test a genuinely representable alias.
    let (ambiguous, _) = execution_fixture(&[
        (registered, ExecutionFamily::SplitResponse),
        (BUILTINS[0], ExecutionFamily::SplitResponse),
    ]);
    assert!(matches!(
        ResponseBucketTable::from_execution_tables(&ambiguous),
        Err(ResponseBucketValidationError::AmbiguousProgramIdentity {
            program_id: ProgramId(0),
            ..
        })
    ));
}

#[test]
fn work_plan_is_bucket_contiguous_generation_safe_and_capacity_bounded() {
    let (tables, mut headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (
            ProgramRef::Registered(ProgramId(0)),
            ExecutionFamily::SplitResponse,
        ),
    ]);
    headers[0].feature_flags |= MaterialFeatureFlags::REGION_ILLUMINATION.0;
    headers[0].generation = 2;
    headers[1].generation = 4;
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    let plan = ShadingWorkPlan::build_with_region_and_buckets_and_execution(
        [
            (9, pixel(1, 4), 0),
            (3, pixel(0, 2), 0),
            (7, pixel(1, 3), 0),
            (10, VisibilityPixel::INVALID, 0),
            (11, pixel(99, 0), 0),
            (12, pixel(0, 2), 1),
            (13, pixel(0, 2), 0),
        ],
        &headers,
        3,
        &buckets,
        Some(&tables),
    )
    .unwrap();
    assert_eq!(
        plan.work
            .iter()
            .map(|w| (w.pixel_index, w.response_kind))
            .collect::<Vec<_>>(),
        [(3, 0), (9, 3), (12, 1)]
    );
    assert_eq!(plan.counts.iter().sum::<u32>(), 3);
    for bucket in buckets.buckets() {
        let i = bucket.bucket_id as usize;
        assert_eq!(plan.offsets[i], plan.counts[..i].iter().sum::<u32>());
        for item in
            &plan.work[plan.offsets[i] as usize..(plan.offsets[i] + plan.counts[i]) as usize]
        {
            assert_eq!(item.shading_class, bucket.bucket_id);
        }
    }
}

#[test]
fn cpu_work_plan_honors_binary_region_and_dynamic_table_offsets() {
    let mut header = fallback_material_header(1);
    header.feature_flags |= MaterialFeatureFlags::REGION_ILLUMINATION.0;
    let table = ResponseBucketTable::builtin();
    let plan = ShadingWorkPlan::build_with_region(
        [
            (4, pixel(0, header.generation), 0),
            (9, pixel(0, header.generation), 1),
        ],
        &[header],
        8,
    )
    .unwrap();
    for (region, expected_pixel) in [(0, 4), (1, 9)] {
        let id = table
            .bucket_id(response_bucket_key_for_header(&header, None, region).unwrap())
            .unwrap() as usize;
        assert_eq!(plan.counts[id], 1);
        assert_eq!(
            plan.work[plan.offsets[id] as usize].pixel_index,
            expected_pixel
        );
    }
    let keys = table
        .buckets()
        .iter()
        .map(|bucket| bucket.key())
        .chain((3..30).map(|id| {
            ResponseBucketKey::new(
                ExecutionFamily::SplitResponse,
                ProgramId(id),
                RenderPhase::Custom,
            )
        }));
    let dynamic = ResponseBucketTable::from_keys(keys);
    let dynamic_plan = ShadingWorkPlan::build_with_region_and_buckets(
        [(4, pixel(0, header.generation), 0)],
        &[header],
        1,
        &dynamic,
    )
    .unwrap();
    assert_eq!(dynamic_plan.offsets.len(), dynamic.len());
    assert_eq!(dynamic_plan.buckets, dynamic);
}

#[test]
fn binary_region_metadata_requires_the_declared_feature_and_valid_selector() {
    let (tables, headers) = execution_fixture(&[(BUILTINS[1], ExecutionFamily::SplitResponse)]);
    let header = headers[0];

    // Metadata from an unrelated or stale visibility row must not override a
    // material-wide NPR response when the header did not publish the feature.
    assert_eq!(
        classify_material_header_with_execution(&header, &tables, 0),
        Ok(BUILTINS[1])
    );

    let mut region_header = header;
    region_header.feature_flags |= MaterialFeatureFlags::REGION_ILLUMINATION.0;
    assert_eq!(
        classify_material_header_with_execution(&region_header, &tables, 0),
        Ok(BUILTINS[0])
    );
    assert_eq!(
        classify_material_header_with_execution(&region_header, &tables, 1),
        Ok(BUILTINS[1])
    );
    for invalid in [2, u32::MAX] {
        assert_eq!(
            classify_material_header_with_execution(&region_header, &tables, invalid),
            Ok(BUILTINS[1])
        );
    }
}

#[test]
fn missing_dispatch_and_live_invalid_execution_do_not_silently_drop_work() {
    let header = fallback_material_header(1);
    let key = response_bucket_key_for_header(&header, None, 0).unwrap();
    assert_eq!(
        ShadingWorkPlan::build_with_region_and_buckets(
            [(7, pixel(0, header.generation), 0)],
            &[header],
            1,
            &ResponseBucketTable::default()
        ),
        Err(ShadingWorkPlanError::MissingBucket {
            pixel_index: 7,
            key
        })
    );
    let invalid = header_with_words([0, 0, 0, 0, 0]);
    assert_eq!(
        ShadingWorkPlan::build([(8, pixel(0, invalid.generation))], &[invalid], 1),
        Err(ShadingWorkPlanError::Classification {
            pixel_index: 8,
            error: ClassificationError::ExecutionContextRequired
        })
    );
    let mut inactive = header;
    inactive.is_active = 0;
    assert_eq!(
        classify_material_header(&inactive),
        Err(ClassificationError::Inactive)
    );
    let mut unsupported = header;
    unsupported.render_domain = u32::MAX;
    assert_eq!(
        classify_material_header(&unsupported),
        Err(ClassificationError::NonSurfaceClass)
    );
}

#[test]
fn continuous_blend_emits_two_verified_arms_and_packed_response_meta() {
    let (mut tables, mut headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (BUILTINS[1], ExecutionFamily::SplitResponse),
    ]);
    let left = ResponseProgramIndex::from_raw(0, tables.programs()).unwrap();
    let right = ResponseProgramIndex::from_raw(1, tables.programs()).unwrap();
    let channels = ChannelMask::one(ResponseChannel::Direct);
    let plan = tables
        .register_plan(
            ShadingPlan::MaskBlend {
                left: BUILTINS[0],
                right: BUILTINS[1],
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::DirectOnly,
            },
            left,
            Some(right),
            ExecutionFamily::Composition,
            ChannelMask::one(ResponseChannel::Direct),
        )
        .unwrap();
    headers[0].execution_plan_index = plan.get();
    headers[0].feature_flags |= MaterialFeatureFlags::ILLUMINATION_BLEND.0;
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    let mut continuous_pixel = pixel(0, headers[0].generation);
    continuous_pixel.coverage_and_flags = 32_768;
    let plan = ShadingWorkPlan::build_with_region_and_buckets_and_execution(
        [(5, continuous_pixel, 0)],
        &headers,
        1,
        &buckets,
        Some(&tables),
    )
    .unwrap();
    let work = &plan.work[0];
    let left_key = ResponseBucketKey::from_program(
        ExecutionFamily::Composition,
        BUILTINS[0],
        RenderPhase::Opaque,
    );
    let right_key = ResponseBucketKey::from_program(
        ExecutionFamily::Composition,
        BUILTINS[1],
        RenderPhase::Stylized,
    );
    assert_eq!(work.shading_class, buckets.bucket_id(left_key).unwrap());
    assert_eq!(
        work.right_shading_class,
        buckets.bucket_id(right_key).unwrap()
    );
    assert_eq!(work.response_kind, response_kind_for_program(BUILTINS[0]));
    assert_eq!(
        work.right_response_kind,
        response_kind_for_program(BUILTINS[1])
    );
    assert_eq!(work.response_meta, 32_768 | (channels.0 << 16));
}

#[test]
fn continuous_blend_rejects_missing_or_malformed_metadata_and_single_plans() {
    let (mut tables, mut headers) = execution_fixture(&[
        (BUILTINS[0], ExecutionFamily::FusedBuiltin),
        (BUILTINS[1], ExecutionFamily::SplitResponse),
    ]);
    let left = ResponseProgramIndex::from_raw(0, tables.programs()).unwrap();
    let right = ResponseProgramIndex::from_raw(1, tables.programs()).unwrap();
    let plan = tables
        .register_plan(
            ShadingPlan::MaskBlend {
                left: BUILTINS[0],
                right: BUILTINS[1],
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::DirectOnly,
            },
            left,
            Some(right),
            ExecutionFamily::Composition,
            ChannelMask::one(ResponseChannel::Direct),
        )
        .unwrap();
    headers[0].execution_plan_index = plan.get();
    let buckets = ResponseBucketTable::from_execution_tables(&tables).unwrap();
    headers[0].feature_flags |= MaterialFeatureFlags::ILLUMINATION_BLEND.0;
    for metadata in [u32::MAX, 1 << 16] {
        let result = ShadingWorkPlan::build_with_region_and_buckets_and_execution(
            [(
                7,
                {
                    let mut value = pixel(0, headers[0].generation);
                    value.coverage_and_flags = metadata;
                    value
                },
                0,
            )],
            &headers,
            1,
            &buckets,
            Some(&tables),
        );
        assert_eq!(
            result,
            Err(ShadingWorkPlanError::Classification {
                pixel_index: 7,
                error: ClassificationError::InvalidContinuousMask,
            })
        );
    }

    let mut single_plan_headers =
        execution_fixture(&[(BUILTINS[0], ExecutionFamily::FusedBuiltin)]).1;
    single_plan_headers[0].feature_flags |= MaterialFeatureFlags::ILLUMINATION_BLEND.0;
    let mut valid_metadata = pixel(0, single_plan_headers[0].generation);
    valid_metadata.coverage_and_flags = 32_768;
    assert_eq!(
        ShadingWorkPlan::build_with_region_and_buckets_and_execution(
            [(8, valid_metadata, 0)],
            &single_plan_headers,
            1,
            &ResponseBucketTable::from_execution_tables(
                &execution_fixture(&[(BUILTINS[0], ExecutionFamily::FusedBuiltin,)]).0,
            )
            .unwrap(),
            Some(&tables),
        ),
        Err(ShadingWorkPlanError::Classification {
            pixel_index: 8,
            error: ClassificationError::InvalidContinuousPlan,
        })
    );
}
