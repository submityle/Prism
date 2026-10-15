//! Contract tests use schema-generated identities, never a copied field table.
use prism_material_pcm::CapabilitySet;
use prism_material_schema::surface_schema;
use prism_render_architecture::{
    abi::{AbiHash, AbiVersion},
    shader_package::{
        ManifestExpectation, PackageState, ShaderPackageId, ShaderPackageManifest,
        ShaderPackageRegistry,
    },
};
use prism_render_material::shading_plan::*;
use prism_render_material::{LobeMask, MaterialExecutionTables, SURFACE_CHANNELS, SURFACE_LOBES};

fn catalog() -> SurfaceFieldCatalog {
    SurfaceFieldCatalog::generated().unwrap()
}

#[test]
fn generated_builtin_manifests_cover_every_contract_without_hand_lists() {
    for program in [
        BuiltinProgram::Lit,
        BuiltinProgram::Npr,
        BuiltinProgram::Unlit,
    ] {
        let manifest = ResponseManifest::from_builtin(program).unwrap();
        manifest.validate(&catalog()).unwrap();
        assert!(!manifest.surface.optional_fields.is_empty() || program == BuiltinProgram::Unlit);
        assert!(!manifest.surface.ignored_fields_with_reason.is_empty());
        assert!(!manifest.surface.ignored_lobes_with_reason.is_empty());
        assert!(manifest.output.produced_channels.0 != 0);
        assert!(manifest.execution.supports_split || manifest.execution.supports_fused);
    }
}

#[test]
fn generated_builtin_contracts_partition_every_field_and_lobe() {
    let all_fields = catalog().fields().cloned().collect::<Vec<_>>();
    let all_lobes = SURFACE_LOBES
        .iter()
        .map(|lobe| lobe.id)
        .collect::<std::collections::BTreeSet<_>>();

    for program in [
        BuiltinProgram::Lit,
        BuiltinProgram::Npr,
        BuiltinProgram::Unlit,
    ] {
        let manifest = ResponseManifest::from_builtin(program).unwrap();
        manifest.validate(&catalog()).unwrap();

        let mapped = manifest
            .surface
            .field_mappings
            .iter()
            .map(|mapping| mapping.field.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let ignored = manifest
            .surface
            .ignored_fields_with_reason
            .iter()
            .map(|(field, _)| field.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            mapped.is_disjoint(&ignored),
            "{program:?} maps and ignores a field"
        );
        assert_eq!(
            mapped.len() + ignored.len(),
            all_fields.len(),
            "{program:?}"
        );
        assert!(all_fields
            .iter()
            .all(|field| mapped.contains(field) || ignored.contains(field)));

        let supported = manifest
            .surface
            .supported_lobes
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let ignored_lobes = manifest
            .surface
            .ignored_lobes_with_reason
            .iter()
            .map(|(lobe, _)| lobe.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            supported.is_disjoint(&ignored_lobes),
            "{program:?} lobe overlap"
        );
        assert_eq!(
            supported.len() + ignored_lobes.len(),
            all_lobes.len(),
            "{program:?}"
        );
        assert_eq!(supported.union(&ignored_lobes).count(), all_lobes.len());
    }
}

#[test]
fn unlit_contract_and_execution_row_are_decode_only_and_unlit_typed() {
    let manifest = ResponseManifest::from_builtin(BuiltinProgram::Unlit).unwrap();
    assert!(manifest.surface.field_mappings.is_empty());
    assert_eq!(
        manifest.surface.ignored_fields_with_reason.len(),
        catalog().fields().count()
    );
    assert!(manifest.surface.supported_lobes.is_empty());
    assert_eq!(
        manifest.surface.ignored_lobes_with_reason.len(),
        SURFACE_LOBES.len()
    );
    assert!(!manifest.lighting.direct_light);
    assert!(!manifest.lighting.shadows);
    assert!(!manifest.lighting.indirect_diffuse);
    assert!(!manifest.lighting.reflection);
    assert!(!manifest.lighting.transmission);
    assert_eq!(
        manifest.output.produced_channels,
        ChannelMask::one(ResponseChannel::Emission).union(ChannelMask::one(ResponseChannel::Alpha))
    );
    assert_eq!(
        manifest.output.blendable_channels,
        ChannelMask::one(ResponseChannel::Emission)
    );
    assert!(!manifest.execution.supports_fused);
    assert!(manifest.execution.supports_split);
    assert_eq!(manifest.execution.max_cost_class, CostClass::Low);
    assert!(manifest.cache.reads.is_empty());
    assert_eq!(
        manifest.cache.writes,
        vec![CacheChannel::Response(ResponseChannel::Emission)]
    );
    assert_eq!(manifest.cache.temporal, TemporalRequirement::None);

    let entry = compiled();
    let packages = packages();
    let mut tables = MaterialExecutionTables::new();
    let package = tables.register_package(&entry, &packages).unwrap();
    let program = tables
        .register_program(
            ProgramRef::Builtin(BuiltinProgram::Unlit),
            &entry,
            package,
            prism_render_architecture::shader_package::ModuleId(7),
            prism_render_architecture::shader_package::PermutationSpace::try_new([1]).unwrap(),
        )
        .unwrap();
    let plan = tables
        .register_plan(
            ShadingPlan::single_builtin(BuiltinProgram::Unlit),
            program,
            None,
            ExecutionFamily::FusedBuiltin,
            ChannelMask::default(),
        )
        .unwrap();
    assert_eq!(tables.gpu_program_rows(), vec![[2, package.get(), 1, 0, 2]]);
    assert_eq!(
        tables.gpu_plan_rows(),
        vec![[
            program.get(),
            u32::MAX,
            ExecutionFamily::FusedBuiltin as u32
        ]]
    );
    assert_eq!(
        tables
            .bind_raw(plan.get(), program.get(), package.get(), 0)
            .unwrap()
            .to_words(),
        [0, 0, 0, 0, 0]
    );
}

// This test program intentionally ignores everything; it is NOT a Lit/NPR
// implementation or a claim that the real responders have semantic coverage.
fn manifest() -> ResponseManifest {
    ResponseManifest {
        surface: SurfaceContract {
            ignored_fields_with_reason: catalog()
                .fields()
                .cloned()
                .map(|field| {
                    (
                        field,
                        "test constant-output program reads no surface fields".into(),
                    )
                })
                .collect(),
            ignored_lobes_with_reason: SURFACE_LOBES
                .iter()
                .map(|lobe| {
                    (
                        lobe.id.into(),
                        "test constant-output program ignores this lobe".into(),
                    )
                })
                .collect(),
            ..Default::default()
        },
        lighting: LightingContract::default(),
        output: OutputContract {
            produced_channels: ChannelMask::radiance(),
            blendable_channels: ChannelMask::radiance(),
        },
        execution: ExecutionContract {
            supports_fused: false,
            supports_split: true,
            max_cost_class: CostClass::Low,
        },
        cache: CacheContract {
            reads: vec![],
            writes: vec![],
            temporal: TemporalRequirement::None,
        },
    }
}

fn expectation() -> ManifestExpectation {
    ManifestExpectation::new(
        AbiVersion::from_parts(1, 0),
        AbiHash::from_bytes(b"response-contract-test"),
    )
}
fn package() -> ShaderPackageManifest {
    ShaderPackageManifest {
        id: ShaderPackageId::new("contract-test"),
        source_revision: "test-revision".into(),
        entry_point: "main".into(),
        abi_version: expectation().abi_version,
        abi_hash: expectation().abi_hash,
        permutation_count: 1,
    }
}

// Minimal executable SPIR-V compute module: void main() { return; }.
// Deliberately not respond_custom math. Binary fixture exercises reflection.
fn binary() -> Vec<u32> {
    vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        (2 << 16) | 17,
        1, // OpCapability Shader
        (3 << 16) | 14,
        0,
        1, // Logical GLSL450 memory model
        (5 << 16) | 15,
        5,
        3,
        0x6e69616d,
        0, // GLCompute main
        (6 << 16) | 16,
        3,
        17,
        1,
        1,
        1, // LocalSize
        (2 << 16) | 19,
        1, // void
        (3 << 16) | 33,
        2,
        1, // void ()
        (5 << 16) | 54,
        1,
        3,
        0,
        2,
        (2 << 16) | 248,
        4,
        (1 << 16) | 253,
        (1 << 16) | 56,
    ]
}
fn compiled() -> CompiledEntry {
    CompiledEntry::from_spirv(package(), binary()).unwrap()
}
fn packages() -> ShaderPackageRegistry {
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(compiled().artifact().clone())
        .unwrap();
    assert!(packages
        .validate(&package().id, &expectation())
        .unwrap()
        .is_accepted());
    packages
}

#[test]
fn custom_spirv_registration_publishes_a_verified_receipt_atomically() {
    let mut responses = ResponseRegistry::new(catalog(), expectation());
    let mut packages = ShaderPackageRegistry::new();
    let receipt = responses
        .register_custom_spirv(ProgramId(7), manifest(), package(), binary(), &mut packages)
        .unwrap();

    assert_eq!(receipt.program, ProgramRef::Registered(ProgramId(7)));
    assert_eq!(receipt.package, package().id);
    assert_eq!(
        receipt.digest,
        CompiledEntry::from_spirv(package(), binary())
            .unwrap()
            .digest()
    );
    assert_eq!(receipt.function_id, 3);
    assert_eq!(responses.len(), 1);
    assert_eq!(packages.len(), 1);
    assert_eq!(
        packages.state_of(&receipt.package),
        Some(PackageState::Ready)
    );
    assert!(responses.resolve(receipt.program, &packages).is_ok());
}

#[test]
fn custom_spirv_registration_rejects_stale_invalid_and_incompatible_inputs_without_response() {
    let mut stale = ResponseRegistry::new(catalog(), expectation());
    let mut stale_packages = packages();
    stale
        .register(registered(9), manifest(), compiled(), &stale_packages)
        .unwrap();
    stale.unregister(registered(9)).unwrap();
    let stale_error = stale.register_custom_spirv(
        ProgramId(9),
        manifest(),
        package(),
        binary(),
        &mut stale_packages,
    );
    assert!(matches!(
        stale_error,
        Err(RegistryError::RetiredProgramId(ProgramId(9)))
    ));
    assert!(stale.is_empty());

    let mut invalid = ResponseRegistry::new(catalog(), expectation());
    let mut invalid_packages = ShaderPackageRegistry::new();
    let invalid_error = invalid.register_custom_spirv(
        ProgramId(10),
        manifest(),
        package(),
        vec![0x0723_0203],
        &mut invalid_packages,
    );
    assert!(matches!(invalid_error, Err(RegistryError::Compile(_))));
    assert!(invalid.is_empty());
    assert!(invalid_packages.is_empty());

    let mut incompatible = ResponseRegistry::new(catalog(), expectation());
    let mut incompatible_packages = ShaderPackageRegistry::new();
    let mut incompatible_manifest = package();
    incompatible_manifest.abi_hash = AbiHash::from_bytes(b"wrong");
    let incompatible_error = incompatible.register_custom_spirv(
        ProgramId(11),
        manifest(),
        incompatible_manifest,
        binary(),
        &mut incompatible_packages,
    );
    assert!(matches!(
        incompatible_error,
        Err(RegistryError::PackageAbi(_))
    ));
    assert!(incompatible.is_empty());
    assert_eq!(incompatible_packages.len(), 1);
    assert_eq!(
        incompatible_packages.state_of(&ShaderPackageId::new("contract-test")),
        Some(PackageState::Rejected)
    );
}
fn context() -> PlanValidationContext {
    PlanValidationContext {
        active_lobes: LobeMask::all(),
        capabilities: CapabilitySet::full(),
        lighting_inputs: LightingContract {
            direct_light: true,
            shadows: true,
            indirect_diffuse: true,
            reflection: true,
            transmission: true,
            temporal_history: true,
        },
        cache_channels: vec![],
        temporal_history_available: true,
    }
}
fn registered(id: u32) -> ProgramRef {
    ProgramRef::Registered(ProgramId(id))
}
fn registry(programs: &[ProgramRef], fused: bool) -> ResponseRegistry {
    let mut registry = ResponseRegistry::new(catalog(), expectation());
    for program in programs {
        let mut manifest = manifest();
        manifest.execution.supports_fused = fused;
        registry
            .register(*program, manifest, compiled(), &packages())
            .unwrap();
    }
    registry
}
fn blend(
    left: ProgramRef,
    right: ProgramRef,
    mask: MaskSource,
    domain: BlendDomain,
) -> ShadingPlan {
    ShadingPlan::MaskBlend {
        left,
        right,
        mask,
        domain,
    }
}

#[test]
fn complete_schema_catalogue_has_every_nonpadding_field() {
    let schema = surface_schema();
    let expected = schema
        .placements(LobeMask::all().bits())
        .into_iter()
        .filter(|field| !field.is_pad)
        .count();
    assert_eq!(catalog().fields().count(), expected);
    let projected = SurfaceFieldCatalog::new(
        schema
            .placements(LobeMask::all().bits())
            .into_iter()
            .filter(|field| !field.is_pad)
            .map(|field| FieldId {
                lobe: field.lobe,
                name: field.name,
            }),
    )
    .unwrap();
    assert_eq!(
        catalog().fields().collect::<Vec<_>>(),
        projected.fields().collect::<Vec<_>>()
    );
    manifest()
        .validate_for_surface(&catalog(), LobeMask::all())
        .unwrap();
}
#[test]
fn incomplete_catalogues_fail_loudly() {
    assert!(matches!(
        SurfaceFieldCatalog::new([]),
        Err(ContractError::EmptyFieldCatalogue)
    ));
    assert!(matches!(
        SurfaceFieldCatalog::new([FieldId::core("a")]),
        Err(ContractError::MissingFieldOwner(_))
    ));
    let mut fields: Vec<_> = catalog().fields().cloned().collect();
    fields.push(fields[0].clone());
    assert!(matches!(
        SurfaceFieldCatalog::new(fields),
        Err(ContractError::DuplicateField(_))
    ));
}
#[test]
fn missing_each_field_is_an_error_even_when_its_lobe_is_inactive() {
    for field in catalog().fields() {
        let mut manifest = manifest();
        manifest
            .surface
            .ignored_fields_with_reason
            .retain(|(id, _)| id != field);
        assert_eq!(
            manifest.validate_for_surface(&catalog(), LobeMask(0)),
            Err(ContractError::MissingField(field.clone()))
        );
    }
}
#[test]
fn every_active_lobe_needs_support_or_explicit_reason() {
    for lobe in &SURFACE_LOBES {
        let mut manifest = manifest();
        manifest
            .surface
            .ignored_lobes_with_reason
            .retain(|(id, _)| id != lobe.id);
        manifest
            .validate_for_surface(&catalog(), LobeMask(0))
            .unwrap();
        assert_eq!(
            manifest.validate_for_surface(&catalog(), LobeMask(1 << lobe.registry_slot)),
            Err(ContractError::UnsupportedActiveLobe(lobe.id.into()))
        );
    }
}
#[test]
fn unknown_active_bits_are_rejected() {
    let bit = 1 << 31;
    assert_eq!(
        manifest().validate_for_surface(&catalog(), LobeMask(bit)),
        Err(ContractError::UnknownActiveLobes(bit))
    );
}
#[test]
fn mapped_fields_require_semantics_and_test_evidence() {
    for field in catalog().fields() {
        let mut manifest = manifest();
        manifest
            .surface
            .ignored_fields_with_reason
            .retain(|(id, _)| id != field);
        manifest.surface.optional_fields.push(field.clone());
        assert_eq!(
            manifest.validate(&catalog()),
            Err(ContractError::MissingMapping(field.clone()))
        );
        manifest.surface.field_mappings.push(FieldMapping {
            field: field.clone(),
            semantic: "test read identity".into(),
            test: "mapped_fields_require_semantics_and_test_evidence".into(),
        });
        if let Some(lobe) = &field.lobe {
            manifest
                .surface
                .ignored_lobes_with_reason
                .retain(|(id, _)| id != lobe);
            manifest.surface.supported_lobes.push(lobe.clone());
        }
        manifest.validate(&catalog()).unwrap();
        manifest.surface.field_mappings[0].test.clear();
        assert_eq!(
            manifest.validate(&catalog()),
            Err(ContractError::InvalidMapping(field.clone()))
        );
    }
}
#[test]
fn required_fields_of_absent_lobes_fail() {
    let field = catalog()
        .fields()
        .find(|field| field.lobe.is_some())
        .unwrap()
        .clone();
    let mut manifest = manifest();
    manifest
        .surface
        .ignored_fields_with_reason
        .retain(|(id, _)| *id != field);
    manifest.surface.required_fields.push(field.clone());
    manifest.surface.field_mappings.push(FieldMapping {
        field: field.clone(),
        semantic: "test read".into(),
        test: "required_fields_of_absent_lobes_fail".into(),
    });
    let lobe = field.lobe.clone().unwrap();
    manifest
        .surface
        .ignored_lobes_with_reason
        .retain(|(id, _)| *id != lobe);
    manifest.surface.supported_lobes.push(lobe);
    assert_eq!(
        manifest.validate_for_surface(&catalog(), LobeMask(0)),
        Err(ContractError::RequiredFieldAbsent(field))
    );
}
#[test]
fn unknown_conflicting_and_empty_declarations_fail() {
    let mut m = manifest();
    m.surface.optional_fields.push(FieldId::core("unknown"));
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::UnknownField(_))
    ));
    let mut m = manifest();
    m.surface
        .optional_fields
        .push(m.surface.ignored_fields_with_reason[0].0.clone());
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::DuplicateField(_))
    ));
    let mut m = manifest();
    m.surface.ignored_fields_with_reason[0].1 = " \t".into();
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::EmptyReason(_))
    ));
    let mut m = manifest();
    m.surface.ignored_lobes_with_reason[0].1.clear();
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::EmptyReason(_))
    ));
    let mut m = manifest();
    m.surface.supported_lobes.push("unknown".into());
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::UnknownLobe(_))
    ));
    let mut m = manifest();
    m.surface.supported_lobes.push(SURFACE_LOBES[0].id.into());
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::DuplicateLobe(_))
    ));
}
#[test]
fn lobe_support_is_not_a_boolean_claim() {
    let mut m = manifest();
    m.surface.ignored_lobes_with_reason.remove(0);
    m.surface.supported_lobes.push(SURFACE_LOBES[0].id.into());
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::SupportedLobeWithoutMapping(_))
    ));
}
#[test]
fn five_contracts_are_validated_not_just_surface() {
    let mut m = manifest();
    m.output.produced_channels = ChannelMask(1 << 31);
    assert_eq!(m.validate(&catalog()), Err(ContractError::InvalidOutput));
    let mut m = manifest();
    m.output.blendable_channels = ChannelMask::one(ResponseChannel::Alpha);
    assert_eq!(m.validate(&catalog()), Err(ContractError::InvalidOutput));
    let mut m = manifest();
    m.execution.supports_split = false;
    assert_eq!(m.validate(&catalog()), Err(ContractError::InvalidExecution));
    let mut m = manifest();
    m.lighting.temporal_history = true;
    assert_eq!(m.validate(&catalog()), Err(ContractError::InvalidTemporal));
    let mut m = manifest();
    m.cache.reads = vec![CacheChannel::SurfaceAux("unknown".into())];
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::UnknownCacheChannel(_))
    ));
    let mut m = manifest();
    let ch = CacheChannel::Response(ResponseChannel::Direct);
    m.cache.reads = vec![ch.clone(), ch];
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::DuplicateCacheChannel(_))
    ));
    let mut m = manifest();
    m.cache.writes = vec![CacheChannel::Response(ResponseChannel::Alpha)];
    assert!(matches!(
        m.validate(&catalog()),
        Err(ContractError::CacheWriteNotProduced(_))
    ));
}
#[test]
fn id_zero_legal_max_reserved_and_stale_is_hard_error() {
    let mut r = registry(&[registered(0)], false);
    assert_eq!(r.len(), 1);
    r.resolve(registered(0), &packages()).unwrap();
    assert_eq!(
        r.resolve(registered(1), &packages()).unwrap_err(),
        RegistryError::StaleProgram(registered(1))
    );
    assert_eq!(
        r.resolve(registered(u32::MAX), &packages()).unwrap_err(),
        RegistryError::ReservedProgramId
    );
    assert_eq!(
        r.register(registered(u32::MAX), manifest(), compiled(), &packages()),
        Err(RegistryError::ReservedProgramId)
    );
    assert_eq!(
        r.register(registered(0), manifest(), compiled(), &packages()),
        Err(RegistryError::DuplicateProgram(registered(0)))
    );
    r.unregister(registered(0)).unwrap();
    assert_eq!(
        r.resolve(registered(0), &packages()).unwrap_err(),
        RegistryError::StaleProgram(registered(0))
    );
    assert_eq!(
        r.register(registered(0), manifest(), compiled(), &packages()),
        Err(RegistryError::RetiredProgramId(ProgramId(0)))
    );
}

#[test]
fn stale_registered_id_fails_plan_validation_instead_of_falling_back() {
    let stale = registered(7);
    let mut r = registry(&[stale], false);
    let packages = packages();
    r.unregister(stale).unwrap();

    let error = r
        .validate_plan(
            &ShadingPlan::Single { program: stale },
            &context(),
            &packages,
        )
        .unwrap_err();
    assert_eq!(error, RegistryError::StaleProgram(stale));
}

#[test]
fn builtin_trio_is_available_only_after_explicit_binding() {
    for builtin in [
        BuiltinProgram::Lit,
        BuiltinProgram::Npr,
        BuiltinProgram::Unlit,
    ] {
        let program = ProgramRef::Builtin(builtin);
        let r = registry(&[program], true);
        let plan = ShadingPlan::Single { program };
        let packages = packages();
        let validated = r.validate_plan(&plan, &context(), &packages).unwrap();
        assert_eq!(validated.execution_family, ExecutionFamily::FusedBuiltin);
        assert!(validated.right.is_none());
    }
}
#[test]
fn registration_failure_is_atomic_and_custom_requires_split() {
    let mut r = registry(&[], false);
    let mut m = manifest();
    m.execution.supports_fused = true;
    assert_eq!(
        r.register(registered(0), m, compiled(), &packages()),
        Err(RegistryError::CustomRequiresSplit)
    );
    let mut m = manifest();
    m.surface.ignored_fields_with_reason.remove(0);
    assert!(matches!(
        r.register(registered(0), m, compiled(), &packages()),
        Err(RegistryError::Contract(_))
    ));
    assert!(r.is_empty());
    r.register(registered(0), manifest(), compiled(), &packages())
        .unwrap();
}
#[test]
fn ready_metadata_is_insufficient_without_compiled_entry() {
    let mut m = package();
    m.entry_point = "not_compiled".into();
    assert_eq!(
        CompiledEntry::from_spirv(m, binary()).unwrap_err(),
        CompiledEntryError::EntryMissing("not_compiled".into())
    );
    assert_eq!(
        CompiledEntry::from_spirv(package(), vec![]).unwrap_err(),
        CompiledEntryError::InvalidHeader
    );
    let mut binary = binary();
    binary.truncate(26); // removes function definition at instruction boundary
    assert_eq!(
        CompiledEntry::from_spirv(package(), binary).unwrap_err(),
        CompiledEntryError::FunctionMissing(3)
    );
}
#[test]
fn reflection_rejects_broken_instructions_and_function_declarations() {
    let mut b = binary();
    b[5] = 0;
    assert!(matches!(
        CompiledEntry::from_spirv(package(), b),
        Err(CompiledEntryError::MalformedInstruction(_))
    ));
    let mut b = binary();
    b.pop();
    assert_eq!(
        CompiledEntry::from_spirv(package(), b).unwrap_err(),
        CompiledEntryError::InvalidFunction(3)
    );
    let mut b = binary();
    b.drain(30..33); // remove OpLabel and return
    assert_eq!(
        CompiledEntry::from_spirv(package(), b).unwrap_err(),
        CompiledEntryError::InvalidFunction(3)
    );
    let entry = compiled();
    assert_eq!(entry.spirv(), binary());
    assert_eq!(entry.function_id(), 3);
}
#[test]
fn package_state_revision_and_abi_are_rechecked_on_resolve() {
    let r = registry(&[registered(0)], false);
    assert_eq!(
        r.resolve(registered(0), &ShaderPackageRegistry::new())
            .unwrap_err(),
        RegistryError::PackageUnavailable
    );
    let mut p = packages();
    p.transition(&package().id, PackageState::Validating)
        .unwrap()
        .unwrap();
    assert_eq!(
        r.resolve(registered(0), &p).unwrap_err(),
        RegistryError::PackageUnavailable
    );
    let mut changed = package();
    changed.source_revision = "new-revision".into();
    let mut p = ShaderPackageRegistry::new();
    p.register(changed).unwrap();
    p.validate(&package().id, &expectation());
    assert_eq!(
        r.resolve(registered(0), &p).unwrap_err(),
        RegistryError::PackageChanged
    );
    let mut incompatible = expectation();
    incompatible.abi_hash = AbiHash::from_bytes(b"another-abi");
    let mut r = ResponseRegistry::new(catalog(), incompatible);
    assert!(matches!(
        r.register(registered(0), manifest(), compiled(), &packages()),
        Err(RegistryError::PackageAbi(_))
    ));
}

#[test]
fn metadata_ready_cannot_forge_executable_validation() {
    let r = registry(&[registered(0)], false);
    let mut metadata = ShaderPackageRegistry::new();
    metadata.register(package()).unwrap();
    metadata.validate(&package().id, &expectation());
    assert_eq!(
        r.resolve(registered(0), &metadata).unwrap_err(),
        RegistryError::PackageNotVerified
    );
}

#[test]
fn valid_code_replacement_with_identical_metadata_is_rejected() {
    let r = registry(&[registered(0)], false);
    let mut words = binary();
    words[18] = 2; // valid LocalSize.x change, no metadata change
    let changed = CompiledEntry::from_spirv(package(), words).unwrap();
    assert_ne!(changed.digest(), compiled().digest());
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(changed.artifact().clone())
        .unwrap();
    packages.validate(&package().id, &expectation());
    assert_eq!(
        r.resolve(registered(0), &packages).unwrap_err(),
        RegistryError::PackageCodeChanged
    );
}

#[test]
fn structurally_reflectable_but_ill_typed_code_is_rejected() {
    let mut words = binary();
    words[27] = 2; // OpFunction result type is a function type, not void
    assert!(matches!(
        CompiledEntry::from_spirv(package(), words),
        Err(CompiledEntryError::Validation(_))
    ));
    let entry = compiled();
    assert_eq!(entry.digest(), entry.clone().digest());
    assert_eq!(entry.spirv().as_ptr(), entry.clone().spirv().as_ptr());
}
#[test]
fn region_selection_materials_execute_both_response_arms() {
    // This is the direct gate for 02 §10.4: a region-selected material must
    // validate both response arms before the runtime can route pixels through
    // the binary selector or the continuous MaskBlend path.
    let r = registry(&[registered(0), registered(1)], false);
    let p = packages();
    let plan = blend(
        registered(0),
        registered(1),
        MaskSource::RegionMask,
        BlendDomain::FullRadiance,
    );
    let v = r.validate_plan(&plan, &context(), &p).unwrap();
    assert_eq!(v.execution_family, ExecutionFamily::Composition);
    assert!(v.right.is_some());
    let plan = blend(
        registered(0),
        registered(2),
        MaskSource::RegionMask,
        BlendDomain::FullRadiance,
    );
    assert_eq!(
        r.validate_plan(&plan, &context(), &p).unwrap_err(),
        RegistryError::StaleProgram(registered(2))
    );
}
#[test]
fn all_mask_variants_and_domains_are_exercised() {
    let a = ProgramRef::Builtin(BuiltinProgram::Lit);
    let b = ProgramRef::Builtin(BuiltinProgram::Npr);
    let r = registry(&[a, b], true);
    let p = packages();
    for mask in [
        MaskSource::Texture("style_mask".into()),
        MaskSource::VertexAttr,
        MaskSource::UvChannel,
        MaskSource::RegionMask,
        MaskSource::Channel(SURFACE_CHANNELS[0].id.into()),
    ] {
        for domain in [
            BlendDomain::FullRadiance,
            BlendDomain::DirectOnly,
            BlendDomain::CustomChannelMask(ChannelMask::one(ResponseChannel::Emission)),
        ] {
            assert_eq!(
                r.validate_plan(&blend(a, b, mask.clone(), domain), &context(), &p)
                    .unwrap()
                    .execution_family,
                ExecutionFamily::FusedBuiltin
            );
        }
    }
    for mask in [
        MaskSource::Texture(" ".into()),
        MaskSource::Channel("unknown".into()),
    ] {
        assert!(r
            .validate_plan(&blend(a, b, mask, BlendDomain::DirectOnly), &context(), &p)
            .is_err());
    }
}
#[test]
fn mask_channel_inherits_exact_generated_capability_requirements() {
    let r = registry(&[registered(0)], false);
    let p = packages();
    let mut c = context();
    c.capabilities = CapabilitySet::EMPTY;
    for channel in &SURFACE_CHANNELS {
        let plan = blend(
            registered(0),
            registered(0),
            MaskSource::Channel(channel.id.into()),
            BlendDomain::DirectOnly,
        );
        let result = r.validate_plan(&plan, &c, &p);
        if let Some(cap) = channel.requires_caps.first() {
            assert_eq!(
                result.unwrap_err(),
                RegistryError::Contract(ContractError::MissingCapability((*cap).into()))
            );
        } else {
            result.unwrap();
        }
    }
}
#[test]
fn invalid_or_nonblendable_domains_fail_without_fallback() {
    let r = registry(&[registered(0)], false);
    let p = packages();
    for channels in [
        ChannelMask(0),
        ChannelMask(1 << 31),
        ChannelMask::one(ResponseChannel::Alpha),
        ChannelMask::one(ResponseChannel::Auxiliary),
    ] {
        let plan = blend(
            registered(0),
            registered(0),
            MaskSource::RegionMask,
            BlendDomain::CustomChannelMask(channels),
        );
        assert_eq!(
            r.validate_plan(&plan, &context(), &p).unwrap_err(),
            RegistryError::Contract(ContractError::InvalidBlendDomain)
        );
    }
    let mut r = ResponseRegistry::new(catalog(), expectation());
    let mut m = manifest();
    m.output.blendable_channels = ChannelMask(0);
    r.register(registered(0), m, compiled(), &p).unwrap();
    let plan = blend(
        registered(0),
        registered(0),
        MaskSource::RegionMask,
        BlendDomain::DirectOnly,
    );
    assert_eq!(
        r.validate_plan(&plan, &context(), &p).unwrap_err(),
        RegistryError::Contract(ContractError::IncompatibleBlendOutput)
    );
}
#[test]
fn lighting_cache_and_temporal_inputs_are_real_requirements() {
    for index in 0..5 {
        let mut m = manifest();
        match index {
            0 => m.lighting.direct_light = true,
            1 => m.lighting.shadows = true,
            2 => m.lighting.indirect_diffuse = true,
            3 => m.lighting.reflection = true,
            _ => m.lighting.transmission = true,
        }
        let mut r = ResponseRegistry::new(catalog(), expectation());
        let p = packages();
        r.register(registered(0), m, compiled(), &p).unwrap();
        let mut c = context();
        c.lighting_inputs = LightingContract::default();
        assert!(matches!(
            r.validate_plan(
                &ShadingPlan::Single {
                    program: registered(0)
                },
                &c,
                &p
            ),
            Err(RegistryError::Contract(ContractError::MissingLighting(_)))
        ));
    }
    let mut m = manifest();
    m.cache.temporal = TemporalRequirement::Required;
    m.cache
        .reads
        .push(CacheChannel::Response(ResponseChannel::Direct));
    let mut r = ResponseRegistry::new(catalog(), expectation());
    let p = packages();
    r.register(registered(0), m, compiled(), &p).unwrap();
    let mut c = context();
    let plan = ShadingPlan::Single {
        program: registered(0),
    };
    assert!(matches!(
        r.validate_plan(&plan, &c, &p),
        Err(RegistryError::Contract(ContractError::MissingCacheChannel(
            _
        )))
    ));
    c.cache_channels
        .push(CacheChannel::Response(ResponseChannel::Direct));
    c.temporal_history_available = false;
    assert!(matches!(
        r.validate_plan(&plan, &c, &p),
        Err(RegistryError::Contract(ContractError::MissingLighting(
            "temporal_history"
        )))
    ));
    c.temporal_history_available = true;
    assert_eq!(
        r.validate_plan(&plan, &c, &p).unwrap().execution_family,
        ExecutionFamily::SplitResponse
    );
}
