//! Bounded S5 consumer-boundary regressions, independent of scene integration.
//!
//! The schema hashes below are caller-supplied package metadata derived from
//! real schema code generation. These tests do not establish that app-side
//! fingerprint generation or FeatureSelection-to-package mapping refreshes it.

use prism_material_schema::{codegen::emit_rust_layout, surface_schema};
use prism_render_architecture::{
    abi::{AbiHash, AbiVersion},
    render_fabric::{MaterialRenderClass as FabricClass, PipelineBinClass, RenderDomain},
    shader_package::{
        ArtifactDigest, DeviceFingerprint, FingerprintMismatch, GraphicsBackend,
        ManifestExpectation, PackageState, PermutationSpace, PersistLoadError, PersistedPsoCache,
        PipelineStateHash, PsoCacheKey, RejectionReason, ShaderPackageId, ShaderPackageManifest,
        ShaderPackageRegistry, ShaderPackageVariantManifest, ValidationOutcome,
    },
};
use prism_render_material::{
    fallback_material_record, BuiltinProgram, CompiledEntry, MaterialExecutionError,
    MaterialExecutionTables, MaterialRenderClass, ProgramRef, RegistryError, ResponseManifest,
    ResponseRegistry, ShadingPlan, SurfaceFieldCatalog, FALLBACK_MATERIAL_HANDLE,
    MATERIAL_ABI_VERSION,
};

// A real executable constant-output fixture, not an implementation of Lit.
const BODY: &str =
    "return RegisteredResponse(vec3(1.0), vec3(2.0), vec3(3.0), vec3(4.0), vec3(5.0), 0.75);";

fn schema_revision_hashes() -> (AbiHash, AbiHash) {
    let schema = surface_schema();
    let mut revised = schema.clone();
    revised.schema_version = schema.schema_version.checked_add(1).unwrap();
    assert_eq!(revised.abi_version, schema.abi_version);
    let original_layout = emit_rust_layout(&schema);
    let revised_layout = emit_rust_layout(&revised);
    let hashes = (
        AbiHash::from_bytes(original_layout.as_bytes()),
        AbiHash::from_bytes(revised_layout.as_bytes()),
    );
    assert_ne!(hashes.0, hashes.1);
    hashes
}

fn package(abi_version: AbiVersion, abi_hash: AbiHash) -> ShaderPackageManifest {
    ShaderPackageManifest {
        id: ShaderPackageId::new("s5-abi-cache-invalidation"),
        source_revision: "unchanged-source".into(),
        entry_point: "main".into(),
        abi_version,
        abi_hash,
        variants: ShaderPackageVariantManifest::from_digests(
            &PermutationSpace::try_new([1]).unwrap(),
            [ArtifactDigest::from_bytes(BODY.as_bytes())],
        )
        .unwrap(),
    }
}

fn expectation(manifest: &ShaderPackageManifest) -> ManifestExpectation {
    ManifestExpectation::new(manifest.abi_version, manifest.abi_hash)
        .with_entry_point(manifest.entry_point.clone())
}

fn compile(manifest: ShaderPackageManifest) -> CompiledEntry {
    CompiledEntry::from_response_wgsl(manifest, vec![BODY.into()]).unwrap()
}

fn assert_same_manifest(actual: &ShaderPackageManifest, expected: &ShaderPackageManifest) {
    assert_eq!(actual.id, expected.id);
    assert_eq!(actual.source_revision, expected.source_revision);
    assert_eq!(actual.entry_point, expected.entry_point);
    assert_eq!(actual.abi_version, expected.abi_version);
    assert_eq!(actual.abi_hash, expected.abi_hash);
    assert_eq!(actual.variants, expected.variants);
}

fn ready_packages(entry: &CompiledEntry) -> ShaderPackageRegistry {
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(entry.artifact().clone())
        .unwrap();
    assert_eq!(
        packages.validate(
            &entry.package_manifest().id,
            &expectation(entry.package_manifest())
        ),
        Some(ValidationOutcome::Accepted)
    );
    assert_eq!(
        packages.state_of(&entry.package_manifest().id),
        Some(PackageState::Ready)
    );
    packages
}

fn responses(entry: &CompiledEntry, packages: &ShaderPackageRegistry) -> ResponseRegistry {
    let mut responses = ResponseRegistry::new(
        SurfaceFieldCatalog::generated().unwrap(),
        expectation(entry.package_manifest()),
    );
    responses
        .register(
            ProgramRef::Builtin(BuiltinProgram::Lit),
            ResponseManifest::from_builtin(BuiltinProgram::Lit).unwrap(),
            entry.clone(),
            packages,
        )
        .unwrap();
    responses
}

fn assert_stale_bindings_rejected(
    original: ShaderPackageManifest,
    revised: ShaderPackageManifest,
    rejection: RejectionReason,
) {
    // Neither program/material nor executable identity changes. Only the ABI
    // metadata selected by the caller differs between package generations.
    assert_eq!(original.id, revised.id);
    assert_eq!(original.source_revision, revised.source_revision);
    assert_eq!(original.entry_point, revised.entry_point);
    assert_eq!(original.variants, revised.variants);
    let old_entry = compile(original.clone());
    let new_entry = compile(revised.clone());
    assert_eq!(old_entry.spirv(), new_entry.spirv());
    assert_eq!(old_entry.digest(), new_entry.digest());
    let old_source = old_entry.response_source().unwrap();
    let new_source = new_entry.response_source().unwrap();
    assert_eq!(old_source.bodies(), new_source.bodies());
    assert_eq!(old_source.standalone_wgsl(), new_source.standalone_wgsl());

    let material = fallback_material_record(FALLBACK_MATERIAL_HANDLE, 1);
    let ShadingPlan::Single { program } = &material.response_plan else {
        panic!("fixture must select one unchanged response program");
    };
    let program = *program;
    let mut old_packages = ready_packages(&old_entry);
    let new_packages = ready_packages(&new_entry);
    let old_responses = responses(&old_entry, &old_packages);
    assert!(old_responses.resolve(program, &old_packages).is_ok());
    assert_eq!(
        old_responses.resolve(program, &new_packages).unwrap_err(),
        RegistryError::PackageChanged
    );

    let mut tables = MaterialExecutionTables::new();
    let slot = tables.register_package(&old_entry, &old_packages).unwrap();
    assert_eq!(tables.register_package(&old_entry, &old_packages), Ok(slot));
    // A stale compiled binding cannot attach to the new ready registry.
    assert_eq!(
        MaterialExecutionTables::new().register_package(&old_entry, &new_packages),
        Err(MaterialExecutionError::PackageChanged)
    );
    // Nor can the new metadata silently alias an already admitted dense slot,
    // even though the package id and verified executable digest are identical.
    assert_eq!(
        tables.register_package(&new_entry, &new_packages),
        Err(MaterialExecutionError::PackageChanged)
    );
    assert_eq!(tables.packages(), 1);
    assert_same_manifest(tables.package(slot).unwrap().manifest(), &original);
    assert_eq!(tables.package(slot).unwrap().digest(), old_entry.digest());

    // Revalidation of the *old* verified artifact under the new expectation
    // must reject specifically the changed ABI component, not code or entry.
    assert_eq!(
        old_packages.validate(&original.id, &expectation(&revised)),
        Some(ValidationOutcome::Rejected(rejection))
    );
    assert_eq!(
        old_packages.state_of(&original.id),
        Some(PackageState::Rejected)
    );
    assert_eq!(
        MaterialExecutionTables::new().register_package(&old_entry, &old_packages),
        Err(MaterialExecutionError::PackageNotReady)
    );

    // Rebuilding bindings under the new ABI succeeds for the same material.
    let new_responses = responses(&new_entry, &new_packages);
    assert!(new_responses.resolve(program, &new_packages).is_ok());
    let mut rebuilt = MaterialExecutionTables::new();
    let new_slot = rebuilt.register_package(&new_entry, &new_packages).unwrap();
    assert_same_manifest(rebuilt.package(new_slot).unwrap().manifest(), &revised);
}

#[test]
fn s5_abi_version_only_change_rejects_stale_response_and_execution_package() {
    let version = AbiVersion::from_parts(u16::try_from(MATERIAL_ABI_VERSION).unwrap(), 0);
    let bumped = AbiVersion::from_parts(version.major().checked_add(1).unwrap(), 0);
    let hash = schema_revision_hashes().0;
    assert_stale_bindings_rejected(
        package(version, hash),
        package(bumped, hash),
        RejectionReason::AbiVersionIncompatible {
            expected: bumped,
            found: version,
        },
    );
}

#[test]
fn s5_schema_revision_only_change_rejects_stale_response_and_execution_package() {
    let version = AbiVersion::from_parts(u16::try_from(MATERIAL_ABI_VERSION).unwrap(), 0);
    let (old_hash, new_hash) = schema_revision_hashes();
    assert_stale_bindings_rejected(
        package(version, old_hash),
        package(version, new_hash),
        RejectionReason::AbiHashMismatch {
            expected: new_hash,
            found: old_hash,
        },
    );
}

#[test]
fn s5_schema_revision_only_change_rejects_persisted_pso_cache() {
    let (old_hash, new_hash) = schema_revision_hashes();
    let old_device = DeviceFingerprint::new(GraphicsBackend::Vulkan, 0x10de, 1, 42, old_hash);
    let new_device = DeviceFingerprint {
        abi_hash: new_hash,
        ..old_device.clone()
    };
    // Opaque nonempty driver fixture: this exercises the production wire
    // loader/fingerprint gate, not a GPU driver or app fingerprint producer.
    let key = PsoCacheKey::new(
        ShaderPackageId::new("s5-abi-cache-invalidation"),
        0,
        PipelineStateHash(0),
    );
    let blob = vec![0x53, 0x35, 0x50, 0x53, 0x4f];
    let mut old_cache = PersistedPsoCache::new(old_device.clone());
    old_cache.insert(key.clone(), blob.clone());
    let bytes = old_cache.encode();
    let loaded = PersistedPsoCache::load_for(&bytes, &old_device).unwrap();
    assert_eq!(loaded, old_cache);
    assert_eq!(loaded.entries()[0].key, key);
    assert_eq!(loaded.entries()[0].blob, blob);
    assert_eq!(
        PersistedPsoCache::load_for(&bytes, &new_device),
        Err(PersistLoadError::Fingerprint(FingerprintMismatch::Abi))
    );
    let mut rebuilt = PersistedPsoCache::new(new_device.clone());
    rebuilt.insert(key, blob);
    assert_eq!(
        PersistedPsoCache::load_for(&rebuilt.encode(), &new_device).unwrap(),
        rebuilt
    );
    assert_eq!(
        PersistedPsoCache::load_for(&rebuilt.encode(), &old_device),
        Err(PersistLoadError::Fingerprint(FingerprintMismatch::Abi))
    );
}

#[test]
fn s5_seven_render_classes_pin_gpu_header_and_fabric_bin_encoding() {
    use MaterialRenderClass as Class;
    let cases = [
        (0, Class::Opaque, FabricClass::Opaque),
        (1, Class::OpaqueTwoSided, FabricClass::OpaqueTwoSided),
        (2, Class::Masked, FabricClass::Masked),
        (3, Class::MaskedTwoSided, FabricClass::MaskedTwoSided),
        (4, Class::Transmissive, FabricClass::Transmissive),
        (5, Class::Transparent, FabricClass::Transparent),
        (6, Class::Additive, FabricClass::Additive),
    ];
    assert_eq!(PipelineBinClass::CLASS_COUNT, 7);
    assert_eq!(PipelineBinClass::COUNT, 35);
    let mut seen = std::collections::BTreeSet::new();
    for (encoded, class, fabric_class) in cases {
        assert_eq!(Class::from_u32(encoded), Some(class));
        assert_eq!(FabricClass::from_u32(encoded), Some(fabric_class));
        assert_eq!(u32::from(fabric_class.index()), encoded);
        for domain in RenderDomain::ALL {
            let mut material = fallback_material_record(FALLBACK_MATERIAL_HANDLE, 1);
            material.render_class = class;
            material.render_domain = domain;
            let header = material.header(0, 0, 1);
            assert_eq!(header.render_class, encoded);
            assert_eq!(header.render_domain, u32::from(domain.index()));
            let decoded_class = FabricClass::from_u32(header.render_class).unwrap();
            let decoded_domain =
                RenderDomain::from_u8(u8::try_from(header.render_domain).unwrap()).unwrap();
            let bin = PipelineBinClass::from_parts(decoded_domain, decoded_class);
            assert_eq!(
                u32::from(bin.index()),
                u32::from(domain.index()) * 7 + encoded
            );
            assert_eq!(bin.decode(), (domain, fabric_class));
            assert_eq!(PipelineBinClass::from_u8(bin.index()), Some(bin));
            assert!(seen.insert(bin.index()));
        }
    }
    assert_eq!(seen.len(), 35);
    for invalid in [7, 255, 256, u32::MAX] {
        assert_eq!(Class::from_u32(invalid), None);
        assert_eq!(FabricClass::from_u32(invalid), None);
    }
    assert_eq!(PipelineBinClass::from_u8(35), None);
    assert_eq!(PipelineBinClass::from_u8(u8::MAX), None);
}
