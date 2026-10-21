use crate::args::{Package, Plan, Validate, Why};
use crate::editable::WorkspaceInput;
use crate::error::CommandError;
use crate::mapping;
use crate::output::{
    container_results, ArtifactResult, FileResult, InspectResult, PackageResult, PlanResult,
    PublishResult, QualityResult, RebuildChangeResult, RebuildExplanationResult, SignatureStatus,
};
use prism_asset::StableGuid;
use prism_asset_build_core::import::{
    CanonicalPngProcessor, ImageMipArtifact, ImportSelector, ImporterRegistry,
};
use prism_asset_build_core::processor::{InputCatalog, ProcessCancellation, ProcessLimits};
use prism_asset_build_core::{
    ArtifactQualityGate, AssetId, BuildExecutor, BuildGraph, BuildInputSnapshot, BuildResources,
    BuildRevision, ContentHash, DerivedDataCache, ImportPlan, ImportProcessor, ImportRecipe, JobId,
    PackageRequest, PublicationDescriptor, ReleaseDescriptor, ReleaseProvenance,
    ReleaseSignatureError, ReleaseSigner, ReleaseTrustStore, ReleaseVerifier, RevisionStore,
    RuntimePackage, SignedReleaseDescriptor, SourceSnapshot,
};
use prism_asset_container::{ContainerReader, RuntimeManifest};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn plan(command: Plan) -> Result<PlanResult, CommandError> {
    let source = snapshot(&command.source, &command.logical_path)?;
    let (source_key, image_key) = artifact_keys(&source, command.asset_id)?;
    Ok(PlanResult {
        source_path: source.logical_path().to_owned(),
        source_artifact_key: source_key.to_hex(),
        image_artifact_key: image_key.to_hex(),
        image_asset_id: command.asset_id.to_string(),
    })
}

pub fn why(command: Why) -> Result<RebuildExplanationResult, CommandError> {
    let revision = BuildRevision(command.revision);
    let inputs = BuildExecutor::default()
        .load_committed_build_inputs(&command.store, revision)
        .map_err(CommandError::RebuildExplanation)?;
    let plan = inputs.plan;
    let changes = plan
        .changes()
        .iter()
        .map(RebuildChangeResult::from)
        .collect();
    Ok(RebuildExplanationResult {
        revision: revision.0,
        direct_jobs: plan
            .dirty_set()
            .direct_jobs()
            .map(|job| job.0.to_string())
            .collect(),
        affected_jobs: plan
            .dirty_set()
            .jobs()
            .map(|job| job.0.to_string())
            .collect(),
        changes,
    })
}

pub fn publish(
    source_path: PathBuf,
    logical_path: String,
    asset_id: u128,
    store_root: PathBuf,
    verify_determinism: bool,
) -> Result<PublishResult, CommandError> {
    let source = snapshot(&source_path, &logical_path)?;
    let source_asset = asset_id.checked_sub(1).ok_or_else(|| {
        CommandError::InvalidInput(
            "asset-id must be at least 1; id-1 is reserved for the source artifact".into(),
        )
    })?;
    let source_job = JobId(1);
    let image_job = JobId(2);
    let store = RevisionStore::open(&store_root)?;
    let mut catalog = InputCatalog::new(&store).map_err(import_error)?;
    catalog
        .insert_source(CanonicalPngProcessor::SOURCE_INPUT, source_job, &source)
        .map_err(import_error)?;
    let mut graph = BuildGraph::new();
    graph
        .add_job(source.build_job(source_job))
        .map_err(|error| {
            CommandError::InvalidInput(format!("source graph setup failed: {error}"))
        })?;
    let plan = plan_import(&catalog, asset_id)?;
    graph.add_job(plan.job).map_err(|error| {
        CommandError::InvalidInput(format!("image graph setup failed: {error}"))
    })?;
    let mut specs = BTreeMap::new();
    specs.insert(source_job, source.artifact_spec(AssetId(source_asset)));
    specs.insert(image_job, plan.spec);
    let mut inputs = BuildInputSnapshot::new().with_source(source_job, &source);
    for (&job, spec) in &specs {
        // Fingerprint routing and processor/output contracts independently
        // from source identity and profile evidence. A source edit must not
        // manufacture a processor-settings or profile change.
        let mut contract = spec.clone();
        if job == source_job {
            // The source processor's settings encode exactly the path/hash
            // already recorded by with_source above.
            contract.canonical_settings.clear();
        } else {
            // Recorded SDK settings bind input descriptors; source evidence
            // is explained separately from the user settings contract.
            contract.canonical_settings = recipe().canonical_settings;
        }
        contract.profile_hash = [0; 32];
        let mut settings = spec.asset.0.to_le_bytes().to_vec();
        settings.extend_from_slice(
            &contract
                .artifact_key(1, &[])
                .map_err(|error| {
                    CommandError::InvalidInput(format!("input contract rejected: {error}"))
                })?
                .0,
        );
        inputs.set_settings(job, settings);
        inputs.set_profile_hash(job, spec.profile_hash);
    }

    let mut ddc = DerivedDataCache::open(store_root.join("ddc"))?;
    let revision = match store.current_revision()? {
        Some(revision) => revision
            .0
            .checked_add(1)
            .ok_or_else(|| CommandError::InvalidInput("revision space is exhausted".into()))?,
        None => 0,
    };
    let executor = BuildExecutor {
        protocol_version: 1,
    };
    let limit = std::thread::available_parallelism()
        .map(|count| count.get().min(64))
        .unwrap_or(1);
    let budget = BuildResources {
        cpu_threads: limit as u64,
        memory_bytes: 512 * 1024 * 1024,
        disk_read_bytes: 2 * CanonicalPngProcessor::MAX_SOURCE_BYTES,
        disk_write_bytes: CanonicalPngProcessor::MAX_ENCODED_BYTES,
        ..BuildResources::default()
    };
    let execution = executor.execute_explained_with_resources(
        &graph,
        &specs,
        &inputs,
        &store,
        &mut ddc,
        BuildRevision(revision),
        limit,
        budget,
        verify_determinism,
    )?;
    let image_descriptor = execution
        .published
        .manifest
        .entries
        .get(&AssetId(asset_id))
        .ok_or_else(|| CommandError::InvalidInput("published image artifact is missing".into()))?;
    let image = ImageMipArtifact::decode(&store.cas().read(image_descriptor.content_hash)?)?;
    Ok(PublishResult {
        revision: execution.published.manifest.revision.0,
        build_manifest_hash: ContentHash::digest(&execution.published.manifest.canonical_bytes())
            .to_hex(),
        source_artifact_key: execution.artifact_keys[&source_job].to_hex(),
        image_artifact_key: image_descriptor.key.to_hex(),
        image_asset_id: asset_id.to_string(),
        image_content_hash: image_descriptor.content_hash.to_hex(),
        image_width: image.width,
        image_height: image.height,
        image_mip_levels: image.levels.len(),
        cache_hits: execution.cache_hits.len(),
        determinism_verified_jobs: verify_determinism.then_some(execution.artifact_keys.len()),
        artifacts: execution
            .published
            .manifest
            .entries
            .iter()
            .map(|(asset, descriptor)| ArtifactResult::new(*asset, descriptor))
            .collect(),
    })
}

pub fn inspect(store_root: PathBuf) -> Result<InspectResult, CommandError> {
    let store = RevisionStore::open(&store_root)?;
    let Some(revision) = store.current_revision()? else {
        return Ok(InspectResult {
            current_revision: None,
            build_manifest_hash: None,
            asset_count: 0,
            artifacts: Vec::new(),
        });
    };
    let manifest = store.manifest(revision)?;
    Ok(InspectResult {
        current_revision: Some(revision.0),
        build_manifest_hash: Some(ContentHash::digest(&manifest.canonical_bytes()).to_hex()),
        asset_count: manifest.entries.len(),
        artifacts: manifest
            .entries
            .iter()
            .map(|(asset, descriptor)| ArtifactResult::new(*asset, descriptor))
            .collect(),
    })
}

pub fn package(command: Package) -> Result<PackageResult, CommandError> {
    let prepared = prepare_package(
        &command.store,
        command.mapping.as_deref(),
        command.asset_id,
        command.logical_path.as_deref(),
        WorkspaceInput {
            workspace: command.workspace.as_deref(),
            roots: &command.root,
            max_snapshot_bytes: command.max_snapshot_bytes,
            max_materialized_payload_bytes: command.max_materialized_payload_bytes,
            profile_hash: command.profile_hash.as_deref(),
            toolchain_fingerprint: command.toolchain_fingerprint.as_deref(),
            environment_fingerprint: command.environment_fingerprint.as_deref(),
        },
    )?;
    write_package(&command.output, &prepared)?;
    if let Some(path) = command.signing_key_file.as_deref() {
        write_release_signature(&command.output, &prepared, path)?;
    }
    let signature = if command.signing_key_file.is_some() {
        SignatureStatus::Signed
    } else {
        // An idempotent package invocation may encounter a previously signed
        // publication. Report the actual output, not just the current flags.
        match fs::metadata(command.output.join("release.sig")) {
            Ok(_) => SignatureStatus::PresentUnverified,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => SignatureStatus::Unsigned,
            Err(error) => return Err(CommandError::Io(error)),
        }
    };
    package_result(prepared, &command.output, signature)
}

pub fn validate(command: Validate) -> Result<PackageResult, CommandError> {
    let prepared = prepare_package(
        &command.store,
        command.mapping.as_deref(),
        command.asset_id,
        command.logical_path.as_deref(),
        WorkspaceInput {
            workspace: command.workspace.as_deref(),
            roots: &command.root,
            max_snapshot_bytes: command.max_snapshot_bytes,
            max_materialized_payload_bytes: command.max_materialized_payload_bytes,
            profile_hash: command.profile_hash.as_deref(),
            toolchain_fingerprint: command.toolchain_fingerprint.as_deref(),
            environment_fingerprint: command.environment_fingerprint.as_deref(),
        },
    )?;
    validate_package_output(&command.package, &prepared)?;
    let signature = validate_release_signature(
        &command.package,
        &prepared,
        command.verification_key_file.as_deref(),
        command.trust_policy_file.as_deref(),
    )?;
    package_result(prepared, &command.package, signature)
}

struct PreparedPackage {
    revision: BuildRevision,
    build_manifest_hash: ContentHash,
    package: RuntimePackage,
    publication: PublicationDescriptor,
    provenance: ReleaseProvenance,
    quality: QualityResult,
}

fn prepare_package(
    store_root: &Path,
    mapping_path: Option<&Path>,
    asset_id: Option<u128>,
    logical_path: Option<&str>,
    editable: WorkspaceInput<'_>,
) -> Result<PreparedPackage, CommandError> {
    let editable_plan = editable.package_plan(mapping_path, asset_id, logical_path)?;
    let mappings = match (editable_plan.as_ref(), mapping_path) {
        (Some(_), _) => Vec::new(),
        (None, Some(path)) => {
            if asset_id.is_some() || logical_path.is_some() {
                return Err(CommandError::InvalidInput(
                    "--mapping cannot be combined with --asset-id or --logical-path".into(),
                ));
            }
            let input = fs::read_to_string(path)?;
            mapping::parse(&input).map_err(|error| {
                CommandError::InvalidInput(format!("mapping {}: {error}", path.display()))
            })?
        }
        (None, None) => {
            let asset_id = asset_id.ok_or_else(|| {
                CommandError::InvalidInput(
                    "legacy package input requires --asset-id and --logical-path".into(),
                )
            })?;
            let logical_path = logical_path.ok_or_else(|| {
                CommandError::InvalidInput(
                    "legacy package input requires --asset-id and --logical-path".into(),
                )
            })?;
            legacy_mappings(asset_id, logical_path)?
        }
    };
    let store = RevisionStore::open(store_root)?;
    let revision = store.current_revision()?.ok_or_else(|| {
        CommandError::InvalidInput("store has no committed build revision".into())
    })?;
    let manifest = store.manifest(revision)?;
    let gate = ArtifactQualityGate::default();
    let report = gate.validate_manifest(&manifest, store.cas())?;

    let build_manifest_hash = ContentHash::digest(&manifest.canonical_bytes());
    let package = match editable_plan {
        Some(plan) => plan.package_manifest(
            &manifest,
            store.cas(),
            "core",
            prism_asset_build_core::PackageOptions::default(),
        )?,
        None => prism_asset_build_core::package_runtime(PackageRequest {
            manifest: &manifest,
            cas: store.cas(),
            mappings: package_mappings(&manifest, mappings)?,
        })?,
    };
    if package.source_manifest_hash != build_manifest_hash {
        return Err(CommandError::InvalidInput(
            "runtime package source manifest hash is inconsistent".into(),
        ));
    }
    let publication = PublicationDescriptor::new(
        revision,
        build_manifest_hash,
        &package.runtime_manifest_bytes,
        &package.containers,
    )
    .map_err(CommandError::Publication)?;
    let provenance = ReleaseProvenance::from_manifest(&manifest, build_manifest_hash, &publication)
        .map_err(|error| CommandError::InvalidInput(error.to_owned()))?;
    Ok(PreparedPackage {
        revision,
        build_manifest_hash,
        package,
        publication,
        provenance,
        quality: report.into(),
    })
}

pub(crate) fn package_mappings(
    manifest: &prism_asset_build_core::Manifest,
    mappings: Vec<prism_asset_build_core::AssetPackageMapping>,
) -> Result<Vec<prism_asset_build_core::AssetPackageMapping>, CommandError> {
    let mut seen = std::collections::BTreeSet::new();
    if let Some(duplicate) = mappings
        .iter()
        .map(|mapping| mapping.asset)
        .find(|asset| !seen.insert(*asset))
    {
        return Err(CommandError::InvalidInput(format!(
            "mapping contains duplicate manifest asset {}",
            duplicate.0
        )));
    }
    let expected = manifest
        .entries
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let actual = mappings
        .iter()
        .map(|mapping| mapping.asset)
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(asset) = actual.difference(&expected).next() {
        return Err(CommandError::InvalidInput(format!(
            "mapping references unknown manifest asset {}",
            asset.0
        )));
    }
    if let Some(asset) = expected.difference(&actual).next() {
        return Err(CommandError::InvalidInput(format!(
            "mapping is missing manifest asset {}",
            asset.0
        )));
    }
    Ok(mappings)
}

fn legacy_mappings(
    image_asset_id: u128,
    logical_path: &str,
) -> Result<Vec<prism_asset_build_core::AssetPackageMapping>, CommandError> {
    let source_asset_id = image_asset_id.checked_sub(1).ok_or_else(|| {
        CommandError::InvalidInput(
            "asset-id must be at least 1; id-1 is reserved for the source artifact".into(),
        )
    })?;
    let image_asset = AssetId(image_asset_id);
    let source_asset = AssetId(source_asset_id);
    let image_guid = StableGuid::from_path(logical_path);
    let source_guid = StableGuid::derive_sub(image_guid, "source");
    Ok(vec![
        prism_asset_build_core::AssetPackageMapping {
            asset: source_asset,
            guid: source_guid,
            container_name: "core".into(),
            priority: 0,
            chunk_priority: 0,
        },
        prism_asset_build_core::AssetPackageMapping {
            asset: image_asset,
            guid: image_guid,
            container_name: "core".into(),
            priority: 0,
            chunk_priority: 1,
        },
    ])
}

fn write_package(output: &Path, prepared: &PreparedPackage) -> Result<(), CommandError> {
    fs::create_dir_all(output)?;
    write_immutable(
        output.join("manifest.bin"),
        &prepared.package.runtime_manifest_bytes,
    )?;
    let publication_bytes = prepared
        .publication
        .encode()
        .map_err(CommandError::Publication)?;
    write_immutable(output.join("publication.bin"), &publication_bytes)?;
    let provenance_bytes = prepared
        .provenance
        .encode()
        .map_err(|error| CommandError::InvalidInput(error.to_owned()))?;
    write_immutable(output.join("provenance.bin"), &provenance_bytes)?;
    let containers = output.join("containers");
    fs::create_dir_all(&containers)?;
    for (name, bytes) in &prepared.package.containers {
        write_immutable(containers.join(name), bytes)?;
    }
    Ok(())
}

fn release_descriptor(prepared: &PreparedPackage) -> Result<ReleaseDescriptor, CommandError> {
    let publication_bytes = prepared
        .publication
        .encode()
        .map_err(CommandError::Publication)?;
    Ok(ReleaseDescriptor {
        revision: prepared.revision,
        manifest_hash: prepared.build_manifest_hash,
        publication_hash: Some(ContentHash::digest(&publication_bytes)),
    })
}

fn read_exact_key<const N: usize>(path: &Path, kind: &str) -> Result<[u8; N], CommandError> {
    let bytes = fs::read(path).map_err(|error| {
        CommandError::InvalidInput(format!(
            "cannot read {kind} file {}: {error}",
            path.display()
        ))
    })?;
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        CommandError::InvalidInput(format!(
            "{kind} file {} must contain exactly {N} raw bytes, got {}",
            path.display(),
            bytes.len()
        ))
    })
}

fn write_release_signature(
    output: &Path,
    prepared: &PreparedPackage,
    signing_key_file: &Path,
) -> Result<(), CommandError> {
    let seed = read_exact_key::<32>(signing_key_file, "Ed25519 signing key")?;
    let signer = ReleaseSigner::from_seed(&seed).map_err(signature_error)?;
    let signed = signer.sign(release_descriptor(prepared)?);
    write_immutable(output.join("release.sig"), &signed.encode())
}

fn validate_release_signature(
    output: &Path,
    prepared: &PreparedPackage,
    verification_key_file: Option<&Path>,
    trust_policy_file: Option<&Path>,
) -> Result<SignatureStatus, CommandError> {
    if verification_key_file.is_some() && trust_policy_file.is_some() {
        return Err(CommandError::InvalidInput(
            "--verification-key-file cannot be combined with --trust-policy-file".into(),
        ));
    }
    let signature_path = output.join("release.sig");
    let signature_bytes = match fs::read(&signature_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if verification_key_file.is_some() || trust_policy_file.is_some() {
                return Err(CommandError::InvalidInput(
                    "release verification options require package release.sig".into(),
                ));
            }
            return Ok(SignatureStatus::Unsigned);
        }
        Err(error) => return Err(CommandError::Io(error)),
    };
    let signed = SignedReleaseDescriptor::decode(&signature_bytes).map_err(|reason| {
        CommandError::InvalidInput(format!("release.sig is invalid: {reason}"))
    })?;
    let expected = release_descriptor(prepared)?;
    if signed.descriptor != expected {
        return Err(CommandError::InvalidInput(
            "release.sig descriptor differs from deterministic package output".into(),
        ));
    }
    match (verification_key_file, trust_policy_file) {
        (Some(key_path), None) => {
            let public_key = read_exact_key::<32>(key_path, "Ed25519 verification key")?;
            ReleaseVerifier::from_public_key(public_key)
                .verify(&signed)
                .map(|()| SignatureStatus::VerifiedPublicKey)
                .map_err(signature_error)
        }
        (None, Some(policy_path)) => {
            let trust = ReleaseTrustStore::open(policy_path).map_err(|error| {
                CommandError::InvalidInput(format!(
                    "cannot open release trust policy {}: {error}",
                    policy_path.display()
                ))
            })?;
            trust.verify(&signed).map(|_| SignatureStatus::VerifiedTrustPolicy).map_err(|error| {
                CommandError::InvalidInput(format!(
                    "release trust policy verification failed: {error}"
                ))
            })
        }
        (None, None) => Err(CommandError::InvalidInput(
            "package contains release.sig; --verification-key-file or --trust-policy-file is required"
                .into(),
        )),
        (Some(_), Some(_)) => unreachable!("verification options are mutually exclusive"),
    }
}

fn signature_error(error: ReleaseSignatureError) -> CommandError {
    CommandError::InvalidInput(format!("release signature verification failed: {error}"))
}

fn validate_package_output(output: &Path, prepared: &PreparedPackage) -> Result<(), CommandError> {
    let runtime_manifest_bytes = fs::read(output.join("manifest.bin"))?;
    let runtime_manifest = RuntimeManifest::decode(&runtime_manifest_bytes)
        .map_err(|_| CommandError::InvalidInput("runtime manifest is invalid".into()))?;
    if runtime_manifest_bytes != prepared.package.runtime_manifest_bytes
        || runtime_manifest != prepared.package.runtime_manifest
    {
        return Err(CommandError::InvalidInput(
            "runtime manifest differs from deterministic package output".into(),
        ));
    }
    let publication_bytes = fs::read(output.join("publication.bin"))?;
    let publication =
        PublicationDescriptor::decode(&publication_bytes).map_err(CommandError::Publication)?;
    let expected_publication = prepared
        .publication
        .encode()
        .map_err(CommandError::Publication)?;
    if publication_bytes != expected_publication || publication != prepared.publication {
        return Err(CommandError::InvalidInput(
            "publication descriptor differs from deterministic package output".into(),
        ));
    }
    let provenance_bytes = fs::read(output.join("provenance.bin"))?;
    let provenance = ReleaseProvenance::decode(&provenance_bytes)
        .map_err(|error| CommandError::InvalidInput(error.to_owned()))?;
    let expected_provenance = prepared
        .provenance
        .encode()
        .map_err(|error| CommandError::InvalidInput(error.to_owned()))?;
    if provenance_bytes != expected_provenance || provenance != prepared.provenance {
        return Err(CommandError::InvalidInput(
            "release provenance differs from deterministic package output".into(),
        ));
    }
    for (name, expected) in &prepared.package.containers {
        let actual = fs::read(output.join("containers").join(name))?;
        if actual != *expected {
            return Err(CommandError::InvalidInput(format!(
                "runtime container {name:?} differs from deterministic package output"
            )));
        }
        let reader = ContainerReader::new(actual).map_err(|_| {
            CommandError::InvalidInput(format!("runtime container {name:?} is invalid"))
        })?;
        let Some(identity) = runtime_manifest.container_identity(name) else {
            return Err(CommandError::InvalidInput(format!(
                "runtime manifest has no identity for container {name:?}"
            )));
        };
        let info = reader.info();
        if identity.byte_len != info.total_len
            || identity.content_hash != info.content_hash
            || identity.generation != info.generation
        {
            return Err(CommandError::InvalidInput(format!(
                "runtime manifest identity does not match container {name:?}"
            )));
        }
    }
    Ok(())
}

fn write_immutable(path: PathBuf, bytes: &[u8]) -> Result<(), CommandError> {
    match fs::read(&path) {
        Ok(existing) if existing == bytes => Ok(()),
        Ok(_) => Err(CommandError::InvalidInput(format!(
            "refusing to overwrite divergent package output {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::write(path, bytes).map_err(CommandError::Io)
        }
        Err(error) => Err(CommandError::Io(error)),
    }
}

fn package_result(
    prepared: PreparedPackage,
    output: &Path,
    signature: SignatureStatus,
) -> Result<PackageResult, CommandError> {
    let publication_bytes = prepared
        .publication
        .encode()
        .map_err(CommandError::Publication)?;
    let provenance_bytes = prepared
        .provenance
        .encode()
        .map_err(|error| CommandError::InvalidInput(error.to_owned()))?;
    let mut files = vec![
        FileResult::new("manifest.bin", &prepared.package.runtime_manifest_bytes),
        FileResult::new("publication.bin", &publication_bytes),
        FileResult::new("provenance.bin", &provenance_bytes),
    ];
    files.extend(
        prepared
            .package
            .containers
            .iter()
            .map(|(name, bytes)| FileResult::new(format!("containers/{name}"), bytes)),
    );
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(PackageResult {
        revision: prepared.revision.0,
        output_directory: output.display().to_string(),
        build_manifest_hash: prepared.build_manifest_hash.to_hex(),
        runtime_manifest_hash: ContentHash::digest(&prepared.package.runtime_manifest_bytes)
            .to_hex(),
        runtime_manifest_bytes: prepared.package.runtime_manifest_bytes.len(),
        container_count: prepared.package.containers.len(),
        containers: container_results(
            &prepared.package.containers,
            &prepared.package.runtime_manifest,
        ),
        files,
        quality: prepared.quality,
        signature,
    })
}

fn snapshot(path: &Path, logical_path: &str) -> Result<SourceSnapshot, CommandError> {
    if logical_path.is_empty() {
        return Err(CommandError::InvalidInput(
            "logical-path must be an explicit non-empty project-relative path".into(),
        ));
    }
    let file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > CanonicalPngProcessor::MAX_SOURCE_BYTES {
        return Err(CommandError::InvalidInput(format!(
            "PNG source must be a regular file of at most {} bytes",
            CanonicalPngProcessor::MAX_SOURCE_BYTES,
        )));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize + 1)
        .map_err(|error| {
            CommandError::InvalidInput(format!("source allocation rejected: {error}"))
        })?;
    file.take(CanonicalPngProcessor::MAX_SOURCE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > CanonicalPngProcessor::MAX_SOURCE_BYTES
        || bytes.len() as u64 != metadata.len()
    {
        return Err(CommandError::InvalidInput(
            "PNG source size changed or exceeded its byte limit".into(),
        ));
    }
    Ok(SourceSnapshot::new(logical_path, bytes)?)
}

fn recipe() -> ImportRecipe {
    ImportRecipe {
        importer_id: CanonicalPngProcessor::IMPORTER_ID,
        importer_version: CanonicalPngProcessor::IMPORTER_VERSION,
        canonical_settings: CanonicalPngProcessor::canonical_settings(),
        profile_hash: [1; 32],
        toolchain_fingerprint: b"prism-png-0.18".to_vec(),
        environment_fingerprint: b"portable".to_vec(),
    }
}

fn artifact_keys(
    source: &SourceSnapshot,
    asset_id: u128,
) -> Result<
    (
        prism_asset_build_core::ArtifactKey,
        prism_asset_build_core::ArtifactKey,
    ),
    CommandError,
> {
    let temporary = tempfile::tempdir()?;
    let store = RevisionStore::open(temporary.path())?;
    let mut catalog = InputCatalog::new(&store).map_err(import_error)?;
    catalog
        .insert_source(CanonicalPngProcessor::SOURCE_INPUT, JobId(1), source)
        .map_err(import_error)?;
    let source_key = source.artifact_key();
    let image_key = plan_import(&catalog, asset_id)?
        .spec
        .artifact_key(1, &[source_key])
        .map_err(|error| CommandError::InvalidInput(format!("cannot derive image key: {error}")))?;
    Ok((source_key, image_key))
}

fn import_error(error: impl std::fmt::Display) -> CommandError {
    CommandError::InvalidInput(format!("import preparation rejected: {error}"))
}

fn plan_import(catalog: &InputCatalog<'_>, asset_id: u128) -> Result<ImportPlan, CommandError> {
    if asset_id == 0 {
        return Err(CommandError::InvalidInput(
            "asset-id must be at least 1".into(),
        ));
    }
    let registry = ImporterRegistry::with_builtin().map_err(import_error)?;
    recipe()
        .plan_from_registry(
            &registry,
            AssetId(asset_id),
            JobId(2),
            catalog,
            vec![CanonicalPngProcessor::SOURCE_INPUT.into()],
            // This command imports PNG content. Logical asset paths need not
            // have a file extension; the analyzer validates the source bytes.
            &ImportSelector::new(None, Some("image/png")),
            ProcessLimits {
                max_output_bytes: CanonicalPngProcessor::MAX_ENCODED_BYTES,
                ..ProcessLimits::default()
            },
            ProcessCancellation::default(),
        )
        .map_err(|error| match error {
            prism_asset_build_core::ImportRecipeError::Processor(source) => {
                CommandError::Build(prism_asset_build_core::BuildExecutionError::Graph(
                    prism_asset_build_core::BuildError::ProcessorFailed {
                        job: JobId(2),
                        source,
                    },
                ))
            }
            error => import_error(error),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use png::Encoder;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-cli-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn png_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[1, 2, 3]).unwrap();
        drop(writer);
        bytes
    }

    #[test]
    fn plan_is_stable_for_same_source() {
        let path = std::env::temp_dir().join(format!("prism-cli-test-{}.png", std::process::id()));
        fs::write(&path, png_bytes()).unwrap();
        let source = snapshot(&path, "textures/test.png").unwrap();
        assert_eq!(
            artifact_keys(&source, 7).unwrap(),
            artifact_keys(&source, 7).unwrap()
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn build_publishes_then_reuses_verified_ddc_artifacts() {
        let root = temp_dir("build-cache");
        let source_path = root.join("hero.png");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source_path, png_bytes()).unwrap();

        publish(
            source_path.clone(),
            "textures/hero.png".into(),
            7,
            root.join("store"),
            false,
        )
        .unwrap();
        publish(
            source_path,
            "textures/hero.png".into(),
            7,
            root.join("store"),
            false,
        )
        .unwrap();

        let store = RevisionStore::open(root.join("store")).unwrap();
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(1)));
        let manifest = store.manifest(BuildRevision(1)).unwrap();
        let descriptor = &manifest.entries[&AssetId(7)];
        let image =
            ImageMipArtifact::decode(&store.cas().read(descriptor.content_hash).unwrap()).unwrap();
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.levels[0].rgba8, vec![1, 2, 3, 255]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_png_does_not_publish_a_revision() {
        let root = temp_dir("invalid-png");
        let source_path = root.join("broken.png");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source_path, b"not a png").unwrap();

        assert!(publish(
            source_path,
            "textures/broken.png".into(),
            7,
            root.join("store"),
            false,
        )
        .is_err());
        let store = RevisionStore::open(root.join("store")).unwrap();
        assert_eq!(store.current_revision().unwrap(), None);
        fs::remove_dir_all(root).unwrap();
    }
}
