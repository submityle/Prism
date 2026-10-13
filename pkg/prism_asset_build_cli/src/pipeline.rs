use crate::args::{Package, Plan, Validate};
use crate::error::CommandError;
use crate::mapping;
use prism_asset::StableGuid;
use prism_asset_build_core::import::{
    CanonicalPngProcessor, ImageArtifact, IMAGE_ARTIFACT_KIND, IMAGE_ARTIFACT_SCHEMA,
};
use prism_asset_build_core::{
    ArtifactQualityGate, AssetId, BuildExecutor, BuildGraph, BuildRevision, ContentHash,
    DerivedDataCache, ImportRecipe, JobId, PackageRequest, PublicationDescriptor, RevisionStore,
    RuntimePackage, SourceSnapshot,
};
use prism_asset_container::{ContainerReader, RuntimeManifest};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn plan(command: Plan) -> Result<(), CommandError> {
    let source = snapshot(&command.source, &command.logical_path)?;
    let (source_key, image_key) = artifact_keys(&source, command.asset_id)?;
    println!("source_path={}", source.logical_path());
    println!("source_artifact_key={}", source_key.to_hex());
    println!("image_artifact_key={}", image_key.to_hex());
    println!("image_asset_id={}", command.asset_id);
    Ok(())
}

pub fn publish(
    source_path: PathBuf,
    logical_path: String,
    asset_id: u128,
    store_root: PathBuf,
) -> Result<(), CommandError> {
    let source = snapshot(&source_path, &logical_path)?;
    let (source_key, image_key) = artifact_keys(&source, asset_id)?;
    let source_job = JobId(1);
    let image_job = JobId(2);
    let mut graph = BuildGraph::new();
    graph
        .add_job(source.build_job(source_job))
        .map_err(|error| {
            CommandError::InvalidInput(format!("source graph setup failed: {error}"))
        })?;
    let plan = recipe()
        .plan(
            AssetId(asset_id),
            image_job,
            source_job,
            CanonicalPngProcessor,
            IMAGE_ARTIFACT_KIND,
            IMAGE_ARTIFACT_SCHEMA,
            4,
        )
        .map_err(|error| CommandError::InvalidInput(format!("import recipe rejected: {error}")))?;
    graph.add_job(plan.job).map_err(|error| {
        CommandError::InvalidInput(format!("image graph setup failed: {error}"))
    })?;
    let mut specs = BTreeMap::new();
    let source_asset = asset_id.checked_sub(1).ok_or_else(|| {
        CommandError::InvalidInput(
            "asset-id must be at least 1; id-1 is reserved for the source artifact".into(),
        )
    })?;
    specs.insert(source_job, source.artifact_spec(AssetId(source_asset)));
    specs.insert(image_job, plan.spec);

    let store = RevisionStore::open(&store_root)?;
    let mut ddc = DerivedDataCache::open(store_root.join("ddc"))?;
    let revision = match store.current_revision()? {
        Some(revision) => revision
            .0
            .checked_add(1)
            .ok_or_else(|| CommandError::InvalidInput("revision space is exhausted".into()))?,
        None => 0,
    };
    let execution = BuildExecutor {
        protocol_version: 1,
    }
    .execute_parallel(&graph, &specs, &store, &mut ddc, BuildRevision(revision))?;
    let image_descriptor = execution
        .published
        .manifest
        .entries
        .get(&AssetId(asset_id))
        .ok_or_else(|| CommandError::InvalidInput("published image artifact is missing".into()))?;
    let image = ImageArtifact::decode(&store.cas().read(image_descriptor.content_hash)?)?;
    println!("revision={}", execution.published.manifest.revision.0);
    println!("source_artifact_key={}", source_key.to_hex());
    println!("image_artifact_key={}", image_key.to_hex());
    println!(
        "image_content_hash={}",
        image_descriptor.content_hash.to_hex()
    );
    println!("image_dimensions={}x{}", image.width, image.height);
    println!("cache_hits={}", execution.cache_hits.len());
    Ok(())
}

pub fn inspect(store_root: PathBuf) -> Result<(), CommandError> {
    let store = RevisionStore::open(&store_root)?;
    let Some(revision) = store.current_revision()? else {
        println!("current_revision=none");
        return Ok(());
    };
    let manifest = store.manifest(revision)?;
    println!("current_revision={}", revision.0);
    println!("asset_count={}", manifest.entries.len());
    for (asset, descriptor) in manifest.entries {
        println!(
            "asset={} key={} kind={} schema={} bytes={} hash={} dependencies={}",
            asset.0,
            descriptor.key.to_hex(),
            descriptor.kind,
            descriptor.schema,
            descriptor.byte_len,
            descriptor.content_hash.to_hex(),
            descriptor.dependencies.len()
        );
    }
    Ok(())
}

pub fn package(command: Package) -> Result<(), CommandError> {
    let prepared = prepare_package(
        &command.store,
        command.mapping.as_deref(),
        command.asset_id,
        command.logical_path.as_deref(),
    )?;
    write_package(&command.output, &prepared)?;
    print_package_summary(&prepared);
    Ok(())
}

pub fn validate(command: Validate) -> Result<(), CommandError> {
    let prepared = prepare_package(
        &command.store,
        command.mapping.as_deref(),
        command.asset_id,
        command.logical_path.as_deref(),
    )?;
    validate_package_output(&command.package, &prepared)?;
    println!("validated_revision={}", prepared.revision.0);
    println!(
        "build_manifest_hash={}",
        prepared.build_manifest_hash.to_hex()
    );
    println!(
        "runtime_manifest_hash={}",
        ContentHash::digest(&prepared.package.runtime_manifest_bytes).to_hex()
    );
    println!("container_count={}", prepared.package.containers.len());
    Ok(())
}

struct PreparedPackage {
    revision: BuildRevision,
    build_manifest_hash: ContentHash,
    package: RuntimePackage,
    publication: PublicationDescriptor,
}

fn prepare_package(
    store_root: &Path,
    mapping_path: Option<&Path>,
    asset_id: Option<u128>,
    logical_path: Option<&str>,
) -> Result<PreparedPackage, CommandError> {
    let mappings = match mapping_path {
        Some(path) => {
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
        None => {
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
    print_quality_report(&report);

    let build_manifest_hash = ContentHash::digest(&manifest.canonical_bytes());
    let package = prism_asset_build_core::package_runtime(PackageRequest {
        manifest: &manifest,
        cas: store.cas(),
        mappings: package_mappings(&manifest, mappings)?,
    })?;
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
    Ok(PreparedPackage {
        revision,
        build_manifest_hash,
        package,
        publication,
    })
}

fn package_mappings(
    manifest: &prism_asset_build_core::Manifest,
    mappings: Vec<prism_asset_build_core::AssetPackageMapping>,
) -> Result<Vec<prism_asset_build_core::AssetPackageMapping>, CommandError> {
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
    let containers = output.join("containers");
    fs::create_dir_all(&containers)?;
    for (name, bytes) in &prepared.package.containers {
        write_immutable(containers.join(name), bytes)?;
    }
    Ok(())
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

fn print_quality_report(report: &prism_asset_build_core::ValidationReport) {
    println!("quality_checked_assets={}", report.checked_assets);
    println!("quality_checked_artifacts={}", report.checked_artifacts);
    println!("quality_checked_blobs={}", report.checked_blobs);
    println!("quality_checked_bytes={}", report.checked_bytes);
    println!(
        "quality_checked_dependency_edges={}",
        report.checked_dependency_edges
    );
}

fn print_package_summary(prepared: &PreparedPackage) {
    println!("packaged_revision={}", prepared.revision.0);
    println!(
        "build_manifest_hash={}",
        prepared.build_manifest_hash.to_hex()
    );
    println!(
        "runtime_manifest_hash={}",
        ContentHash::digest(&prepared.package.runtime_manifest_bytes).to_hex()
    );
    println!(
        "runtime_manifest_bytes={}",
        prepared.package.runtime_manifest_bytes.len()
    );
    println!("container_count={}", prepared.package.containers.len());
    for (name, bytes) in &prepared.package.containers {
        let info = ContainerReader::new(bytes.clone())
            .expect("package output was just encoded")
            .info();
        println!(
            "container={} bytes={} hash={} entries={}",
            name,
            bytes.len(),
            ContentHash::digest(bytes).to_hex(),
            info.entry_count
        );
    }
}

fn snapshot(path: &Path, logical_path: &str) -> Result<SourceSnapshot, CommandError> {
    if logical_path.is_empty() {
        return Err(CommandError::InvalidInput(
            "logical-path must be an explicit non-empty project-relative path".into(),
        ));
    }
    Ok(SourceSnapshot::new(logical_path, fs::read(path)?)?)
}

fn recipe() -> ImportRecipe {
    ImportRecipe {
        importer_id: 0x504e47,
        importer_version: 1,
        canonical_settings: b"rgba8".to_vec(),
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
    let source_key = source.artifact_key();
    let image_key = recipe()
        .artifact_spec(
            AssetId(asset_id),
            IMAGE_ARTIFACT_KIND,
            IMAGE_ARTIFACT_SCHEMA,
            4,
        )
        .artifact_key(1, &[source_key])
        .map_err(|error| CommandError::InvalidInput(format!("cannot derive image key: {error}")))?;
    Ok((source_key, image_key))
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
        )
        .unwrap();
        publish(
            source_path,
            "textures/hero.png".into(),
            7,
            root.join("store"),
        )
        .unwrap();

        let store = RevisionStore::open(root.join("store")).unwrap();
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(1)));
        let manifest = store.manifest(BuildRevision(1)).unwrap();
        let descriptor = &manifest.entries[&AssetId(7)];
        let image =
            ImageArtifact::decode(&store.cas().read(descriptor.content_hash).unwrap()).unwrap();
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.rgba8, vec![1, 2, 3, 255]);
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
        )
        .is_err());
        let store = RevisionStore::open(root.join("store")).unwrap();
        assert_eq!(store.current_revision().unwrap(), None);
        fs::remove_dir_all(root).unwrap();
    }
}
