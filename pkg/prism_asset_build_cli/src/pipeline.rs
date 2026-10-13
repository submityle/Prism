use crate::args::Plan;
use crate::error::CommandError;
use prism_asset_build_core::import::{
    CanonicalPngProcessor, ImageArtifact, IMAGE_ARTIFACT_KIND, IMAGE_ARTIFACT_SCHEMA,
};
use prism_asset_build_core::{
    AssetId, BuildExecutor, BuildGraph, BuildRevision, DerivedDataCache, ImportRecipe, JobId,
    RevisionStore, SourceSnapshot,
};
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
