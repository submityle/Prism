//! CLI orchestration for deterministic runtime Patch creation.
//!
//! The build core owns all diffing, payload selection, manifest construction,
//! and descriptor binding. This module only loads the already-published base
//! package and writes the resulting incremental publication directory.

use crate::args::Patch;
use crate::error::CommandError;
use crate::mapping;
use crate::pipeline::package_mappings;
use prism_asset_build_core::{
    package_runtime_patch, AssetPackageMapping, BuildRevision, ContentHash, ReleaseProvenance,
    RevisionStore, RuntimePackage, RuntimePatchRequest,
};
use prism_asset_container::{ContainerReader, RuntimeManifest};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Builds and writes one deterministic runtime patch.
pub fn run(command: Patch) -> Result<(), CommandError> {
    let store = RevisionStore::open(&command.store)?;
    let base_revision = BuildRevision(command.base_revision);
    let target_revision = store.current_revision()?.ok_or_else(|| {
        CommandError::InvalidInput("store has no committed target revision".into())
    })?;
    if target_revision <= base_revision {
        return Err(CommandError::InvalidInput(
            "target current revision must be newer than base revision".into(),
        ));
    }
    let base_manifest = store.manifest(base_revision)?;
    let target_manifest = store.manifest(target_revision)?;
    let base_mapping = read_mapping(&command.base_mapping, &base_manifest)?;
    let target_mapping = read_mapping(&command.mapping, &target_manifest)?;
    let base_package =
        load_published_package(&command.base_package, &base_manifest, &base_mapping)?;
    let patch = package_runtime_patch(RuntimePatchRequest {
        base_manifest: &base_manifest,
        target_manifest: &target_manifest,
        base_package: &base_package,
        cas: store.cas(),
        target_mappings: target_mapping,
    })
    .map_err(|error| CommandError::InvalidInput(format!("runtime patch failed: {error}")))?;

    let provenance = ReleaseProvenance::decode(&patch.target_provenance_bytes)
        .map_err(|error| CommandError::InvalidInput(format!("patch provenance: {error}")))?;
    if provenance != patch.target_provenance
        || provenance
            .encode()
            .map_err(|error| CommandError::InvalidInput(format!("patch provenance: {error}")))?
            != patch.target_provenance_bytes
    {
        return Err(CommandError::InvalidInput(
            "patch provenance bytes do not match the generated target provenance".into(),
        ));
    }

    validate_output_shape(&command.output, patch.payload.containers.keys())?;
    write_immutable(
        command.output.join("descriptor.bin"),
        &patch.descriptor_bytes,
    )?;
    write_immutable(
        command.output.join("patch_manifest.bin"),
        &patch.payload.runtime_manifest_bytes,
    )?;
    write_immutable(
        command.output.join("target_manifest.bin"),
        &patch.target_runtime_manifest_bytes,
    )?;
    write_immutable(
        command.output.join("provenance.bin"),
        &patch.target_provenance_bytes,
    )?;
    let containers = command.output.join("containers");
    fs::create_dir_all(&containers)?;
    for (name, bytes) in &patch.payload.containers {
        write_immutable(containers.join(name), bytes)?;
    }

    println!("base_revision={}", patch.descriptor.base_revision);
    println!("target_revision={}", patch.descriptor.target_revision);
    println!("removed_routes={}", patch.descriptor.removed_guids.len());
    println!("patch_container_count={}", patch.payload.containers.len());
    println!(
        "patch_manifest_hash={}",
        ContentHash::digest(&patch.payload.runtime_manifest_bytes).to_hex()
    );
    println!(
        "target_manifest_hash={}",
        ContentHash::digest(&patch.target_runtime_manifest_bytes).to_hex()
    );
    Ok(())
}

fn validate_output_shape<'a>(
    output: &Path,
    container_names: impl IntoIterator<Item = &'a String>,
) -> Result<(), CommandError> {
    if !output.exists() {
        return Ok(());
    }
    if !output.is_dir() {
        return Err(CommandError::InvalidInput(format!(
            "patch output {} is not a directory",
            output.display()
        )));
    }
    let expected_containers = container_names
        .into_iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(output)? {
        let entry = entry?;
        let name = entry.file_name();
        let allowed = name == "descriptor.bin"
            || name == "patch_manifest.bin"
            || name == "target_manifest.bin"
            || name == "provenance.bin"
            || name == "containers";
        if !allowed {
            return Err(CommandError::InvalidInput(format!(
                "patch output contains unexpected entry {}",
                entry.path().display()
            )));
        }
    }
    let containers = output.join("containers");
    if !containers.exists() {
        return Ok(());
    }
    if !containers.is_dir() {
        return Err(CommandError::InvalidInput(format!(
            "patch container output {} is not a directory",
            containers.display()
        )));
    }
    for entry in fs::read_dir(&containers)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !expected_containers.contains(&name) {
            return Err(CommandError::InvalidInput(format!(
                "patch output contains unexpected container {}",
                entry.path().display()
            )));
        }
        if !entry.file_type()?.is_file() {
            return Err(CommandError::InvalidInput(format!(
                "patch container output {} is not a regular file",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn read_mapping(
    path: &Path,
    manifest: &prism_asset_build_core::Manifest,
) -> Result<Vec<AssetPackageMapping>, CommandError> {
    let input = fs::read_to_string(path)?;
    let mappings = mapping::parse(&input).map_err(|error| {
        CommandError::InvalidInput(format!("mapping {}: {error}", path.display()))
    })?;
    package_mappings(manifest, mappings)
}

fn load_published_package(
    root: &Path,
    manifest: &prism_asset_build_core::Manifest,
    mappings: &[AssetPackageMapping],
) -> Result<RuntimePackage, CommandError> {
    let runtime_manifest_bytes = fs::read(root.join("manifest.bin"))?;
    let runtime_manifest = RuntimeManifest::decode(&runtime_manifest_bytes)
        .map_err(|error| CommandError::InvalidInput(format!("base runtime manifest: {error}")))?;
    let mut containers = BTreeMap::new();
    let mut container_guids = BTreeMap::new();
    for route in runtime_manifest.routes() {
        if !containers.contains_key(&route.container_name) {
            let container_path = root.join("containers").join(&route.container_name);
            let bytes = fs::read(&container_path)?;
            let reader = ContainerReader::new(bytes.clone()).map_err(|error| {
                CommandError::InvalidInput(format!(
                    "base container {}: {error}",
                    container_path.display()
                ))
            })?;
            let info = reader.info();
            let Some(identity) = runtime_manifest.container_identity(&route.container_name) else {
                return Err(CommandError::InvalidInput(format!(
                    "base manifest has no identity for container {:?}",
                    route.container_name
                )));
            };
            if identity.byte_len != info.total_len
                || identity.content_hash != info.content_hash
                || identity.generation != info.generation
            {
                return Err(CommandError::InvalidInput(format!(
                    "base container {:?} does not match its manifest identity",
                    route.container_name
                )));
            }
            container_guids.insert(
                route.container_name.clone(),
                reader.guids().collect::<BTreeSet<_>>(),
            );
            containers.insert(route.container_name.clone(), bytes);
        }
        if !container_guids
            .get(&route.container_name)
            .is_some_and(|guids| guids.contains(&route.container_guid))
        {
            return Err(CommandError::InvalidInput(format!(
                "base route {:?} is absent from container {:?}",
                route.guid, route.container_name
            )));
        }
    }

    let mut asset_guids = BTreeMap::new();
    for mapping in mappings {
        if runtime_manifest.route(mapping.guid).is_none() {
            return Err(CommandError::InvalidInput(format!(
                "base mapping asset {} references a missing runtime route",
                mapping.asset.0
            )));
        }
        if asset_guids.insert(mapping.asset, mapping.guid).is_some() {
            return Err(CommandError::InvalidInput(format!(
                "base mapping contains duplicate asset {}",
                mapping.asset.0
            )));
        }
    }
    let unique_guids = asset_guids.values().copied().collect::<BTreeSet<_>>();
    if asset_guids.len() != manifest.entries.len()
        || unique_guids.len() != asset_guids.len()
        || unique_guids.len() != runtime_manifest.routes().count()
        || manifest
            .entries
            .keys()
            .any(|asset| !asset_guids.contains_key(asset))
    {
        return Err(CommandError::InvalidInput(
            "base mapping does not cover the base build manifest one-to-one".into(),
        ));
    }
    Ok(RuntimePackage {
        source_manifest_hash: ContentHash::digest(&manifest.canonical_bytes()),
        asset_guids,
        containers,
        runtime_manifest,
        runtime_manifest_bytes,
    })
}

fn write_immutable(path: PathBuf, bytes: &[u8]) -> Result<(), CommandError> {
    match fs::read(&path) {
        Ok(existing) if existing == bytes => return Ok(()),
        Ok(_) => {
            return Err(CommandError::InvalidInput(format!(
                "refusing to overwrite divergent patch output {}",
                path.display()
            )));
        }
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
            return Err(CommandError::Io(error));
        }
        Err(_) => {}
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            if let Err(error) = file.write_all(bytes) {
                let _ = fs::remove_file(&path);
                return Err(CommandError::Io(error));
            }
            file.sync_all().map_err(CommandError::Io)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => match fs::read(&path) {
            Ok(existing) if existing == bytes => Ok(()),
            Ok(_) => Err(CommandError::InvalidInput(format!(
                "refusing to overwrite divergent patch output {}",
                path.display()
            ))),
            Err(error) => Err(CommandError::Io(error)),
        },
        Err(error) => Err(CommandError::Io(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_output_writer_is_immutable() {
        let root =
            std::env::temp_dir().join(format!("prism-cli-patch-writer-{}", std::process::id()));
        let path = root.join("nested/output.bin");
        write_immutable(path.clone(), b"stable").unwrap();
        write_immutable(path.clone(), b"stable").unwrap();
        assert!(write_immutable(path, b"different").is_err());
        let _ = fs::remove_dir_all(root);
    }
}
