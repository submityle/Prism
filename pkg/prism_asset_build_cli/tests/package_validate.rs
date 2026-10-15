use png::Encoder;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const LOGICAL_PATH: &str = "textures/integration.png";
const ASSET_ID: &str = "7";

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "prism-cli-integration-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after epoch")
            .as_nanos()
    ))
}

fn png_bytes() -> Vec<u8> {
    png_bytes_with([11, 22, 33])
}

fn png_bytes_with(rgb: [u8; 3]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 1, 1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("PNG header should encode");
    writer
        .write_image_data(&rgb)
        .expect("PNG pixel should encode");
    drop(writer);
    bytes
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"))
        .args(args)
        .output()
        .expect("CLI process should start")
}

fn build(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let source = root.join("integration.png");
    let store = root.join("store");
    let package = root.join("package");
    fs::create_dir_all(root).expect("test root should be created");
    fs::write(&source, png_bytes()).expect("source PNG should be written");
    let source_arg = source.to_str().expect("temporary paths are UTF-8");
    let store_arg = store.to_str().expect("temporary paths are UTF-8");
    let output = run(&[
        "build",
        "--source",
        source_arg,
        "--logical-path",
        LOGICAL_PATH,
        "--asset-id",
        ASSET_ID,
        "--store",
        store_arg,
    ]);
    assert!(
        output.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (source, store, package)
}

fn package(store: &Path, output: &Path) -> Output {
    run(&[
        "package",
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
        "--output",
        output.to_str().expect("temporary paths are UTF-8"),
        "--asset-id",
        ASSET_ID,
        "--logical-path",
        LOGICAL_PATH,
    ])
}

fn validate(store: &Path, package: &Path) -> Output {
    run(&[
        "validate",
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
        "--package",
        package.to_str().expect("temporary paths are UTF-8"),
        "--asset-id",
        ASSET_ID,
        "--logical-path",
        LOGICAL_PATH,
    ])
}

fn explicit_mapping(root: &Path) -> PathBuf {
    let path = root.join("mapping.txt");
    fs::write(
        &path,
        format!(
            "version=1\nasset=6;guid=sub:{LOGICAL_PATH}#source;container=core;priority=0;chunk_priority=0\nasset=7;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=1\n"
        ),
    )
    .expect("mapping should be written");
    path
}

fn package_with_mapping(store: &Path, output: &Path, mapping: &Path) -> Output {
    run(&[
        "package",
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
        "--output",
        output.to_str().expect("temporary paths are UTF-8"),
        "--mapping",
        mapping.to_str().expect("temporary paths are UTF-8"),
    ])
}

fn validate_with_mapping(store: &Path, package: &Path, mapping: &Path) -> Output {
    run(&[
        "validate",
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
        "--package",
        package.to_str().expect("temporary paths are UTF-8"),
        "--mapping",
        mapping.to_str().expect("temporary paths are UTF-8"),
    ])
}

fn patch(
    store: &Path,
    base_revision: &str,
    base_package: &Path,
    base_mapping: &Path,
    mapping: &Path,
    output: &Path,
) -> Output {
    run(&[
        "patch",
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
        "--base-revision",
        base_revision,
        "--base-package",
        base_package.to_str().expect("temporary paths are UTF-8"),
        "--base-mapping",
        base_mapping.to_str().expect("temporary paths are UTF-8"),
        "--mapping",
        mapping.to_str().expect("temporary paths are UTF-8"),
        "--output",
        output.to_str().expect("temporary paths are UTF-8"),
    ])
}

#[test]
fn cli_build_package_validate_round_trip() {
    let root = temp_dir("round-trip");
    let (_source, store, package_dir) = build(&root);

    let packaged = package(&store, &package_dir);
    assert!(
        packaged.status.success(),
        "package failed: {}",
        String::from_utf8_lossy(&packaged.stderr)
    );
    let summary = String::from_utf8_lossy(&packaged.stdout);
    assert!(summary.contains("quality_checked_assets=2"), "{summary}");
    assert!(summary.contains("container=core "), "{summary}");
    assert!(package_dir.join("manifest.bin").is_file());
    assert!(package_dir.join("publication.bin").is_file());
    assert!(package_dir.join("containers/core").is_file());

    let validated = validate(&store, &package_dir);
    assert!(
        validated.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&validated.stderr)
    );
    assert!(String::from_utf8_lossy(&validated.stdout).contains("validated_revision=0"));
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_package_validate_accepts_explicit_multi_asset_mapping() {
    let root = temp_dir("explicit-mapping");
    let (_source, store, package_dir) = build(&root);
    let mapping = explicit_mapping(&root);

    let packaged = package_with_mapping(&store, &package_dir, &mapping);
    assert!(
        packaged.status.success(),
        "package failed: {}",
        String::from_utf8_lossy(&packaged.stderr)
    );
    let validated = validate_with_mapping(&store, &package_dir, &mapping);
    assert!(
        validated.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&validated.stderr)
    );
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_package_rejects_unknown_asset_in_mapping() {
    let root = temp_dir("invalid-mapping");
    let (_source, store, package_dir) = build(&root);
    let mapping = root.join("mapping.txt");
    fs::write(
        &mapping,
        "version=1\nasset=999;guid=path:textures/unknown.png;container=core;priority=0;chunk_priority=0\n",
    )
    .expect("mapping should be written");

    let packaged = package_with_mapping(&store, &package_dir, &mapping);
    assert!(
        !packaged.status.success(),
        "unknown mapping asset was accepted"
    );
    assert!(String::from_utf8_lossy(&packaged.stderr).contains("unknown manifest asset"));
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_package_rejects_duplicate_asset_in_mapping() {
    let root = temp_dir("duplicate-mapping");
    let (_source, store, package_dir) = build(&root);
    let mapping = root.join("mapping.txt");
    fs::write(
        &mapping,
        format!(
            "version=1\nasset=6;guid=sub:{LOGICAL_PATH}#source;container=core;priority=0;chunk_priority=0\nasset=6;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=1\nasset=7;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=1\n"
        ),
    )
    .expect("mapping should be written");

    let packaged = package_with_mapping(&store, &package_dir, &mapping);
    assert!(!packaged.status.success(), "duplicate mapping was accepted");
    let stderr = String::from_utf8_lossy(&packaged.stderr);
    assert!(stderr.contains("duplicate asset"), "{stderr}");
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_validate_rejects_tampered_runtime_container() {
    let root = temp_dir("tamper");
    let (_source, store, package_dir) = build(&root);

    let packaged = package(&store, &package_dir);
    assert!(
        packaged.status.success(),
        "package failed: {}",
        String::from_utf8_lossy(&packaged.stderr)
    );
    let container_path = package_dir.join("containers/core");
    let mut bytes = fs::read(&container_path).expect("runtime container should exist");
    let last = bytes.last_mut().expect("container should not be empty");
    *last ^= 0x40;
    fs::write(&container_path, bytes).expect("tampered container should be written");

    let validated = validate(&store, &package_dir);
    assert!(!validated.status.success(), "tampered package was accepted");
    assert!(String::from_utf8_lossy(&validated.stderr).contains("failed"));
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_patch_emits_incremental_manifests_and_containers() {
    let root = temp_dir("patch");
    let (source, store, base_package) = build(&root);
    let mapping = explicit_mapping(&root);
    let packaged = package_with_mapping(&store, &base_package, &mapping);
    assert!(
        packaged.status.success(),
        "base package failed: {}",
        String::from_utf8_lossy(&packaged.stderr)
    );

    fs::write(&source, png_bytes_with([44, 55, 66])).expect("changed PNG should be written");
    let rebuilt = run(&[
        "build",
        "--source",
        source.to_str().expect("temporary paths are UTF-8"),
        "--logical-path",
        LOGICAL_PATH,
        "--asset-id",
        ASSET_ID,
        "--store",
        store.to_str().expect("temporary paths are UTF-8"),
    ]);
    assert!(
        rebuilt.status.success(),
        "target build failed: {}",
        String::from_utf8_lossy(&rebuilt.stderr)
    );

    let patch_dir = root.join("patch");
    let emitted = patch(&store, "0", &base_package, &mapping, &mapping, &patch_dir);
    assert!(
        emitted.status.success(),
        "patch failed: {}",
        String::from_utf8_lossy(&emitted.stderr)
    );
    assert!(patch_dir.join("descriptor.bin").is_file());
    assert!(patch_dir.join("patch_manifest.bin").is_file());
    assert!(patch_dir.join("target_manifest.bin").is_file());
    assert!(patch_dir.join("containers").is_dir());
    assert!(String::from_utf8_lossy(&emitted.stdout).contains("target_revision=1"));

    let rerun = patch(&store, "0", &base_package, &mapping, &mapping, &patch_dir);
    assert!(
        rerun.status.success(),
        "identical patch rerun failed: {}",
        String::from_utf8_lossy(&rerun.stderr)
    );

    let invalid_base_mapping = root.join("invalid-base-mapping.txt");
    fs::write(
        &invalid_base_mapping,
        format!(
            "version=1\nasset=6;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=0\nasset=7;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=1\n"
        ),
    )
    .expect("invalid base mapping should be written");
    let invalid = patch(
        &store,
        "0",
        &base_package,
        &invalid_base_mapping,
        &mapping,
        &root.join("invalid-patch"),
    );
    assert!(
        !invalid.status.success(),
        "duplicate base GUID was accepted"
    );
    assert!(
        String::from_utf8_lossy(&invalid.stderr).contains("one-to-one"),
        "{}",
        String::from_utf8_lossy(&invalid.stderr)
    );

    fs::write(patch_dir.join("containers/stale"), b"stale")
        .expect("stale patch output should be written");
    let stale = patch(&store, "0", &base_package, &mapping, &mapping, &patch_dir);
    assert!(!stale.status.success(), "stale patch output was accepted");
    assert!(
        String::from_utf8_lossy(&stale.stderr).contains("unexpected container"),
        "{}",
        String::from_utf8_lossy(&stale.stderr)
    );
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}

#[test]
fn cli_validate_rejects_tampered_publication_descriptor() {
    let root = temp_dir("publication-tamper");
    let (_source, store, package_dir) = build(&root);

    let packaged = package(&store, &package_dir);
    assert!(
        packaged.status.success(),
        "package failed: {}",
        String::from_utf8_lossy(&packaged.stderr)
    );
    let publication_path = package_dir.join("publication.bin");
    let mut bytes = fs::read(&publication_path).expect("publication should exist");
    let last = bytes.last_mut().expect("publication should not be empty");
    *last ^= 0x01;
    fs::write(publication_path, bytes).expect("tampered publication should be written");

    let validated = validate(&store, &package_dir);
    assert!(
        !validated.status.success(),
        "tampered publication was accepted"
    );
    assert!(String::from_utf8_lossy(&validated.stderr).contains("failed"));
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}
