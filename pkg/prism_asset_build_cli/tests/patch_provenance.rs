use png::Encoder;
use prism_asset_build_core::{ContentHash, ReleaseProvenance};
use prism_asset_container::RuntimeManifest;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const LOGICAL_PATH: &str = "textures/provenance.png";
const ASSET_ID: &str = "7";

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "prism-cli-patch-provenance-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after epoch")
            .as_nanos()
    ))
}

fn png_bytes(rgb: [u8; 3]) -> Vec<u8> {
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

fn mapping(root: &Path) -> PathBuf {
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

fn assert_success(output: Output, context: &str) {
    assert!(
        output.status.success(),
        "{context}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn patch_emits_canonical_target_provenance_and_rejects_tampering() {
    let root = temp_dir();
    fs::create_dir_all(&root).expect("test root should be created");
    let source = root.join("source.png");
    let store = root.join("store");
    let base_package = root.join("base");
    let patch_dir = root.join("patch");
    let mapping = mapping(&root);

    fs::write(&source, png_bytes([1, 2, 3])).expect("base PNG should be written");
    assert_success(
        run(&[
            "build",
            "--source",
            source.to_str().unwrap(),
            "--logical-path",
            LOGICAL_PATH,
            "--asset-id",
            ASSET_ID,
            "--store",
            store.to_str().unwrap(),
        ]),
        "base build",
    );
    assert_success(
        run(&[
            "package",
            "--store",
            store.to_str().unwrap(),
            "--output",
            base_package.to_str().unwrap(),
            "--mapping",
            mapping.to_str().unwrap(),
        ]),
        "base package",
    );

    fs::write(&source, png_bytes([9, 8, 7])).expect("target PNG should be written");
    assert_success(
        run(&[
            "build",
            "--source",
            source.to_str().unwrap(),
            "--logical-path",
            LOGICAL_PATH,
            "--asset-id",
            ASSET_ID,
            "--store",
            store.to_str().unwrap(),
        ]),
        "target build",
    );
    assert_success(
        run(&[
            "patch",
            "--store",
            store.to_str().unwrap(),
            "--base-revision",
            "0",
            "--base-package",
            base_package.to_str().unwrap(),
            "--base-mapping",
            mapping.to_str().unwrap(),
            "--mapping",
            mapping.to_str().unwrap(),
            "--output",
            patch_dir.to_str().unwrap(),
        ]),
        "patch",
    );

    let provenance_bytes =
        fs::read(patch_dir.join("provenance.bin")).expect("patch must emit target provenance.bin");
    let provenance = ReleaseProvenance::decode(&provenance_bytes).expect("provenance is canonical");
    assert_eq!(provenance.revision.0, 1);
    assert_eq!(
        provenance.runtime_manifest_hash,
        ContentHash::digest(&fs::read(patch_dir.join("target_manifest.bin")).unwrap())
    );
    let target_manifest =
        RuntimeManifest::decode(&fs::read(patch_dir.join("target_manifest.bin")).unwrap())
            .expect("target manifest is canonical");
    assert_eq!(provenance.containers.len(), 1);
    assert_eq!(
        provenance.containers[0].name,
        target_manifest.routes().next().unwrap().container_name
    );

    let original = provenance_bytes.clone();
    let mut tampered = provenance_bytes;
    tampered[0] ^= 1;
    fs::write(patch_dir.join("provenance.bin"), &tampered).expect("tamper should be written");
    let rejected = run(&[
        "patch",
        "--store",
        store.to_str().unwrap(),
        "--base-revision",
        "0",
        "--base-package",
        base_package.to_str().unwrap(),
        "--base-mapping",
        mapping.to_str().unwrap(),
        "--mapping",
        mapping.to_str().unwrap(),
        "--output",
        patch_dir.to_str().unwrap(),
    ]);
    assert!(
        !rejected.status.success(),
        "tampered provenance was overwritten"
    );
    assert_eq!(
        fs::read(patch_dir.join("provenance.bin")).unwrap(),
        tampered
    );

    fs::write(patch_dir.join("provenance.bin"), original).expect("provenance should be restored");
    fs::remove_dir_all(root).expect("test outputs should be cleaned up");
}
