//! End-to-end fresh-build determinism verification through both CLI entry points.

use prism_asset_build_core::{BuildRevision, RevisionStore};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Fixture(PathBuf);

impl Fixture {
    fn new(command: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-cli-determinism-{command}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir_all(&root).unwrap();
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 1, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[17, 29, 41, 255]).unwrap();
        }
        fs::write(root.join("source.png"), png).unwrap();
        Self(root)
    }

    fn run(&self, command: &str, verify: bool) -> String {
        let mut process = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"));
        process
            .arg(command)
            .arg("--source")
            .arg(self.0.join("source.png"))
            .args([
                "--logical-path",
                "textures/determinism.png",
                "--asset-id",
                "7",
            ])
            .arg("--store")
            .arg(self.0.join("store"));
        if verify {
            process.arg("--verify-determinism");
        }
        let output = process.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn verify_cli(command: &str) {
    let fixture = Fixture::new(command);
    let cold = fixture.run(command, true);
    assert!(cold.contains("revision=0\n"));
    assert!(cold.contains("cache_hits=0\n"));
    assert!(cold.contains("determinism_verified_jobs=2\n"));

    let cached = fixture.run(command, false);
    assert!(cached.contains("revision=1\n"));
    assert!(cached.contains("cache_hits=2\n"));
    assert!(!cached.contains("determinism_verified_jobs="));

    let verified = fixture.run(command, true);
    assert!(verified.contains("revision=2\n"));
    assert!(verified.contains("cache_hits=0\n"));
    assert!(verified.contains("determinism_verified_jobs=2\n"));

    let store = RevisionStore::open(fixture.0.join("store")).unwrap();
    let baseline = store.manifest(BuildRevision(0)).unwrap();
    for revision in [1, 2] {
        let manifest = store.manifest(BuildRevision(revision)).unwrap();
        assert_eq!(manifest.entries, baseline.entries);
        for descriptor in manifest.entries.values() {
            assert_eq!(
                store.cas().read(descriptor.content_hash).unwrap().len() as u64,
                descriptor.byte_len
            );
        }
    }
}

#[test]
fn build_verifies_cold_and_warm_outputs_without_using_ddc_hits() {
    verify_cli("build");
}

#[test]
fn import_verifies_cold_and_warm_outputs_without_using_ddc_hits() {
    verify_cli("import");
}
