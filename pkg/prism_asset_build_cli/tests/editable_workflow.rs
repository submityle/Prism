//! Exercises the installed command surface, including JSONL framing. Fixtures
//! use saved .prism packages and workspace payload CAS; their external source
//! PNG never exists.

use prism_asset::StableGuid;
use prism_asset_build_core::import::{
    CanonicalPngProcessor, ImageMipArtifact, IMAGE_ARTIFACT_KIND,
};
use prism_asset_build_core::{
    AssetId, BuildRevision, ContentHash, EditableAsset, EditableDependency,
    EditableDependencyKind as Kind, EditablePayload, EditableSubAsset, EditableSubAssetReference,
    EditableWorkspace, RevisionStore, SourceRecord,
};
use prism_asset_container::{ContainerReader, RuntimeManifest};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

fn invoke(args: &[&str], success: bool) -> Value {
    invoke_with_exit(args, if success { 0 } else { 1 })
}

fn invoke_with_exit(args: &[&str], expected_exit: i32) -> Value {
    let success = expected_exit == 0;
    let output = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"))
        .arg("--json")
        .args(args)
        .output()
        .unwrap();
    assert!(output.stderr.is_empty(), "{output:?}");
    assert_eq!(output.status.code(), Some(expected_exit), "{output:?}");
    let events = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2);
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["schema_version"], 1);
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["command"], args[0]);
    }
    assert_eq!(events[0]["event"], "command_started");
    assert_eq!(
        events[1]["event"],
        if success {
            "command_succeeded"
        } else {
            "command_failed"
        }
    );
    if success {
        assert!(events[1]["error"].is_null());
        events[1]["result"].clone()
    } else {
        assert!(events[1]["result"].is_null());
        events[1]["error"].clone()
    }
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn image_payload(seed: u8) -> Vec<u8> {
    // PNG exists only in memory to construct the initial mip layout. The saved
    // texels below intentionally differ from both source and generated mips.
    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, 2, 2);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[1, 2, 3, 255].repeat(4)).unwrap();
    }
    let mut image = CanonicalPngProcessor::import_mip_artifact(&png).unwrap();
    image.levels[0].rgba8 = [seed, 31, 47, 255].repeat(4);
    image.levels[1].rgba8 = vec![seed + 1, 33, 49, 255];
    image.encode().unwrap()
}

fn asset(id: u128, payload: Vec<u8>) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        IMAGE_ARTIFACT_KIND,
        0,
        b"saved image settings".to_vec(),
        SourceRecord::external(
            "missing-source.png",
            "image/png",
            ContentHash::digest(b"source is deliberately unavailable"),
        )
        .unwrap(),
        vec![EditableSubAsset {
            id: 4,
            name: "surface".into(),
            kind: IMAGE_ARTIFACT_KIND,
        }],
        Vec::new(),
        Vec::new(),
        payload,
        None,
    )
    .unwrap()
}

struct Fixture {
    root: PathBuf,
    workspace: PathBuf,
    store: PathBuf,
    package: PathBuf,
    saved: EditableAsset,
}

impl Fixture {
    fn new(soft: bool) -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-cli-editable-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        // Wall clocks can return the same timestamp to parallel test threads.
        // Never share a root or allow one fixture's Drop to remove another's.
        fs::create_dir(&root).unwrap();
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        asset(8, image_payload(61))
            .write_to_path(workspace.join("dependency.prism"))
            .unwrap();
        let mut saved = asset(7, image_payload(17));
        saved.dependencies = vec![
            EditableDependency {
                asset: AssetId(8),
                kind: Kind::Hard,
            },
            EditableDependency {
                asset: AssetId(9),
                kind: Kind::EditorOnly,
            },
        ];
        if soft {
            saved.dependencies.push(EditableDependency {
                asset: AssetId(99),
                kind: Kind::Soft,
            });
            saved.subasset_references.push(EditableSubAssetReference {
                asset: AssetId(99),
                subasset: 1234,
                kind: Kind::Soft,
            });
        }
        saved.write_to_path(workspace.join("edited.prism")).unwrap();
        // An unregistered editor-only kind must never be cooked.
        let mut editor_only = asset(9, b"editor data".to_vec());
        editor_only.kind = 42;
        editor_only
            .write_to_path(workspace.join("editor.prism"))
            .unwrap();
        Self {
            store: root.join("store"),
            package: root.join("package"),
            root,
            workspace,
            saved,
        }
    }

    fn invoke(&self, command: &str, extra: &[&str], success: bool) -> Value {
        let mut args = vec![command, "--workspace", path(&self.workspace), "--root", "7"];
        if command != "editable-plan" {
            args.extend(["--store", path(&self.store)]);
        }
        if command == "package" {
            args.extend(["--output", path(&self.package)]);
        } else if command == "validate" {
            args.extend(["--package", path(&self.package)]);
        }
        args.extend_from_slice(extra);
        invoke(&args, success)
    }

    fn revision(&self) -> Option<BuildRevision> {
        RevisionStore::open(&self.store)
            .unwrap()
            .current_revision()
            .unwrap()
    }

    fn externalize(&mut self, name: &str) -> Vec<u8> {
        let file = self.workspace.join(name);
        let mut saved = EditableAsset::read_from_path(&file).unwrap();
        let bytes = saved.authoring_payload.inline_bytes().unwrap().to_vec();
        let workspace = EditableWorkspace::open(&self.workspace).unwrap();
        let lease = workspace
            .store_payload(bytes.as_slice(), bytes.len() as u64)
            .unwrap();
        saved.authoring_payload = EditablePayload::External(lease.reference());
        saved.write_to_path(file).unwrap();
        if saved.asset == self.saved.asset {
            self.saved = saved;
        }
        bytes
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn saved_mips_and_missing_soft_target_roundtrip_without_source_png() {
    let fixture = Fixture::new(true);
    let saved_bytes = fs::read(fixture.workspace.join("edited.prism")).unwrap();
    let profile = "ab".repeat(32);
    let options = [
        "--profile-hash",
        profile.as_str(),
        "--toolchain-fingerprint",
        "explicit-test-toolchain",
        "--environment-fingerprint",
        "explicit-test-environment",
    ];
    let plan = fixture.invoke("editable-plan", &options, true);
    assert_eq!(plan["execution_order"], json!(["8", "7"]));
    assert_eq!(plan["assets"].as_array().unwrap().len(), 2);
    assert!(!fixture.store.exists());
    let mut build_options = options.to_vec();
    build_options.push("--verify-determinism");
    let build = fixture.invoke("editable-build", &build_options, true);
    assert_eq!(build["revision"], 0);
    assert_eq!(build["determinism_verified_jobs"], 2);
    assert_eq!(build["cache_hits"], 0);
    let packaged = fixture.invoke("package", &options, true);
    let validated = fixture.invoke("validate", &options, true);
    assert_eq!(
        packaged["runtime_manifest_hash"],
        validated["runtime_manifest_hash"]
    );
    let manifest =
        RuntimeManifest::decode(&fs::read(fixture.package.join("manifest.bin")).unwrap()).unwrap();
    assert_eq!(manifest.routes().count(), 2);
    let guid = StableGuid::from_u128(7);
    let metadata = manifest.metadata(guid).unwrap();
    assert_eq!(metadata.hard_dependencies, vec![StableGuid::from_u128(8)]);
    assert_eq!(
        metadata.optional_dependencies,
        vec![StableGuid::from_u128(99)]
    );
    assert!(manifest
        .routes()
        .all(|route| route.guid != StableGuid::from_u128(99)));
    let route = manifest.routes().find(|route| route.guid == guid).unwrap();
    let reader = ContainerReader::new(
        fs::read(
            fixture
                .package
                .join("containers")
                .join(&route.container_name),
        )
        .unwrap(),
    )
    .unwrap();
    let cooked = reader.read_all(route.container_guid).unwrap();
    assert_eq!(
        Some(cooked.as_slice()),
        fixture.saved.authoring_payload.inline_bytes()
    );
    assert_eq!(
        ImageMipArtifact::decode(&cooked).unwrap().levels[1].rgba8,
        [18, 33, 49, 255]
    );
    assert_eq!(
        fs::read(fixture.workspace.join("edited.prism")).unwrap(),
        saved_bytes
    );
    assert!(!fixture.workspace.join("missing-source.png").exists());
    assert!(!fixture.root.join("missing-source.png").exists());
    // Packaging must use the exact explicit fingerprints of the committed plan.
    assert_eq!(
        fixture.invoke("validate", &[], false)["code"],
        "editable_build_error"
    );
}

#[test]
fn editable_revision_is_consumable_by_existing_multiasset_mapping() {
    let fixture = Fixture::new(false);
    fixture.invoke("editable-build", &[], true);
    let mapping = fixture.root.join("mapping.txt");
    fs::write(&mapping, "version=1\nasset=7;guid=raw:00000000000000000000000000000007;container=core;priority=0;chunk_priority=0\nasset=8;guid=raw:00000000000000000000000000000008;container=core;priority=0;chunk_priority=0\n").unwrap();
    invoke(
        &[
            "package",
            "--store",
            path(&fixture.store),
            "--mapping",
            path(&mapping),
            "--output",
            path(&fixture.package),
        ],
        true,
    );
    invoke(
        &[
            "validate",
            "--store",
            path(&fixture.store),
            "--mapping",
            path(&mapping),
            "--package",
            path(&fixture.package),
        ],
        true,
    );
}

#[test]
fn canonical_roots_cache_determinism_and_payload_invalidation() {
    let fixture = Fixture::new(false);
    let first = fixture.invoke("editable-plan", &["--root", "8"], true);
    let reordered = invoke(
        &[
            "editable-plan",
            "--workspace",
            path(&fixture.workspace),
            "--root",
            "8",
            "--root",
            "7",
            "--root",
            "8",
        ],
        true,
    );
    assert_eq!(first, reordered);
    let cold = fixture.invoke(
        "editable-build",
        &["--max-concurrency", "2", "--cpu-threads", "2"],
        true,
    );
    let warm = fixture.invoke("editable-build", &[], true);
    assert_eq!(warm["revision"], 1);
    assert_eq!(warm["cache_hits"], 2);
    assert_eq!(cold["artifacts"], warm["artifacts"]);
    let verified = fixture.invoke("editable-build", &["--verify-determinism"], true);
    assert_eq!(verified["revision"], 2);
    assert_eq!(verified["cache_hits"], 0);
    assert_eq!(verified["determinism_verified_jobs"], 2);
    assert_eq!(cold["artifacts"], verified["artifacts"]);
    let mut changed = fixture.saved.clone();
    changed.authoring_payload = image_payload(81).into();
    changed
        .write_to_path(fixture.workspace.join("edited.prism"))
        .unwrap();
    let rebuilt = fixture.invoke("editable-build", &[], true);
    assert_eq!(rebuilt["revision"], 3);
    assert_eq!(rebuilt["cache_hits"], 1);
    assert_ne!(
        cold["artifacts"][0]["artifact_key"],
        rebuilt["artifacts"][0]["artifact_key"]
    );
    assert_eq!(cold["artifacts"][1], rebuilt["artifacts"][1]);
}

#[test]
fn budgets_and_invalid_selections_never_publish() {
    let fixture = Fixture::new(false);
    fixture.invoke("editable-build", &[], true);
    let cases: &[&[&str]] = &[
        &["--max-snapshot-bytes", "0"],
        &["--max-snapshot-bytes", "1"],
        &["--max-snapshot-bytes", "536870913"],
        &["--max-concurrency", "0"],
        &["--cpu-threads", "0"],
        &["--memory-bytes", "1"],
        &["--root", "9999"],
        &["--root", "0"],
        &["--profile-hash", "invalid"],
    ];
    for options in cases {
        fixture.invoke("editable-build", options, false);
        assert_eq!(fixture.revision(), Some(BuildRevision(0)), "{options:?}");
    }
    invoke(
        &["editable-plan", "--workspace", path(&fixture.workspace)],
        false,
    );
    let mut missing = fixture.saved.clone();
    missing.dependencies.push(EditableDependency {
        asset: AssetId(123),
        kind: Kind::Hard,
    });
    missing
        .write_to_path(fixture.workspace.join("edited.prism"))
        .unwrap();
    fixture.invoke("editable-build", &[], false);
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
}

#[test]
fn failed_cook_malformed_package_and_duplicate_ids_preserve_publication() {
    let fixture = Fixture::new(false);
    fixture.invoke("editable-build", &[], true);
    let original = fixture.saved.encode().unwrap();
    let mut invalid = fixture.saved.clone();
    invalid.authoring_payload = b"not a mip artifact".to_vec().into();
    invalid
        .write_to_path(fixture.workspace.join("edited.prism"))
        .unwrap();
    let invalid_bytes = invalid.encode().unwrap();
    assert_eq!(
        fixture.invoke("editable-build", &[], false)["code"],
        "editable_build_error"
    );
    assert_eq!(
        fs::read(fixture.workspace.join("edited.prism")).unwrap(),
        invalid_bytes
    );
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
    fs::write(fixture.workspace.join("edited.prism"), b"malformed").unwrap();
    fixture.invoke("editable-build", &[], false);
    assert_eq!(
        fs::read(fixture.workspace.join("edited.prism")).unwrap(),
        b"malformed"
    );
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
    fs::write(fixture.workspace.join("edited.prism"), &original).unwrap();
    fs::write(fixture.workspace.join("duplicate.prism"), &original).unwrap();
    fixture.invoke("editable-build", &[], false);
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
    let fresh = Fixture::new(false);
    invalid
        .write_to_path(fresh.workspace.join("edited.prism"))
        .unwrap();
    fresh.invoke("editable-build", &[], false);
    assert_eq!(fresh.revision(), None);
}

#[test]
fn workspace_packaging_rejects_mixed_modes_and_changed_saved_payload() {
    let fixture = Fixture::new(true);
    fixture.invoke("editable-build", &[], true);
    for command in ["package", "validate"] {
        for options in [
            vec!["--mapping", "nonexistent.mapping"],
            vec!["--asset-id", "7"],
            vec!["--logical-path", "unused.png"],
        ] {
            assert_eq!(
                fixture.invoke(command, &options, false)["code"],
                "invalid_input"
            );
        }
    }
    assert!(!fixture.package.exists());
    invoke(
        &[
            "package",
            "--store",
            path(&fixture.store),
            "--output",
            path(&fixture.package),
            "--root",
            "7",
        ],
        false,
    );
    fixture.invoke("package", &[], true);
    let runtime = fs::read(fixture.package.join("manifest.bin")).unwrap();
    let mut changed = fixture.saved.clone();
    changed.authoring_payload = image_payload(81).into();
    changed
        .write_to_path(fixture.workspace.join("edited.prism"))
        .unwrap();
    let rejected_output = fixture.root.join("rejected-package");
    invoke(
        &[
            "package",
            "--store",
            path(&fixture.store),
            "--workspace",
            path(&fixture.workspace),
            "--root",
            "7",
            "--output",
            path(&rejected_output),
        ],
        false,
    );
    assert!(!rejected_output.exists());
    assert_eq!(
        fixture.invoke("validate", &[], false)["code"],
        "editable_build_error"
    );
    assert_eq!(
        fs::read(fixture.package.join("manifest.bin")).unwrap(),
        runtime
    );
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
}

#[test]
fn external_payloads_build_package_validate_and_reuse_inline_ddc_keys() {
    let mut fixture = Fixture::new(false);
    let inline_plan = fixture.invoke("editable-plan", &[], true);
    let inline_build = fixture.invoke("editable-build", &[], true);
    assert_eq!(inline_build["cache_hits"], 0);
    let root_bytes = fixture.externalize("edited.prism");
    let dependency_bytes = fixture.externalize("dependency.prism");
    let exact = (root_bytes.len() + dependency_bytes.len()).to_string();
    let options = ["--max-materialized-payload-bytes", exact.as_str()];
    let saved_bytes = fs::read(fixture.workspace.join("edited.prism")).unwrap();
    assert_eq!(fixture.invoke("editable-plan", &options, true), inline_plan);
    let external_build = fixture.invoke("editable-build", &options, true);
    assert_eq!(external_build["revision"], 1);
    assert_eq!(external_build["cache_hits"], 2);
    assert_eq!(external_build["artifacts"], inline_build["artifacts"]);
    let packaged = fixture.invoke("package", &options, true);
    let validated = fixture.invoke("validate", &options, true);
    assert_eq!(
        packaged["runtime_manifest_hash"],
        validated["runtime_manifest_hash"]
    );
    let manifest =
        RuntimeManifest::decode(&fs::read(fixture.package.join("manifest.bin")).unwrap()).unwrap();
    assert_eq!(manifest.routes().count(), 2);
    for (id, expected) in [(7, &root_bytes), (8, &dependency_bytes)] {
        let route = manifest
            .routes()
            .find(|route| route.guid == StableGuid::from_u128(id))
            .unwrap();
        let reader = ContainerReader::new(
            fs::read(
                fixture
                    .package
                    .join("containers")
                    .join(&route.container_name),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(reader.read_all(route.container_guid).unwrap(), *expected);
    }
    assert_eq!(
        fs::read(fixture.workspace.join("edited.prism")).unwrap(),
        saved_bytes
    );
    assert!(!fixture.workspace.join("missing-source.png").exists());
}

#[test]
fn materialized_payload_aggregate_budget_is_exact_for_every_workspace_command() {
    let mut fixture = Fixture::new(false);
    let root_bytes = fixture.externalize("edited.prism");
    let dependency =
        EditableAsset::read_from_path(fixture.workspace.join("dependency.prism")).unwrap();
    // Mix external and inline inputs. Each fits individually; only their sum
    // exceeds the smaller budget. The editor-only payload is excluded.
    let total = root_bytes.len() as u64 + dependency.authoring_payload.byte_len();
    let below = total - 1;
    assert!(below >= root_bytes.len() as u64);
    assert!(below >= dependency.authoring_payload.byte_len());
    let exact = total.to_string();
    let below = below.to_string();
    let low_options = ["--max-materialized-payload-bytes", below.as_str()];
    let exact_options = ["--max-materialized-payload-bytes", exact.as_str()];
    let message = format!("selected payloads need {total} materialized bytes, limit is {below}");
    let package_bytes = fs::read(fixture.workspace.join("edited.prism")).unwrap();
    for command in ["editable-plan", "editable-build"] {
        let error = fixture.invoke(command, &low_options, false);
        assert_eq!(error["code"], "editable_build_error");
        assert!(error["message"].as_str().unwrap().contains(&message));
        assert!(!fixture.store.exists());
        assert!(!fixture.package.exists());
    }
    fixture.invoke("editable-plan", &exact_options, true);
    assert!(!fixture.store.exists());
    fixture.invoke("editable-build", &exact_options, true);
    for command in ["package", "validate"] {
        let error = fixture.invoke(command, &low_options, false);
        assert_eq!(error["code"], "editable_build_error");
        assert!(error["message"].as_str().unwrap().contains(&message));
        assert!(!fixture.package.exists());
        assert_eq!(fixture.revision(), Some(BuildRevision(0)));
    }
    fixture.invoke("package", &exact_options, true);
    fixture.invoke("validate", &exact_options, true);
    let runtime = fs::read(fixture.package.join("manifest.bin")).unwrap();
    for command in ["editable-build", "package", "validate"] {
        fixture.invoke(command, &low_options, false);
        assert_eq!(fixture.revision(), Some(BuildRevision(0)));
        assert_eq!(
            fs::read(fixture.package.join("manifest.bin")).unwrap(),
            runtime
        );
    }
    assert_eq!(
        fs::read(fixture.workspace.join("edited.prism")).unwrap(),
        package_bytes
    );
}

#[test]
fn shared_external_payload_counts_once_through_cli_workspace_commands() {
    let mut fixture = Fixture::new(false);
    let mut dependency =
        EditableAsset::read_from_path(fixture.workspace.join("dependency.prism")).unwrap();
    dependency.authoring_payload = fixture.saved.authoring_payload.clone();
    dependency
        .write_to_path(fixture.workspace.join("dependency.prism"))
        .unwrap();
    let bytes = fixture.externalize("edited.prism");
    assert_eq!(fixture.externalize("dependency.prism"), bytes);
    let exact = bytes.len().to_string();
    let options = ["--max-materialized-payload-bytes", exact.as_str()];
    for command in ["editable-plan", "editable-build", "package", "validate"] {
        fixture.invoke(command, &options, true);
    }
    assert_eq!(fixture.revision(), Some(BuildRevision(0)));
}

#[test]
fn materialized_payload_budget_rejects_zero_and_above_four_gib() {
    let fixture = Fixture::new(false);
    for command in ["editable-plan", "editable-build", "package", "validate"] {
        for limit in ["0", "4294967297"] {
            let error =
                fixture.invoke(command, &["--max-materialized-payload-bytes", limit], false);
            assert_eq!(error["code"], "editable_build_error");
            assert!(error["message"]
                .as_str()
                .unwrap()
                .contains("invalid materialized payload budget"));
            assert!(!fixture.store.exists());
            assert!(!fixture.package.exists());
        }
    }
    // The upper bound is inclusive and need not match the tiny input size.
    let maximum = ["--max-materialized-payload-bytes", "4294967296"];
    for command in ["editable-plan", "editable-build", "package", "validate"] {
        fixture.invoke(command, &maximum, true);
    }
}

#[test]
fn materialized_payload_budget_requires_workspace() {
    let fixture = Fixture::new(false);
    for command in ["editable-plan", "editable-build"] {
        let mut args = vec![
            command,
            "--root",
            "7",
            "--max-materialized-payload-bytes",
            "1024",
        ];
        if command == "editable-build" {
            args.extend(["--store", path(&fixture.store)]);
        }
        let error = invoke_with_exit(&args, 2);
        assert_eq!(error["code"], "invalid_arguments");
        assert!(error["message"].as_str().unwrap().contains("--workspace"));
    }
    for (command, output_flag) in [("package", "--output"), ("validate", "--package")] {
        // Neither legacy identity nor a mapping may silently discard an
        // editable-only budget when --workspace is absent.
        for identity in [
            vec!["--asset-id", "7", "--logical-path", "missing-source.png"],
            vec!["--mapping", "nonexistent.mapping"],
        ] {
            let mut args = vec![
                command,
                "--store",
                path(&fixture.store),
                output_flag,
                path(&fixture.package),
                "--max-materialized-payload-bytes",
                "1024",
            ];
            args.extend(identity);
            let error = invoke(&args, false);
            assert_eq!(error["code"], "invalid_input");
            assert!(error["message"]
                .as_str()
                .unwrap()
                .contains("require --workspace"));
        }
        assert!(!fixture.store.exists());
        assert!(!fixture.package.exists());
    }
}
