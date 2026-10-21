//! Real CLI invocations over standalone snapshots, with no workspace or store.

use prism_asset_build_core::{
    AssetId, ContentHash, EditableAsset, EditableDependency, EditableDependencyKind as Kind,
    EditablePayload, EditablePayloadRef, EditableSubAsset, EditableSubAssetReference, SourceRecord,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const MAX_INPUT_BYTES: u64 = 8 * 1024 * 1024;
const PRIVATE_PAYLOAD: &[u8] = b"PRIVATE_PAYLOAD_OLD";
const PRIVATE_SETTINGS: &[u8] = b"PRIVATE_SETTINGS_OLD";
const PRIVATE_PREVIEW: &[u8] = b"PRIVATE_PREVIEW_DATA";
const PRIVATE_SOURCE: &[u8] = b"PRIVATE_EMBEDDED_SOURCE_DATA";

fn subasset(id: u64, name: &str) -> EditableSubAsset {
    EditableSubAsset {
        id,
        name: name.into(),
        kind: 42,
    }
}

fn dependency(asset: u128, kind: Kind) -> EditableDependency {
    EditableDependency {
        asset: AssetId(asset),
        kind,
    }
}

fn reference(asset: u128, subasset: u64, kind: Kind) -> EditableSubAssetReference {
    EditableSubAssetReference {
        asset: AssetId(asset),
        subasset,
        kind,
    }
}

fn asset() -> EditableAsset {
    EditableAsset::new(
        AssetId(7),
        42,
        0,
        PRIVATE_SETTINGS.to_vec(),
        // This source is never resolved or opened by editable-diff.
        SourceRecord::external(
            "missing-source.bin",
            "test/raw",
            ContentHash::digest(b"missing"),
        )
        .unwrap(),
        vec![subasset(4, "part")],
        Vec::new(),
        Vec::new(),
        PRIVATE_PAYLOAD.to_vec(),
        None,
    )
    .unwrap()
}

#[derive(Debug, Eq, PartialEq)]
enum Entry {
    File(Vec<u8>),
    Directory,
    Symlink(PathBuf),
}

fn tree(root: &Path) -> BTreeMap<PathBuf, Entry> {
    fn visit(root: &Path, directory: &Path, entries: &mut BTreeMap<PathBuf, Entry>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            let key = path.strip_prefix(root).unwrap().to_path_buf();
            let metadata = fs::symlink_metadata(&path).unwrap();
            if metadata.file_type().is_symlink() {
                entries.insert(key, Entry::Symlink(fs::read_link(path).unwrap()));
            } else if metadata.is_dir() {
                entries.insert(key, Entry::Directory);
                visit(root, &path, entries);
            } else {
                entries.insert(key, Entry::File(fs::read(path).unwrap()));
            }
        }
    }
    let mut entries = BTreeMap::new();
    visit(root, root, &mut entries);
    entries
}

struct Fixture {
    root: TempDir,
    before: PathBuf,
    after: PathBuf,
}

impl Fixture {
    fn new(before: &EditableAsset, after: &EditableAsset) -> Self {
        let root = tempfile::tempdir().unwrap();
        let before_path = root.path().join("before.prism");
        let after_path = root.path().join("after.prism");
        before.write_to_path(&before_path).unwrap();
        after.write_to_path(&after_path).unwrap();
        Self {
            root,
            before: before_path,
            after: after_path,
        }
    }

    fn run(&self, before: &Path, after: &Path, extra: &[&str], json: bool) -> Output {
        let initial = tree(self.root.path());
        let mut command = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"));
        command.current_dir(self.root.path());
        if json {
            command.arg("--json");
        }
        let output = command
            .arg("editable-diff")
            .arg("--before")
            .arg(before)
            .arg("--after")
            .arg(after)
            .args(extra)
            .output()
            .unwrap();
        assert_eq!(
            initial,
            tree(self.root.path()),
            "CLI changed its inputs or created output"
        );
        // Check both presentation modes for accidental opaque-content logging.
        for private in [
            PRIVATE_PAYLOAD,
            PRIVATE_SETTINGS,
            PRIVATE_PREVIEW,
            PRIVATE_SOURCE,
            b"PRIVATE_PAYLOAD_NEW",
            b"PRIVATE_SETTINGS_NEW",
        ] {
            let marker = std::str::from_utf8(private).unwrap();
            assert!(
                !String::from_utf8_lossy(&output.stdout).contains(marker),
                "{output:?}"
            );
            assert!(
                !String::from_utf8_lossy(&output.stderr).contains(marker),
                "{output:?}"
            );
        }
        output
    }

    fn json(&self, before: &Path, after: &Path, extra: &[&str], exit: i32) -> Value {
        let output = self.run(before, after, extra, true);
        assert_eq!(output.status.code(), Some(exit), "{output:?}");
        assert!(output.stderr.is_empty(), "{output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.ends_with('\n'));
        let events: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 2, "{text}");
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(event["schema_version"], 1);
            assert_eq!(event["sequence"], sequence);
            assert_eq!(event["command"], "editable-diff");
            assert_eq!(event.as_object().unwrap().len(), 7);
        }
        assert_eq!(events[0]["event"], "command_started");
        assert!(events[0]["result"].is_null());
        assert!(events[0]["error"].is_null());
        assert!(events[0]["exit_code"].is_null());
        assert_eq!(events[1]["exit_code"], exit);
        if exit == 0 {
            assert_eq!(events[1]["event"], "command_succeeded");
            assert!(events[1]["error"].is_null());
            events[1]["result"].clone()
        } else {
            assert_eq!(events[1]["event"], "command_failed");
            assert!(events[1]["result"].is_null());
            events[1]["error"].clone()
        }
    }

    fn compare(&self) -> Value {
        self.json(&self.before, &self.after, &[], 0)
    }
}

fn blob(bytes: &[u8]) -> Value {
    json!({"bytes": bytes.len(), "content_hash": ContentHash::digest(bytes).to_hex()})
}

fn payload_summary(payload: &EditablePayload) -> Value {
    json!({
        "bytes": payload.byte_len(),
        "content_hash": payload.content_hash().to_hex(),
        "storage": match payload {
            EditablePayload::Inline(_) => "inline",
            EditablePayload::External(_) => "external",
        },
    })
}

#[test]
fn payload_storage_changes_are_nonempty_editor_only_without_resolving_cas() {
    let before = asset();
    let mut after = before.clone();
    after.authoring_payload = EditablePayload::External(EditablePayloadRef {
        content_hash: before.authoring_payload.content_hash(),
        byte_len: before.authoring_payload.byte_len(),
    });
    let fixture = Fixture::new(&before, &after);
    let impact = json!({"cook": false, "editor_import": true});
    for (old_path, new_path, old, new) in [
        (&fixture.before, &fixture.after, &before, &after),
        (&fixture.after, &fixture.before, &after, &before),
    ] {
        let result = fixture.json(old_path, new_path, &[], 0);
        assert_eq!(result["identical"], false);
        assert_eq!(result["impact"], impact);
        assert_eq!(
            result["authoring_payload"],
            json!({
                "before": payload_summary(&old.authoring_payload),
                "after": payload_summary(&new.authoring_payload),
                "impact": impact,
            })
        );
        assert_ne!(
            result["before"]["content_hash"],
            result["after"]["content_hash"]
        );
        let text = fixture.run(old_path, new_path, &[], false);
        assert!(text.status.success(), "{text:?}");
        let text = String::from_utf8(text.stdout).unwrap();
        assert!(text.starts_with("identical=false cook=false editor_import=true\n"));
        assert!(text.contains("\"storage\":\"inline\""));
        assert!(text.contains("\"storage\":\"external\""));
    }
    let identical = fixture.json(&fixture.after, &fixture.after, &[], 0);
    assert_eq!(identical["identical"], true);
    assert!(identical["authoring_payload"].is_null());
    assert_eq!(
        identical["impact"],
        json!({"cook": false, "editor_import": false})
    );
}

#[test]
fn external_content_changes_compare_hash_and_length_with_encoded_input_budget() {
    let mut before = asset();
    let reference = EditablePayloadRef {
        content_hash: ContentHash::digest(b"unavailable external content"),
        byte_len: 1024 * 1024 * 1024,
    };
    before.authoring_payload = EditablePayload::External(reference);
    for changed in [
        EditablePayloadRef {
            content_hash: ContentHash::digest(b"different external content"),
            ..reference
        },
        EditablePayloadRef {
            byte_len: reference.byte_len + 1,
            ..reference
        },
    ] {
        let mut after = before.clone();
        after.authoring_payload = EditablePayload::External(changed);
        let fixture = Fixture::new(&before, &after);
        let limit = before
            .encode()
            .unwrap()
            .len()
            .max(after.encode().unwrap().len())
            .to_string();
        // Budget covers the small snapshot encoding, not the referenced GiB.
        // Fixture verifies no CAS/store/workspace or other output is created.
        let result = fixture.json(
            &fixture.before,
            &fixture.after,
            &["--max-input-bytes", &limit],
            0,
        );
        let impact = json!({"cook": true, "editor_import": false});
        assert_eq!(result["identical"], false);
        assert_eq!(result["impact"], impact);
        assert_eq!(result["authoring_payload"]["impact"], impact);
        assert_eq!(
            result["authoring_payload"]["before"],
            payload_summary(&before.authoring_payload)
        );
        assert_eq!(
            result["authoring_payload"]["after"],
            payload_summary(&after.authoring_payload)
        );
    }
}

#[test]
fn changed_payload_content_and_storage_affect_both_domains() {
    let before = asset();
    let mut after = before.clone();
    after.authoring_payload = EditablePayload::External(EditablePayloadRef {
        content_hash: ContentHash::digest(b"different payload"),
        byte_len: before.authoring_payload.byte_len(),
    });
    let fixture = Fixture::new(&before, &after);
    let result = fixture.compare();
    let impact = json!({"cook": true, "editor_import": true});
    assert_eq!(result["impact"], impact);
    assert_eq!(result["authoring_payload"]["impact"], impact);
    assert_eq!(
        result["authoring_payload"]["after"],
        payload_summary(&after.authoring_payload)
    );
}

#[test]
fn identical_snapshots_and_same_path_have_stable_empty_diff() {
    let before = asset();
    let fixture = Fixture::new(&before, &before);
    let result = fixture.compare();
    assert_eq!(result["identical"], true);
    assert_eq!(
        result["impact"],
        json!({"cook": false, "editor_import": false})
    );
    assert_eq!(result["before"], result["after"]);
    assert_eq!(result["before"]["asset_id"], "7");
    assert_eq!(
        result["before"]["content_hash"],
        ContentHash::digest(&before.encode().unwrap()).to_hex()
    );
    for field in [
        "asset_id",
        "kind",
        "flags",
        "source",
        "canonical_settings",
        "authoring_payload",
        "preview",
    ] {
        assert!(result[field].is_null(), "{field}");
    }
    assert_eq!(result["subassets"], json!([]));
    assert_eq!(result["dependencies"], json!({"added": [], "removed": []}));
    assert_eq!(
        result["subasset_references"],
        json!({"added": [], "removed": []})
    );
    assert_eq!(
        fixture.json(&fixture.before, &fixture.before, &[], 0),
        result
    );
    let first = fixture.run(&fixture.before, &fixture.after, &[], true);
    let second = fixture.run(&fixture.before, &fixture.after, &[], true);
    assert_eq!(first.stdout, second.stdout);
    let text = fixture.run(&fixture.before, &fixture.after, &[], false);
    assert!(text.status.success());
    assert!(text.stderr.is_empty());
    assert!(String::from_utf8(text.stdout)
        .unwrap()
        .starts_with("identical=true cook=false editor_import=false\n"));
}

#[test]
fn metadata_only_diff_summarizes_source_preview_and_editor_edges_in_both_directions() {
    let before = asset();
    let mut after = before.clone();
    after.source = SourceRecord::embedded(
        "moved\nforged=record.bin",
        "other/format",
        PRIVATE_SOURCE.to_vec(),
    )
    .unwrap();
    after.preview = Some(PRIVATE_PREVIEW.to_vec());
    after.dependencies.push(dependency(77, Kind::EditorOnly));
    after
        .subasset_references
        .push(reference(77, 99, Kind::EditorOnly));
    let fixture = Fixture::new(&before, &after);
    let result = fixture.compare();
    let impact = json!({"cook": false, "editor_import": true});
    assert_eq!(result["identical"], false);
    assert_eq!(result["impact"], impact);
    assert_eq!(result["source"]["impact"], impact);
    assert_eq!(
        result["source"]["after"]["embedded_source"],
        blob(PRIVATE_SOURCE)
    );
    assert_eq!(result["preview"]["after"], blob(PRIVATE_PREVIEW));
    assert!(result["preview"]["before"].is_null());
    assert_eq!(
        result["dependencies"]["added"],
        json!([{"asset_id": "77", "kind": "editor_only", "impact": impact}])
    );
    assert_eq!(
        result["subasset_references"]["added"][0]["subasset_id"],
        "99"
    );
    assert_eq!(result["subasset_references"]["added"][0]["impact"], impact);
    let reversed = fixture.json(&fixture.after, &fixture.before, &[], 0);
    assert_eq!(reversed["impact"], impact);
    assert_eq!(
        reversed["dependencies"]["removed"],
        result["dependencies"]["added"]
    );
    assert_eq!(
        reversed["subasset_references"]["removed"],
        result["subasset_references"]["added"]
    );
    assert!(reversed["preview"]["after"].is_null());
    let text = fixture.run(&fixture.before, &fixture.after, &[], false);
    assert!(text.status.success());
    assert!(text.stderr.is_empty());
    let text = String::from_utf8(text.stdout).unwrap();
    assert!(text.starts_with("identical=false cook=false editor_import=true\n"));
    assert_eq!(text.lines().count(), 7);
    assert!(!text.lines().any(|line| line.starts_with("forged=")));
    assert!(text.contains(r"moved\nforged=record.bin"));
}

#[test]
fn cook_changes_report_stable_id_matching_scalars_opaque_hashes_and_edge_transitions() {
    let mut before = asset();
    before.subassets = vec![subasset(2, "removed"), subasset(4, "part")];
    before.dependencies = vec![dependency(2, Kind::Hard), dependency(3, Kind::Soft)];
    before.subasset_references = vec![reference(2, 4, Kind::Hard), reference(3, 8, Kind::Soft)];
    let mut after = before.clone();
    after.asset = AssetId(8);
    after.kind = 43;
    after.flags = 9;
    after.canonical_settings = b"PRIVATE_SETTINGS_NEW".to_vec();
    after.authoring_payload = b"PRIVATE_PAYLOAD_NEW".to_vec().into();
    assert_eq!(
        before.authoring_payload.byte_len(),
        after.authoring_payload.byte_len()
    );
    after.subassets = vec![subasset(4, "renamed"), subasset(8, "added")];
    after.dependencies = vec![dependency(2, Kind::Soft), dependency(5, Kind::Hard)];
    after.subasset_references = vec![reference(2, 4, Kind::Soft), reference(5, 9, Kind::Hard)];
    let fixture = Fixture::new(&before, &after);
    let result = fixture.compare();
    let impact = json!({"cook": true, "editor_import": false});
    assert_eq!(result["impact"], impact);
    assert_eq!(
        result["asset_id"],
        json!({"before": "7", "after": "8", "impact": impact})
    );
    assert_eq!(result["kind"]["after"], "43");
    assert_eq!(result["flags"]["after"], 9);
    for (field, old, new) in [
        (
            "canonical_settings",
            before.canonical_settings.as_slice(),
            after.canonical_settings.as_slice(),
        ),
        (
            "authoring_payload",
            before.authoring_payload.inline_bytes().unwrap(),
            after.authoring_payload.inline_bytes().unwrap(),
        ),
    ] {
        let expected = |bytes: &[u8]| {
            if field == "authoring_payload" {
                payload_summary(&EditablePayload::Inline(bytes.to_vec()))
            } else {
                blob(bytes)
            }
        };
        assert_eq!(result[field]["before"], expected(old));
        assert_eq!(result[field]["after"], expected(new));
        assert_ne!(
            result[field]["before"]["content_hash"],
            result[field]["after"]["content_hash"]
        );
        assert_eq!(result[field]["impact"], impact);
    }
    let changes = result["subassets"].as_array().unwrap();
    assert_eq!(
        changes
            .iter()
            .map(|value| value["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["2", "4", "8"]
    );
    assert_eq!(
        changes
            .iter()
            .map(|value| value["change"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["removed", "modified", "added"]
    );
    assert_eq!(changes[1]["before"]["name"], "part");
    assert_eq!(changes[1]["after"]["name"], "renamed");
    assert!(changes[0]["after"].is_null());
    assert!(changes[2]["before"].is_null());
    for field in ["dependencies", "subasset_references"] {
        assert_eq!(result[field]["removed"].as_array().unwrap().len(), 2);
        assert_eq!(result[field]["added"].as_array().unwrap().len(), 2);
        assert_eq!(result[field]["removed"][0]["asset_id"], "2");
        assert_eq!(result[field]["removed"][0]["kind"], "hard");
        assert_eq!(result[field]["added"][0]["kind"], "soft");
        assert_eq!(result[field]["added"][0]["impact"], impact);
    }
    let text = fixture.run(&fixture.before, &fixture.after, &[], false);
    assert!(text.status.success());
    assert!(text.stderr.is_empty());
    let text = String::from_utf8(text.stdout).unwrap();
    assert!(text.starts_with("identical=false cook=true editor_import=false\n"));
    assert!(text.contains("subasset={\"id\":\"4\",\"change\":\"modified\""));
    assert!(text.contains("dependency_added="));
    assert!(text.contains("reference_removed="));
}

#[test]
fn absent_and_empty_preview_are_distinct_without_cook_impact() {
    let before = asset();
    let mut after = before.clone();
    after.preview = Some(Vec::new());
    let fixture = Fixture::new(&before, &after);
    let result = fixture.compare();
    assert_eq!(
        result["impact"],
        json!({"cook": false, "editor_import": true})
    );
    assert!(result["preview"]["before"].is_null());
    assert_eq!(result["preview"]["after"], blob(b""));
}

#[test]
fn strict_decode_rejects_invalid_truncated_and_trailing_input_on_either_side() {
    let saved = asset();
    let fixture = Fixture::new(&saved, &saved);
    let canonical = saved.encode().unwrap();
    let mut trailing = canonical.clone();
    trailing.push(0);
    for bytes in [
        b"invalid package".to_vec(),
        canonical[..canonical.len() - 1].to_vec(),
        trailing,
    ] {
        fs::write(&fixture.after, bytes).unwrap();
        for (before, after) in [
            (&fixture.before, &fixture.after),
            (&fixture.after, &fixture.before),
        ] {
            let error = fixture.json(before, after, &[], 1);
            assert_eq!(error["code"], "editable_error");
        }
        let text = fixture.run(&fixture.before, &fixture.after, &[], false);
        assert_eq!(text.status.code(), Some(1));
        assert!(text.stdout.is_empty());
        assert!(!text.stderr.is_empty());
    }
}

#[test]
fn budgets_are_per_file_and_can_only_tighten_with_no_output_side_effects() {
    let saved = asset();
    let fixture = Fixture::new(&saved, &saved);
    let exact = saved.encode().unwrap().len().to_string();
    fixture.json(
        &fixture.before,
        &fixture.after,
        &["--max-input-bytes", &exact],
        0,
    );
    let short = (exact.parse::<u64>().unwrap() - 1).to_string();
    for limit in [short.as_str(), "0", "8388609", "18446744073709551615"] {
        let error = fixture.json(
            &fixture.before,
            &fixture.after,
            &["--max-input-bytes", limit],
            1,
        );
        assert_eq!(error["code"], "invalid_input");
    }
    let oversized = fixture.root.path().join("oversized.prism");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(MAX_INPUT_BYTES + 1)
        .unwrap();
    for (before, after) in [(&oversized, &fixture.after), (&fixture.before, &oversized)] {
        let error = fixture.json(before, after, &[], 1);
        assert_eq!(error["code"], "invalid_input");
        assert!(error["message"]
            .as_str()
            .unwrap()
            .contains("max-input-bytes"));
    }
    let error = fixture.json(
        &fixture.before,
        &fixture.after,
        &["--max-input-bytes", "not-a-number"],
        2,
    );
    assert_eq!(error["code"], "invalid_arguments");
}

#[test]
fn missing_non_regular_and_wrong_extension_inputs_are_read_only_failures() {
    let fixture = Fixture::new(&asset(), &asset());
    let directory = fixture.root.path().join("directory.prism");
    fs::create_dir(&directory).unwrap();
    let wrong_extension = fixture.root.path().join("asset.bin");
    fs::copy(&fixture.before, &wrong_extension).unwrap();
    let missing = fixture.root.path().join("missing.prism");
    for (input, code) in [
        (&directory, "invalid_input"),
        (&wrong_extension, "invalid_input"),
        (&missing, "io_error"),
    ] {
        assert_eq!(fixture.json(input, &fixture.after, &[], 1)["code"], code);
        assert_eq!(fixture.json(&fixture.before, input, &[], 1)["code"], code);
    }
}

#[cfg(unix)]
#[test]
fn symlinks_and_broken_symlinks_are_rejected_on_both_sides() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new(&asset(), &asset());
    let link = fixture.root.path().join("link.prism");
    let broken = fixture.root.path().join("broken.prism");
    symlink(&fixture.before, &link).unwrap();
    symlink(fixture.root.path().join("missing.prism"), &broken).unwrap();
    for input in [&link, &broken] {
        assert_eq!(
            fixture.json(input, &fixture.after, &[], 1)["code"],
            "invalid_input"
        );
        assert_eq!(
            fixture.json(&fixture.before, input, &[], 1)["code"],
            "invalid_input"
        );
    }
}

#[test]
fn collection_change_budget_rejects_incomplete_results() {
    let before = asset();
    let mut after = before.clone();
    after.subassets = (1..=65_536)
        .map(|id| subasset(id, &format!("part-{id}")))
        .collect();
    // 65,535 additions plus the changed ID 4 name and this edge exceed the
    // combined 65,536 record budget, while each input collection remains valid.
    after.dependencies.push(dependency(999_999, Kind::Soft));
    let fixture = Fixture::new(&before, &after);
    let error = fixture.json(&fixture.before, &fixture.after, &[], 1);
    assert_eq!(error["code"], "invalid_input");
    assert!(error["message"].as_str().unwrap().contains("65536"));
}

#[test]
fn escaped_output_budget_is_enforced_before_any_success_in_json_or_text() {
    let mut before = asset();
    let mut after = before.clone();
    // Both encoded files are below 1 MiB, but JSON escaping each control byte
    // as six bytes expands the full before/after result beyond 8 MiB.
    let escaped = "\u{0001}".repeat(400);
    before.subassets = (1..=2000)
        .map(|id| subasset(id, &format!("before-{id}-{escaped}")))
        .collect();
    after.subassets = (1..=2000)
        .map(|id| subasset(id, &format!("after-{id}-{escaped}")))
        .collect();
    assert!(before.encode().unwrap().len() < 1024 * 1024);
    assert!(after.encode().unwrap().len() < 1024 * 1024);
    let fixture = Fixture::new(&before, &after);
    let error = fixture.json(&fixture.before, &fixture.after, &[], 1);
    assert_eq!(error["code"], "invalid_input");
    assert!(error["message"].as_str().unwrap().contains("output budget"));
    let text = fixture.run(&fixture.before, &fixture.after, &[], false);
    assert_eq!(text.status.code(), Some(1));
    assert!(text.stdout.is_empty());
    assert!(text.stderr.len() < 1024);
}
