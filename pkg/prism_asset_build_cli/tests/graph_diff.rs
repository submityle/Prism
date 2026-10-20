//! Process-boundary queries over real published revisions. Mutations below are
//! fault injection in fixtures; the CLI must never initialize or repair them.
use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, ContentHash, Manifest, RevisionStore,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const JOURNAL_HEADER: usize = b"PRISM-TXN-JNL\0\x02".len();
const FRAME: usize = 8 + 99 + 32;

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    store: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::Builder::new()
            .prefix("prism-query-")
            .tempdir()
            .unwrap();
        let root = directory.path().to_path_buf();
        Self {
            store: root.join("store"),
            root,
            _directory: directory,
        }
    }
    fn publish(
        &self,
        revision: u64,
        mut entries: Vec<(u128, ArtifactDescriptor, Vec<u8>)>,
        reverse: bool,
    ) {
        let store = RevisionStore::open(&self.store).unwrap();
        let mut transaction = store.begin(
            BuildRevision(revision),
            revision.checked_sub(1).map(BuildRevision),
        );
        if reverse {
            entries.reverse();
        }
        for (asset, descriptor, bytes) in entries {
            transaction
                .stage(AssetId(asset), descriptor, bytes)
                .unwrap();
        }
        transaction.commit().unwrap();
    }
    fn populate(&self, reverse: bool) {
        self.publish(0, base(), reverse);
        self.publish(1, target(), reverse);
    }
    fn graph(&self, extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "graph".into(),
            "--store".into(),
            self.store.to_str().unwrap().into(),
            "--revision".into(),
            "0".into(),
        ];
        args.extend(extra.iter().map(|arg| (*arg).into()));
        args
    }
    fn diff(&self, base: &str, target: &str) -> Vec<String> {
        [
            "diff",
            "--store",
            self.store.to_str().unwrap(),
            "--base-revision",
            base,
            "--target-revision",
            target,
        ]
        .map(str::to_owned)
        .to_vec()
    }
    fn manifest(&self) -> PathBuf {
        self.store.join("revisions/0.manifest")
    }
    fn journal(&self) -> PathBuf {
        self.store.join("transactions/journal.bin")
    }
}
fn entry(
    asset: u128,
    key: u8,
    bytes: &[u8],
    dependencies: &[u8],
) -> (u128, ArtifactDescriptor, Vec<u8>) {
    (
        asset,
        ArtifactDescriptor::for_bytes(
            ArtifactKey([key; 32]),
            99,
            1,
            4,
            dependencies
                .iter()
                .map(|key| ArtifactKey([*key; 32]))
                .collect(),
            bytes,
        ),
        bytes.to_vec(),
    )
}
fn base() -> Vec<(u128, ArtifactDescriptor, Vec<u8>)> {
    vec![
        entry(2, 1, b"abc", &[]),
        entry(10, 1, b"abc", &[]),
        entry(20, 2, b"de", &[1]),
        entry(30, 3, b"f", &[2, 1]),
        entry(40, 4, b"gone", &[]),
        entry(60, 6, b"same", &[]),
    ]
}
fn target() -> Vec<(u128, ArtifactDescriptor, Vec<u8>)> {
    let mut changed = entry(20, 8, b"de", &[]);
    changed.1.kind = 100;
    changed.1.schema = 2;
    changed.1.alignment = 8;
    vec![
        entry(2, 1, b"abc", &[]),
        entry(10, 1, b"abc", &[]),
        changed,
        entry(30, 9, b"fresh", &[8, 1]),
        entry(50, 10, b"fresh", &[]),
        entry(60, 6, b"same", &[]),
    ]
}
fn run(args: &[String], json: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"));
    if json {
        command.arg("--json");
    }
    command.args(args).output().unwrap()
}
fn terminal(args: &[String], exit: i32) -> Value {
    let output = run(args, true);
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
        assert_eq!(event["command"], args[0]);
        assert_eq!(event.as_object().unwrap().len(), 7);
    }
    assert_eq!(events[0]["event"], "command_started");
    assert!(events[0]["result"].is_null());
    assert_eq!(events[1]["exit_code"], exit);
    assert_eq!(
        events[1]["event"],
        if exit == 0 {
            "command_succeeded"
        } else {
            "command_failed"
        }
    );
    if exit == 0 {
        assert!(events[1]["error"].is_null());
    } else {
        assert!(events[1]["result"].is_null());
        assert!(events[1]["error"]["message"].as_str().unwrap().len() > 0);
    }
    events[1].clone()
}
fn success(args: &[String]) -> Value {
    terminal(args, 0)["result"].clone()
}
fn tree(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    fn visit(root: &Path, at: &Path, result: &mut BTreeMap<PathBuf, Option<Vec<u8>>>) {
        result.insert(at.strip_prefix(root).unwrap().to_path_buf(), None);
        for entry in fs::read_dir(at).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, result);
            } else {
                result.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    Some(fs::read(path).unwrap()),
                );
            }
        }
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result);
    result
}
fn assert_failure_without_writes(f: &Fixture, args: &[String]) -> Value {
    let before = tree(&f.root);
    let event = terminal(args, 1);
    assert_eq!(tree(&f.root), before);
    event
}

#[test]
fn graph_full_forward_reverse_aliases_and_dependency_order() {
    let f = Fixture::new();
    f.populate(false);
    let before = tree(&f.root);
    let graph = success(&f.graph(&[]));
    assert_eq!(graph["revision"], 0);
    assert_eq!(graph["direction"], "all");
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 5);
    assert_eq!(graph["nodes"][0]["asset_ids"], json!(["2", "10"]));
    assert_eq!(graph["edges"].as_array().unwrap().len(), 3);
    assert_eq!(
        graph["edges"][1]["dependency_key"],
        ArtifactKey([2; 32]).to_hex()
    );
    assert_eq!(graph["edges"][2]["dependency_index"], 1);
    let forward = success(&f.graph(&["--asset-id", "30"]));
    assert_eq!(forward["nodes"].as_array().unwrap().len(), 3);
    let reverse = success(&f.graph(&["--asset-id", "10", "--reverse"]));
    assert_eq!(reverse["direction"], "dependents");
    assert_eq!(reverse["nodes"], forward["nodes"]);
    let leaf = success(&f.graph(&["--asset-id", "60", "--reverse"]));
    assert_eq!(leaf["nodes"].as_array().unwrap().len(), 1);
    assert!(leaf["edges"].as_array().unwrap().is_empty());
    let human = run(&f.graph(&[]), false);
    assert!(human.status.success());
    let text = String::from_utf8(human.stdout).unwrap();
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("node "))
            .count(),
        5
    );
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("edge "))
            .count(),
        3
    );
    assert_eq!(tree(&f.root), before);
    assert_failure_without_writes(&f, &f.graph(&["--asset-id", "999"]));
    assert_failure_without_writes(&f, &f.graph(&["--reverse"]));
}

#[test]
fn diff_full_descriptors_unique_content_metrics_equal_and_reverse() {
    let f = Fixture::new();
    f.populate(false);
    let before = tree(&f.root);
    let result = success(&f.diff("0", "1"));
    assert_eq!(
        (
            result["added"].clone(),
            result["removed"].clone(),
            result["changed"].clone(),
            result["unchanged"].clone()
        ),
        (json!(1), json!(1), json!(2), json!(3))
    );
    assert_eq!(result["base_bytes"], 17);
    assert_eq!(result["target_bytes"], 22);
    assert_eq!(result["download_bytes"], 5);
    assert_eq!(result["install_growth_bytes"], 5);
    assert_eq!(
        result["new_content"],
        json!([ContentHash::digest(b"fresh").to_hex()])
    );
    let entries = result["entries"].as_array().unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry["asset_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["20", "30", "40", "50"]
    );
    assert_eq!(
        entries[0]["changed_fields"],
        json!([
            "artifact_key",
            "kind",
            "schema",
            "alignment",
            "dependencies"
        ])
    );
    assert_eq!(entries[0]["content_reused"], true);
    assert_eq!(
        entries[0]["base"]["dependencies"],
        json!([ArtifactKey([1; 32]).to_hex()])
    );
    assert_eq!(entries[0]["target"]["schema"], 2);
    assert_eq!(entries[0]["target"]["storage_class"], "shared");
    assert_eq!(
        entries[1]["changed_fields"],
        json!(["artifact_key", "content_hash", "bytes", "dependencies"])
    );
    assert!(entries[2]["target"].is_null());
    assert!(entries[3]["base"].is_null());
    let reverse = success(&f.diff("1", "0"));
    assert_eq!(reverse["download_bytes"], 5);
    assert_eq!(reverse["install_growth_bytes"], -5);
    let equal = success(&f.diff("0", "0"));
    assert!(equal["entries"].as_array().unwrap().is_empty());
    assert_eq!(equal["unchanged"], 6);
    assert_eq!(equal["download_bytes"], 0);
    let human = run(&f.diff("0", "1"), false);
    assert!(human.status.success());
    let text = String::from_utf8(human.stdout).unwrap();
    assert!(text.contains("download_bytes=5 install_growth_bytes=5"));
    assert!(text.contains("changed_fields=artifact_key,kind,schema,alignment,dependencies"));
    assert_eq!(tree(&f.root), before);
}

#[test]
fn output_is_identical_across_insertion_order_and_process_reopen() {
    let a = Fixture::new();
    let b = Fixture::new();
    a.populate(false);
    b.populate(true);
    for (left, right) in [
        (a.graph(&[]), b.graph(&[])),
        (
            a.graph(&["--asset-id", "2", "--reverse"]),
            b.graph(&["--asset-id", "2", "--reverse"]),
        ),
        (a.diff("0", "1"), b.diff("0", "1")),
    ] {
        for json in [false, true] {
            let first = run(&left, json);
            assert!(first.status.success());
            assert_eq!(first.stdout, run(&left, json).stdout);
            assert_eq!(first.stdout, run(&right, json).stdout);
        }
    }
}

#[test]
fn missing_store_revision_manifest_and_journal_never_create_files() {
    let f = Fixture::new();
    assert_failure_without_writes(&f, &f.graph(&[]));
    assert_failure_without_writes(&f, &f.diff("0", "1"));
    assert!(!f.store.exists());
    f.populate(false);
    assert_failure_without_writes(&f, &f.diff("0", "999"));
    let mut graph = f.graph(&[]);
    graph[4] = "999".into();
    assert_failure_without_writes(&f, &graph);
    fs::remove_file(f.manifest()).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
    assert_failure_without_writes(&f, &f.diff("0", "1"));
    fs::remove_file(f.journal()).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
}

#[test]
fn historical_queries_need_neither_current_cas_ddc_nor_input_sidecars() {
    let f = Fixture::new();
    f.populate(false);
    fs::remove_file(f.store.join("CURRENT")).unwrap();
    fs::remove_dir_all(f.store.join("cas")).unwrap();
    let before = tree(&f.root);
    success(&f.graph(&[]));
    success(&f.diff("0", "1"));
    assert_eq!(tree(&f.root), before);
    assert!(!f.store.join("cas").exists());
    assert!(!f.store.join("ddc").exists());
}

#[test]
fn tampered_manifest_or_complete_journal_frame_is_rejected_without_repair() {
    let f = Fixture::new();
    f.populate(false);
    let original = fs::read(f.manifest()).unwrap();
    let mut damaged = original.clone();
    *damaged.last_mut().unwrap() ^= 1;
    fs::write(f.manifest(), damaged).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
    assert_failure_without_writes(&f, &f.diff("0", "1"));
    fs::write(f.manifest(), original).unwrap();
    let mut bytes = fs::read(f.journal()).unwrap();
    bytes[JOURNAL_HEADER + 8 + 34] ^= 1;
    fs::write(f.journal(), bytes).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
    assert_failure_without_writes(&f, &f.diff("0", "1"));
}

#[test]
fn prepare_current_without_commit_rejected_partial_tail_preserved() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let journal = fs::read(f.journal()).unwrap();
    fs::write(f.journal(), &journal[..JOURNAL_HEADER + FRAME]).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
    assert!(RevisionStore::read_committed_manifest(&f.store, BuildRevision(0)).is_err());
    let mut partial = journal.clone();
    partial.extend_from_slice(b"TXN");
    fs::write(f.journal(), partial).unwrap();
    let before = tree(&f.root);
    success(&f.graph(&[]));
    assert_eq!(tree(&f.root), before);
    let mut partial_commit = journal[..JOURNAL_HEADER + FRAME].to_vec();
    partial_commit.extend_from_slice(&journal[JOURNAL_HEADER + FRAME..journal.len() - 1]);
    fs::write(f.journal(), partial_commit).unwrap();
    assert_failure_without_writes(&f, &f.graph(&[]));
}

fn rechecksum(bytes: &mut [u8], frame_offset: usize) {
    let digest = ContentHash::digest(&bytes[frame_offset + 8..frame_offset + 107]);
    bytes[frame_offset + 107..frame_offset + FRAME].copy_from_slice(&digest.0);
}

#[test]
fn checksummed_binding_mismatch_and_post_commit_records_are_rejected() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let journal = fs::read(f.journal()).unwrap();
    for field in [1usize, 34, 66] {
        let mut changed = journal.clone();
        let commit = JOURNAL_HEADER + FRAME;
        changed[commit + 8 + field] ^= 1;
        rechecksum(&mut changed, commit);
        fs::write(f.journal(), changed).unwrap();
        assert_failure_without_writes(&f, &f.graph(&[]));
    }
    // Recovery rejects all three as well: there is no outstanding Prepare
    // after a durable Commit, and committed revisions cannot be re-prepared.
    for kind in [1u8, 2, 3] {
        let mut changed = journal.clone();
        let offset = changed.len();
        changed.extend_from_slice(&journal[JOURNAL_HEADER + FRAME..]);
        changed[offset + 8] = kind;
        rechecksum(&mut changed, offset);
        fs::write(f.journal(), changed).unwrap();
        assert_failure_without_writes(&f, &f.graph(&[]));
        assert!(RevisionStore::open(&f.store).is_err());
    }
}

fn bind_manifest(f: &Fixture, manifest: &[u8], original_journal: &[u8]) {
    fs::write(f.manifest(), manifest).unwrap();
    let hash = ContentHash::digest(manifest);
    let mut journal = original_journal.to_vec();
    for offset in [JOURNAL_HEADER, JOURNAL_HEADER + FRAME] {
        journal[offset + 8 + 34..offset + 8 + 66].copy_from_slice(&hash.0);
        rechecksum(&mut journal, offset);
    }
    fs::write(f.journal(), journal).unwrap();
}

#[test]
fn correctly_bound_noncanonical_or_invalid_manifests_are_rejected_by_core() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let original = fs::read(f.manifest()).unwrap();
    let journal = fs::read(f.journal()).unwrap();
    let mut trailing = original.clone();
    trailing.push(0);
    let mut wrong_revision = original.clone();
    let header = b"PRISM-MANIFEST\0\x02".len();
    wrong_revision[header] = 1;
    let mut huge_count = original.clone();
    huge_count[header + 8..header + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut wrong_order = original.clone();
    let start = header + 16;
    let first = wrong_order[start..start + 121].to_vec();
    wrong_order.copy_within(start + 121..start + 242, start);
    wrong_order[start + 121..start + 242].copy_from_slice(&first);
    for manifest in [trailing, wrong_revision, huge_count, wrong_order] {
        bind_manifest(&f, &manifest, &journal);
        assert_failure_without_writes(&f, &f.graph(&[]));
    }
    for case in 0..5 {
        let mut entries = vec![entry(1, 1, b"a", &[]), entry(2, 2, b"b", &[1])];
        match case {
            0 => entries[1].1.dependencies.push(ArtifactKey([99; 32])),
            1 => entries[1].1.dependencies.push(ArtifactKey([1; 32])),
            2 => entries[0].1.dependencies.push(ArtifactKey([2; 32])),
            3 => entries[1].1.key = ArtifactKey([1; 32]),
            _ => {
                entries[1].1.storage_class =
                    prism_asset_build_core::ArtifactStorageClass::DriverLocal
            }
        }
        let manifest = Manifest {
            revision: BuildRevision(0),
            entries: entries
                .into_iter()
                .map(|(id, d, _)| (AssetId(id), d))
                .collect(),
        };
        bind_manifest(&f, &manifest.canonical_bytes(), &journal);
        assert_failure_without_writes(&f, &f.graph(&[]));
        assert_failure_without_writes(&f, &f.diff("0", "0"));
    }
}

#[test]
fn empty_committed_revision_queries_are_complete() {
    let f = Fixture::new();
    f.publish(0, vec![], false);
    let graph = success(&f.graph(&[]));
    assert_eq!(graph["nodes"], json!([]));
    assert_eq!(graph["edges"], json!([]));
    let diff = success(&f.diff("0", "0"));
    assert_eq!(diff["entries"], json!([]));
    assert_eq!(diff["base_bytes"], 0);
    let (manifest, hash) =
        RevisionStore::read_committed_manifest(&f.store, BuildRevision(0)).unwrap();
    assert_eq!(hash, ContentHash::digest(&manifest.canonical_bytes()));
}

#[test]
fn aborted_attempt_then_new_prepare_commit_is_queryable() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let original = fs::read(f.journal()).unwrap();
    let mut aborted = original[JOURNAL_HEADER..].to_vec();
    // A prior failed attempt has a distinct transaction identity.
    aborted[8 + 1] ^= 1;
    aborted[FRAME + 8 + 1] ^= 1;
    aborted[FRAME + 8] = 3;
    rechecksum(&mut aborted, 0);
    rechecksum(&mut aborted, FRAME);
    let mut journal = original[..JOURNAL_HEADER].to_vec();
    journal.extend_from_slice(&aborted);
    journal.extend_from_slice(&original[JOURNAL_HEADER..]);
    fs::write(f.journal(), journal).unwrap();
    let before = tree(&f.root);
    success(&f.graph(&[]));
    assert_eq!(tree(&f.root), before);
    // Normal recovery also accepts the same abort/retry history.
    RevisionStore::open(&f.store).unwrap();
    let before = tree(&f.root);
    success(&f.graph(&[]));
    assert_eq!(tree(&f.root), before);
}

#[test]
fn input_admission_rejects_sparse_oversize_files_without_reading_or_repair() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let original = fs::read(f.manifest()).unwrap();
    let oversized_manifest = 64 * 1024 * 1024 + 1;
    fs::OpenOptions::new()
        .write(true)
        .open(f.manifest())
        .unwrap()
        .set_len(oversized_manifest)
        .unwrap();
    let result = terminal(&f.graph(&[]), 1);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("64 MiB"));
    assert_eq!(
        fs::metadata(f.manifest()).unwrap().len(),
        oversized_manifest
    );
    fs::write(f.manifest(), original).unwrap();
    let oversized_journal = 256 * 1024 * 1024 + 1;
    fs::OpenOptions::new()
        .write(true)
        .open(f.journal())
        .unwrap()
        .set_len(oversized_journal)
        .unwrap();
    let result = terminal(&f.diff("0", "0"), 1);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("256 MiB"));
    assert_eq!(fs::metadata(f.journal()).unwrap().len(), oversized_journal);
}

#[test]
fn byte_overflow_or_conflicting_content_lengths_never_return_partial_diff() {
    let f = Fixture::new();
    f.publish(0, base(), false);
    let journal = fs::read(f.journal()).unwrap();
    for conflicting_hash in [false, true] {
        let mut a = entry(1, 1, b"a", &[]).1;
        let mut b = entry(2, 2, b"bb", &[]).1;
        if conflicting_hash {
            b.content_hash = a.content_hash;
        } else {
            a.byte_len = u64::MAX;
        }
        let manifest = Manifest {
            revision: BuildRevision(0),
            entries: BTreeMap::from([(AssetId(1), a), (AssetId(2), b)]),
        };
        bind_manifest(&f, &manifest.canonical_bytes(), &journal);
        assert_failure_without_writes(&f, &f.diff("0", "0"));
    }
}
