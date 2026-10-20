//! Process-boundary coverage for cache inspection and reclamation. Fixtures
//! use the authoritative store APIs; no test writes a second DDC wire format.

use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, CasError, ContentAddressedStore,
    DerivedDataCache, RevisionStore,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    store: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-cache-{label}-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&root).unwrap();
        Self {
            store: root.join("store"),
            root,
        }
    }

    fn descriptor(key: u8, bytes: &[u8]) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 7, 1, 4, Vec::new(), bytes)
    }

    fn publish(&self, revision: u64, asset: u128, descriptor: ArtifactDescriptor, bytes: &[u8]) {
        let store = RevisionStore::open(&self.store).unwrap();
        let mut transaction = store.begin(
            BuildRevision(revision),
            revision.checked_sub(1).map(BuildRevision),
        );
        transaction
            .stage(AssetId(asset), descriptor, bytes.to_vec())
            .unwrap();
        transaction.commit().unwrap();
    }

    fn cache(&self, descriptor: ArtifactDescriptor) {
        let mut ddc = DerivedDataCache::open(self.store.join("ddc")).unwrap();
        ddc.put_descriptor(descriptor).unwrap();
    }

    fn args(&self, operation: &str, extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "cache".to_owned(),
            operation.to_owned(),
            "--store".to_owned(),
            self.store.to_string_lossy().into_owned(),
        ];
        args.extend(extra.iter().map(|argument| (*argument).to_owned()));
        args
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
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
    let events: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.len(), 2, "{text}");
    assert_eq!(events[0]["event"], "command_started");
    assert_eq!(events[1]["command"], "cache");
    assert_eq!(events[1]["exit_code"], exit);
    events[1].clone()
}

fn success(args: &[String]) -> Value {
    let event = terminal(args, 0);
    assert_eq!(event["event"], "command_succeeded");
    event["result"].clone()
}

fn failure(args: &[String]) -> Value {
    let event = terminal(args, 1);
    assert_eq!(event["event"], "command_failed");
    assert!(event["result"].is_null());
    event
}

fn tree(root: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
    fn visit(root: &Path, path: &Path, out: &mut Vec<(PathBuf, Option<Vec<u8>>)>) {
        out.push((path.strip_prefix(root).unwrap().to_path_buf(), None));
        let mut entries: Vec<_> = fs::read_dir(path).unwrap().map(Result::unwrap).collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let child = entry.path();
            if entry.file_type().unwrap().is_dir() {
                visit(root, &child, out);
            } else {
                out.push((
                    child.strip_prefix(root).unwrap().to_path_buf(),
                    Some(fs::read(child).unwrap()),
                ));
            }
        }
    }
    let mut out = Vec::new();
    visit(root, root, &mut out);
    out
}

#[test]
fn status_and_verify_are_read_only_and_reject_complete_oversize_indexes() {
    let fixture = Fixture::new("readonly");
    let descriptor = Fixture::descriptor(1, b"published");
    fixture.publish(0, 1, descriptor.clone(), b"published");
    fixture.cache(descriptor);
    let before = tree(&fixture.root);

    let status = success(&fixture.args("status", &[]));
    assert_eq!(status["ddc_entries"], 1);
    assert_eq!(status["cas_blobs"], 1);
    let verification = success(&fixture.args("verify", &[]));
    assert_eq!(verification["checked_blobs"], 1);
    assert_eq!(verification["checked_current_descriptors"], 1);
    assert_eq!(tree(&fixture.root), before);

    let index_size = fs::metadata(fixture.store.join("ddc/index.ddc"))
        .unwrap()
        .len();
    assert!(index_size > 0);
    let max_index_bytes = (index_size - 1).to_string();
    let event = failure(&fixture.args("status", &["--max-index-bytes", &max_index_bytes]));
    assert_eq!(event["error"]["code"], "ddc_error");
    assert_eq!(tree(&fixture.root), before);
}

#[test]
fn missing_store_queries_fail_without_initializing_any_cache_path() {
    let fixture = Fixture::new("missing");
    let before = tree(&fixture.root);
    for operation in ["status", "verify"] {
        let event = failure(&fixture.args(operation, &[]));
        assert_eq!(event["error"]["code"], "io_error");
        assert_eq!(tree(&fixture.root), before);
    }
}

#[test]
fn verify_reports_corruption_without_repairing_it() {
    let fixture = Fixture::new("corrupt");
    let descriptor = Fixture::descriptor(2, b"verified bytes");
    fixture.publish(0, 1, descriptor.clone(), b"verified bytes");
    fixture.cache(descriptor.clone());
    let store = RevisionStore::open(&fixture.store).unwrap();
    let blob = store.cas().blob_path(descriptor.content_hash);
    fs::write(&blob, b"corrupted").unwrap();
    let before = fs::read(&blob).unwrap();

    let event = failure(&fixture.args("verify", &[]));
    assert_eq!(event["error"]["code"], "cas_error");
    assert_eq!(fs::read(blob).unwrap(), before);
}

#[test]
fn prune_removes_deleted_uncommitted_cache_descriptors_but_keeps_committed_history() {
    let fixture = Fixture::new("prune-closure");
    let historical = Fixture::descriptor(3, b"historical");
    let current = Fixture::descriptor(4, b"current");
    let orphan = Fixture::descriptor(5, b"uncommitted cache output");
    fixture.publish(0, 1, historical.clone(), b"historical");
    fixture.publish(1, 1, current.clone(), b"current");
    fixture.cache(historical.clone());
    fixture.cache(current.clone());
    let store = RevisionStore::open(&fixture.store).unwrap();
    assert_eq!(
        store.cas().put(b"uncommitted cache output").unwrap(),
        orphan.content_hash
    );
    fixture.cache(orphan.clone());

    let result = success(&fixture.args("prune", &["--max-entries", "100", "--grace-seconds", "0"]));
    assert_eq!(result["cas_deleted"], 1);
    assert_eq!(result["removed_missing_keys"], json!([orphan.key.to_hex()]));
    assert_eq!(result["ddc_entries"], 2);
    assert!(store.cas().blob_path(historical.content_hash).is_file());
    assert!(store.cas().blob_path(current.content_hash).is_file());
    assert!(!store.cas().blob_path(orphan.content_hash).exists());
    let verified = success(&fixture.args("verify", &[]));
    assert_eq!(verified["checked_blobs"], 2);
}

#[test]
fn prune_defers_while_another_process_holds_the_build_output_lease() {
    let fixture = Fixture::new("lease");
    let descriptor = Fixture::descriptor(6, b"published");
    fixture.publish(0, 1, descriptor, b"published");
    let store = RevisionStore::open(&fixture.store).unwrap();
    let orphan = store.cas().put(b"protected unpublished output").unwrap();
    let lease = store.protect_build_outputs().unwrap();

    let deferred =
        success(&fixture.args("prune", &["--max-entries", "100", "--grace-seconds", "0"]));
    assert_eq!(deferred["deferred_due_to_active_build"], true);
    assert!(store.cas().blob_path(orphan).exists());
    drop(lease);

    let completed =
        success(&fixture.args("prune", &["--max-entries", "100", "--grace-seconds", "0"]));
    assert_eq!(completed["deferred_due_to_active_build"], false);
    assert_eq!(completed["cas_deleted"], 1);
    assert!(!store.cas().blob_path(orphan).exists());
}

#[test]
fn nonregular_cas_roots_and_shards_are_rejected_without_following_them() {
    let fixture = Fixture::new("nonregular");
    fs::create_dir_all(&fixture.store).unwrap();
    fs::write(fixture.store.join("cas"), b"not a directory").unwrap();
    let event = failure(&fixture.args("status", &[]));
    assert_eq!(event["error"]["code"], "invalid_input");

    let root = fixture.root.join("standalone-cas");
    fs::create_dir(&root).unwrap();
    let hash = prism_asset_build_core::ContentHash::digest(b"x");
    fs::write(root.join(&hash.to_hex()[..2]), b"not a shard").unwrap();
    assert!(matches!(
        ContentAddressedStore::verify_at(&root, hash),
        Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
    ));

    let ddc_root = fixture.root.join("nonregular-ddc");
    fs::write(&ddc_root, b"not a directory").unwrap();
    assert!(matches!(
        DerivedDataCache::read_snapshot(&ddc_root, 1024),
        Err(prism_asset_build_core::DdcError::InvalidFormat(_))
    ));

    let ddc_with_directory_index = fixture.root.join("ddc-directory-index");
    fs::create_dir(&ddc_with_directory_index).unwrap();
    fs::create_dir(ddc_with_directory_index.join("index.ddc")).unwrap();
    assert!(matches!(
        DerivedDataCache::read_snapshot(&ddc_with_directory_index, 1024),
        Err(prism_asset_build_core::DdcError::InvalidFormat(_))
    ));
}

#[cfg(unix)]
#[test]
fn symlinked_cache_roots_are_rejected_without_traversal() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new("symlink");
    fs::create_dir_all(&fixture.store).unwrap();
    let outside = fixture.root.join("outside");
    fs::create_dir(&outside).unwrap();
    symlink(&outside, fixture.store.join("cas")).unwrap();
    let event = failure(&fixture.args("status", &[]));
    assert_eq!(event["error"]["code"], "invalid_input");

    let ddc_link = fixture.root.join("ddc-link");
    symlink(&outside, &ddc_link).unwrap();
    assert!(matches!(
        DerivedDataCache::read_snapshot(&ddc_link, 1024),
        Err(prism_asset_build_core::DdcError::InvalidFormat(_))
    ));
    let ddc_root = fixture.root.join("ddc-root");
    fs::create_dir(&ddc_root).unwrap();
    let outside_index = outside.join("index.ddc");
    fs::write(&outside_index, b"not an index").unwrap();
    symlink(&outside_index, ddc_root.join("index.ddc")).unwrap();
    assert!(matches!(
        DerivedDataCache::read_snapshot(&ddc_root, 1024),
        Err(prism_asset_build_core::DdcError::InvalidFormat(_))
    ));

    let cas_root = fixture.root.join("cas-root");
    fs::create_dir(&cas_root).unwrap();
    let hash = prism_asset_build_core::ContentHash::digest(b"shard payload");
    let shard_name = &hash.to_hex()[..2];
    let real_shard = outside.join("real-shard");
    fs::create_dir(&real_shard).unwrap();
    fs::write(real_shard.join(&hash.to_hex()[2..]), b"shard payload").unwrap();
    symlink(&real_shard, cas_root.join(shard_name)).unwrap();
    assert!(matches!(
        ContentAddressedStore::verify_at(&cas_root, hash),
        Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
    ));

    let cas_with_blob_link = fixture.root.join("cas-blob-link");
    fs::create_dir(&cas_with_blob_link).unwrap();
    let blob_hash = prism_asset_build_core::ContentHash::digest(b"linked blob");
    let blob_hex = blob_hash.to_hex();
    fs::create_dir(cas_with_blob_link.join(&blob_hex[..2])).unwrap();
    let outside_blob = outside.join("linked-blob");
    fs::write(&outside_blob, b"linked blob").unwrap();
    symlink(
        &outside_blob,
        cas_with_blob_link.join(&blob_hex[..2]).join(&blob_hex[2..]),
    )
    .unwrap();
    assert!(matches!(
        ContentAddressedStore::verify_at(&cas_with_blob_link, blob_hash),
        Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
    ));
}
