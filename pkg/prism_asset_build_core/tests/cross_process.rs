//! Independent-process acceptance of local publication and stale-writer gates.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use fs4::fs_std::FileExt;
use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, DdcError, DerivedDataCache,
    PublishError, ReleaseKeyStatus, ReleaseSigner, ReleaseTrustError, ReleaseTrustStore,
    RevisionStore, TrustPolicy,
};

const TIMEOUT: Duration = Duration::from_secs(30);
const ROOT_ENV: &str = "PRISM_BUILD_PROCESS_TEST_ROOT";
const JOB_ENV: &str = "PRISM_BUILD_PROCESS_TEST_JOB";
const INDEX_ENV: &str = "PRISM_BUILD_PROCESS_TEST_INDEX";

/// Reaped even if an assertion fails; child stdout is drained concurrently so
/// a failed child can never deadlock its parent on a full pipe.
struct Worker {
    child: Child,
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    transcript: String,
}

impl Worker {
    fn spawn(root: &Path, job: &str, index: u8) -> Self {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "subprocess_worker",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ROOT_ENV, root)
            .env(JOB_ENV, job)
            .env(INDEX_ENV, index.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(format!("stdout read failed: {error}"));
                        break;
                    }
                }
            }
        });
        Self {
            child,
            lines,
            reader: Some(reader),
            transcript: String::new(),
        }
    }

    fn ready(&mut self) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            let line = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| {
                    panic!("child did not become ready: {error}\n{}", self.transcript)
                });
            self.transcript.push_str(&line);
            self.transcript.push('\n');
            if line.ends_with("PRISM_READY") {
                break;
            }
        }
    }

    fn release(&mut self) {
        self.child
            .stdin
            .take()
            .unwrap()
            .write_all(b"publish\n")
            .unwrap();
    }

    fn finish(mut self) -> String {
        let deadline = Instant::now() + TIMEOUT;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "child publication timed out\n{}",
                self.transcript
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        self.reader.take().unwrap().join().unwrap();
        for line in self.lines.try_iter() {
            self.transcript.push_str(&line);
            self.transcript.push('\n');
        }
        assert!(
            status.success(),
            "child failed: {status}\n{}",
            self.transcript
        );
        self.transcript.clone()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn rendezvous() {
    println!("PRISM_READY");
    std::io::stdout().flush().unwrap();
    let mut command = String::new();
    std::io::stdin().read_line(&mut command).unwrap();
    assert_eq!(
        command, "publish\n",
        "parent must explicitly release publication"
    );
}

fn descriptor(index: u8, bytes: &[u8]) -> ArtifactDescriptor {
    ArtifactDescriptor::for_bytes(ArtifactKey([index; 32]), 4, 1, 16, vec![], bytes)
}

// The parent invokes this exact test in a new executable for each worker.
// The ordinary test run has no worker environment and performs no child work.
#[test]
fn subprocess_worker() {
    let Some(root) = std::env::var_os(ROOT_ENV) else {
        return;
    };
    let job = std::env::var(JOB_ENV).unwrap();
    let index: u8 = std::env::var(INDEX_ENV).unwrap().parse().unwrap();
    match job.as_str() {
        "hold-lock" => {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(root)
                .unwrap();
            FileExt::lock_exclusive(&file).unwrap();
            rendezvous();
            FileExt::unlock(&file).unwrap();
        }
        "ddc" => {
            let mut cache = DerivedDataCache::open(root).unwrap();
            assert!(cache.is_empty());
            rendezvous();
            for offset in 0..16 {
                let key = index * 16 + offset + 1;
                cache
                    .put(ArtifactKey([key; 32]), descriptor(key, &[key]))
                    .unwrap();
            }
            println!("PRISM_OK");
        }
        "revision" => {
            let store = RevisionStore::open(root).unwrap();
            assert_eq!(store.current_revision().unwrap(), None);
            let mut transaction = store.begin(BuildRevision(0), None);
            transaction
                .stage(
                    AssetId(u128::from(index) + 1),
                    descriptor(index, &[index]),
                    vec![index],
                )
                .unwrap();
            rendezvous();
            match transaction.commit() {
                Ok(_) => println!("PRISM_WINNER_{index}"),
                Err(PublishError::Conflict { .. }) => println!("PRISM_CONFLICT"),
                other => panic!("unexpected publication result: {other:?}"),
            }
        }
        "ddc-conflict" => {
            let mut cache = DerivedDataCache::open(root).unwrap();
            assert!(cache.is_empty());
            rendezvous();
            match cache.put(ArtifactKey([71; 32]), descriptor(71, &[index])) {
                Ok(()) => println!("PRISM_WINNER_{index}"),
                Err(DdcError::KeyConflict { .. }) => println!("PRISM_CONFLICT"),
                other => panic!("unexpected cache conflict result: {other:?}"),
            }
        }
        "trust" => {
            let mut trust = ReleaseTrustStore::open(root).unwrap();
            assert_eq!(trust.policy().policy_revision(), 1);
            let key = ReleaseSigner::from_seed(&[58; 32]).unwrap().public_key();
            rendezvous();
            let result = if index == 0 {
                trust.revoke_key(key.key_id)
            } else {
                trust.advance_minimum_revision(BuildRevision(2))
            };
            match result {
                Ok(()) => println!("PRISM_WINNER_{index}"),
                Err(ReleaseTrustError::PolicyChanged {
                    expected: 1,
                    actual: 2,
                }) => println!("PRISM_CONFLICT"),
                other => panic!("unexpected trust result: {other:?}"),
            }
        }
        _ => panic!("unknown child job: {job}"),
    }
}

fn run_contenders(root: &Path, job: &str, count: u8) -> Vec<String> {
    let mut workers: Vec<_> = (0..count)
        .map(|index| Worker::spawn(root, job, index))
        .collect();
    // All handles/transactions were opened from the same stale state before
    // any publication starts. No wall-clock delay is used as concurrency proof.
    for worker in &mut workers {
        worker.ready();
    }
    for worker in &mut workers {
        worker.release();
    }
    workers.into_iter().map(Worker::finish).collect()
}

fn winner(outputs: &[String]) -> u8 {
    assert_eq!(
        outputs
            .iter()
            .filter(|output| output.contains("PRISM_CONFLICT"))
            .count(),
        1,
        "{outputs:?}"
    );
    let winners: Vec<_> = outputs
        .iter()
        .enumerate()
        .filter(|(index, output)| output.contains(&format!("PRISM_WINNER_{index}")))
        .map(|(index, _)| index as u8)
        .collect();
    assert_eq!(winners.len(), 1, "{outputs:?}");
    winners[0]
}

/// Probe the actual stable lock file while another process holds it, then
/// terminate that process without running its Rust destructors. `wait` proves
/// process death; neither lock contention nor recovery is inferred from time.
fn terminate_lock_holder(lock_path: &Path) {
    let mut worker = Worker::spawn(lock_path, "hold-lock", 0);
    worker.ready();
    let probe = OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    assert!(!FileExt::try_lock_exclusive(&probe).unwrap());
    worker.child.kill().unwrap();
    assert!(!worker.child.wait().unwrap().success());
    assert!(FileExt::try_lock_exclusive(&probe).unwrap());
    FileExt::unlock(&probe).unwrap();
}

#[test]
fn killed_process_releases_ddc_lock_without_losing_committed_entries() {
    let root = tempfile::tempdir().unwrap();
    let mut cache = DerivedDataCache::open(root.path()).unwrap();
    cache
        .put(ArtifactKey([9; 32]), descriptor(9, &[9]))
        .unwrap();
    terminate_lock_holder(&root.path().join(".index.ddc.lock"));
    cache
        .put(ArtifactKey([10; 32]), descriptor(10, &[10]))
        .unwrap();
    let cache = DerivedDataCache::open(root.path()).unwrap();
    assert_eq!(cache.len(), 2);
    for index in [9, 10] {
        assert_eq!(
            cache.get(ArtifactKey([index; 32])),
            Some(descriptor(index, &[index]))
        );
    }
}

#[test]
fn killed_process_releases_revision_lock_without_losing_committed_manifest() {
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut first = store.begin(BuildRevision(0), None);
    first
        .stage(AssetId(1), descriptor(7, &[7]), vec![7])
        .unwrap();
    first.commit().unwrap();
    terminate_lock_holder(&root.path().join(".publish.lock"));
    let mut next = store.begin(BuildRevision(1), Some(BuildRevision(0)));
    next.stage(AssetId(1), descriptor(8, &[8]), vec![8])
        .unwrap();
    next.commit().unwrap();
    let reopened = RevisionStore::open(root.path()).unwrap();
    assert_eq!(reopened.current_revision().unwrap(), Some(BuildRevision(1)));
    for (revision, index) in [(0, 7), (1, 8)] {
        let manifest = reopened.manifest(BuildRevision(revision)).unwrap();
        assert_eq!(
            manifest.entries.get(&AssetId(1)),
            Some(&descriptor(index, &[index]))
        );
        assert_eq!(
            reopened
                .cas()
                .read(descriptor(index, &[index]).content_hash)
                .unwrap(),
            vec![index]
        );
    }
}

#[test]
fn killed_process_releases_trust_lock_without_weakening_policy() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("trust.policy");
    let key = ReleaseSigner::from_seed(&[58; 32]).unwrap().public_key();
    let policy = TrustPolicy::new(0, BuildRevision(4));
    let mut trust = ReleaseTrustStore::create(&path, policy).unwrap();
    trust.add_key(key).unwrap();
    terminate_lock_holder(&root.path().join("trust.policy.lock"));
    trust.revoke_key(key.key_id).unwrap();
    let reopened = ReleaseTrustStore::open(&path).unwrap();
    assert_eq!(reopened.policy().policy_revision(), 2);
    assert_eq!(reopened.policy().minimum_revision(), BuildRevision(4));
    assert_eq!(
        reopened.policy().key_status(key.key_id),
        Some(ReleaseKeyStatus::Revoked)
    );
}

#[test]
fn independent_processes_merge_stale_ddc_writers_without_lost_updates() {
    let root = tempfile::tempdir().unwrap();
    let outputs = run_contenders(root.path(), "ddc", 4);
    assert!(outputs.iter().all(|output| output.contains("PRISM_OK")));
    let cache = DerivedDataCache::open(root.path()).unwrap();
    assert_eq!(cache.len(), 64);
    for key in 1..=64 {
        assert_eq!(
            cache.get(ArtifactKey([key; 32])),
            Some(descriptor(key, &[key]))
        );
    }
}

#[test]
fn independent_processes_publish_exactly_one_expected_revision() {
    let root = tempfile::tempdir().unwrap();
    let index = winner(&run_contenders(root.path(), "revision", 2));
    let store = RevisionStore::open(root.path()).unwrap();
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
    let expected = descriptor(index, &[index]);
    let manifest = store.manifest(BuildRevision(0)).unwrap();
    assert_eq!(manifest.entries.len(), 1);
    assert_eq!(
        manifest.entries.get(&AssetId(u128::from(index) + 1)),
        Some(&expected)
    );
    assert_eq!(
        store.cas().read(expected.content_hash).unwrap(),
        vec![index]
    );
}

#[test]
fn independent_processes_cannot_replace_an_immutable_ddc_key() {
    let root = tempfile::tempdir().unwrap();
    let index = winner(&run_contenders(root.path(), "ddc-conflict", 2));
    let cache = DerivedDataCache::open(root.path()).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.get(ArtifactKey([71; 32])),
        Some(descriptor(71, &[index]))
    );
}

#[test]
fn independent_processes_reject_stale_trust_policy_replacement() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("trust.policy");
    let key = ReleaseSigner::from_seed(&[58; 32]).unwrap().public_key();
    let mut policy = TrustPolicy::new(0, BuildRevision(0));
    policy.add_key(key).unwrap();
    ReleaseTrustStore::create(&path, policy).unwrap();
    let index = winner(&run_contenders(&path, "trust", 2));
    let mut trust = ReleaseTrustStore::open(&path).unwrap();
    assert_eq!(trust.policy().policy_revision(), 2);
    assert_eq!(
        trust.policy().minimum_revision(),
        BuildRevision(if index == 0 { 0 } else { 2 })
    );
    assert_eq!(trust.policy().public_key(key.key_id), Some(key));
    assert_eq!(
        trust.policy().key_status(key.key_id),
        Some(if index == 0 {
            ReleaseKeyStatus::Revoked
        } else {
            ReleaseKeyStatus::Active
        })
    );
    // An explicit retry from a fresh snapshot must preserve the first writer's
    // security decision: neither a revocation nor the release floor is lost.
    if index == 0 {
        trust.advance_minimum_revision(BuildRevision(2)).unwrap();
    } else {
        trust.revoke_key(key.key_id).unwrap();
    }
    let trust = ReleaseTrustStore::open(path).unwrap();
    assert_eq!(trust.policy().policy_revision(), 3);
    assert_eq!(trust.policy().minimum_revision(), BuildRevision(2));
    assert_eq!(
        trust.policy().key_status(key.key_id),
        Some(ReleaseKeyStatus::Revoked)
    );
}
