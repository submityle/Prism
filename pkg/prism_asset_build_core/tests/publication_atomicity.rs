//! Integration-level publication-boundary regression tests.

use prism_asset_build_core::{
    ArtifactSpec, AssetId, BuildError, BuildExecutor, BuildGraph, BuildJob, BuildRevision,
    DerivedDataCache, JobId, ProcessorError, RevisionStore,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("prism-asset-build-atomicity-{id}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn spec(asset: u128, settings: &[u8]) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(asset),
        processor_id: 0x50524f434553534f52,
        processor_version: 1,
        canonical_settings: settings.to_vec(),
        profile_hash: [1; 32],
        toolchain_fingerprint: b"test-toolchain".to_vec(),
        environment_fingerprint: b"test-environment".to_vec(),
        kind: 0x4152544946414354,
        schema: 1,
        alignment: 4,
    }
}

#[test]
fn failed_later_job_does_not_publish_a_new_revision() {
    let root = temp_dir();
    let mut graph = BuildGraph::new();
    graph
        .add_job(BuildJob::new(JobId(1), [], |_: &[&[u8]]| {
            Ok(b"completed dependency".to_vec())
        }))
        .unwrap();
    graph
        .add_job(BuildJob::new(JobId(2), [JobId(1)], |_: &[&[u8]]| {
            Err(ProcessorError::new("later stage failed"))
        }))
        .unwrap();

    let mut specs = BTreeMap::new();
    specs.insert(JobId(1), spec(41, b"first"));
    specs.insert(JobId(2), spec(42, b"second"));
    let store = RevisionStore::open(&root).unwrap();
    let mut ddc = DerivedDataCache::open(root.join("ddc")).unwrap();

    let result = BuildExecutor {
        protocol_version: 1,
    }
    .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0));

    assert!(matches!(
        result,
        Err(prism_asset_build_core::BuildExecutionError::Graph(
            BuildError::ProcessorFailed { job: JobId(2), .. }
        ))
    ));
    assert_eq!(store.current_revision().unwrap(), None);
    fs::remove_dir_all(root).unwrap();
}
