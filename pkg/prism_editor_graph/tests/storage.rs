//! Real files, exact-byte conflicts and explicit host adoption of async results.
extern crate alloc;
use alloc::sync::Arc;
use prism_editor_graph::*;
use std::{
    fs::{self, File, FileTimes},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    },
    task::Poll,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(10);
static SERIAL: AtomicU64 = AtomicU64::new(1);
struct Project(PathBuf);
impl Project {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-graph-files-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self) -> PathBuf {
        self.0.join("camera.graph")
    }
    fn service(&self, max_requests: usize) -> GraphStorageService {
        GraphStorageService::new(
            self.path(),
            GraphStorageLimits {
                max_requests,
                ..Default::default()
            },
        )
        .unwrap()
    }
}
impl Drop for Project {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn document(rate: u64) -> GraphDocument {
    let mut graph = Graph::new();
    let ty = PortType::named("host.scalar");
    let discarded = graph
        .add_node(NodeSpec::new("discarded", vec![]).with_port(PortSpec::output("out", ty.clone())))
        .unwrap();
    graph.remove_node(discarded).unwrap();
    let a = graph
        .add_node(
            NodeSpec::new(format!("节点-{rate}"), vec![0, 255, 3])
                .with_port(PortSpec::input("in", ty.clone()))
                .with_port(PortSpec::output("out", ty.clone())),
        )
        .unwrap();
    let b = graph
        .add_node(
            NodeSpec::new("sink", vec![128])
                .with_port(
                    PortSpec::input("in", ty.clone())
                        .with_connection_policy(ConnectionPolicy::Multiple),
                )
                .with_port(PortSpec::output("out", ty)),
        )
        .unwrap();
    let edge = graph
        .connect(
            graph.port_named(a, "out").unwrap(),
            graph.port_named(b, "in").unwrap(),
        )
        .unwrap();
    graph.disconnect(edge).unwrap();
    graph
        .connect(
            graph.port_named(a, "out").unwrap(),
            graph.port_named(b, "in").unwrap(),
        )
        .unwrap();
    let discarded_group = graph.add_group(GroupSpec::new("discarded")).unwrap();
    graph.remove_group(discarded_group).unwrap();
    let child = graph.add_group(GroupSpec::new("child")).unwrap();
    let parent = graph.add_group(GroupSpec::new("parent")).unwrap();
    graph.set_group_parent(child, Some(parent)).unwrap();
    graph.set_node_group(a, Some(child)).unwrap();
    graph.set_node_group(b, Some(parent)).unwrap();
    GraphDocument::new(
        graph,
        [
            (
                a,
                GraphPoint {
                    x: -1_000_000,
                    y: 1_000_000,
                },
            ),
            (b, GraphPoint { x: 314, y: -271 }),
        ]
        .into(),
        GraphDocumentLimits::default(),
    )
    .unwrap()
}
fn bytes(rate: u64) -> Vec<u8> {
    document(rate)
        .encode(GraphDocumentLimits::default(), 8 * 1024 * 1024)
        .unwrap()
}
fn token() -> GraphSnapshotToken {
    GraphEditor::new(GraphAuthoringLimits::default())
        .unwrap()
        .snapshot_token()
}

#[test]
fn full_document_restores_groups_ports_highwater_positions_and_evaluation() {
    let original = document(120);
    let limits = GraphDocumentLimits::default();
    let encoded = original.encode(limits, 4096).unwrap();
    let restored = GraphDocument::decode(&encoded, limits, 4096).unwrap();
    assert_eq!(restored.next_ids(), original.next_ids());
    assert_eq!(
        restored.positions().collect::<Vec<_>>(),
        original.positions().collect::<Vec<_>>()
    );
    let a = NodeId::from_raw(2);
    let b = NodeId::from_raw(3);
    let child = GroupId::from_raw(2);
    let parent = GroupId::from_raw(3);
    let graph = restored.graph();
    assert_eq!(graph.node(a).unwrap().payload(), &[0, 255, 3]);
    assert_eq!(
        graph.node(a).unwrap().ports(),
        &[PortId::from_raw(2), PortId::from_raw(3)]
    );
    assert_eq!(graph.group(child).unwrap().parent(), Some(parent));
    assert_eq!(
        graph.group(parent).unwrap().children().collect::<Vec<_>>(),
        vec![child]
    );
    assert_eq!(graph.group_nodes(parent, true).unwrap(), vec![a, b]);
    assert_eq!(graph.incoming_edges(b).len(), 1);
    assert_eq!(graph.port_named(b, "in"), Some(PortId::from_raw(4)));
    let mut visited = Vec::new();
    graph
        .evaluate(|context| {
            visited.push(context.node().id());
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(visited, vec![a, b]);
    assert_eq!(restored.encode(limits, 4096).unwrap(), encoded);
    let (next_node, next_port, next_edge, next_group) = restored.next_ids();
    let (mut graph, _) = restored.into_parts();
    let added = graph
        .add_node(
            NodeSpec::new("after-load", vec![])
                .with_port(PortSpec::input("in", PortType::named("host.scalar"))),
        )
        .unwrap();
    assert_eq!(added, next_node);
    assert_eq!(graph.port_named(added, "in"), Some(next_port));
    assert_eq!(
        graph
            .connect(graph.port_named(a, "out").unwrap(), next_port)
            .unwrap(),
        next_edge
    );
    assert_eq!(
        graph.add_group(GroupSpec::new("after-load")).unwrap(),
        next_group
    );
}

#[test]
fn adopted_grouped_document_undo_restores_membership_and_save_tokens_are_conservative() {
    let mut editor =
        GraphEditor::from_document(document(120), GraphAuthoringLimits::default()).unwrap();
    let other = GraphEditor::from_document(document(120), GraphAuthoringLimits::default()).unwrap();
    assert_ne!(editor.snapshot_token(), other.snapshot_token());
    let before = editor
        .snapshot(GraphDocumentLimits::default(), 4096)
        .unwrap();
    assert_eq!(editor.undo_depth(), 0);
    assert!(editor
        .execute(&GraphCommand::Edit(vec![GraphEdit::DeleteNode(
            NodeId::from_raw(999)
        )]))
        .is_err());
    assert_eq!(editor.snapshot_token(), before.token);
    let id = NodeId::from_raw(2);
    editor
        .execute(&GraphCommand::Edit(vec![GraphEdit::DeleteNode(id)]))
        .unwrap();
    let deleted = editor.snapshot_token();
    assert_ne!(deleted, before.token);
    editor.execute(&GraphCommand::Undo).unwrap();
    assert_ne!(editor.snapshot_token(), before.token);
    assert_ne!(editor.snapshot_token(), deleted);
    assert_eq!(
        editor
            .graph()
            .group_nodes(GroupId::from_raw(2), false)
            .unwrap(),
        vec![id]
    );
    assert_eq!(
        editor
            .snapshot(GraphDocumentLimits::default(), 4096)
            .unwrap()
            .bytes,
        before.bytes
    );
    editor.execute(&GraphCommand::Redo).unwrap();
    assert_ne!(editor.snapshot_token(), deleted);
}

#[test]
fn document_policy_positions_and_byte_boundaries_are_enforced() {
    let limits = GraphDocumentLimits::default();
    let doc = document(120);
    let encoded = doc.encode(limits, 4096).unwrap();
    assert!(doc.encode(limits, encoded.len()).is_ok());
    assert!(doc.encode(limits, encoded.len() - 1).is_err());
    assert!(GraphDocument::decode(&encoded, limits, encoded.len() - 1).is_err());
    assert!(GraphDocument::decode(
        &encoded,
        GraphDocumentLimits {
            groups: 1,
            ..limits
        },
        4096
    )
    .is_err());
    assert!(GraphDocument::decode(
        &encoded,
        GraphDocumentLimits {
            graph: GraphAuthoringLimits {
                ports_per_node: 1,
                ..limits.graph
            },
            ..limits
        },
        4096
    )
    .is_err());
    let (graph, mut positions) = document(120).into_parts();
    positions.remove(&NodeId::from_raw(2));
    assert!(GraphDocument::new(graph, positions, limits).is_err());
    let (graph, mut positions) = document(120).into_parts();
    positions.insert(NodeId::from_raw(2), GraphPoint { x: i32::MIN, y: 0 });
    assert!(GraphDocument::new(graph, positions, limits).is_err());
    let (graph, mut positions) = document(120).into_parts();
    positions.insert(NodeId::from_raw(999), GraphPoint::default());
    assert!(GraphDocument::new(graph, positions, limits).is_err());
}

#[test]
fn malformed_document_identity_versions_utf8_and_all_truncations_are_rejected() {
    let original = bytes(120);
    let limits = GraphDocumentLimits::default();
    for end in 0..original.len() {
        assert!(
            GraphDocument::decode(&original[..end], limits, 4096).is_err(),
            "prefix {end}"
        );
    }
    for (offset, replacement) in [
        (4, vec![2]),                          // unsupported version
        (6, 0u64.to_le_bytes().to_vec()),      // zero node highwater
        (14, 1u64.to_le_bytes().to_vec()),     // port highwater below live IDs
        (22, 1u64.to_le_bytes().to_vec()),     // edge highwater below live IDs
        (30, 1u64.to_le_bytes().to_vec()),     // group highwater below live IDs
        (50, 0u64.to_le_bytes().to_vec()),     // nil node identity
        (50, u64::MAX.to_le_bytes().to_vec()), // exhausted live identity
        (62, vec![255]),                       // invalid UTF8 in node name
    ] {
        let mut bad = original.clone();
        bad[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert!(
            GraphDocument::decode(&bad, limits, 4096).is_err(),
            "offset {offset}"
        );
    }
    // Mutation corpus must either reject or reconstruct a valid canonical graph.
    for offset in 0..original.len() {
        let mut changed = original.clone();
        changed[offset] ^= 0xff;
        if let Ok(doc) = GraphDocument::decode(&changed, limits, 4096) {
            doc.graph().validate().unwrap();
            assert_eq!(doc.encode(limits, 4096).unwrap(), changed);
        }
    }
}

#[test]
fn decode_revalidates_port_uniqueness_types_group_cycles_and_edge_endpoints() {
    let original = bytes(120);
    let limits = GraphDocumentLimits::default();
    let mut cursor = 50 + 8;
    fn skip_blob(bytes: &[u8], cursor: &mut usize) {
        let size = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap()) as usize;
        *cursor += 4 + size;
    }
    skip_blob(&original, &mut cursor); // node name
    skip_blob(&original, &mut cursor); // opaque payload
    cursor += 8 + 8 + 4; // coordinates, group, port count
    let first_port = cursor;
    cursor += 8;
    skip_blob(&original, &mut cursor);
    skip_blob(&original, &mut cursor);
    let direction = cursor;
    let second_port = cursor + 2;
    let edge = original.len() - 24;
    let child = edge - (8 + 4 + "child".len() + 8) - (8 + 4 + "parent".len() + 8);
    let type_offset = original
        .windows(b"host.scalar".len())
        .enumerate()
        .filter(|(_, value)| *value == b"host.scalar")
        .nth(2)
        .unwrap()
        .0;
    for (offset, replacement) in [
        (second_port, original[first_port..first_port + 8].to_vec()),
        (direction, vec![2]),
        (direction + 1, vec![2]),
        (type_offset, b"host.vector".to_vec()),
        (child + 8 + 4 + "child".len(), 2u64.to_le_bytes().to_vec()),
        (edge + 8, 2u64.to_le_bytes().to_vec()),
        (edge + 16, 999u64.to_le_bytes().to_vec()),
    ] {
        let mut bad = original.clone();
        bad[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert!(
            GraphDocument::decode(&bad, limits, 4096).is_err(),
            "offset {offset}"
        );
    }
    let mut cycle = original;
    cycle[46..50].copy_from_slice(&2u32.to_le_bytes());
    cycle[22..30].copy_from_slice(&4u64.to_le_bytes());
    cycle.extend_from_slice(&3u64.to_le_bytes());
    cycle.extend_from_slice(&5u64.to_le_bytes()); // sink output
    cycle.extend_from_slice(&2u64.to_le_bytes()); // source input
    assert!(GraphDocument::decode(&cycle, limits, 4096).is_err());
}
fn wait(request: &mut GraphStorageRequest) -> Result<GraphStorageResult, GraphStorageError> {
    let until = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(result) = request.poll() {
            return result;
        }
        assert!(
            Instant::now() < until,
            "worker did not finish request {}",
            request.id()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn saved(request: &mut GraphStorageRequest) -> GraphSaveReceipt {
    match wait(request).unwrap() {
        GraphStorageResult::Saved(receipt) => receipt,
        other => panic!("{other:?}"),
    }
}
fn loaded(request: &mut GraphStorageRequest) -> GraphLoadCandidate {
    match wait(request).unwrap() {
        GraphStorageResult::Loaded(candidate) => candidate,
        other => panic!("{other:?}"),
    }
}
fn error(request: &mut GraphStorageRequest, expected: GraphStorageError) {
    assert_eq!(wait(request).unwrap_err(), expected);
}

#[test]
fn real_worker_roundtrip_preserves_document_exact_bytes_and_snapshot_identity() {
    let project = Project::new();
    let service = project.service(2);
    let snapshot_token = token();
    let original = bytes(120);
    let mut request = service
        .save(
            original.clone(),
            snapshot_token.clone(),
            GraphFileRevision::Missing,
            TIMEOUT,
        )
        .unwrap();
    let receipt = saved(&mut request);
    assert_eq!(receipt.snapshot_token, snapshot_token);
    assert_eq!(
        receipt.revision,
        GraphFileRevision::Present(original.clone().into())
    );
    assert_eq!(receipt.durability, GraphSaveDurability::Synced);
    assert!(!receipt.deadline_exceeded);
    assert_eq!(request.cancel(), GraphCancelOutcome::TooLate);
    assert!(matches!(
        request.poll(),
        Poll::Ready(Err(GraphStorageError::Consumed))
    ));
    let mut load = service.load(TIMEOUT).unwrap();
    assert!(load.id() > request.id());
    let candidate = loaded(&mut load);
    assert_eq!(
        candidate
            .document
            .encode(GraphDocumentLimits::default(), 8 * 1024 * 1024)
            .unwrap(),
        bytes(120)
    );
    assert_eq!(candidate.revision, receipt.revision);
    assert_eq!(fs::read(project.path()).unwrap(), original);
    assert_eq!(
        fs::read_dir(&project.0).unwrap().count(),
        2,
        "only document and sidecar lock"
    );
}

#[test]
fn same_size_same_mtime_edit_is_a_conflict_and_deleted_files_are_distinct_revisions() {
    let project = Project::new();
    let service = project.service(2);
    let old = bytes(120);
    let external = bytes(240);
    assert_eq!(old.len(), external.len());
    fs::write(project.path(), &old).unwrap();
    let candidate = loaded(&mut service.load(TIMEOUT).unwrap());
    let previous_time = fs::metadata(project.path()).unwrap().modified().unwrap();
    fs::write(project.path(), &external).unwrap();
    File::options()
        .write(true)
        .open(project.path())
        .unwrap()
        .set_times(FileTimes::new().set_modified(previous_time))
        .unwrap();
    assert_eq!(
        fs::metadata(project.path()).unwrap().modified().unwrap(),
        previous_time
    );
    error(
        &mut service
            .save(bytes(480), token(), candidate.revision.clone(), TIMEOUT)
            .unwrap(),
        GraphStorageError::Conflict,
    );
    assert_eq!(fs::read(project.path()).unwrap(), external);
    fs::remove_file(project.path()).unwrap();
    error(
        &mut service
            .save(bytes(480), token(), candidate.revision, TIMEOUT)
            .unwrap(),
        GraphStorageError::Conflict,
    );
    let receipt = saved(
        &mut service
            .save(bytes(480), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    error(
        &mut service
            .save(bytes(960), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
        GraphStorageError::Conflict,
    );
    assert_eq!(
        receipt.revision,
        GraphFileRevision::Present(bytes(480).into())
    );
}

#[test]
fn independent_workers_do_not_silently_overwrite_one_anothers_revision() {
    let project = Project::new();
    let a = project.service(2);
    let b = project.service(2);
    let mut first = a
        .save(bytes(120), token(), GraphFileRevision::Missing, TIMEOUT)
        .unwrap();
    let mut second = b
        .save(bytes(240), token(), GraphFileRevision::Missing, TIMEOUT)
        .unwrap();
    let results = [wait(&mut first), wait(&mut second)];
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Ok(GraphStorageResult::Saved(_))))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(
                result,
                Err(GraphStorageError::Busy | GraphStorageError::Conflict)
            ))
            .count(),
        1
    );
    let actual = fs::read(project.path()).unwrap();
    assert!(actual == bytes(120) || actual == bytes(240));
}

#[test]
fn output_slots_and_input_capacity_remain_bounded_until_request_handles_drop() {
    let project = Project::new();
    let service = project.service(2);
    fs::write(project.path(), bytes(120)).unwrap();
    let mut first = service.load(TIMEOUT).unwrap();
    let mut second = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        service.load(TIMEOUT),
        Err(GraphStorageError::Busy)
    ));
    loaded(&mut first);
    loaded(&mut second);
    assert!(
        matches!(service.load(TIMEOUT), Err(GraphStorageError::Busy)),
        "consumed handles still reserve their slots"
    );
    drop(first);
    let mut third = service.load(TIMEOUT).unwrap();
    loaded(&mut third);
    let small = GraphStorageService::new(
        project.path(),
        GraphStorageLimits {
            max_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap();
    let mut oversized_capacity = Vec::with_capacity(257);
    oversized_capacity.extend_from_slice(b"tiny");
    assert!(matches!(
        small.save(
            oversized_capacity,
            token(),
            GraphFileRevision::Missing,
            TIMEOUT
        ),
        Err(GraphStorageError::ByteLimit)
    ));
    assert!(matches!(
        small.save(
            vec![],
            token(),
            GraphFileRevision::Present(vec![0; 257].into()),
            TIMEOUT
        ),
        Err(GraphStorageError::ByteLimit)
    ));
    File::create(project.path()).unwrap().set_len(257).unwrap();
    error(
        &mut small.load(TIMEOUT).unwrap(),
        GraphStorageError::ByteLimit,
    );
}

#[test]
fn malformed_and_policy_rejected_loads_and_saves_preserve_the_file() {
    let project = Project::new();
    let service = project.service(2);
    let original = bytes(120);
    fs::write(project.path(), &original).unwrap();
    for bad in [
        vec![],
        b"wrong format".to_vec(),
        original[..original.len() - 1].to_vec(),
        {
            let mut bad = original.clone();
            bad.push(0);
            bad
        },
        {
            let mut bad = original.clone();
            bad[38..42].copy_from_slice(&u32::MAX.to_le_bytes());
            bad
        },
    ] {
        let mut save = service
            .save(
                bad.clone(),
                token(),
                GraphFileRevision::Present(original.clone().into()),
                TIMEOUT,
            )
            .unwrap();
        assert!(matches!(wait(&mut save), Err(GraphStorageError::Codec(_))));
        assert_eq!(fs::read(project.path()).unwrap(), original);
        fs::write(project.path(), &bad).unwrap();
        assert!(matches!(
            wait(&mut service.load(TIMEOUT).unwrap()),
            Err(GraphStorageError::Codec(_))
        ));
        assert_eq!(fs::read(project.path()).unwrap(), bad);
        fs::write(project.path(), &original).unwrap();
    }
    let restrictive = GraphStorageService::new(
        project.path(),
        GraphStorageLimits {
            document: GraphDocumentLimits {
                groups: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        wait(&mut restrictive.load(TIMEOUT).unwrap()),
        Err(GraphStorageError::Codec(GraphAuthoringError::Limit))
    ));
}

#[test]
fn cancellation_and_zero_timeout_fence_results_without_replacing_bytes() {
    let project = Project::new();
    let service = project.service(2);
    let original = bytes(120);
    fs::write(project.path(), &original).unwrap();
    let mut cancelled = service.load(TIMEOUT).unwrap();
    assert_eq!(cancelled.cancel(), GraphCancelOutcome::Cancelled);
    error(&mut cancelled, GraphStorageError::Cancelled);
    drop(cancelled);
    let mut expired = service
        .save(
            bytes(240),
            token(),
            GraphFileRevision::Present(original.clone().into()),
            Duration::ZERO,
        )
        .unwrap();
    error(&mut expired, GraphStorageError::TimedOut);
    assert!(matches!(
        expired.poll(),
        Poll::Ready(Err(GraphStorageError::Consumed))
    ));
    assert_eq!(fs::read(project.path()).unwrap(), original);
}

#[test]
fn old_load_and_save_results_require_host_generation_and_snapshot_checks() {
    let project = Project::new();
    let service = project.service(2);
    fs::write(project.path(), bytes(120)).unwrap();
    let mut old_load = service.load(TIMEOUT).unwrap();
    let old = loaded(&mut old_load);
    let mut editor =
        GraphEditor::from_document(old.document, GraphAuthoringLimits::default()).unwrap();
    let snapshot = editor.snapshot_token();
    let mut save = service
        .save(bytes(120), snapshot.clone(), old.revision, TIMEOUT)
        .unwrap();
    drop(old_load);
    editor
        .execute(&GraphCommand::Edit(vec![GraphEdit::DeleteNode(
            NodeId::from_raw(2),
        )]))
        .unwrap();
    let receipt = saved(&mut save);
    assert_eq!(receipt.snapshot_token, snapshot);
    assert_ne!(
        receipt.snapshot_token,
        editor.snapshot_token(),
        "host must not confirm this old save as current"
    );
    assert_eq!(
        editor.graph().node_count(),
        1,
        "worker did not replace live editor"
    );
    let mut new_load = service.load(TIMEOUT).unwrap();
    assert!(new_load.id() > save.id());
    let newer = loaded(&mut new_load);
    assert_eq!(newer.document.graph().node_count(), 2);
    assert_eq!(
        editor.graph().node_count(),
        1,
        "load candidate is not auto-adopted"
    );
}

#[test]
fn uncoordinated_readers_observe_only_complete_atomic_generations() {
    let project = Project::new();
    let service = project.service(1);
    let old = bytes(120);
    let new = bytes(240);
    fs::write(project.path(), &old).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let reader_barrier = barrier.clone();
    let path = project.path();
    let reader_old = old.clone();
    let reader_new = new.clone();
    let reader = std::thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..4096 {
            let actual = fs::read(&path).unwrap();
            assert!(
                actual == reader_old || actual == reader_new,
                "partial generation"
            );
        }
    });
    barrier.wait();
    let mut revision = GraphFileRevision::Present(old.clone().into());
    for index in 0..16 {
        let next = if index % 2 == 0 {
            new.clone()
        } else {
            old.clone()
        };
        revision = saved(&mut service.save(next, token(), revision, TIMEOUT).unwrap()).revision;
    }
    reader.join().unwrap();
    assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
}

#[test]
fn sidecar_lock_errors_missing_parent_and_missing_file_are_explicit() {
    let project = Project::new();
    let service = project.service(2);
    error(
        &mut service.load(TIMEOUT).unwrap(),
        GraphStorageError::Missing,
    );
    let lock = File::options()
        .read(true)
        .write(true)
        .open(project.0.join("camera.graph.prism-graph.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    error(
        &mut service
            .save(bytes(120), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
        GraphStorageError::Busy,
    );
    assert!(!project.path().exists());
    drop(lock);
    saved(
        &mut service
            .save(bytes(120), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    let missing = GraphStorageService::new(
        project.0.join("not-created/file.graph"),
        GraphStorageLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        wait(&mut missing.load(TIMEOUT).unwrap()),
        Err(GraphStorageError::Io {
            operation: "resolve parent",
            kind: std::io::ErrorKind::NotFound
        })
    ));
    assert!(matches!(
        GraphStorageService::new("relative.graph", GraphStorageLimits::default()),
        Err(GraphStorageError::InvalidConfiguration)
    ));
}

#[test]
fn independent_process_loads_only_real_saved_document_bytes() {
    const CHILD: &str = "PRISM_GRAPH_STORAGE_CHILD_FILE";
    if let Some(path) = std::env::var_os(CHILD) {
        let service =
            GraphStorageService::new(PathBuf::from(path), GraphStorageLimits::default()).unwrap();
        let candidate = loaded(&mut service.load(TIMEOUT).unwrap());
        assert_eq!(
            candidate
                .document
                .encode(GraphDocumentLimits::default(), 8 * 1024 * 1024)
                .unwrap(),
            bytes(120)
        );
        saved(
            &mut service
                .save(bytes(240), token(), candidate.revision, TIMEOUT)
                .unwrap(),
        );
        return;
    }
    let project = Project::new();
    saved(
        &mut project
            .service(1)
            .save(bytes(120), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("independent_process_loads_only_real_saved_document_bytes")
        .env(CHILD, project.path())
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
}

#[cfg(unix)]
#[test]
fn target_and_lock_symlinks_or_directories_are_rejected() {
    use std::os::unix::fs::symlink;
    let project = Project::new();
    let outside = Project::new();
    fs::write(outside.path(), bytes(120)).unwrap();
    symlink(outside.path(), project.path()).unwrap();
    let service = project.service(1);
    error(
        &mut service.load(TIMEOUT).unwrap(),
        GraphStorageError::UnsafePath,
    );
    fs::remove_file(project.path()).unwrap();
    fs::create_dir(project.path()).unwrap();
    error(
        &mut service
            .save(bytes(240), token(), GraphFileRevision::Missing, TIMEOUT)
            .unwrap(),
        GraphStorageError::UnsafePath,
    );
    fs::remove_dir(project.path()).unwrap();
    fs::remove_file(project.0.join("camera.graph.prism-graph.lock")).unwrap();
    symlink(
        outside.path(),
        project.0.join("camera.graph.prism-graph.lock"),
    )
    .unwrap();
    error(
        &mut service.load(TIMEOUT).unwrap(),
        GraphStorageError::UnsafePath,
    );
    assert_eq!(fs::read(outside.path()).unwrap(), bytes(120));
}
