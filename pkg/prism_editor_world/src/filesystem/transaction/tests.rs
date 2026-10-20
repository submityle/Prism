use super::super::tests::{run, Temp};
use super::*;
use crate::{
    CellCoord, EditorWorld, EditorWorldData, EditorWorldLimits, ResidencyDemand,
    WorldTransactionEdit,
};
use std::{
    process::{Child, Command},
    sync::mpsc,
    thread,
    time::Instant,
};

const TIMEOUT: Duration = Duration::from_secs(10);
fn limits() -> WorldTransactionLimits {
    WorldTransactionLimits::new(16, 1024, 4096, 4, 16384).unwrap()
}
fn provider(root: &Path) -> FileWorldTransactionProvider {
    FileWorldTransactionProvider::new(root, limits()).unwrap()
}
fn idle(p: &FileWorldTransactionProvider) {
    let until = Instant::now() + TIMEOUT;
    while p.outstanding_operations() != 0 || p.reserved_bytes() != 0 {
        assert!(Instant::now() < until, "retained worker");
        thread::yield_now();
    }
}
fn cells() -> [WorldFileTarget; 2] {
    let mut w = EditorWorld::new(StableGuid::from_path("world/transaction")).unwrap();
    [
        WorldFileTarget::Cell(
            w.add_cell(CellCoord::new(-1, 0, 0), [])
                .unwrap()
                .file()
                .clone(),
        ),
        WorldFileTarget::Cell(
            w.add_cell(CellCoord::new(2, 0, 0), [])
                .unwrap()
                .file()
                .clone(),
        ),
    ]
}
fn actor(cell: CellCoord) -> WorldFileTarget {
    WorldFileTarget::Actor(
        ActorFileIdentity::new(StableGuid::from_path("actor/moved"), cell).unwrap(),
    )
}
fn put(t: WorldFileTarget, base: ActorRevisionBase, r: u64, b: &[u8]) -> WorldFileChange {
    WorldFileChange::put(t, base, r, b.into()).unwrap()
}
fn request(g: u64) -> WorldTransactionRequest {
    let [a, b] = cells();
    let base = if g == 0 {
        ActorRevisionBase::Missing
    } else {
        ActorRevisionBase::Revision(g)
    };
    let mut changes = vec![
        put(
            a.clone(),
            base,
            g + 1,
            if g == 0 {
                b"cell-a"
            } else {
                b"cell-a/moved-out"
            },
        ),
        put(
            b.clone(),
            base,
            g + 1,
            if g == 0 {
                b"cell-b"
            } else {
                b"cell-b/moved-in"
            },
        ),
    ];
    if g == 0 {
        changes.push(put(actor(a.cell()), base, 1, b"actor"));
    }
    if g == 1 {
        changes.push(WorldFileChange::delete(actor(a.cell()), 1).unwrap());
        changes.push(put(
            actor(b.cell()),
            ActorRevisionBase::Missing,
            2,
            b"moved",
        ));
    }
    WorldTransactionRequest::new(StableGuid::from_u128(100 + g as u128), g, changes).unwrap()
}
fn commit(
    p: &FileWorldTransactionProvider,
    r: WorldTransactionRequest,
) -> Result<WorldTransactionReceipt, FileCellIoError> {
    let result = run(p.commit(r, TIMEOUT).unwrap());
    idle(p);
    result
}
fn read(p: &FileWorldTransactionProvider) -> WorldTransactionSnapshot {
    let result = run(p.load_snapshot(TIMEOUT).unwrap()).unwrap();
    idle(p);
    result
}
fn hook(
    p: &FileWorldTransactionProvider,
    phase: TestPhase,
) -> (mpsc::Receiver<()>, mpsc::SyncSender<()>) {
    let (tx, rx) = mpsc::sync_channel(1);
    let (release, wait) = mpsc::sync_channel(1);
    let wait = Mutex::new(wait);
    *lock(&p.io.cell.shared.hook) = Some(Arc::new(move |at| {
        if at == phase {
            tx.send(()).unwrap();
            lock(&wait).recv_timeout(TIMEOUT).unwrap();
        }
        Ok(())
    }));
    (rx, release)
}

#[test]
fn snapshot_move_cas_idempotency_and_slot_reuse() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    assert_eq!(read(&p).generation(), 0);
    commit(&p, request(0)).unwrap();
    let before = read(&p);
    let moved = commit(&provider(&temp.0), request(1)).unwrap();
    assert_eq!(moved.generation(), 2);
    let after = read(&p);
    assert_eq!(after.files().len(), 3);
    assert!(after
        .files()
        .iter()
        .any(|f| f.target() == &actor(cells()[1].cell()) && f.bytes() == b"moved"));
    assert!(!after
        .files()
        .iter()
        .any(|f| f.target() == &actor(cells()[0].cell())));
    assert_eq!(
        before
            .files()
            .iter()
            .find(|f| matches!(f.target(), WorldFileTarget::Actor(_)))
            .unwrap()
            .bytes(),
        b"actor"
    );
    commit(&p, request(1)).unwrap(); // exact retry recovers a real receipt
    let mut divergent = request(1);
    divergent.changes[0].value.as_mut().unwrap().1 = Arc::from(b"divergent".as_slice());
    assert!(matches!(
        commit(&p, divergent),
        Err(FileCellIoError::RevisionConflict { .. })
    ));
    assert!(matches!(
        commit(&p, request(0)),
        Err(FileCellIoError::RevisionConflict { .. })
    ));
    for g in 2..6 {
        commit(&p, request(g)).unwrap();
    }
    assert_eq!(before.generation(), 1);
    assert_eq!(before.files().len(), 3);
    assert_eq!(read(&p).generation(), 6);
    for slot in [0, 1] {
        assert!(
            fs::read_dir(temp.0.join(STORE).join(format!("slot{slot}")))
                .unwrap()
                .count()
                <= 4
        );
    }
}

#[test]
fn target_cas_and_global_actor_ownership_reject_whole_batch() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let mut wrong = request(1);
    wrong.changes[0].base = ActorRevisionBase::Revision(99);
    assert!(matches!(
        commit(&p, wrong),
        Err(FileCellIoError::RevisionConflict {
            current: Some(1),
            ..
        })
    ));
    assert_eq!(read(&p).generation(), 1);
    let duplicate = WorldTransactionRequest::new(
        StableGuid::from_u128(900),
        1,
        vec![put(
            actor(cells()[1].cell()),
            ActorRevisionBase::Missing,
            2,
            b"duplicate",
        )],
    )
    .unwrap();
    assert!(matches!(
        commit(&p, duplicate),
        Err(FileCellIoError::InvalidRequest)
    ));
    let orphan = WorldTransactionRequest::new(
        StableGuid::from_u128(901),
        1,
        vec![WorldFileChange::delete(cells()[0].clone(), 1).unwrap()],
    )
    .unwrap();
    assert!(matches!(
        commit(&p, orphan),
        Err(FileCellIoError::InvalidRequest)
    ));
    assert_eq!(read(&p).generation(), 1);
}

#[test]
fn readers_and_competing_providers_cannot_observe_prepared_mix() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let other = provider(&temp.0);
    let (ready, release) = hook(&p, TestPhase::TransactionPrepared);
    let future = p.commit(request(1), TIMEOUT).unwrap();
    ready.recv_timeout(TIMEOUT).unwrap();
    assert!(matches!(
        run(other.load_snapshot(TIMEOUT).unwrap()),
        Err(FileCellIoError::Busy)
    ));
    assert!(matches!(
        commit(&other, request(1)),
        Err(FileCellIoError::Busy)
    ));
    release.send(()).unwrap();
    run(future).unwrap();
    idle(&p);
    assert_eq!(read(&other).generation(), 2);
}

#[test]
fn sorted_stripes_and_legacy_root_isolation() {
    let temp = Temp::new();
    let legacy =
        FileActorIoProvider::new(&temp.0, ActorIoLimits::new(1024, 2, 2048).unwrap()).unwrap();
    let p = provider(&temp.0);
    let store = Directory::child(&temp.0, STORE, false).unwrap();
    let locks = Directory::child(&store.path, "locks", true).unwrap();
    let mut key = Vec::new();
    format::target(&mut key, &request(0).changes[0].target);
    let stripe = format::digest(&key) % 64;
    let guard = locks
        .acquire_path(&locks.path.join(stripe.to_string()), true)
        .unwrap();
    assert!(matches!(commit(&p, request(0)), Err(FileCellIoError::Busy)));
    drop(guard);
    let WorldFileTarget::Actor(a) = actor(cells()[0].cell()) else {
        unreachable!()
    };
    assert!(matches!(
        run(legacy.load_actor(a, 100, TIMEOUT).unwrap()),
        Err(FileCellIoError::TransactionalRoot)
    ));
    commit(&p, request(0)).unwrap();
    let temp2 = Temp::new();
    fs::create_dir(temp2.0.join("cells")).unwrap();
    assert!(matches!(
        FileWorldTransactionProvider::new(&temp2.0, limits()),
        Err(FileCellIoError::InvalidConfiguration)
    ));
}

#[test]
fn corruption_and_failed_staging_never_publish() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    *lock(&p.io.cell.shared.hook) = Some(Arc::new(|phase| {
        if phase == TestPhase::TransactionPrepared {
            Err(io::Error::other("injected stage failure"))
        } else {
            Ok(())
        }
    }));
    assert!(matches!(
        commit(&p, request(1)),
        Err(FileCellIoError::Io { .. })
    ));
    assert_eq!(read(&p).generation(), 1);
    *lock(&p.io.cell.shared.hook) = None;
    fs::write(temp.0.join(STORE).join("slot1/0"), b"corrupt").unwrap();
    assert!(run(p.load_snapshot(TIMEOUT).unwrap()).is_err());
    idle(&p);
    assert!(commit(&p, request(1)).is_err());
}

#[test]
fn decided_journal_replays_and_corrupt_candidate_fails_closed() {
    for phase in [
        TestPhase::TransactionDecided,
        TestPhase::TransactionPublished,
    ] {
        let temp = Temp::new();
        let p = provider(&temp.0);
        commit(&p, request(0)).unwrap();
        *lock(&p.io.cell.shared.hook) = Some(Arc::new(move |at| {
            if at == phase {
                Err(io::Error::other("injected publication failure"))
            } else {
                Ok(())
            }
        }));
        assert!(matches!(
            commit(&p, request(1)),
            Err(FileCellIoError::CommitUncertain(_))
        ));
        assert!(temp.0.join(STORE).join("journal").exists());
        let reopened = provider(&temp.0);
        assert_eq!(read(&reopened).generation(), 2);
        commit(&reopened, request(1)).unwrap();
        assert!(!temp.0.join(STORE).join("journal").exists());
    }
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    *lock(&p.io.cell.shared.hook) = Some(Arc::new(|at| {
        if at == TestPhase::TransactionDecided {
            Err(io::Error::other("decided"))
        } else {
            Ok(())
        }
    }));
    assert!(commit(&p, request(1)).is_err());
    fs::write(temp.0.join(STORE).join("slot0/1"), b"broken").unwrap();
    assert!(run(provider(&temp.0).load_snapshot(TIMEOUT).unwrap()).is_err());
    assert!(temp.0.join(STORE).join("journal").exists());
}

#[test]
fn cancellation_timeout_and_admission_before_decision() {
    for cancel in [true, false] {
        let temp = Temp::new();
        let p = provider(&temp.0);
        commit(&p, request(0)).unwrap();
        let (ready, release) = hook(&p, TestPhase::TransactionPrepared);
        let timeout = if cancel {
            TIMEOUT
        } else {
            Duration::from_secs(2)
        };
        let start = Instant::now();
        let future = p.commit(request(1), timeout).unwrap();
        ready.recv_timeout(TIMEOUT).unwrap();
        assert!(p.reserved_bytes() >= 4096);
        if cancel {
            drop(future);
            release.send(()).unwrap();
        } else {
            while start.elapsed() <= timeout {
                thread::sleep(Duration::from_millis(2));
            }
            release.send(()).unwrap();
            assert!(matches!(run(future), Err(FileCellIoError::TimedOut)));
        }
        idle(&p);
        assert_eq!(read(&p).generation(), 1);
    }
    let temp = Temp::new();
    let p = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(16, 1024, 4096, 1, 8192).unwrap(),
    )
    .unwrap();
    let (ready, release) = hook(&p, TestPhase::TransactionPrepared);
    let future = p.commit(request(0), TIMEOUT).unwrap();
    ready.recv_timeout(TIMEOUT).unwrap();
    assert!(matches!(
        p.load_snapshot(TIMEOUT),
        Err(FileCellIoError::Capacity)
    ));
    release.send(()).unwrap();
    run(future).unwrap();
    idle(&p);
    assert!(p.commit(request(1), Duration::ZERO).is_err());
    let oversized = WorldTransactionRequest::new(
        StableGuid::from_u128(70),
        1,
        vec![put(
            cells()[0].clone(),
            ActorRevisionBase::Revision(1),
            2,
            &[0; 1025],
        )],
    )
    .unwrap();
    assert!(matches!(
        p.commit(oversized, TIMEOUT),
        Err(FileCellIoError::TooLarge { .. })
    ));
}

#[test]
fn timeout_after_decision_completes_publication() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let (ready, release) = hook(&p, TestPhase::TransactionDecided);
    let start = Instant::now();
    let future = p.commit(request(1), Duration::from_secs(2)).unwrap();
    ready.recv_timeout(TIMEOUT).unwrap();
    while start.elapsed() <= Duration::from_secs(2) {
        thread::sleep(Duration::from_millis(2));
    }
    release.send(()).unwrap();
    run(future).unwrap();
    idle(&p);
    assert_eq!(read(&p).generation(), 2);
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn transaction_process_child() {
    let Some(root) = std::env::var_os("PRISM_WORLD_TX_CHILD") else {
        return;
    };
    let phase = std::env::var("PRISM_WORLD_TX_PHASE").unwrap();
    let p = provider(Path::new(&root));
    *lock(&p.io.cell.shared.hook) = Some(Arc::new(move |at| {
        let stop = match phase.as_str() {
            "prepared" => TestPhase::TransactionPrepared,
            "decided" => TestPhase::TransactionDecided,
            _ => TestPhase::TransactionPublished,
        };
        if at == stop {
            fs::write(Path::new(&root).join("ready"), b"ready").unwrap();
            loop {
                thread::park_timeout(Duration::from_secs(1));
            }
        }
        Ok(())
    }));
    commit(&p, request(1)).unwrap();
    panic!("child should be killed at checkpoint");
}

#[test]
fn killed_process_recovery_at_each_publication_boundary() {
    for (phase, generation) in [("prepared", 1), ("decided", 2), ("published", 2)] {
        let temp = Temp::new();
        let p = provider(&temp.0);
        commit(&p, request(0)).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "filesystem::transaction::tests::transaction_process_child",
                "--nocapture",
            ])
            .env("PRISM_WORLD_TX_CHILD", &temp.0)
            .env("PRISM_WORLD_TX_PHASE", phase)
            .spawn()
            .unwrap();
        let mut child = ChildGuard(child);
        let until = Instant::now() + TIMEOUT;
        while !temp.0.join("ready").exists() {
            assert!(Instant::now() < until, "child failed to prepare");
            assert!(child.0.try_wait().unwrap().is_none());
            thread::sleep(Duration::from_millis(5));
        }
        assert!(matches!(
            run(p.load_snapshot(TIMEOUT).unwrap()),
            Err(FileCellIoError::Busy)
        ));
        idle(&p);
        child.0.kill().unwrap();
        assert!(!child.0.wait().unwrap().success());
        let recovered = read(&provider(&temp.0));
        assert_eq!(recovered.generation(), generation);
        let expected = request(generation - 1);
        let files: Vec<_> = expected
            .changes
            .iter()
            .filter_map(|c| {
                c.value.as_ref().map(|(revision, bytes)| WorldFileSnapshot {
                    target: c.target.clone(),
                    revision: *revision,
                    bytes: bytes.clone(),
                })
            })
            .collect();
        assert_eq!(recovered.transaction_id(), expected.id());
        assert_eq!(recovered.files(), files, "mixed generation after {phase}");
        assert!(!temp.0.join(STORE).join("journal").exists());
        commit(&p, request(1)).unwrap();
        assert_eq!(read(&p).generation(), 2);
    }
}

fn resident() -> (EditorWorldData, Vec<crate::CellLockToken>) {
    let mut w = EditorWorldData::new(
        StableGuid::from_path("world/transaction"),
        EditorWorldLimits::new(4, 1024, 16, 1, 1).unwrap(),
    )
    .unwrap();
    let mut demand = ResidencyDemand::new();
    for t in cells() {
        w.add_cell(t.cell(), []).unwrap();
        demand.require(t.cell());
    }
    let (_, tokens) = w.begin_residency(&demand, 0).unwrap();
    for t in tokens {
        w.complete_load(t, b"local".to_vec()).unwrap();
    }
    let leases = cells()
        .iter()
        .map(|c| {
            w.acquire_lock(c.cell(), StableGuid::from_path("owner"), 0, 100)
                .unwrap()
        })
        .collect();
    w.register_actor(StableGuid::from_path("actor/moved"), cells()[0].cell())
        .unwrap();
    (w, leases)
}
fn edits() -> Vec<WorldTransactionEdit> {
    vec![
        WorldTransactionEdit::Cell {
            cell: cells()[0].cell(),
            payload: Arc::from(b"source sans actor".as_slice()),
        },
        WorldTransactionEdit::Cell {
            cell: cells()[1].cell(),
            payload: Arc::from(b"destination with actor".as_slice()),
        },
        WorldTransactionEdit::Actor {
            actor: StableGuid::from_path("actor/moved"),
            cell: cells()[1].cell(),
            payload: Arc::from(b"new actor".as_slice()),
        },
    ]
}

#[test]
fn world_move_receipt_and_explicit_restart_restore() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let (w, leases) = resident();
    let mut w = w;
    w.discard_edits_and_restore_transaction_snapshot(&read(&p), &leases, 1)
        .unwrap();
    let pending = w
        .prepare_world_transaction(
            StableGuid::from_u128(800),
            &read(&p),
            &edits(),
            &leases,
            2,
            &p,
        )
        .unwrap();
    assert!(cells()
        .iter()
        .all(|c| w.model().cell(c.cell()).unwrap().is_dirty()));
    let receipt = commit(&p, pending.request().clone()).unwrap();
    w.commit_world_transaction(&pending, &receipt, 3).unwrap();
    assert!(cells()
        .iter()
        .all(|c| !w.model().cell(c.cell()).unwrap().is_dirty()));
    assert_eq!(
        w.actor_record(StableGuid::from_path("actor/moved"))
            .unwrap()
            .identity()
            .cell(),
        cells()[1].cell()
    );
    assert_eq!(
        w.actor_payload(StableGuid::from_path("actor/moved")),
        Some(b"new actor".as_slice())
    );
    let (mut restarted, leases) = resident();
    restarted
        .discard_edits_and_restore_transaction_snapshot(&read(&p), &leases, 4)
        .unwrap();
    assert_eq!(
        restarted.actor_payload(StableGuid::from_path("actor/moved")),
        Some(b"new actor".as_slice())
    );
}

#[test]
fn stale_edit_and_expired_lease_receipts_keep_both_cells_dirty() {
    for stale in [true, false] {
        let temp = Temp::new();
        let p = provider(&temp.0);
        commit(&p, request(0)).unwrap();
        let (mut w, leases) = resident();
        w.discard_edits_and_restore_transaction_snapshot(&read(&p), &leases, 1)
            .unwrap();
        let pending = w
            .prepare_world_transaction(
                StableGuid::from_u128(801),
                &read(&p),
                &edits(),
                &leases,
                2,
                &p,
            )
            .unwrap();
        let receipt = commit(&p, pending.request().clone()).unwrap();
        if stale {
            w.replace_payload(
                cells()[0].cell(),
                &leases[0],
                3,
                b"intervening edit".to_vec(),
            )
            .unwrap();
        }
        assert!(w
            .commit_world_transaction(&pending, &receipt, if stale { 4 } else { 101 })
            .is_err());
        assert!(cells()
            .iter()
            .all(|c| w.model().cell(c.cell()).unwrap().is_dirty()));
        assert_eq!(
            w.actor_record(StableGuid::from_path("actor/moved"))
                .unwrap()
                .identity()
                .cell(),
            cells()[0].cell()
        );
    }
}

#[test]
fn new_world_create_delete_and_wrong_root_receipt() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    let (mut w, leases) = resident();
    let mut create = edits();
    if let WorldTransactionEdit::Actor { cell, .. } = &mut create[2] {
        *cell = cells()[0].cell();
    }
    let pending = w
        .prepare_world_transaction(
            StableGuid::from_u128(820),
            &read(&p),
            &create,
            &leases,
            1,
            &p,
        )
        .unwrap();
    let other = Temp::new();
    let other = provider(&other.0);
    let wrong = commit(&other, pending.request().clone()).unwrap();
    assert!(w.commit_world_transaction(&pending, &wrong, 2).is_err());
    let receipt = commit(&p, pending.request().clone()).unwrap();
    w.commit_world_transaction(&pending, &receipt, 2).unwrap();
    assert_eq!(
        w.actor_record(StableGuid::from_path("actor/moved"))
            .unwrap()
            .persisted_revision(),
        Some(1)
    );
    let old = read(&p);
    let delete = vec![
        create[0].clone(),
        WorldTransactionEdit::DeleteActor(StableGuid::from_path("actor/moved")),
    ];
    let pending = w
        .prepare_world_transaction(StableGuid::from_u128(821), &old, &delete, &leases, 3, &p)
        .unwrap();
    let receipt = commit(&p, pending.request().clone()).unwrap();
    w.commit_world_transaction(&pending, &receipt, 4).unwrap();
    assert!(w
        .actor_record(StableGuid::from_path("actor/moved"))
        .is_none());
    assert_eq!(read(&p).files().len(), 2);
    assert!(w
        .discard_edits_and_restore_transaction_snapshot(&old, &leases, 5)
        .is_err());
    assert!(w
        .discard_edits_and_restore_transaction_snapshot(&read(&other), &leases, 5)
        .is_err());
}

#[test]
fn uncommitted_actor_omission_and_missing_destination_are_rejected_before_dirty() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    let (mut w, leases) = resident();
    let only_cells = edits()[..2].to_vec();
    assert!(w
        .prepare_world_transaction(
            StableGuid::from_u128(830),
            &read(&p),
            &only_cells,
            &leases,
            1,
            &p
        )
        .is_err());
    assert!(cells()
        .iter()
        .all(|c| !w.model().cell(c.cell()).unwrap().is_dirty()));
    commit(&p, request(0)).unwrap();
    w.discard_edits_and_restore_transaction_snapshot(&read(&p), &leases, 1)
        .unwrap();
    let partial = vec![edits()[0].clone(), edits()[2].clone()];
    assert!(w
        .prepare_world_transaction(
            StableGuid::from_u128(831),
            &read(&p),
            &partial,
            &leases,
            2,
            &p
        )
        .is_err());
    assert!(cells()
        .iter()
        .all(|c| !w.model().cell(c.cell()).unwrap().is_dirty()));
}

#[test]
fn actor_move_revisions_advance_in_either_coordinate_direction() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    commit(&p, request(1)).unwrap();
    let back = |revision| {
        WorldTransactionRequest::new(
            StableGuid::from_u128(840 + revision as u128),
            2,
            vec![
                put(
                    cells()[0].clone(),
                    ActorRevisionBase::Revision(2),
                    3,
                    b"back member",
                ),
                put(
                    cells()[1].clone(),
                    ActorRevisionBase::Revision(2),
                    3,
                    b"removed member",
                ),
                WorldFileChange::delete(actor(cells()[1].cell()), 2).unwrap(),
                put(
                    actor(cells()[0].cell()),
                    ActorRevisionBase::Missing,
                    revision,
                    b"back",
                ),
            ],
        )
        .unwrap()
    };
    assert!(matches!(
        commit(&p, back(1)),
        Err(FileCellIoError::RevisionConflict { .. })
    ));
    commit(&p, back(3)).unwrap();
    assert!(read(&p)
        .files()
        .iter()
        .any(|f| f.target() == &actor(cells()[0].cell()) && f.revision() == 3));
}

#[test]
fn bounds_and_metadata_decoder_reject_before_unbounded_allocations() {
    assert!(WorldTransactionLimits::new(0, 1, 1, 1, 2).is_err());
    assert!(WorldTransactionLimits::new(MAX_ENTRIES + 1, 1, 1, 1, 2).is_err());
    assert!(WorldTransactionRequest::new(StableGuid::from_u128(0), 0, vec![]).is_err());
    assert!(WorldTransactionRequest::new(
        StableGuid::from_u128(1),
        u64::MAX,
        vec![put(cells()[0].clone(), ActorRevisionBase::Missing, 0, b"")]
    )
    .is_err());
    let duplicated = put(cells()[0].clone(), ActorRevisionBase::Missing, 1, b"");
    assert!(WorldTransactionRequest::new(
        StableGuid::from_u128(1),
        0,
        vec![duplicated.clone(), duplicated]
    )
    .is_err());
    let mut bad_identity = match cells()[0].clone() {
        WorldFileTarget::Cell(c) => c,
        _ => unreachable!(),
    };
    bad_identity.stable_id = StableGuid::from_u128(0);
    assert!(WorldFileChange::put(
        WorldFileTarget::Cell(bad_identity),
        ActorRevisionBase::Missing,
        1,
        Arc::from([])
    )
    .is_err());
    assert!(format::decode_manifest(&u32::MAX.to_le_bytes(), limits()).is_err());
    assert!(format::decode_manifest(&[0; 3], limits()).is_err());
    let mut bytes = 1u32.to_le_bytes().to_vec();
    format::target(&mut bytes, &cells()[0]);
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(u64::MAX.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    assert!(matches!(
        format::decode_manifest(&bytes, limits()),
        Err(FileCellIoError::TooLarge { .. })
    ));
    let temp = Temp::new();
    let p = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(2, 1024, 4096, 1, 16384).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        commit(&p, request(0)),
        Err(FileCellIoError::Capacity)
    ));
    assert_eq!(read(&p).generation(), 0);
}

#[test]
fn full_snapshot_bytes_and_request_count_have_independent_bounds() {
    let temp = Temp::new();
    let p = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(16, 1024, 1024, 4, 8192).unwrap(),
    )
    .unwrap();
    let r = WorldTransactionRequest::new(
        StableGuid::from_u128(860),
        0,
        vec![put(
            cells()[0].clone(),
            ActorRevisionBase::Missing,
            1,
            &[0; 600],
        )],
    )
    .unwrap();
    commit(&p, r).unwrap();
    let r = WorldTransactionRequest::new(
        StableGuid::from_u128(861),
        1,
        vec![put(
            cells()[1].clone(),
            ActorRevisionBase::Missing,
            1,
            &[0; 600],
        )],
    )
    .unwrap();
    assert!(matches!(commit(&p, r), Err(FileCellIoError::Capacity)));
    assert_eq!(read(&p).generation(), 1);
    let temp = Temp::new();
    let p = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(16, 1024, 4096, 1, 16384).unwrap(),
    )
    .unwrap();
    let (ready, release) = hook(&p, TestPhase::TransactionPrepared);
    let future = p.commit(request(0), TIMEOUT).unwrap();
    ready.recv_timeout(TIMEOUT).unwrap();
    let bytes = p.reserved_bytes();
    assert!(matches!(
        p.load_snapshot(TIMEOUT),
        Err(FileCellIoError::Capacity)
    ));
    assert_eq!(p.reserved_bytes(), bytes);
    release.send(()).unwrap();
    run(future).unwrap();
    idle(&p);
}

#[test]
fn cancellation_drop_waits_for_decided_publication_fence() {
    let temp = Temp::new();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let (ready, release) = hook(&p, TestPhase::TransactionDecided);
    let future = p.commit(request(1), TIMEOUT).unwrap();
    ready.recv_timeout(TIMEOUT).unwrap();
    let (start, started) = mpsc::sync_channel(1);
    let (done, completed) = mpsc::sync_channel(1);
    let join = thread::spawn(move || {
        start.send(()).unwrap();
        drop(future);
        done.send(()).unwrap();
    });
    started.recv_timeout(TIMEOUT).unwrap();
    assert!(completed.recv_timeout(Duration::from_millis(20)).is_err());
    release.send(()).unwrap();
    completed.recv_timeout(TIMEOUT).unwrap();
    join.join().unwrap();
    idle(&p);
    assert_eq!(read(&p).generation(), 2);
}

#[test]
fn corrupt_journal_and_manifest_refuse_replay() {
    for path in ["journal", "slot0/manifest"] {
        let temp = Temp::new();
        let p = provider(&temp.0);
        commit(&p, request(0)).unwrap();
        *lock(&p.io.cell.shared.hook) = Some(Arc::new(|at| {
            if at == TestPhase::TransactionDecided {
                Err(io::Error::other("decided"))
            } else {
                Ok(())
            }
        }));
        assert!(commit(&p, request(1)).is_err());
        let file = temp.0.join(STORE).join(path);
        let mut bytes = fs::read(&file).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(file, bytes).unwrap();
        assert!(matches!(
            run(p.load_snapshot(TIMEOUT).unwrap()),
            Err(FileCellIoError::Corrupt(_))
        ));
        idle(&p);
        assert!(temp.0.join(STORE).join("journal").exists());
    }
}

#[cfg(unix)]
#[test]
fn symlink_stage_refused_and_external_file_preserved() {
    let temp = Temp::new();
    let outside = Temp::new();
    let file = outside.0.join("keep");
    fs::write(&file, b"keep").unwrap();
    let p = provider(&temp.0);
    commit(&p, request(0)).unwrap();
    let slot = Directory::child(&temp.0.join(STORE), "slot0", true).unwrap();
    std::os::unix::fs::symlink(&file, slot.path.join("0")).unwrap();
    assert!(matches!(
        commit(&p, request(1)),
        Err(FileCellIoError::UnsafePath(_))
    ));
    assert_eq!(fs::read(&file).unwrap(), b"keep");
    assert_eq!(read(&p).generation(), 1);
}
