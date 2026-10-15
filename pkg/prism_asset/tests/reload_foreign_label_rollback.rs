//! Rejected reloads must not consume another resident source's label ownership.

#![cfg(feature = "std")]

use std::sync::Arc;

use prism_asset::{
    Asset, AssetEvent, AssetLoader, AssetPath, AssetServer, AssetTypeId, DepRequest, LoadContext,
    LoadError, LoadState, MemSource,
};

const CANDIDATE_PATH: &str = "candidate.foreign_rollback";
const FOREIGN_PATH: &str = "foreign.foreign_rollback";
const FOREIGN_LABEL_PATH: &str = "foreign.foreign_rollback#old";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Source(u8);

impl Asset for Source {
    const TYPE_NAME: &'static str = "reload_foreign_label_rollback::Source";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Label(u8);

impl Asset for Label {
    const TYPE_NAME: &'static str = "reload_foreign_label_rollback::Label";
}

struct SourceLoader;

impl AssetLoader for SourceLoader {
    type Asset = Source;

    fn extensions(&self) -> &'static [&'static str] {
        &["foreign_rollback"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Source, LoadError> {
        let revision = *bytes
            .first()
            .ok_or_else(|| LoadError::Malformed("missing revision".into()))?;
        match revision {
            1 => {}
            2 => ctx.add_labeled_asset("old", Label(revision)),
            3 => {
                // Resolve an existing, still source-owned foreign label while
                // staging a new output. Reject only after that resolution, so
                // neither the new output nor its dependency is a real consumer.
                ctx.add_labeled_asset_with_deps(
                    "provisional",
                    Label(revision),
                    vec![DepRequest::typed(
                        AssetPath::parse(FOREIGN_LABEL_PATH),
                        AssetTypeId::of::<Label>(),
                    )],
                );
                ctx.add_labeled_asset("provisional", Label(revision));
            }
            _ => return Err(LoadError::Malformed("unexpected revision".into())),
        }
        Ok(Source(revision))
    }
}

fn mount(server: &AssetServer, candidate_revision: u8) {
    let mut source = MemSource::new();
    source.insert(CANDIDATE_PATH, [candidate_revision]);
    source.insert(FOREIGN_PATH, [2]);
    server.mount("mem", Arc::new(source), 0);
}

fn collect(server: &AssetServer) {
    // Advance deterministic reclaim points, not wall-clock time.
    for _ in 0..8 {
        server.collect_releases();
    }
}

#[test]
fn failed_reload_does_not_promote_uncommitted_foreign_label() {
    let server = AssetServer::new();
    server.register_loader(SourceLoader);
    mount(&server, 1);

    // Keep BOTH source roots alive throughout rejection and reclamation.
    let candidate_source = server.load::<Source>(CANDIDATE_PATH);
    let foreign_source = server.load::<Source>(FOREIGN_PATH);
    assert_eq!(server.total_assets(), 3);

    // Reading an Added id/snapshot does not resolve a label or promote its
    // ownership. In particular, do not call load[_untyped/soft] on this path.
    let (events, snapshots) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(events.len(), 1);
    let label_id = match events[0] {
        AssetEvent::Added { id } => id,
        ref event => panic!("expected the foreign label's Added event, got {event:?}"),
    };
    assert_eq!(snapshots, vec![(label_id, Label(2))]);
    server.drain_events_with_snapshots::<Source>().unwrap();

    collect(&server);
    assert_eq!(server.with_asset(label_id, |value| value.0), Some(2));
    assert_eq!(server.load_state(label_id.untyped()), LoadState::Loaded);
    assert_eq!(server.total_assets(), 3);

    mount(&server, 3);
    let reloaded = server.reload(CANDIDATE_PATH).unwrap();
    assert_eq!(reloaded.id(), candidate_source.untyped_id());
    drop(reloaded);

    // The duplicate rejects the candidate after resolving its foreign label
    // dependency; no source value or staged labeled value may be published.
    assert_eq!(
        server.with_asset(candidate_source.id(), |value| value.0),
        Some(1)
    );
    assert_eq!(
        server.with_asset(foreign_source.id(), |value| value.0),
        Some(2)
    );
    let (events, snapshots) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(events.len(), 1);
    assert!(
        events[0].is_failed(),
        "the provisional slot must be rejected"
    );
    assert_ne!(events[0].id(), label_id);
    assert!(snapshots.is_empty());
    assert!(server
        .drain_events_with_snapshots::<Source>()
        .unwrap()
        .0
        .is_empty());
    assert_eq!(server.with_asset(label_id, |value| value.0), Some(2));

    collect(&server);
    for (handle, expected) in [(&candidate_source, 1), (&foreign_source, 2)] {
        assert_eq!(server.load_state(handle.untyped_id()), LoadState::Loaded);
        assert_eq!(
            server.with_asset(handle.id(), |value| value.0),
            Some(expected)
        );
    }
    assert_eq!(
        server.with_asset(label_id, |value| value.0),
        Some(2),
        "rejected staging must preserve the foreign source's old label through GC"
    );
    assert_eq!(server.load_state(label_id.untyped()), LoadState::Loaded);

    // Resolve only AFTER checking post-GC survival, then verify original identity.
    let resolved = server.load_untyped(FOREIGN_LABEL_PATH).unwrap();
    assert_eq!(resolved.id(), label_id.untyped());
    drop(resolved);
    drop(candidate_source);
    drop(foreign_source);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}
