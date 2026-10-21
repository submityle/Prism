//! Reimport resolves workspace payloads before callbacks and publishes packages
//! only after input admission, semantic reconciliation and output admission.

use prism_asset_build_core::{
    AssetId, ContentAddressedStore, ContentHash, EditableAsset, EditableAssetError,
    EditableDependencyKind, EditablePayload, EditablePayloadReferenceRewriter,
    EditableReimportOutput, EditableReimporter, EditableSubAsset, EditableSubAssetRedirect,
    EditableSubAssetReference, EditableSubAssetReferenceRewrite, EditableWorkspace,
    EditableWorkspaceReimport, LocalFileSourceProvider, SourceRecord, SourceSnapshot,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use std::cell::Cell;
use std::fs;
use tempfile::{tempdir, TempDir};

const TARGET: AssetId = AssetId(10);
const CONSUMER: AssetId = AssetId(20);
const REDIRECTS: &[EditableSubAssetRedirect] = &[EditableSubAssetRedirect { from: 9, to: 12 }];

fn subassets(id: u64) -> Vec<EditableSubAsset> {
    vec![EditableSubAsset {
        id,
        name: format!("part-{id}"),
        kind: 42,
    }]
}

fn target(payload: impl Into<EditablePayload>) -> EditableAsset {
    EditableAsset::new(
        TARGET,
        42,
        5,
        b"original settings".to_vec(),
        SourceRecord::external(
            "sources/hero.input",
            "test/raw",
            ContentHash::digest(b"old source"),
        )
        .unwrap(),
        subassets(9),
        Vec::new(),
        Vec::new(),
        payload,
        Some(b"old preview".to_vec()),
    )
    .unwrap()
}

fn consumer(payload: impl Into<EditablePayload>) -> EditableAsset {
    EditableAsset::new(
        CONSUMER,
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer.input", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: TARGET,
            subasset: 9,
            kind: EditableDependencyKind::Hard,
        }],
        payload,
        None,
    )
    .unwrap()
}

struct Importer {
    calls: Cell<usize>,
    expected: EditableAsset,
    expected_payload: Vec<u8>,
    next_subassets: Vec<EditableSubAsset>,
    output_len: usize,
}

impl Importer {
    fn new(previous: &EditableAsset, payload: &[u8], output_len: usize, next_id: u64) -> Self {
        Self {
            calls: Cell::new(0),
            expected: previous.clone(),
            expected_payload: payload.to_vec(),
            next_subassets: subassets(next_id),
            output_len,
        }
    }
}

impl EditableReimporter for Importer {
    fn reimport(
        &self,
        previous: &EditableAsset,
        previous_payload: &[u8],
        source: &SourceSnapshot,
    ) -> Result<EditableReimportOutput, EditableAssetError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(previous, &self.expected);
        assert_eq!(previous_payload, self.expected_payload);
        assert_eq!(
            ContentHash::digest(previous_payload),
            previous.authoring_payload.content_hash()
        );
        assert_eq!(source.logical_path(), "sources/hero.input");
        assert_eq!(source.bytes(), b"new source");
        assert_eq!(source.content_hash(), ContentHash::digest(b"new source"));
        Ok(EditableReimportOutput {
            canonical_settings: b"reimported settings".to_vec(),
            subassets: self.next_subassets.clone(),
            dependencies: Vec::new(),
            subasset_references: Vec::new(),
            authoring_payload: vec![0xA7; self.output_len],
            preview: Some(b"reimported preview".to_vec()),
        })
    }
}

struct Rewriter {
    calls: Cell<usize>,
    expected: EditableAsset,
    expected_payload: Vec<u8>,
    output: Vec<u8>,
}

impl Rewriter {
    fn new(previous: &EditableAsset, payload: &[u8], output: &[u8]) -> Self {
        Self {
            calls: Cell::new(0),
            expected: previous.clone(),
            expected_payload: payload.to_vec(),
            output: output.to_vec(),
        }
    }
}

impl EditablePayloadReferenceRewriter for Rewriter {
    fn rewrite_authoring_payload(
        &self,
        asset: &EditableAsset,
        payload: &[u8],
        rewrites: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(asset, &self.expected);
        assert!(matches!(
            asset.authoring_payload,
            EditablePayload::External(_)
        ));
        assert_eq!(payload, self.expected_payload);
        assert_eq!(
            ContentHash::digest(payload),
            asset.authoring_payload.content_hash()
        );
        assert_eq!(
            rewrites,
            &[EditableSubAssetReferenceRewrite {
                asset: TARGET,
                from: 9,
                to: 12
            }]
        );
        Ok(self.output.clone())
    }
}

struct UnexpectedRewrite;

impl EditablePayloadReferenceRewriter for UnexpectedRewrite {
    fn rewrite_authoring_payload(
        &self,
        _: &EditableAsset,
        _: &[u8],
        _: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError> {
        panic!("a package without redirected references must not invoke the payload rewriter");
    }
}

struct Fixture {
    directory: TempDir,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join("sources")).unwrap();
        fs::write(directory.path().join("sources/hero.input"), b"new source").unwrap();
        Self { directory }
    }

    fn workspace(&self) -> EditableWorkspace {
        EditableWorkspace::open(self.directory.path()).unwrap()
    }

    fn write(&self, name: &str, asset: &EditableAsset) {
        asset
            .write_to_path(self.directory.path().join(name))
            .unwrap();
    }

    fn read(&self, name: &str) -> EditableAsset {
        EditableAsset::read_from_path(self.directory.path().join(name)).unwrap()
    }

    fn external_target(&self, bytes: &[u8]) -> EditableAsset {
        let lease = self.workspace().store_payload(bytes, 1024).unwrap();
        let asset = target(EditablePayload::External(lease.reference()));
        self.write("hero.prism", &asset);
        asset
    }

    fn external_consumer(&self, bytes: &[u8]) -> EditableAsset {
        let lease = self.workspace().store_payload(bytes, 1024).unwrap();
        let asset = consumer(EditablePayload::External(lease.reference()));
        self.write("consumer.prism", &asset);
        asset
    }

    fn reimport<R: EditablePayloadReferenceRewriter>(
        &self,
        importer: &Importer,
        rewriter: &R,
        redirects: &[EditableSubAssetRedirect],
        budget: u64,
    ) -> Result<EditableWorkspaceReimport, EditableAssetError> {
        self.workspace().reimport_with_payload_budget(
            "hero.prism",
            &LocalFileSourceProvider::new(self.directory.path()).unwrap(),
            importer,
            rewriter,
            redirects,
            budget,
        )
    }

    fn package_bytes(&self, names: &[&str]) -> Vec<Vec<u8>> {
        names
            .iter()
            .map(|name| fs::read(self.directory.path().join(name)).unwrap())
            .collect()
    }

    fn assert_unchanged(&self, names: &[&str], before: &[Vec<u8>]) {
        assert_eq!(self.package_bytes(names), before);
        // Reopening runs recovery too; an aborted operation must not leave a
        // transaction that publishes replacement bytes on the next open.
        let _reopened = self.workspace();
        assert_eq!(self.package_bytes(names), before);
    }

    fn corrupt_payload(&self, asset: &EditableAsset) {
        let EditablePayload::External(reference) = asset.authoring_payload else {
            panic!("external fixture")
        };
        let cas =
            ContentAddressedStore::open(self.directory.path().join(".prism-editable-payloads"))
                .unwrap();
        // Preserve the declared length so the reader must verify the hash.
        fs::write(
            cas.blob_path(reference.content_hash),
            vec![0xEE; reference.byte_len as usize],
        )
        .unwrap();
    }
}

#[test]
fn external_previous_is_verified_before_callback_and_small_reimport_stays_external() {
    let fixture = Fixture::new();
    let previous = fixture.external_target(b"previous authoring bytes");
    let importer = Importer::new(&previous, b"previous authoring bytes", 7, 9);
    let result = fixture
        .reimport(&importer, &UnexpectedRewrite, &[], 1024)
        .unwrap();
    assert_eq!(importer.calls.get(), 1);
    assert_eq!(result.updated_assets, [TARGET]);
    let saved = fixture.read("hero.prism");
    assert_eq!(
        (saved.asset, saved.kind, saved.flags),
        (previous.asset, previous.kind, previous.flags)
    );
    assert_eq!(saved.source.uri, previous.source.uri);
    assert_eq!(saved.source.format, previous.source.format);
    assert_eq!(
        saved.source.content_hash,
        ContentHash::digest(b"new source")
    );
    assert_eq!(saved.canonical_settings, b"reimported settings");
    assert_eq!(
        saved.preview.as_deref(),
        Some(b"reimported preview".as_slice())
    );
    assert!(matches!(
        saved.authoring_payload,
        EditablePayload::External(_)
    ));
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&TARGET], saved);
    assert_eq!(
        snapshot
            .resolve_payload(&saved.authoring_payload, 7)
            .unwrap()
            .as_ref(),
        &[0xA7; 7]
    );
}

#[test]
fn external_consumer_redirect_rewrites_verified_bytes_and_persists_external_output() {
    let fixture = Fixture::new();
    let previous = target(b"target".to_vec());
    fixture.write("hero.prism", &previous);
    let old_consumer = fixture.external_consumer(b"ref:9");
    let importer = Importer::new(&previous, b"target", 8, 12);
    let rewriter = Rewriter::new(&old_consumer, b"ref:9", b"ref:12");
    // Inputs total 11 bytes; callback outputs total exactly 14 bytes.
    let result = fixture
        .reimport(&importer, &rewriter, REDIRECTS, 14)
        .unwrap();
    assert_eq!(result.updated_assets, [TARGET, CONSUMER]);
    assert_eq!((importer.calls.get(), rewriter.calls.get()), (1, 1));
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&TARGET].subassets, subassets(12));
    let saved = &snapshot[&CONSUMER];
    assert_eq!(
        saved.subasset_references,
        [EditableSubAssetReference {
            asset: TARGET,
            subasset: 12,
            kind: EditableDependencyKind::Hard,
        }]
    );
    assert!(matches!(
        saved.authoring_payload,
        EditablePayload::External(_)
    ));
    assert_ne!(saved.authoring_payload, old_consumer.authoring_payload);
    assert_eq!(
        snapshot
            .resolve_payload(&saved.authoring_payload, 6)
            .unwrap()
            .as_ref(),
        b"ref:12"
    );
    assert_eq!(
        snapshot
            .resolve_payload(&snapshot[&TARGET].authoring_payload, 8)
            .unwrap()
            .as_ref(),
        &[0xA7; 8]
    );
}

#[test]
fn aggregate_input_preflight_precedes_blob_verification_and_callbacks_without_publication() {
    let fixture = Fixture::new();
    let previous = fixture.external_target(b"target!!");
    let old_consumer = fixture.external_consumer(b"ref:9");
    let importer = Importer::new(&previous, b"target!!", 1, 12);
    let rewriter = Rewriter::new(&old_consumer, b"ref:9", b"ref:12");
    let names = ["hero.prism", "consumer.prism"];
    let before = fixture.package_bytes(&names);
    fixture.corrupt_payload(&previous);
    // Either selected payload alone fits. Their aggregate does not; even the
    // corrupted first input must remain unread until aggregate admission.
    assert_eq!(
        fixture.reimport(&importer, &rewriter, REDIRECTS, 12),
        Err(EditableAssetError::PayloadBudgetExceeded {
            requested: 13,
            limit: 12
        })
    );
    assert_eq!((importer.calls.get(), rewriter.calls.get()), (0, 0));
    fixture.assert_unchanged(&names, &before);
    assert_eq!(
        fixture.reimport(&importer, &rewriter, REDIRECTS, 13),
        Err(EditableAssetError::InvalidPayloadReference)
    );
    assert_eq!((importer.calls.get(), rewriter.calls.get()), (0, 0));
    fixture.assert_unchanged(&names, &before);
}

#[test]
fn corrupted_previous_or_redirect_consumer_is_rejected_before_any_callback() {
    for corrupt_consumer in [false, true] {
        let fixture = Fixture::new();
        let previous = fixture.external_target(b"target");
        let old_consumer = fixture.external_consumer(b"ref:9");
        let importer = Importer::new(&previous, b"target", 8, 12);
        let rewriter = Rewriter::new(&old_consumer, b"ref:9", b"ref:12");
        let names = ["hero.prism", "consumer.prism"];
        let before = fixture.package_bytes(&names);
        fixture.corrupt_payload(if corrupt_consumer {
            &old_consumer
        } else {
            &previous
        });
        assert_eq!(
            fixture.reimport(&importer, &rewriter, REDIRECTS, 1024),
            Err(EditableAssetError::InvalidPayloadReference)
        );
        assert_eq!((importer.calls.get(), rewriter.calls.get()), (0, 0));
        fixture.assert_unchanged(&names, &before);
    }
}

#[test]
fn shared_external_input_counts_once_and_input_output_budgets_are_separate() {
    let fixture = Fixture::new();
    let previous = fixture.external_target(b"ref:9");
    let old_consumer = fixture.external_consumer(b"ref:9");
    assert_eq!(previous.authoring_payload, old_consumer.authoring_payload);
    let importer = Importer::new(&previous, b"ref:9", 1, 12);
    let rewriter = Rewriter::new(&old_consumer, b"ref:9", b"r");
    // One unique 5-byte input and two output bytes fit separate 5-byte bounds.
    let result = fixture
        .reimport(&importer, &rewriter, REDIRECTS, 5)
        .unwrap();
    assert_eq!(result.updated_assets, [TARGET, CONSUMER]);
    assert_eq!((importer.calls.get(), rewriter.calls.get()), (1, 1));
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(
        snapshot
            .resolve_payload(&snapshot[&CONSUMER].authoring_payload, 1)
            .unwrap()
            .as_ref(),
        b"r"
    );
    assert_eq!(
        snapshot
            .resolve_payload(&snapshot[&TARGET].authoring_payload, 1)
            .unwrap()
            .as_ref(),
        &[0xA7]
    );
}

#[test]
fn importer_output_over_budget_keeps_the_previous_external_package() {
    let fixture = Fixture::new();
    let previous = fixture.external_target(b"p");
    let importer = Importer::new(&previous, b"p", 9, 9);
    let names = ["hero.prism"];
    let before = fixture.package_bytes(&names);
    assert_eq!(
        fixture.reimport(&importer, &UnexpectedRewrite, &[], 8),
        Err(EditableAssetError::PayloadBudgetExceeded {
            requested: 9,
            limit: 8
        })
    );
    assert_eq!(importer.calls.get(), 1);
    fixture.assert_unchanged(&names, &before);
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(
        snapshot
            .resolve_payload(&snapshot[&TARGET].authoring_payload, 1)
            .unwrap()
            .as_ref(),
        b"p"
    );
}

#[test]
fn aggregate_callback_output_over_budget_keeps_target_and_consumer_unchanged() {
    let fixture = Fixture::new();
    let previous = fixture.external_target(b"target");
    let old_consumer = fixture.external_consumer(b"ref:9");
    let importer = Importer::new(&previous, b"target", 8, 12);
    let rewriter = Rewriter::new(&old_consumer, b"ref:9", b"ref:12");
    let names = ["hero.prism", "consumer.prism"];
    let before = fixture.package_bytes(&names);
    // Both outputs fit individually. The aggregate output (14) exceeds 13.
    assert_eq!(
        fixture.reimport(&importer, &rewriter, REDIRECTS, 13),
        Err(EditableAssetError::PayloadBudgetExceeded {
            requested: 14,
            limit: 13
        })
    );
    assert_eq!((importer.calls.get(), rewriter.calls.get()), (1, 1));
    fixture.assert_unchanged(&names, &before);
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&TARGET], previous);
    assert_eq!(snapshot[&CONSUMER], old_consumer);
}

#[test]
fn output_above_32_mib_becomes_external_and_keeps_the_saved_package_small() {
    let fixture = Fixture::new();
    let previous = target(b"p".to_vec());
    fixture.write("hero.prism", &previous);
    let length = 32 * 1024 * 1024 + 1;
    let importer = Importer::new(&previous, b"p", length, 9);
    let result = fixture
        .reimport(&importer, &UnexpectedRewrite, &[], length as u64)
        .unwrap();
    assert_eq!(result.updated_assets, [TARGET]);
    assert_eq!(importer.calls.get(), 1);
    assert!(
        fs::metadata(fixture.directory.path().join("hero.prism"))
            .unwrap()
            .len()
            < 4096
    );
    let saved = fixture.read("hero.prism");
    let EditablePayload::External(reference) = saved.authoring_payload else {
        panic!("payloads above the inline limit must be external");
    };
    assert_eq!(reference.byte_len, length as u64);
    let snapshot = fixture
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    let bytes = snapshot
        .resolve_payload(&snapshot[&TARGET].authoring_payload, length as u64)
        .unwrap();
    assert_eq!(bytes.len(), length);
    assert!(bytes.iter().all(|&byte| byte == 0xA7));
    assert_eq!(ContentHash::digest(&bytes), reference.content_hash);
}
