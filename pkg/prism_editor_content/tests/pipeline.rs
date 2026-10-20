//! End-to-end coverage for the bounded asset-build pipeline protocol.

use prism_editor_content::pipeline::{
    Pipeline, PipelineAsset, PipelineError, PipelineLimits, PipelineOutput, PipelineRequest,
};
use prism_editor_content::validation::{
    ValidatedCatalog, ValidationLimits, ValidationOperation, ValidationPolicy, ValidationRegistry,
};
use prism_editor_content::{AssetId, AssetKind, AssetRecord, ContentCatalog};
use std::task::Poll;
use std::{
    fs,
    path::PathBuf,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00, 0x00, 0xb5,
    0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64,
    0xf8, 0x0f, 0x00, 0x01, 0x05, 0x01, 0x01, 0x27, 0x18, 0xe3, 0x66, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

fn temp() -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let p = std::env::temp_dir().join(format!("prism-pipeline-{n}"));
    fs::create_dir_all(&p).expect("temp");
    p
}
fn cli() -> PathBuf {
    std::env::var_os("PRISM_ASSET_BUILD_CLI")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/private/tmp/prism-asset-build-check/debug/prism_asset_build_cli")
        })
}
fn checked<'a>(c: &'a ContentCatalog, r: &'a ValidationRegistry) -> ValidatedCatalog<'a> {
    r.validate(c, ValidationOperation::Cook)
        .expect("validate")
        .authorize(c, r, ValidationOperation::Cook, ValidationPolicy::default())
        .expect("authorize")
}
fn wait_unchecked(
    mut job: prism_editor_content::pipeline::PipelineJob,
) -> Result<prism_editor_content::pipeline::PipelineCompletion, PipelineError> {
    for _ in 0..10000 {
        match job.poll() {
            Poll::Ready(r) => return r,
            Poll::Pending => thread::sleep(Duration::from_millis(2)),
        }
    }
    job.cancel();
    panic!("pipeline polling bound exceeded")
}
fn wait_checked(
    mut job: prism_editor_content::pipeline::PipelineJob,
    catalog: &ContentCatalog,
) -> Result<prism_editor_content::pipeline::PipelineCompletion, PipelineError> {
    for _ in 0..10000 {
        match job.poll_checked(catalog) {
            Poll::Ready(r) => return r,
            Poll::Pending => thread::sleep(Duration::from_millis(2)),
        }
    }
    job.cancel();
    panic!("checked pipeline polling bound exceeded")
}
fn catalog() -> (ContentCatalog, ValidationRegistry) {
    let mut c = ContentCatalog::new();
    c.insert(
        AssetRecord::new(
            AssetId::from_raw(7),
            "textures/hero.png",
            AssetKind::new("texture").expect("kind"),
        )
        .expect("record"),
    )
    .expect("insert");
    (c, ValidationRegistry::new(ValidationLimits::default()))
}

#[test]
fn real_png_import_package_validate_inspect() {
    let root = temp();
    let source = root.join("textures/hero.png");
    fs::create_dir_all(source.parent().expect("parent")).expect("parent");
    fs::write(&source, PNG).expect("png");
    let store = root.join("store");
    let output = root.join("package");
    let (c, r) = catalog();
    let auth = checked(&c, &r);
    let p = Pipeline::new(cli(), root.clone(), PipelineLimits::default()).expect("pipeline");
    let asset = PipelineAsset {
        asset: AssetId::from_raw(7),
        logical_path: "textures/hero.png".into(),
    };
    let result = wait_checked(
        p.submit_checked(
            &auth,
            PipelineRequest::Import {
                source: asset.clone(),
                store: store.clone(),
                verify_determinism: false,
            },
        )
        .expect("import"),
        &c,
    )
    .expect("import result");
    assert!(matches!(result.output, PipelineOutput::Import(_)));
    result.ensure_current(&c).expect("fence");
    let result = wait_checked(
        p.submit_checked(
            &auth,
            PipelineRequest::Build {
                source: asset.clone(),
                store: store.clone(),
                verify_determinism: false,
            },
        )
        .expect("build"),
        &c,
    )
    .expect("build result");
    assert!(matches!(result.output, PipelineOutput::Build(_)));
    let result = wait_checked(
        p.submit_checked(
            &auth,
            PipelineRequest::Package {
                source: asset.clone(),
                store: store.clone(),
                output: output.clone(),
            },
        )
        .expect("package"),
        &c,
    )
    .expect("package result");
    assert!(matches!(result.output, PipelineOutput::Package(_)));
    let result = wait_checked(
        p.submit_checked(
            &auth,
            PipelineRequest::Validate {
                source: asset,
                store: store.clone(),
                package: output,
            },
        )
        .expect("validate"),
        &c,
    )
    .expect("validate result");
    assert!(matches!(result.output, PipelineOutput::Validate(_)));
    let result = wait_unchecked(
        p.submit_unvalidated(PipelineRequest::Inspect { store })
            .expect("inspect"),
    )
    .expect("inspect result");
    assert!(matches!(result.output, PipelineOutput::Inspect(_)));
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn checked_submission_rejects_path_substitution() {
    let root = temp();
    let (c, r) = catalog();
    let auth = checked(&c, &r);
    let p = Pipeline::new(cli(), root.clone(), PipelineLimits::default()).expect("pipeline");
    let result = p.submit_checked(
        &auth,
        PipelineRequest::Import {
            source: PipelineAsset {
                asset: AssetId::from_raw(7),
                logical_path: "textures/other.png".into(),
            },
            store: root.join("store"),
            verify_determinism: false,
        },
    );
    assert!(matches!(result, Err(PipelineError::UnauthorizedAsset)));
    fs::remove_dir_all(root).expect("cleanup");
}

#[cfg(unix)]
#[test]
fn cancellation_reaps_fixture_without_success() {
    use std::os::unix::fs::PermissionsExt;
    let root = temp();
    let executable = root.join("sleep-fixture");
    fs::write(&executable, b"#!/bin/sh\nexec /bin/sleep 30\n").expect("fixture");
    let mut permissions = fs::metadata(&executable).expect("metadata").permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&executable, permissions).expect("permissions");
    let pipeline =
        Pipeline::new(executable, root.clone(), PipelineLimits::default()).expect("pipeline");
    let job = pipeline
        .submit_unvalidated(PipelineRequest::Inspect {
            store: root.join("store"),
        })
        .expect("submit");
    for _ in 0..1000 {
        if job.process_id().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert!(job.process_id().is_some());
    job.cancel();
    assert!(matches!(wait_unchecked(job), Err(PipelineError::Cancelled)));
    fs::remove_dir_all(root).expect("cleanup");
}
