//! Scene Project bootstrap and durable editor-document integration tests.

#![cfg(feature = "desktop")]

extern crate alloc;

use alloc::{sync::Arc, task::Wake};
use core::task::{Context, Poll, Waker};
use std::fs;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use prism_editor_app::{
    DesktopDocumentSession, DesktopError, DesktopGpu, DesktopRenderer, DesktopViewportBounds,
    DesktopWork, DocumentSchemaCatalog, SceneProject, SceneProjectConfig, SceneProjectOpen,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_reflect::{Reflect, TypeRegistry};

const MANIFEST_FILE: &str = "prism.project";
const SCENE_FILE: &str = "scene/cells/0_0_0.prismcell";
const MANIFEST_MAGIC_BYTES: usize = 8;
const MANIFEST_VERSION_OFFSET: usize = MANIFEST_MAGIC_BYTES;
const MANIFEST_GUID_OFFSET: usize = MANIFEST_VERSION_OFFSET + size_of::<u32>();
const MANIFEST_MIN_BYTES: usize = MANIFEST_GUID_OFFSET + size_of::<u128>();

#[derive(Clone, Debug, PartialEq, Reflect)]
struct ProjectModel {
    title: String,
}

fn project_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<ProjectModel>();
    registry
}

fn project_catalog() -> DocumentSchemaCatalog {
    let mut catalog = DocumentSchemaCatalog::new();
    catalog
        .register(std::any::type_name::<ProjectModel>(), 1, project_registry)
        .expect("register project model schema");
    catalog
}

struct PersistenceRenderer;

impl DesktopRenderer for PersistenceRenderer {
    fn initialize(&mut self, _gpu: &DesktopGpu) -> Result<(), DesktopError> {
        Ok(())
    }

    fn encode(
        &mut self,
        _gpu: &DesktopGpu,
        _ui: &mut prism_editor_app::EditorDesktopUi,
        _encoder: &mut wgpu::CommandEncoder,
        _target: &wgpu::TextureView,
        _bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError> {
        Ok(())
    }

    fn pointer(
        &mut self,
        _gpu: &DesktopGpu,
        _ui: &mut prism_editor_app::EditorDesktopUi,
        _position: [f32; 2],
        _mode: prism_editor_ui::SelectionMode,
    ) -> Result<bool, DesktopError> {
        Ok(false)
    }

    fn poll(
        &mut self,
        _gpu: &DesktopGpu,
        _ui: &mut prism_editor_app::EditorDesktopUi,
    ) -> Result<DesktopWork, DesktopError> {
        Ok(DesktopWork::default())
    }

    fn suspend(&mut self) {}
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn wait_for_save(session: &mut DesktopDocumentSession, ui: &mut prism_editor_app::EditorDesktopUi) {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut renderer = PersistenceRenderer;
    loop {
        match session
            .poll(&mut context, ui, &mut renderer)
            .expect("save completion is acknowledged")
        {
            Poll::Ready(_) => return,
            Poll::Pending => {
                assert!(Instant::now() < deadline, "scene project save timed out");
                thread::park_timeout(Duration::from_millis(5));
            }
        }
    }
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-scene-project-{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create private test directory {path:?}: {error}"),
            }
        }
    }

    fn project(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only the private scene-project test directory");
    }
}

fn create(path: &Path) -> SceneProjectOpen {
    SceneProject::create(path)
        .unwrap_or_else(|error| panic!("create scene project at {path:?}: {error}"))
}

fn open(path: &Path) -> SceneProjectOpen {
    SceneProject::open(path)
        .unwrap_or_else(|error| panic!("open scene project at {path:?}: {error}"))
}

fn assert_create_rejected(path: &Path) {
    let Err(error) = SceneProject::create(path) else {
        panic!("unsafe scene project create unexpectedly succeeded at {path:?}");
    };
    assert!(
        !error.to_string().is_empty(),
        "scene project create errors must have a diagnostic"
    );
}

fn assert_open_rejected(path: &Path) {
    let Err(error) = SceneProject::open(path) else {
        panic!("unsafe scene project open unexpectedly succeeded at {path:?}");
    };
    assert!(
        !error.to_string().is_empty(),
        "scene project open errors must have a diagnostic"
    );
}

fn manifest(path: &Path) -> PathBuf {
    path.join(MANIFEST_FILE)
}

fn scene(path: &Path) -> PathBuf {
    path.join(SCENE_FILE)
}

fn mutate_manifest(path: &Path, mutate: impl FnOnce(&mut Vec<u8>)) {
    let manifest = manifest(path);
    let mut bytes = fs::read(&manifest).expect("read private project manifest");
    assert!(
        bytes.len() >= MANIFEST_MIN_BYTES,
        "created manifest must contain magic, version and stable GUID"
    );
    mutate(&mut bytes);
    fs::write(manifest, bytes).expect("replace private project manifest");
}

fn assert_clean_empty_project(project: &SceneProjectOpen, expected_root: &Path) {
    assert_eq!(project.root(), expected_root);
    assert!(!project.project_guid().is_nil());
    assert_eq!(project.document_session().durable_revision(), Some(0));
    assert!(!project.document_session().is_busy());
    assert!(!project.document_session().reconciliation_required());
    assert!(!project.ui().has_uncommitted_edit());
    assert!(!project.ui().controller().is_dirty());
    assert_eq!(project.ui().controller().model_revision(), 0);
    assert_eq!(project.ui().controller().document_objects().len(), 0);
    assert_eq!(project.ui().controller().document_history_lengths(), (0, 0));
}

#[test]
fn create_publishes_a_durable_empty_scene_and_open_restores_it_cleanly() {
    let temp = TempRoot::new("round-trip");
    let project_path = temp.project("project");

    let created = create(&project_path);
    let canonical = fs::canonicalize(&project_path).expect("canonical project root");
    assert_clean_empty_project(&created, &canonical);
    assert!(manifest(&project_path).is_file());
    assert_eq!(
        &fs::read(scene(&project_path)).expect("read durable scene cell")[..8],
        b"PEWCELL\0"
    );
    let guid = created.project_guid();
    drop(created);

    let reopened = open(&project_path);
    assert_clean_empty_project(&reopened, &canonical);
    assert_eq!(reopened.project_guid(), guid);
}

#[test]
fn authored_document_object_survives_project_save_and_reopen() {
    let temp = TempRoot::new("authored-round-trip");
    let project_path = temp.project("project");
    let project = SceneProject::create_with(
        &project_path,
        project_catalog(),
        SceneProjectConfig::default(),
    )
    .expect("create project with its schema catalog");
    let (mut ui, mut session) = project.into_desktop();

    let identity = SelectionItem::entity(0xfeed);
    ui.controller_mut()
        .register_object(
            identity.clone(),
            Box::new(ProjectModel {
                title: "Saved ECS-backed scene".into(),
            }),
            project_registry(),
        )
        .expect("register authored object");
    let mut row = OutlinerNode::new(OutlinerId::from_raw(77), None, "Scene object");
    row.set_selection(identity.clone());
    ui.controller_mut()
        .register_outliner_node(row)
        .expect("register authored Outliner row");

    session
        .save(&mut ui)
        .expect("submit authored document save");
    wait_for_save(&mut session, &mut ui);
    assert_eq!(session.durable_revision(), Some(1));
    drop((ui, session));

    let reopened = SceneProject::open_with(
        &project_path,
        project_catalog(),
        SceneProjectConfig::default(),
    )
    .expect("reopen authored project");
    assert_eq!(reopened.document_session().durable_revision(), Some(1));
    let controller = reopened.ui().controller();
    assert_eq!(controller.document_objects().len(), 1);
    assert_eq!(controller.document_outliner_nodes().len(), 1);
    assert_eq!(
        controller.document_outliner_nodes()[0].selection(),
        Some(&identity)
    );
    let mut restored = ProjectModel {
        title: String::new(),
    };
    restored
        .apply(&*controller.snapshot(&identity).expect("restored object"))
        .expect("apply reflected object state");
    assert_eq!(restored.title, "Saved ECS-backed scene");
}

#[test]
fn repeated_create_does_not_overwrite_manifest_or_scene() {
    let temp = TempRoot::new("repeat-create");
    let project_path = temp.project("project");
    drop(create(&project_path));
    let manifest_before = fs::read(manifest(&project_path)).expect("read created manifest");
    let scene_before = fs::read(scene(&project_path)).expect("read created scene");

    assert_create_rejected(&project_path);

    assert_eq!(
        fs::read(manifest(&project_path)).expect("read preserved manifest"),
        manifest_before
    );
    assert_eq!(
        fs::read(scene(&project_path)).expect("read preserved scene"),
        scene_before
    );
    open(&project_path);
}

#[test]
fn persisted_project_guid_survives_a_project_directory_move() {
    let temp = TempRoot::new("moved-project");
    let original = temp.project("original");
    let created = create(&original);
    let guid = created.project_guid();
    drop(created);

    let moved = temp.project("moved");
    fs::rename(&original, &moved).expect("move private project directory");
    let reopened = open(&moved);

    assert_eq!(reopened.project_guid(), guid);
    assert_eq!(
        reopened.root(),
        fs::canonicalize(&moved).expect("canonical moved project root")
    );
    assert_clean_empty_project(&reopened, reopened.root());
}

#[test]
fn create_rejects_a_nonempty_directory_without_touching_existing_data() {
    let temp = TempRoot::new("nonempty");
    let project_path = temp.project("project");
    fs::create_dir(&project_path).expect("create existing directory");
    let sentinel = project_path.join("keep.bin");
    fs::write(&sentinel, b"owned by caller").expect("write existing data");

    assert_create_rejected(&project_path);

    assert_eq!(
        fs::read(&sentinel).expect("read preserved existing data"),
        b"owned by caller"
    );
    assert!(!manifest(&project_path).exists());
    assert!(!project_path.join("scene").exists());
}

#[test]
fn open_rejects_bad_magic_unsupported_version_nil_guid_and_trailing_bytes() {
    let temp = TempRoot::new("bad-manifest");

    let bad_magic = temp.project("bad-magic");
    drop(create(&bad_magic));
    mutate_manifest(&bad_magic, |bytes| bytes[0] ^= 0xff);
    assert_open_rejected(&bad_magic);

    let bad_version = temp.project("bad-version");
    drop(create(&bad_version));
    mutate_manifest(&bad_version, |bytes| {
        bytes[MANIFEST_VERSION_OFFSET..MANIFEST_GUID_OFFSET]
            .copy_from_slice(&u32::MAX.to_le_bytes());
    });
    assert_open_rejected(&bad_version);

    let nil_guid = temp.project("nil-guid");
    drop(create(&nil_guid));
    mutate_manifest(&nil_guid, |bytes| {
        bytes[MANIFEST_GUID_OFFSET..MANIFEST_MIN_BYTES].fill(0);
    });
    assert_open_rejected(&nil_guid);

    let truncated = temp.project("truncated");
    drop(create(&truncated));
    mutate_manifest(&truncated, |bytes| bytes.truncate(MANIFEST_MIN_BYTES - 1));
    assert_open_rejected(&truncated);

    let trailing = temp.project("trailing");
    drop(create(&trailing));
    mutate_manifest(&trailing, |bytes| {
        bytes.extend_from_slice(b"not-manifest-data");
    });
    assert_open_rejected(&trailing);
}

#[test]
fn open_rejects_a_missing_durable_scene() {
    let temp = TempRoot::new("missing-scene");
    let project_path = temp.project("project");
    drop(create(&project_path));
    fs::remove_file(scene(&project_path)).expect("remove private scene cell");

    assert_open_rejected(&project_path);
}

#[cfg(unix)]
#[test]
fn create_and_open_reject_a_symlink_project_root() {
    let temp = TempRoot::new("root-symlink");
    let real_create_root = temp.project("real-create-root");
    fs::create_dir(&real_create_root).expect("create real target root");
    let create_link = temp.project("create-link");
    symlink(&real_create_root, &create_link).expect("create project-root symlink");
    assert_create_rejected(&create_link);
    assert!(fs::read_dir(&real_create_root)
        .expect("read untouched target root")
        .next()
        .is_none());

    let real_open_root = temp.project("real-open-root");
    drop(create(&real_open_root));
    let manifest_before = fs::read(manifest(&real_open_root)).expect("read target manifest");
    let open_link = temp.project("open-link");
    symlink(&real_open_root, &open_link).expect("create open-root symlink");
    assert_open_rejected(&open_link);
    assert_eq!(
        fs::read(manifest(&real_open_root)).expect("read untouched target manifest"),
        manifest_before
    );
}

#[cfg(unix)]
#[test]
fn open_rejects_symlink_manifest_and_scene_directory() {
    let temp = TempRoot::new("internal-symlink");
    let outside = TempRoot::new("outside");

    let manifest_project = temp.project("manifest-project");
    drop(create(&manifest_project));
    let outside_manifest = outside.project("outside.project");
    fs::rename(manifest(&manifest_project), &outside_manifest)
        .expect("move manifest to private outside root");
    symlink(&outside_manifest, manifest(&manifest_project)).expect("symlink project manifest");
    assert_open_rejected(&manifest_project);

    let scene_project = temp.project("scene-project");
    drop(create(&scene_project));
    let scene_root = scene_project.join("scene");
    let outside_scene = outside.project("outside-scene");
    fs::rename(&scene_root, &outside_scene).expect("move scene to private outside root");
    symlink(&outside_scene, &scene_root).expect("symlink scene root");
    assert_open_rejected(&scene_project);
}
