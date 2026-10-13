//! Black-box hot-reload regression tests.
//!
//! This test intentionally stays outside `src/server.rs`: it exercises the
//! public loading contract with a source overlay and proves that a failed
//! reload does not destroy the last known-good payload.

#![cfg(feature = "std")]

extern crate alloc;

use alloc::sync::Arc;
use prism_asset::{
    Asset, AssetLoader, AssetPath, AssetReader, AssetServer, AssetTypeId, LoadContext, LoadError,
    MemSource, ReadError,
};

#[derive(Debug, PartialEq, Eq)]
struct ValueAsset {
    value: u8,
}

impl Asset for ValueAsset {
    const TYPE_NAME: &'static str = "prism_asset::tests::hot_reload_correctness::ValueAsset";
}

struct ValueLoader;

impl AssetLoader for ValueLoader {
    type Asset = ValueAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["value"]
    }

    fn load(&self, bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
        bytes
            .first()
            .copied()
            .map(|value| ValueAsset { value })
            .ok_or_else(|| LoadError::Malformed("empty value asset".into()))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct OtherValueAsset;

impl Asset for OtherValueAsset {
    const TYPE_NAME: &'static str = "prism_asset::tests::hot_reload_correctness::OtherValueAsset";
}

struct OtherValueLoader;

impl AssetLoader for OtherValueLoader {
    type Asset = OtherValueAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["value"]
    }

    fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
        Ok(OtherValueAsset)
    }
}

struct FailingReader;

impl AssetReader for FailingReader {
    fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
        Err(ReadError::Io("simulated reload read failure".into()))
    }

    fn metadata(&self, _path: &str) -> Result<prism_asset::AssetMeta, ReadError> {
        Err(ReadError::Io("simulated reload metadata failure".into()))
    }
}

struct PanickingReader;

impl AssetReader for PanickingReader {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if path == "panic.value" {
            panic!("simulated reader panic")
        }
        Err(ReadError::NotFound)
    }

    fn metadata(&self, _path: &str) -> Result<prism_asset::AssetMeta, ReadError> {
        Ok(prism_asset::AssetMeta {
            size: 0,
            modified_ms: None,
            is_dir: false,
        })
    }
}

#[derive(Debug)]
struct RootAsset;

impl Asset for RootAsset {
    const TYPE_NAME: &'static str = "prism_asset::tests::hot_reload_correctness::RootAsset";
}

#[derive(Debug, PartialEq, Eq)]
struct DerivedAsset(u8);

impl Asset for DerivedAsset {
    const TYPE_NAME: &'static str = "prism_asset::tests::hot_reload_correctness::DerivedAsset";
}

struct DerivedLoader;

impl AssetLoader for DerivedLoader {
    type Asset = DerivedAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["derived"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
        ctx.add_typed_dependency(
            AssetPath::parse("leaf.dep"),
            AssetTypeId::of::<DependencyAsset>(),
        );
        bytes
            .first()
            .copied()
            .map(DerivedAsset)
            .ok_or_else(|| LoadError::Malformed("empty derived asset".into()))
    }
}

#[derive(Debug)]
struct DependencyAsset;

impl Asset for DependencyAsset {
    const TYPE_NAME: &'static str = "prism_asset::tests::hot_reload_correctness::DependencyAsset";
}

struct RootLoader;

impl AssetLoader for RootLoader {
    type Asset = RootAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["root"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
        let path = match bytes.first() {
            Some(b'a') => "a.dep",
            Some(b'b') => "b.dep",
            _ => return Err(LoadError::Malformed("unknown dependency selector".into())),
        };
        ctx.add_typed_dependency(AssetPath::parse(path), AssetTypeId::of::<DependencyAsset>());
        Ok(RootAsset)
    }
}

struct DependencyLoader;

impl AssetLoader for DependencyLoader {
    type Asset = DependencyAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["dep"]
    }

    fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
        Ok(DependencyAsset)
    }
}

#[test]
fn failed_reload_keeps_last_known_good_value_and_ready_state() {
    let server = AssetServer::new();
    server.register_loader(ValueLoader);

    let mut source = MemSource::new();
    source.insert("hero.value", [7_u8]);
    server.mount("base", Arc::new(source), 0);

    let handle = server.load::<ValueAsset>(AssetPath::parse("hero.value"));
    let id = handle.id();
    assert_eq!(
        server.load_state(handle.untyped_id()),
        prism_asset::LoadState::Loaded
    );
    assert_eq!(server.with_asset(id, |asset| asset.value), Some(7));

    // The higher-priority mount owns the path during reload and fails hard.
    // This models an I/O or decode failure without changing the server API.
    server.mount("reload-failure", Arc::new(FailingReader), 100);
    let reloaded = server
        .reload("hero.value")
        .expect("failed reload still returns the resident handle");

    assert_eq!(reloaded.id(), handle.untyped_id());
    assert_eq!(
        server.load_state(handle.untyped_id()),
        prism_asset::LoadState::Loaded
    );
    assert_eq!(server.with_asset(id, |asset| asset.value), Some(7));
}

#[test]
fn successful_reload_replaces_the_retained_dependency_closure() {
    let server = AssetServer::new();
    server.register_loader(RootLoader);
    server.register_loader(DependencyLoader);

    let mut source = MemSource::new();
    source.insert("hero.root", [b'a']);
    source.insert("a.dep", [0_u8]);
    source.insert("b.dep", [0_u8]);
    server.mount("mem", Arc::new(source), 0);

    let root = server.load::<RootAsset>("hero.root");
    assert_eq!(server.total_assets(), 2);

    let mut revised_source = MemSource::new();
    revised_source.insert("hero.root", [b'b']);
    revised_source.insert("a.dep", [0_u8]);
    revised_source.insert("b.dep", [0_u8]);
    server.mount("mem", Arc::new(revised_source), 0);
    assert_eq!(
        server.reload("hero.root").map(|handle| handle.id()),
        Some(root.untyped_id())
    );

    // The new dependency is retained and the old one is no longer in the
    // owner's closure. A zero-grace collection reclaims exactly the old leaf.
    assert_eq!(server.total_assets(), 3);
    assert_eq!(server.collect_releases(), 1);
    assert_eq!(server.total_assets(), 2);
    assert_eq!(
        server.load_state(root.untyped_id()),
        prism_asset::LoadState::Loaded
    );
}

#[test]
fn typed_path_conflict_returns_failed_handle_instead_of_panicking() {
    let server = AssetServer::new();
    server.register_loader(ValueLoader);
    server.register_loader(OtherValueLoader);
    let mut source = MemSource::new();
    source.insert("same.value", [7_u8]);
    server.mount("mem", Arc::new(source), 0);

    let first = server.load::<ValueAsset>("same.value");
    assert_eq!(server.with_asset(first.id(), |asset| asset.value), Some(7));
    let conflicting = server.load::<OtherValueAsset>("same.value");
    assert_eq!(
        server.load_state(conflicting.untyped_id()),
        prism_asset::LoadState::Failed(
            server
                .load_state(conflicting.untyped_id())
                .error()
                .expect("type conflict is recorded")
        )
    );
}

#[test]
fn reader_panic_fails_one_load_and_does_not_poison_server() {
    let server = AssetServer::new();
    server.register_loader(ValueLoader);
    server.mount("panic", Arc::new(PanickingReader), 100);
    let failed = server.load::<ValueAsset>("panic.value");
    assert!(server.load_state(failed.untyped_id()).is_failed());

    let mut source = MemSource::new();
    source.insert("ok.value", [9_u8]);
    server.mount("base", Arc::new(source), 0);
    let ok = server.load::<ValueAsset>("ok.value");
    assert_eq!(server.with_asset(ok.id(), |asset| asset.value), Some(9));
}

#[test]
fn failed_dependency_fails_parent_asset() {
    let server = AssetServer::new();
    server.register_loader(RootLoader);
    server.register_loader(DependencyLoader);
    let mut source = MemSource::new();
    source.insert("hero.root", [b'a']);
    server.mount("mem", Arc::new(source), 0);

    let root = server.load::<RootAsset>("hero.root");
    assert!(server.load_state(root.untyped_id()).is_failed());
    assert!(server.recursive_state(root.untyped_id()).is_failed());
}

#[test]
fn reloading_dependency_reassembles_its_loaded_dependents_first() {
    let server = AssetServer::new();
    server.register_loader(DerivedLoader);
    server.register_loader(DependencyLoader);
    let mut source = MemSource::new();
    source.insert("hero.derived", [1_u8]);
    source.insert("leaf.dep", [0_u8]);
    server.mount("mem", Arc::new(source), 0);

    let derived = server.load::<DerivedAsset>("hero.derived");
    let mut revised_source = MemSource::new();
    revised_source.insert("hero.derived", [2_u8]);
    revised_source.insert("leaf.dep", [1_u8]);
    server.mount("mem", Arc::new(revised_source), 0);

    let leaf = server
        .load_untyped("leaf.dep")
        .expect("dependency remains resident through its dependent");
    assert_eq!(server.reload("leaf.dep").map(|h| h.id()), Some(leaf.id()));
    assert_eq!(server.with_asset(derived.id(), |asset| asset.0), Some(2));
}
