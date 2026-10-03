//! Unit tests for the filesystem facade. Each test provisions its own uniquely
//! named scratch path under [`std::env::temp_dir`] and removes it on exit.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fs::{self, dirs, path};

/// Monotonic counter so concurrent tests never share a scratch name.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Build a unique scratch path (not created) under the system temp dir.
fn unique_temp(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("prism_platform_test_{}_{}_{}", std::process::id(), tag, n);
    dirs::temp_dir().join(name)
}

/// RAII guard that removes a path (file or directory tree) when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = fs::remove_dir_all(&self.0);
        } else if self.0.exists() {
            let _ = fs::file::remove_file(&self.0);
        }
    }
}

#[test]
fn file_write_read_roundtrip() {
    let file = unique_temp("roundtrip.bin");
    let _guard = Scratch(file.clone());

    let payload = b"prism platform fs \x00\x01\x02 round trip";
    fs::file::write(&file, payload).expect("write");
    assert!(fs::file::exists(&file).expect("exists"));

    let read = fs::file::read(&file).expect("read");
    assert_eq!(read, payload);

    let meta = fs::file::metadata(&file).expect("metadata");
    assert!(meta.is_file);
    assert!(!meta.is_dir);
    assert_eq!(meta.len, payload.len() as u64);
    assert!(meta.modified.is_some());

    fs::file::append(&file, b"!").expect("append");
    let text = fs::file::read(&file).expect("read bytes len");
    assert_eq!(text.len(), payload.len() + 1);

    fs::file::remove_file(&file).expect("remove");
    assert!(!fs::file::exists(&file).expect("exists after remove"));
}

#[test]
fn file_copy_and_rename() {
    let root = unique_temp("copytree");
    let _guard = Scratch(root.clone());
    fs::create_dir_all(&root).expect("create root");

    let a = root.join("a.txt");
    let b = root.join("b.txt");
    let c = root.join("c.txt");
    fs::file::write(&a, b"hello").expect("write a");

    let copied = fs::file::copy(&a, &b).expect("copy");
    assert_eq!(copied, 5);
    assert_eq!(fs::file::read(&b).expect("read b"), b"hello");

    fs::file::rename(&b, &c).expect("rename");
    assert!(!fs::file::exists(&b).expect("b gone"));
    assert_eq!(fs::file::read(&c).expect("read c"), b"hello");
}

#[test]
fn dir_create_list_walk_remove() {
    let root = unique_temp("tree");
    let _guard = Scratch(root.clone());

    let nested = root.join("sub").join("deep");
    fs::create_dir_all(&nested).expect("create_dir_all");
    fs::file::write(root.join("top.txt"), b"t").expect("top file");
    fs::file::write(nested.join("leaf.txt"), b"l").expect("leaf file");

    let mut top = fs::read_dir(&root).expect("read_dir");
    top.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].name, "sub");
    assert!(top[0].is_dir);
    assert_eq!(top[1].name, "top.txt");
    assert!(!top[1].is_dir);

    let all = fs::walk(&root).expect("walk");
    let names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"sub"));
    assert!(names.contains(&"deep"));
    assert!(names.contains(&"leaf.txt"));
    assert!(names.contains(&"top.txt"));
    // Four entries total: sub/, sub/deep/, sub/deep/leaf.txt, top.txt.
    assert_eq!(all.len(), 4);

    fs::remove_dir_all(&root).expect("remove_dir_all");
    assert!(!fs::file::exists(&root).expect("root gone"));
}

#[test]
fn path_lexical_normalization() {
    assert_eq!(path::normalize("a/./b/../c"), PathBuf::from("a/c"));
    assert_eq!(path::normalize("a/b/../../.."), PathBuf::from(".."));
    assert_eq!(path::normalize("./"), PathBuf::from("."));
    assert_eq!(path::normalize("foo/bar/.."), PathBuf::from("foo"));
    // Absolute paths can never ascend above the root.
    assert_eq!(path::normalize("/a/../.."), PathBuf::from("/"));
    assert_eq!(path::normalize("/a/b/./c/../d"), PathBuf::from("/a/b/d"));
}

#[test]
fn path_component_helpers() {
    assert_eq!(path::extension("dir/file.tar.gz").as_deref(), Some("gz"));
    assert_eq!(
        path::file_stem("dir/file.tar.gz").as_deref(),
        Some("file.tar")
    );
    assert_eq!(path::parent("dir/file.txt"), Some(PathBuf::from("dir")));
    assert!(path::is_absolute("/etc"));
    assert!(!path::is_absolute("etc"));
    assert_eq!(path::join("a/b", "c"), PathBuf::from("a/b/c"));
    assert_eq!(path::join("a/b", "/c"), PathBuf::from("/c"));
}

#[test]
fn standard_dirs_are_resolvable() {
    // On the three desktop targets these must resolve to a non-empty path.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    {
        for (label, dir) in [
            ("home", dirs::home_dir()),
            ("config", dirs::config_dir()),
            ("data", dirs::data_dir()),
            ("cache", dirs::cache_dir()),
        ] {
            let path = dir.unwrap_or_else(|e| panic!("{label}_dir: {e}"));
            assert!(!path.as_os_str().is_empty(), "{label}_dir empty");
        }
    }

    assert!(!dirs::temp_dir().as_os_str().is_empty());
    assert!(!dirs::current_dir().expect("cwd").as_os_str().is_empty());
    assert!(!dirs::current_exe().expect("exe").as_os_str().is_empty());
}
