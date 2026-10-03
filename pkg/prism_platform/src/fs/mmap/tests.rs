//! Unit tests for the M4 memory-mapping layer.
//!
//! On the host they run on these exercise the real OS backend end-to-end: map
//! a file read-only and compare bytes, map a sub-range at a non-aligned offset,
//! map read-write and observe the change land in the file, and the empty-file
//! guard.

use std::io::Write as _;

use super::{mmap_supported, Mmap, MmapError, MmapMut};
use crate::fs::file::OpenOptions;

/// Create a uniquely-named scratch file under the OS temp dir with `contents`.
fn scratch(tag: &str, contents: &[u8]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("prism_mmap_{tag}_{nanos}_{:?}.bin", std::thread::current().id()));
    let mut f = std::fs::File::create(&path).expect("create scratch file");
    f.write_all(contents).expect("write scratch file");
    f.sync_all().expect("sync scratch file");
    path
}

#[test]
fn supported_is_true_on_this_host() {
    // Every CI/dev target this suite runs on (Linux/macOS/Windows) has real
    // mmap; the fallback is wasm-only.
    assert!(mmap_supported(), "expected a zero-copy mmap backend on this host");
}

#[test]
fn read_only_maps_whole_file() {
    let data: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
    let path = scratch("ro", &data);

    let map = Mmap::map(&path).expect("map read-only");
    assert_eq!(map.len(), data.len());
    assert!(!map.is_empty());
    assert_eq!(&map[..], &data[..]);
    assert_eq!(map.as_slice(), &data[..]);

    drop(map);
    std::fs::remove_file(&path).ok();
}

#[test]
fn read_only_sub_range_at_unaligned_offset() {
    let data: Vec<u8> = (0u8..200).collect();
    let path = scratch("sub", &data);

    // Offset 37 is deliberately not page-aligned; the backend must align the
    // underlying mapping internally and still expose exactly [37, 37+64).
    let file = OpenOptions::new().read(true).open(&path).expect("open");
    let map = Mmap::from_file(file, 37, 64).expect("map sub-range");
    assert_eq!(map.len(), 64);
    assert_eq!(&map[..], &data[37..37 + 64]);

    drop(map);
    std::fs::remove_file(&path).ok();
}

#[test]
fn read_write_mapping_persists_to_file() {
    let data = vec![0u8; 4096];
    let path = scratch("rw", &data);

    {
        let mut map = MmapMut::map(&path).expect("map read-write");
        assert_eq!(map.len(), 4096);
        for (i, byte) in map.as_mut_slice().iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        map.flush().expect("flush dirty pages");
    }

    let readback = std::fs::read(&path).expect("read back file");
    let expected: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    assert_eq!(readback, expected);

    std::fs::remove_file(&path).ok();
}

#[test]
fn empty_file_is_rejected() {
    let path = scratch("empty", &[]);
    let err = Mmap::map(&path).expect_err("empty file cannot be mapped");
    assert!(matches!(err, MmapError::Empty), "got {err:?}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn out_of_range_is_rejected() {
    let path = scratch("oor", &[1, 2, 3, 4]);
    let file = OpenOptions::new().read(true).open(&path).expect("open");
    let err = Mmap::from_file(file, 2, 1000).expect_err("range past EOF rejected");
    assert!(matches!(err, MmapError::InvalidArgument), "got {err:?}");
    std::fs::remove_file(&path).ok();
}
