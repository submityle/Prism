//! Verified readers retain the checked file and expose only its admitted bytes.

use prism_asset_build_core::cas::VerifiedCasReader;
use prism_asset_build_core::{CasError, ContentAddressedStore, ContentHash};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use tempfile::tempdir;

const BLOCK: usize = 64 * 1024;

fn pattern(offset: usize) -> u8 {
    ((offset * 17 + offset / BLOCK) % 251) as u8
}

fn assert_io<T>(result: Result<T, CasError>, kind: io::ErrorKind) {
    match result {
        Err(CasError::Io(error)) => assert_eq!(error.kind(), kind),
        Err(other) => panic!("expected {kind:?}, got {other:?}"),
        Ok(_) => panic!("expected {kind:?}, got success"),
    }
}

#[test]
fn large_cross_block_input_streams_without_an_owned_payload_buffer() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let length = 16 * 1024 * 1024 + 37;
    let mut writer = cas.begin_write(length as u64).unwrap();
    let mut block = [0; BLOCK];
    let mut offset = 0;
    while offset < length {
        let count = block.len().min(length - offset);
        for (index, byte) in block[..count].iter_mut().enumerate() {
            *byte = pattern(offset + index);
        }
        writer.write_all(&block[..count]).unwrap();
        offset += count;
    }
    let stored = writer.finish().unwrap();
    let mut reader: VerifiedCasReader = cas
        .open_verified(stored.content_hash, length as u64)
        .unwrap();
    assert_eq!(reader.content_hash(), stored.content_hash);
    assert_eq!(reader.byte_len(), length as u64);
    assert_eq!(reader.stream_position().unwrap(), 0);
    // An odd request size crosses verification blocks and leaves a short tail.
    let mut output = [0; 8191];
    offset = 0;
    while offset < length {
        let expected = output.len().min(length - offset);
        let count = reader.read(&mut output).unwrap();
        assert_eq!(count, expected);
        assert!(output[..count]
            .iter()
            .enumerate()
            .all(|(index, &byte)| byte == pattern(offset + index)));
        offset += count;
    }
    assert_eq!(reader.read(&mut output).unwrap(), 0);
    assert_eq!(reader.stream_position().unwrap(), length as u64);
    reader.seek(SeekFrom::Start((BLOCK - 3) as u64)).unwrap();
    reader.read_exact(&mut output[..11]).unwrap();
    assert_eq!(
        output[..11],
        std::array::from_fn::<_, 11, _>(|i| pattern(BLOCK - 3 + i))
    );
}

#[test]
fn empty_blob_accepts_zero_budget_and_only_zero_seeks() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = cas.put(&[]).unwrap();
    let mut reader = cas.open_verified(hash, 0).unwrap();
    assert_eq!(reader.content_hash(), ContentHash::digest(&[]));
    assert_eq!(reader.byte_len(), 0);
    let mut output = [0xAB; 8];
    assert_eq!(reader.read(&mut output).unwrap(), 0);
    assert_eq!(reader.read(&mut []).unwrap(), 0);
    assert_eq!(output, [0xAB; 8]);
    for seek in [SeekFrom::Start(0), SeekFrom::Current(0), SeekFrom::End(0)] {
        assert_eq!(reader.seek(seek).unwrap(), 0);
    }
    for seek in [SeekFrom::Start(1), SeekFrom::Current(-1), SeekFrom::End(1)] {
        assert_eq!(
            reader.seek(seek).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(reader.stream_position().unwrap(), 0);
    }
}

#[test]
fn exact_budget_succeeds_and_smaller_budgets_fail_before_hashing() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let bytes = b"bounded input";
    let hash = cas.put(bytes).unwrap();
    assert_eq!(
        cas.open_verified(hash, bytes.len() as u64)
            .unwrap()
            .byte_len(),
        bytes.len() as u64
    );
    assert_io(cas.open_verified(hash, 0), io::ErrorKind::InvalidInput);
    assert_io(
        cas.open_verified(hash, bytes.len() as u64 - 1),
        io::ErrorKind::InvalidInput,
    );
    // A corrupt occupant larger than the caller's budget is rejected on its
    // actual opened-file length, before hashing can report corruption.
    fs::write(cas.blob_path(hash), b"a much larger corrupt occupant").unwrap();
    assert_io(
        cas.open_verified(hash, bytes.len() as u64),
        io::ErrorKind::InvalidInput,
    );
    assert!(matches!(
        cas.open_verified(hash, u64::MAX),
        Err(CasError::Corrupt { .. })
    ));
}

#[test]
fn verification_checks_first_middle_last_and_changed_file_lengths() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let bytes = (0..(3 * BLOCK + 19)).map(pattern).collect::<Vec<_>>();
    let hash = cas.put(&bytes).unwrap();
    for index in [0, BLOCK + 7, bytes.len() - 1] {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 1;
        fs::write(cas.blob_path(hash), &corrupt).unwrap();
        assert!(matches!(
            cas.open_verified(hash, bytes.len() as u64),
            Err(CasError::Corrupt { expected, actual })
                if expected == hash && actual == ContentHash::digest(&corrupt)
        ));
    }
    for length in [0, BLOCK, bytes.len() - 1, bytes.len() + 1] {
        let mut corrupt = bytes.clone();
        corrupt.resize(length, 0xFF);
        fs::write(cas.blob_path(hash), &corrupt).unwrap();
        assert!(matches!(
            cas.open_verified(hash, u64::MAX),
            Err(CasError::Corrupt { expected, actual })
                if expected == hash && actual == ContentHash::digest(&corrupt)
        ));
    }
}

#[test]
fn seeks_and_reads_obey_logical_eof_and_invalid_seeks_preserve_position() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = cas.put(b"0123456789").unwrap();
    let mut reader = cas.open_verified(hash, 10).unwrap();
    assert_eq!(reader.seek(SeekFrom::Start(3)).unwrap(), 3);
    let mut pair = [0; 2];
    reader.read_exact(&mut pair).unwrap();
    assert_eq!(&pair, b"34");
    assert_eq!(reader.seek(SeekFrom::Current(-2)).unwrap(), 3);
    for seek in [
        SeekFrom::Start(11),
        SeekFrom::Start(u64::MAX),
        SeekFrom::Current(-4),
        SeekFrom::Current(i64::MIN),
        SeekFrom::Current(i64::MAX),
        SeekFrom::End(1),
        SeekFrom::End(-11),
        SeekFrom::End(i64::MIN),
        SeekFrom::End(i64::MAX),
    ] {
        assert_eq!(
            reader.seek(seek).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(reader.stream_position().unwrap(), 3);
    }
    reader.read_exact(&mut pair).unwrap();
    assert_eq!(&pair, b"34");
    assert_eq!(reader.seek(SeekFrom::End(-2)).unwrap(), 8);
    let mut tail = [0xFF; 8];
    assert_eq!(reader.read(&mut tail).unwrap(), 2);
    assert_eq!(&tail[..2], b"89");
    assert_eq!(tail[2..], [0xFF; 6]);
    assert_eq!(reader.read(&mut tail).unwrap(), 0);
    assert_eq!(reader.seek(SeekFrom::End(0)).unwrap(), 10);
    assert_eq!(reader.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(reader.seek(SeekFrom::End(-10)).unwrap(), 0);
    let mut whole = [0; 10];
    reader.read_exact(&mut whole).unwrap();
    assert_eq!(&whole, b"0123456789");
}

#[test]
fn truncation_after_open_returns_short_read_error_instead_of_successful_eof() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = cas.put(b"0123456789").unwrap();
    let mut reader = cas.open_verified(hash, 10).unwrap();
    let mut prefix = [0; 3];
    reader.read_exact(&mut prefix).unwrap();
    File::options()
        .write(true)
        .open(cas.blob_path(hash))
        .unwrap()
        .set_len(5)
        .unwrap();
    // Deliberately violate CAS immutability: two bytes are delivered before
    // the underlying file ends, and the logical position tracks that prefix.
    let mut output = [0xFF; 6];
    assert_eq!(
        reader.read_exact(&mut output).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(&output[..2], b"34");
    assert_eq!(output[2..], [0xFF; 4]);
    assert_eq!(reader.stream_position().unwrap(), 5);
    assert_eq!(reader.read(&mut []).unwrap(), 0);
    assert_eq!(
        reader.read(&mut output).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(reader.seek(SeekFrom::Start(4)).unwrap(), 4);
    let mut one = [0];
    reader.read_exact(&mut one).unwrap();
    assert_eq!(&one, b"4");
}

#[test]
fn append_after_open_cannot_expand_the_verified_logical_range() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = cas.put(b"verified").unwrap();
    let mut reader = cas.open_verified(hash, 8).unwrap();
    OpenOptions::new()
        .append(true)
        .open(cas.blob_path(hash))
        .unwrap()
        .write_all(b"unverified suffix")
        .unwrap();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"verified");
    assert_eq!(reader.byte_len(), 8);
    assert_eq!(reader.seek(SeekFrom::End(0)).unwrap(), 8);
    assert_eq!(
        reader.seek(SeekFrom::Start(9)).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(reader.read(&mut [0; 1]).unwrap(), 0);
}

#[test]
fn missing_and_non_regular_blob_paths_are_rejected() {
    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = ContentHash::digest(b"absent");
    assert_io(cas.open_verified(hash, 100), io::ErrorKind::NotFound);
    let blob = cas.blob_path(hash);
    fs::create_dir_all(&blob).unwrap();
    assert_io(cas.open_verified(hash, 100), io::ErrorKind::InvalidData);
}

#[cfg(unix)]
#[test]
fn opened_reader_keeps_the_verified_inode_after_atomic_path_replacement() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    let hash = cas.put(b"original").unwrap();
    let path = cas.blob_path(hash);
    let original_inode = fs::metadata(&path).unwrap().ino();
    let mut reader = cas.open_verified(hash, 8).unwrap();
    let mut replacement = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    replacement.write_all(b"replaced").unwrap();
    fs::rename(replacement.path(), &path).unwrap();
    assert_ne!(fs::metadata(&path).unwrap().ino(), original_inode);
    assert_eq!(fs::read(&path).unwrap(), b"replaced");
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"original");
    reader.seek(SeekFrom::End(-4)).unwrap();
    let mut tail = [0; 4];
    reader.read_exact(&mut tail).unwrap();
    assert_eq!(&tail, b"inal");
    assert_eq!(reader.content_hash(), hash);
    assert!(matches!(
        cas.open_verified(hash, 8),
        Err(CasError::Corrupt { .. })
    ));
}

#[cfg(unix)]
#[test]
fn symlink_blob_and_shard_and_root_are_rejected_by_ordinary_path_checks() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path().join("cas")).unwrap();
    let hash = cas.put(b"actual payload").unwrap();
    let path = cas.blob_path(hash);
    let real_blob = directory.path().join("real-blob");
    fs::rename(&path, &real_blob).unwrap();
    symlink(&real_blob, &path).unwrap();
    assert_io(cas.open_verified(hash, 100), io::ErrorKind::InvalidData);
    fs::remove_file(&path).unwrap();
    fs::rename(&real_blob, &path).unwrap();

    let shard = path.parent().unwrap();
    let real_shard = directory.path().join("real-shard");
    fs::rename(shard, &real_shard).unwrap();
    symlink(&real_shard, shard).unwrap();
    assert_io(cas.open_verified(hash, 100), io::ErrorKind::InvalidData);
    fs::remove_file(shard).unwrap();
    fs::rename(&real_shard, shard).unwrap();

    let root_alias = directory.path().join("root-alias");
    symlink(cas.root(), &root_alias).unwrap();
    let alias = ContentAddressedStore::open(&root_alias).unwrap();
    assert_io(alias.open_verified(hash, 100), io::ErrorKind::InvalidData);
}
