//! Tests for the relocatable, offset-pointer containers (`reloc`, §24.5).
//!
//! The suite pins the fixed little-endian byte layout against hand-computed
//! oracles, exercises every bounds check, and — crucially for a *relocatable*
//! format — verifies that a blob stays valid after being `memcpy`'d to a fresh
//! allocation and after being embedded at an arbitrary offset inside a larger
//! buffer (the "copy the bytes anywhere, use in place" contract).

extern crate alloc;

use super::{
    OffsetPtr, OffsetSlice, Reloc, RelocError, RelocMap, RelocMapView, RelocVec, RelocVecView,
};
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Reloc scalar encode/decode round-trips (hand-checked little-endian oracles).
// ---------------------------------------------------------------------------

/// Encode a value into a fresh `Vec` of exactly `T::SIZE` bytes.
fn enc<T: Reloc>(v: T) -> Vec<u8> {
    let mut buf = alloc::vec![0u8; T::SIZE];
    v.encode(&mut buf);
    buf
}

#[test]
fn scalar_little_endian_oracles() {
    // Fixed, hand-computed little-endian byte strings.
    assert_eq!(enc(0x12u8), [0x12]);
    assert_eq!(enc(0x1234u16), [0x34, 0x12]);
    assert_eq!(enc(0x1234_5678u32), [0x78, 0x56, 0x34, 0x12]);
    assert_eq!(
        enc(0x0102_0304_0506_0708u64),
        [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    );
    assert_eq!(enc(-1i8), [0xFF]);
    assert_eq!(enc(-1i16), [0xFF, 0xFF]);
    assert_eq!(enc(-2i32), [0xFE, 0xFF, 0xFF, 0xFF]);
    // IEEE-754: 1.0f32 = 0x3F800000, 1.0f64 = 0x3FF0000000000000.
    assert_eq!(enc(1.0f32), [0x00, 0x00, 0x80, 0x3F]);
    assert_eq!(
        enc(1.0f64),
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F]
    );
    assert_eq!(enc(true), [0x01]);
    assert_eq!(enc(false), [0x00]);
}

#[test]
fn scalar_round_trips() {
    macro_rules! rt {
        ($t:ty, $($v:expr),*) => {
            $(
                let buf = enc::<$t>($v);
                assert_eq!(<$t>::decode(&buf), $v);
            )*
        };
    }
    rt!(u8, 0, 1, 255);
    rt!(u16, 0, 1, 40000, u16::MAX);
    rt!(u32, 0, 1, 0xDEAD_BEEF, u32::MAX);
    rt!(u64, 0, 1, 0x0123_4567_89AB_CDEF, u64::MAX);
    rt!(i8, 0, -1, 127, -128);
    rt!(i16, 0, -1, i16::MIN, i16::MAX);
    rt!(i32, 0, -1, i32::MIN, i32::MAX);
    rt!(i64, 0, -1, i64::MIN, i64::MAX);
    rt!(f32, 0.0, -0.0, 1.5, -2.25, f32::INFINITY);
    rt!(f64, 0.0, -0.0, 123.456_789_012_345, f64::NEG_INFINITY);
    rt!(bool, true, false);
}

#[test]
fn reloc_sizes_are_fixed() {
    assert_eq!(<u8 as Reloc>::SIZE, 1);
    assert_eq!(<u64 as Reloc>::SIZE, 8);
    assert_eq!(<bool as Reloc>::SIZE, 1);
    assert_eq!(<OffsetPtr<u32> as Reloc>::SIZE, 4);
    assert_eq!(<OffsetSlice<u32> as Reloc>::SIZE, 8);
}

// ---------------------------------------------------------------------------
// OffsetPtr / OffsetSlice resolution against a hand-laid blob.
// ---------------------------------------------------------------------------

#[test]
fn offset_ptr_resolves_forward_and_backward() {
    // blob: [u32 @0 = 0xAABBCCDD][u32 @4 = 0x11223344]
    let mut blob = Vec::new();
    blob.extend_from_slice(&0xAABB_CCDDu32.to_le_bytes());
    blob.extend_from_slice(&0x1122_3344u32.to_le_bytes());

    // A pointer *stored at field_pos 0* that targets byte 4 => offset +4.
    let p_fwd = OffsetPtr::<u32>::from_raw(4);
    assert_eq!(p_fwd.get(&blob, 0).unwrap(), Some(0x1122_3344));

    // A pointer stored at field_pos 4 that targets byte 0 => offset -4.
    let p_back = OffsetPtr::<u32>::from_raw(-4);
    assert_eq!(p_back.get(&blob, 4).unwrap(), Some(0xAABB_CCDD));
}

#[test]
fn offset_ptr_null_is_none() {
    let blob = [0u8; 8];
    let p = OffsetPtr::<u32>::NULL;
    assert!(p.is_null());
    assert_eq!(p.get(&blob, 0).unwrap(), None);
}

#[test]
fn offset_ptr_out_of_bounds_and_overflow() {
    let blob = [0u8; 8];
    // Target just past the end of the 8-byte blob.
    let p = OffsetPtr::<u32>::from_raw(8);
    assert_eq!(p.get(&blob, 4), Err(RelocError::OutOfBounds));
    // Negative resolved position overflows below zero.
    let p = OffsetPtr::<u32>::from_raw(-4);
    assert_eq!(p.get(&blob, 0), Err(RelocError::Overflow));
}

#[test]
fn offset_slice_indexing_and_bounds() {
    // Three u16 elements starting at byte 4, pointer stored at field_pos 0.
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0u8; 4]); // padding / header stand-in
    for v in [10u16, 20, 30] {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    let s = OffsetSlice::<u16>::from_raw(4, 3);
    assert_eq!(s.len(), 3);
    assert!(!s.is_empty());
    assert_eq!(s.get(&blob, 0, 0).unwrap(), Some(10));
    assert_eq!(s.get(&blob, 0, 1).unwrap(), Some(20));
    assert_eq!(s.get(&blob, 0, 2).unwrap(), Some(30));
    // Index past the end returns None (not an error).
    assert_eq!(s.get(&blob, 0, 3).unwrap(), None);

    // A slice claiming four elements runs one u16 past the blob.
    let bad = OffsetSlice::<u16>::from_raw(4, 4);
    assert_eq!(bad.get(&blob, 0, 3), Err(RelocError::OutOfBounds));

    // The empty slice is always fine.
    let empty = OffsetSlice::<u16>::EMPTY;
    assert!(empty.is_empty());
    assert_eq!(empty.get(&blob, 0, 0).unwrap(), None);
}

// ---------------------------------------------------------------------------
// RelocVec: build / read / iterate, plus relocation.
// ---------------------------------------------------------------------------

#[test]
fn reloc_vec_build_and_read() {
    let mut v = RelocVec::<u32>::new();
    assert!(v.is_empty());
    for x in [100u32, 200, 300, 400] {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    assert!(!v.is_empty());
    assert_eq!(v.get(0), Some(100));
    assert_eq!(v.get(3), Some(400));
    assert_eq!(v.get(4), None);
    let collected: Vec<u32> = v.iter().collect();
    assert_eq!(collected, alloc::vec![100, 200, 300, 400]);
}

#[test]
fn reloc_vec_from_slice_matches_oracle_bytes() {
    let v = RelocVec::<u16>::from_slice(&[1u16, 2, 3]);
    // Header (16 bytes): magic b"RVC1" | elem_size 2 | offset +8 | len 3,
    // then packed elements 1,2,3 as little-endian u16.
    let expected: Vec<u8> = {
        let mut b = Vec::new();
        b.extend_from_slice(&0x3143_5652u32.to_le_bytes()); // magic
        b.extend_from_slice(&2u32.to_le_bytes()); // elem_size
        b.extend_from_slice(&8i32.to_le_bytes()); // self-rel offset to data
        b.extend_from_slice(&3u32.to_le_bytes()); // len
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes());
        b
    };
    assert_eq!(v.as_bytes(), expected.as_slice());
}

#[test]
fn reloc_vec_survives_memcpy_to_fresh_allocation() {
    let v = RelocVec::<u32>::from_slice(&[7, 8, 9, 10, 11]);
    // "memcpy": copy the raw bytes into a brand-new, independently-allocated Vec.
    let copied: Vec<u8> = v.as_bytes().to_vec();
    drop(v); // the original backing store is gone.
    let view = RelocVecView::<u32>::new(&copied).unwrap();
    assert_eq!(view.len(), 5);
    let got: Vec<u32> = view.iter().collect();
    assert_eq!(got, alloc::vec![7, 8, 9, 10, 11]);
    assert_eq!(view.get(2), Some(9));
}

#[test]
fn reloc_vec_survives_embedding_at_arbitrary_offset() {
    let v = RelocVec::<i32>::from_slice(&[-1, -2, -3]);
    let blob = v.into_bytes();
    // Embed the blob 37 bytes into a larger buffer (simulating a packed asset
    // file). Because every internal reference is *self-relative*, the embedded
    // sub-blob must still parse without any fix-up.
    const PAD: usize = 37;
    let mut big = alloc::vec![0xA5u8; PAD];
    big.extend_from_slice(&blob);
    big.extend_from_slice(&[0x5Au8; 11]); // trailing padding
    let sub = &big[PAD..PAD + blob.len()];
    let view = RelocVecView::<i32>::new(sub).unwrap();
    let got: Vec<i32> = view.iter().collect();
    assert_eq!(got, alloc::vec![-1, -2, -3]);
}

#[test]
fn reloc_vec_from_bytes_rejects_corruption() {
    let v = RelocVec::<u32>::from_slice(&[1, 2, 3]);
    let good = v.as_bytes().to_vec();

    // Truncated below the header.
    assert_eq!(
        RelocVec::<u32>::from_bytes(&good[..8]).err(),
        Some(RelocError::OutOfBounds)
    );
    // Bad magic.
    let mut bad_magic = good.clone();
    bad_magic[0] ^= 0xFF;
    assert_eq!(
        RelocVec::<u32>::from_bytes(&bad_magic).err(),
        Some(RelocError::BadMagic)
    );
    // Opening a u32 blob as a u16 vector is a size mismatch.
    assert_eq!(
        RelocVec::<u16>::from_bytes(&good).err(),
        Some(RelocError::SizeMismatch)
    );
    // Header claims 3 elements but the element region is truncated.
    let truncated = &good[..good.len() - 2];
    assert_eq!(
        RelocVec::<u32>::from_bytes(truncated).err(),
        Some(RelocError::OutOfBounds)
    );
}

#[test]
fn reloc_vec_empty_round_trips() {
    let v = RelocVec::<u64>::new();
    assert_eq!(v.as_bytes().len(), 16);
    let copied = v.as_bytes().to_vec();
    let view = RelocVecView::<u64>::new(&copied).unwrap();
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
    assert_eq!(view.iter().count(), 0);
}

// ---------------------------------------------------------------------------
// RelocMap: sorted baking, dedup, binary-search lookup, relocation.
// ---------------------------------------------------------------------------

#[test]
fn reloc_map_sorts_and_binary_searches() {
    // Deliberately unsorted input.
    let m = RelocMap::<u32, u32>::from_pairs(&[(30, 300), (10, 100), (20, 200)]);
    assert_eq!(m.len(), 3);
    assert!(!m.is_empty());
    assert_eq!(m.get(&10), Some(100));
    assert_eq!(m.get(&20), Some(200));
    assert_eq!(m.get(&30), Some(300));
    assert_eq!(m.get(&15), None);
    assert_eq!(m.get(&40), None);
    assert!(m.contains_key(&20));
    assert!(!m.contains_key(&99));

    // Iteration is in ascending key order regardless of input order.
    let entries: Vec<(u32, u32)> = m.iter().collect();
    assert_eq!(entries, alloc::vec![(10, 100), (20, 200), (30, 300)]);
}

#[test]
fn reloc_map_last_duplicate_wins() {
    // Two pairs with key 5; the later one must win (map-insert semantics).
    let m = RelocMap::<u32, u32>::from_pairs(&[(5, 1), (7, 70), (5, 999), (5, 42)]);
    assert_eq!(m.len(), 2); // keys 5 and 7
    assert_eq!(m.get(&5), Some(42));
    assert_eq!(m.get(&7), Some(70));
}

#[test]
fn reloc_map_survives_memcpy() {
    let m = RelocMap::<u32, i32>::from_pairs(&[(1, -10), (2, -20), (3, -30), (4, -40)]);
    let copied = m.as_bytes().to_vec();
    drop(m);
    let view = RelocMapView::<u32, i32>::new(&copied).unwrap();
    assert_eq!(view.len(), 4);
    assert_eq!(view.get(&3), Some(-30));
    assert_eq!(view.get(&4), Some(-40));
    assert_eq!(view.get(&5), None);
    let entries: Vec<(u32, i32)> = view.iter().collect();
    assert_eq!(entries, alloc::vec![(1, -10), (2, -20), (3, -30), (4, -40)]);
}

#[test]
fn reloc_map_survives_embedding_at_arbitrary_offset() {
    let m = RelocMap::<u16, u16>::from_pairs(&[(100, 1), (200, 2), (300, 3)]);
    let blob = m.into_bytes();
    const PAD: usize = 23;
    let mut big = alloc::vec![0u8; PAD];
    big.extend_from_slice(&blob);
    big.extend_from_slice(&[0xFFu8; 5]);
    let sub = &big[PAD..PAD + blob.len()];
    let view = RelocMapView::<u16, u16>::new(sub).unwrap();
    assert_eq!(view.get(&200), Some(2));
    let entries: Vec<(u16, u16)> = view.iter().collect();
    assert_eq!(entries, alloc::vec![(100, 1), (200, 2), (300, 3)]);
}

#[test]
fn reloc_map_from_bytes_rejects_corruption() {
    let m = RelocMap::<u32, u32>::from_pairs(&[(1, 10), (2, 20)]);
    let good = m.as_bytes().to_vec();

    assert_eq!(
        RelocMap::<u32, u32>::from_bytes(&good[..16]).err(),
        Some(RelocError::OutOfBounds)
    );
    let mut bad_magic = good.clone();
    bad_magic[0] ^= 0xFF;
    assert_eq!(
        RelocMap::<u32, u32>::from_bytes(&bad_magic).err(),
        Some(RelocError::BadMagic)
    );
    // Opening with a mismatched value size.
    assert_eq!(
        RelocMap::<u32, u16>::from_bytes(&good).err(),
        Some(RelocError::SizeMismatch)
    );
    // Truncated value region.
    let truncated = &good[..good.len() - 1];
    assert_eq!(
        RelocMap::<u32, u32>::from_bytes(truncated).err(),
        Some(RelocError::OutOfBounds)
    );
}

#[test]
fn reloc_map_empty() {
    let m = RelocMap::<u32, u32>::from_pairs(&[]);
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
    assert_eq!(m.get(&1), None);
    let copied = m.as_bytes().to_vec();
    let view = RelocMapView::<u32, u32>::new(&copied).unwrap();
    assert!(view.is_empty());
    assert_eq!(view.iter().count(), 0);
}

#[test]
fn reloc_error_messages_are_distinct() {
    let msgs = [
        RelocError::OutOfBounds.as_str(),
        RelocError::BadMagic.as_str(),
        RelocError::SizeMismatch.as_str(),
        RelocError::Overflow.as_str(),
    ];
    // All four descriptions are distinct and non-empty.
    for (i, a) in msgs.iter().enumerate() {
        assert!(!a.is_empty());
        for b in &msgs[i + 1..] {
            assert_ne!(a, b);
        }
    }
}
