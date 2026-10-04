//! Oracle-backed tests for the [`guard`](super) hardening containers.
//!
//! [`GuardedBuffer`] is checked against a plain shadow `Vec<u8>` payload: for
//! any in-bounds write/read sequence the buffer must agree with the shadow, and
//! every out-of-bounds or post-free access must be reported as the matching
//! [`GuardError`]. [`GuardedPool`] is checked against a shadow map of live
//! handles so that use-after-free, double free, and dangling handles are all
//! surfaced exactly when the oracle says the handle is invalid.

use super::canary::GuardedBuffer;
use super::pool::{GuardHandle, GuardedPool};
use super::{GuardConfig, GuardError};

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use crate::hash::HashMap;

// ----- GuardedBuffer -------------------------------------------------------

#[test]
fn fresh_buffer_is_zeroed_and_valid() {
    let buf = GuardedBuffer::new(8);
    assert_eq!(buf.len(), 8);
    assert!(!buf.is_empty());
    assert!(!buf.is_freed());
    assert_eq!(buf.payload().unwrap(), &[0u8; 8]);
    assert_eq!(buf.validate(), Ok(()));
}

#[test]
fn write_and_read_round_trip_matches_shadow() {
    let mut buf = GuardedBuffer::new(16);
    let mut shadow = vec![0u8; 16];
    let mut state: u64 = 0xDEAD_BEEF_1234_5678;
    for _ in 0..2000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let off = (state as usize) % 16;
        let max = 16 - off;
        let n = ((state >> 8) as usize % max) + 1;
        let byte = (state >> 16) as u8;
        let src = vec![byte; n];
        assert_eq!(buf.write_at(off, &src), Ok(()));
        shadow[off..off + n].copy_from_slice(&src);
        assert_eq!(buf.payload().unwrap(), shadow.as_slice());
        let mut out = vec![0u8; n];
        assert_eq!(buf.read_at(off, &mut out), Ok(()));
        assert_eq!(out, src);
        // Canaries stay intact through every in-bounds write.
        assert_eq!(buf.validate(), Ok(()));
    }
}

#[test]
fn write_past_payload_is_overflow() {
    let mut buf = GuardedBuffer::new(4);
    assert_eq!(buf.write_at(2, &[1, 2, 3]), Err(GuardError::Overflow));
    assert_eq!(buf.write_at(4, &[1]), Err(GuardError::Overflow));
    assert_eq!(buf.write_at(usize::MAX, &[1]), Err(GuardError::Overflow));
    // A failed write must not have touched the payload or canaries.
    assert_eq!(buf.payload().unwrap(), &[0u8; 4]);
    assert_eq!(buf.validate(), Ok(()));
}

#[test]
fn read_past_payload_is_overflow() {
    let buf = GuardedBuffer::new(4);
    let mut out = [0u8; 3];
    assert_eq!(buf.read_at(2, &mut out), Err(GuardError::Overflow));
}

#[test]
fn trailing_canary_corruption_is_overflow() {
    let mut buf = GuardedBuffer::new(4);
    let last = buf.block().len() - 1;
    buf.block_mut()[last] ^= 0xFF; // scribble into the trailing redzone
    assert_eq!(buf.validate(), Err(GuardError::Overflow));
    assert_eq!(buf.free(), Err(GuardError::Overflow));
}

#[test]
fn leading_canary_corruption_is_underflow() {
    let mut buf = GuardedBuffer::new(4);
    buf.block_mut()[0] ^= 0xFF; // scribble into the leading redzone
    assert_eq!(buf.validate(), Err(GuardError::Underflow));
    assert_eq!(buf.free(), Err(GuardError::Underflow));
}

#[test]
fn free_poisons_and_blocks_further_use() {
    let mut buf = GuardedBuffer::new(4);
    buf.write_at(0, &[1, 2, 3, 4]).unwrap();
    assert_eq!(buf.free(), Ok(()));
    assert!(buf.is_freed());
    // Payload is poisoned with the configured byte.
    let poison = GuardConfig::DEFAULT_POISON_BYTE;
    assert!(buf.block()[buf.payload_range()].iter().all(|&b| b == poison));
    // Any use after free is reported.
    assert_eq!(buf.payload(), Err(GuardError::UseAfterFree));
    assert_eq!(buf.payload_mut(), Err(GuardError::UseAfterFree));
    assert_eq!(buf.write_at(0, &[9]), Err(GuardError::UseAfterFree));
    let mut out = [0u8; 1];
    assert_eq!(buf.read_at(0, &mut out), Err(GuardError::UseAfterFree));
    assert_eq!(buf.validate(), Err(GuardError::UseAfterFree));
}

#[test]
fn double_free_is_detected() {
    let mut buf = GuardedBuffer::new(2);
    assert_eq!(buf.free(), Ok(()));
    assert_eq!(buf.free(), Err(GuardError::DoubleFree));
    assert_eq!(buf.free(), Err(GuardError::DoubleFree));
}

#[test]
fn zero_redzone_disables_canary_but_keeps_poison() {
    let mut buf = GuardedBuffer::with_config(4, GuardConfig::new(0));
    assert_eq!(buf.config().redzone_len(), 0);
    buf.write_at(0, &[1, 2, 3, 4]).unwrap();
    assert_eq!(buf.validate(), Ok(()));
    assert_eq!(buf.free(), Ok(()));
    assert_eq!(buf.free(), Err(GuardError::DoubleFree));
}

#[test]
fn custom_canary_and_poison_bytes_are_honoured() {
    let config = GuardConfig::new(8)
        .with_canary_byte(0xAB)
        .with_poison_byte(0xCD);
    let mut buf = GuardedBuffer::with_config(4, config);
    assert!(buf.block()[..8].iter().all(|&b| b == 0xAB));
    assert_eq!(buf.validate(), Ok(()));
    buf.free().unwrap();
    assert!(buf.block()[buf.payload_range()].iter().all(|&b| b == 0xCD));
}

#[test]
fn empty_payload_is_handled() {
    let mut buf = GuardedBuffer::new(0);
    assert!(buf.is_empty());
    assert_eq!(buf.payload().unwrap(), &[] as &[u8]);
    assert_eq!(buf.validate(), Ok(()));
    assert_eq!(buf.write_at(0, &[]), Ok(()));
    assert_eq!(buf.write_at(0, &[1]), Err(GuardError::Overflow));
    assert_eq!(buf.free(), Ok(()));
}

// ----- GuardedPool ---------------------------------------------------------

#[test]
fn insert_get_remove_basic() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    assert!(pool.is_empty());
    let a = pool.insert(10);
    let b = pool.insert(20);
    assert_eq!(pool.len(), 2);
    assert_eq!(*pool.get(a), 10);
    assert_eq!(*pool.get(b), 20);
    *pool.get_mut(a) += 5;
    assert_eq!(*pool.get(a), 15);
    assert_eq!(pool.remove(b), 20);
    assert_eq!(pool.len(), 1);
    assert!(pool.contains(a));
    assert!(!pool.contains(b));
}

#[test]
fn stale_handle_after_recycle_is_use_after_free() {
    let mut pool: GuardedPool<&'static str> = GuardedPool::new();
    let a = pool.insert("first");
    assert_eq!(pool.remove(a), "first");
    // The slot is recycled by the next insert, with a bumped generation.
    let b = pool.insert("second");
    assert_eq!(a.index(), b.index(), "free list should reuse the slot");
    assert_ne!(a.generation(), b.generation(), "generation must bump");
    assert_eq!(pool.try_get(a), Err(GuardError::UseAfterFree));
    assert_eq!(pool.try_get(b), Ok(&"second"));
}

#[test]
fn double_free_is_detected_on_pool() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let a = pool.insert(1);
    assert_eq!(pool.try_remove(a), Ok(1));
    assert_eq!(pool.try_remove(a), Err(GuardError::DoubleFree));
}

#[test]
fn dangling_handle_is_detected() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    pool.insert(1);
    let bogus: GuardHandle<u32> = GuardHandle::from_raw_for_test(999, 1);
    assert_eq!(pool.try_get(bogus), Err(GuardError::DanglingHandle));
    assert_eq!(pool.try_remove(bogus), Err(GuardError::DanglingHandle));
}

#[test]
#[should_panic(expected = "use-after-free")]
fn get_panics_on_use_after_free() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let a = pool.insert(1);
    pool.remove(a);
    pool.insert(2); // recycle the slot
    let _ = pool.get(a);
}

#[test]
#[should_panic(expected = "double free")]
fn remove_panics_on_double_free() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let a = pool.insert(1);
    pool.remove(a);
    pool.remove(a);
}

#[test]
#[should_panic(expected = "dangling handle")]
fn remove_panics_on_dangling_handle() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let bogus: GuardHandle<u32> = GuardHandle::from_raw_for_test(7, 1);
    pool.remove(bogus);
}

#[test]
fn clear_invalidates_all_handles() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let a = pool.insert(1);
    let b = pool.insert(2);
    pool.clear();
    assert!(pool.is_empty());
    assert_eq!(pool.try_get(a), Err(GuardError::DanglingHandle));
    assert_eq!(pool.try_get(b), Err(GuardError::DanglingHandle));
}

#[test]
fn iter_and_values_visit_only_live_slots() {
    let mut pool: GuardedPool<u32> = GuardedPool::new();
    let a = pool.insert(10);
    let b = pool.insert(20);
    let c = pool.insert(30);
    pool.remove(b);
    let mut collected: Vec<u32> = pool.values().copied().collect();
    collected.sort_unstable();
    assert_eq!(collected, [10, 30]);
    for (handle, value) in pool.iter() {
        assert_eq!(pool.get(handle), value);
    }
    assert_eq!(*pool.get(a), 10);
    assert_eq!(*pool.get(c), 30);
}

#[test]
fn matches_oracle_over_a_random_lifecycle() {
    // Shadow model: every handle ever issued, mapped to its expected value and
    // whether it is still live. The pool must agree with this map on every
    // access, including rejecting stale handles after their slot is recycled.
    let mut pool: GuardedPool<u64> = GuardedPool::new();
    let mut live: HashMap<GuardHandle<u64>, u64> = HashMap::default();
    let mut dead: Vec<GuardHandle<u64>> = Vec::new();
    let mut state: u64 = 0x0BAD_F00D_CAFE_1234;
    let mut next_value: u64 = 0;

    for _ in 0..6000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let choice = state % 3;
        if choice == 0 || live.is_empty() {
            // Insert.
            let value = next_value;
            next_value += 1;
            let handle = pool.insert(value);
            assert_eq!(pool.try_get(handle), Ok(&value));
            live.insert(handle, value);
        } else {
            // Pick a live handle to remove.
            let keys: Vec<GuardHandle<u64>> = live.keys().copied().collect();
            let idx = (state >> 11) as usize % keys.len();
            let handle = keys[idx];
            let expected = live.remove(&handle).unwrap();
            assert_eq!(pool.try_remove(handle), Ok(expected));
            // Removing again now reports a stale handle.
            assert!(matches!(
                pool.try_remove(handle),
                Err(GuardError::DoubleFree | GuardError::UseAfterFree)
            ));
            dead.push(handle);
        }

        assert_eq!(pool.len(), live.len());
    }

    // Every live handle still resolves to its value.
    for (handle, value) in &live {
        assert_eq!(pool.try_get(*handle), Ok(value));
    }
    // No dead handle resolves to a live value.
    for handle in &dead {
        assert!(pool.try_get(*handle).is_err() || !live.contains_key(handle));
    }
}
