//! A fast, non-cryptographic hasher (`FxHasher`, the Firefox/`rustc` hash).

use core::hash::{BuildHasherDefault, Hasher};

const SEED64: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A fast non-cryptographic hasher. Not DoS-resistant; use only for internal
/// maps keyed by trusted data (type ids, entity ids, interned strings).
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED64);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            self.add(word);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A [`BuildHasher`](core::hash::BuildHasher) that produces [`FxHasher`]s.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
