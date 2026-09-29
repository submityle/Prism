//! Deterministic per-frame hash chaining for capture streams.
//!
//! A capture is a strictly ordered sequence of frames. To make replay
//! verification cheap and tamper-evident we fold every frame into a rolling
//! digest so that a single 32-byte value fingerprints the entire ordered
//! stream. The construction mirrors the `FNV`-1a lane mixing used by
//! [`crate::abi::AbiHash`] and deliberately avoids any external crypto crate:
//! it is a content fingerprint for detecting drift and reordering, not a
//! cryptographic `MAC`.
//!
//! The chain has three properties that the tests below pin down:
//!
//! * **Determinism** – the same seed and the same ordered frame bytes always
//!   yield the same digest, on any host.
//! * **Order sensitivity** – swapping two frames changes the digest, because
//!   each step folds the previous digest and the running position into the
//!   next one.
//! * **Byte sensitivity** – flipping a single byte in any frame changes the
//!   digest, because frame bytes are length-framed before being absorbed.

use crate::abi::{AbiHash, AbiHashBuilder};

/// Domain-separation tag mixed into the seed and every step so capture chain
/// digests never collide with bare [`AbiHash`] content hashes of the same
/// bytes.
const CHAIN_DOMAIN: &[u8] = b"prism.capture.hash-chain.v1";

/// A rolling digest over an ordered sequence of frame byte blobs.
///
/// Construct with [`FrameHashChain::seeded`], fold each frame with
/// [`FrameHashChain::absorb`], and read the summary with
/// [`FrameHashChain::chain_hash`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameHashChain {
    state: AbiHash,
    count: u64,
}

impl FrameHashChain {
    /// Starts a chain from a deterministic seed.
    ///
    /// The seed is folded together with the domain tag so two captures that
    /// used different random seeds start from distinct states even before any
    /// frame is absorbed.
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        let mut builder = AbiHashBuilder::new();
        builder.write_framed(CHAIN_DOMAIN);
        builder.write_framed(&seed.to_le_bytes());
        Self {
            state: builder.finish(),
            count: 0,
        }
    }

    /// Folds one frame's bytes into the chain and returns the new digest.
    ///
    /// The returned value is the per-frame hash `mix(prev_hash, frame_bytes)`
    /// after this frame; keeping the running position (`count`) in the mix
    /// means an empty frame at position 3 differs from an empty frame at
    /// position 7, so reordering is always observable.
    pub fn absorb(&mut self, frame_bytes: &[u8]) -> AbiHash {
        let mut builder = AbiHashBuilder::new();
        builder.write_framed(CHAIN_DOMAIN);
        builder.write_framed(&self.count.to_le_bytes());
        builder.write_framed(&self.state.0);
        builder.write_framed(frame_bytes);
        self.state = builder.finish();
        self.count = self.count.wrapping_add(1);
        self.state
    }

    /// Returns the current rolling digest.
    ///
    /// For a freshly seeded chain this is the seed digest; after absorbing
    /// frames it summarizes the whole ordered sequence.
    #[must_use]
    pub fn chain_hash(&self) -> AbiHash {
        self.state
    }

    /// Returns how many frames have been folded into the chain.
    #[must_use]
    pub fn absorbed(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn digest_of(seed: u64, frames: &[&[u8]]) -> AbiHash {
        let mut chain = FrameHashChain::seeded(seed);
        for frame in frames {
            chain.absorb(frame);
        }
        chain.chain_hash()
    }

    #[test]
    fn empty_chain_is_seed_digest() {
        let chain = FrameHashChain::seeded(7);
        assert_eq!(chain.absorbed(), 0);
        // The seed digest must be non-zero and stable.
        assert_ne!(chain.chain_hash(), AbiHash::ZERO);
        assert_eq!(chain.chain_hash(), FrameHashChain::seeded(7).chain_hash());
    }

    #[test]
    fn distinct_seeds_diverge_before_any_frame() {
        assert_ne!(
            FrameHashChain::seeded(1).chain_hash(),
            FrameHashChain::seeded(2).chain_hash()
        );
    }

    #[test]
    fn chain_is_deterministic() {
        let a = digest_of(99, &[b"frame-0", b"frame-1", b"frame-2"]);
        let b = digest_of(99, &[b"frame-0", b"frame-1", b"frame-2"]);
        assert_eq!(a, b);
    }

    #[test]
    fn chain_is_order_sensitive() {
        let forward = digest_of(0, &[b"a", b"b", b"c"]);
        let swapped = digest_of(0, &[b"a", b"c", b"b"]);
        assert_ne!(forward, swapped);
    }

    #[test]
    fn chain_is_byte_sensitive() {
        let original = digest_of(0, &[b"payload", b"tail"]);
        let mutated = digest_of(0, &[b"payloae", b"tail"]);
        assert_ne!(original, mutated);
    }

    #[test]
    fn framing_is_significant() {
        // Same concatenated bytes, different frame boundaries => different digest.
        let grouped = digest_of(0, &[b"ab", b"c"]);
        let regrouped = digest_of(0, &[b"a", b"bc"]);
        assert_ne!(grouped, regrouped);
    }

    #[test]
    fn seed_changes_result() {
        let seed_a = digest_of(1, &[b"same", b"frames"]);
        let seed_b = digest_of(2, &[b"same", b"frames"]);
        assert_ne!(seed_a, seed_b);
    }

    #[test]
    fn per_frame_hash_updates_each_step() {
        let mut chain = FrameHashChain::seeded(5);
        let mut seen: Vec<AbiHash> = Vec::new();
        seen.push(chain.chain_hash());
        for frame in [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()] {
            let step = chain.absorb(frame);
            // The returned per-frame hash equals the running digest.
            assert_eq!(step, chain.chain_hash());
            seen.push(step);
        }
        // Every intermediate digest is distinct.
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(seen[i], seen[j], "digests {i} and {j} collided");
            }
        }
        assert_eq!(chain.absorbed(), 3);
    }

    #[test]
    fn empty_frames_still_advance_position() {
        // Two empty frames must not cancel out; position mixing keeps them distinct.
        let one_empty = digest_of(0, &[b""]);
        let two_empty = digest_of(0, &[b"", b""]);
        assert_ne!(one_empty, two_empty);
        assert_ne!(one_empty, FrameHashChain::seeded(0).chain_hash());
    }
}
