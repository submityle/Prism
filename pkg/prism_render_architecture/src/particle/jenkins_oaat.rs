//! Jenkins one-at-a-time hash (Bob Jenkins' `OAAT`): a tiny, pure-integer
//! 32-bit non-cryptographic hash for cheaply bucketing small keys such as
//! particle identifiers, atlas tile names, or `GPU` resource lookup strings
//! (design § hashing helpers).
//!
//! The hash walks the input one byte at a time. For every byte it folds the
//! byte into a running 32-bit accumulator with an add, a left-shift-and-add,
//! and an exclusive-or of a right-shifted copy. After the whole stream has been
//! consumed a three-step finishing mix (another shift-add, a shift-xor, and a
//! final shift-add) spreads the accumulated bits so that every input bit
//! influences every output bit, giving good avalanche for such a small
//! construction. Every operation is a wrapping integer add, a shift, or an
//! exclusive-or; there are no floating-point or transcendental operations and
//! no lookup tables, so the result is bit-for-bit identical on every platform.
//!
//! Two entry points are provided. The one-shot [`jenkins_oaat`] hashes a single
//! byte slice. The streaming [`JenkinsOaat`] accumulator folds bytes across
//! several [`JenkinsOaat::update`] calls and applies the finishing mix only in
//! [`JenkinsOaat::finalize`]; feeding the same bytes in any chunking therefore
//! yields exactly the same digest as the one-shot function. For the empty input
//! the accumulator stays at `0` and, because the finishing mix maps `0` to `0`,
//! the digest is `0`.
//!
//! Scope: this is a hash for hash tables and bucketing, not a checksum and not
//! a message authentication code. The Jenkins one-at-a-time hash is *not*
//! cryptographically secure; collisions and preimages are easy to construct on
//! purpose, so it must never be used to authenticate data or guard against a
//! malicious adversary. For content-addressing or security use a real
//! cryptographic hash instead.

/// Folds a single byte into the running accumulator (the per-byte loop body).
///
/// This performs the first three steps of the algorithm: add the byte, add a
/// left-shifted copy, then exclusive-or in a right-shifted copy. The finishing
/// mix is deliberately *not* applied here so that the streaming accumulator can
/// defer it to [`JenkinsOaat::finalize`].
#[inline]
fn fold_byte(mut hash: u32, byte: u8) -> u32 {
    hash = hash.wrapping_add(byte as u32);
    hash = hash.wrapping_add(hash << 10);
    hash ^= hash >> 6;
    hash
}

/// Applies the three-step finishing mix to a folded accumulator.
///
/// This is the tail of the algorithm run once after every byte has been folded:
/// a shift-add, a shift-xor, and a final shift-add. It maps `0` to `0`, so an
/// empty input hashes to `0`.
#[inline]
fn finish(mut hash: u32) -> u32 {
    hash = hash.wrapping_add(hash << 3);
    hash ^= hash >> 11;
    hash = hash.wrapping_add(hash << 15);
    hash
}

/// Computes the Jenkins one-at-a-time 32-bit hash of `data` in a single call.
///
/// Equivalent to constructing a [`JenkinsOaat`], calling
/// [`JenkinsOaat::update`] once with `data`, and then
/// [`JenkinsOaat::finalize`]. Returns `0` for the empty slice.
pub fn jenkins_oaat(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    for &b in data {
        hash = fold_byte(hash, b);
    }
    finish(hash)
}

/// Streaming Jenkins one-at-a-time hash accumulator.
///
/// Bytes are folded incrementally with [`update`](JenkinsOaat::update); the
/// finishing mix is applied only by [`finalize`](JenkinsOaat::finalize). Because
/// the per-byte loop is associative over concatenation, splitting the input into
/// any sequence of [`update`](JenkinsOaat::update) calls produces the same
/// digest as the one-shot [`jenkins_oaat`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JenkinsOaat {
    hash: u32,
}

impl JenkinsOaat {
    /// Creates a fresh accumulator seeded at `0`.
    #[inline]
    pub fn new() -> Self {
        Self { hash: 0 }
    }

    /// Folds every byte of `data` into the running accumulator.
    ///
    /// This runs only the per-byte loop (add, shift-add, shift-xor); it does
    /// *not* apply the finishing mix.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.hash = fold_byte(self.hash, b);
        }
    }

    /// Applies the finishing mix and returns the final 32-bit digest.
    ///
    /// The accumulator is consumed by value so a finished digest cannot be
    /// accidentally fed more bytes.
    #[inline]
    pub fn finalize(self) -> u32 {
        finish(self.hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Hard reference vectors --------------------------------------

    #[test]
    fn ref_single_a() {
        assert_eq!(jenkins_oaat(b"a"), 0xCA2E_9442);
    }

    #[test]
    fn ref_single_a_decimal() {
        assert_eq!(jenkins_oaat(b"a"), 3_392_050_242);
    }

    #[test]
    fn ref_fox() {
        assert_eq!(
            jenkins_oaat(b"The quick brown fox jumps over the lazy dog"),
            0x519E_91F5
        );
    }

    #[test]
    fn ref_empty_is_zero() {
        assert_eq!(jenkins_oaat(b""), 0);
    }

    #[test]
    fn ref_empty_decimal() {
        assert_eq!(jenkins_oaat(&[]), 0u32);
    }

    // ----- Empty / single-byte behaviour -------------------------------

    #[test]
    fn empty_slice_literal() {
        let empty: &[u8] = &[];
        assert_eq!(jenkins_oaat(empty), 0);
    }

    #[test]
    fn single_zero_byte_is_zero() {
        // A single 0x00 byte folds to 0, matching the empty-input digest: the
        // whole mixing chain maps 0 to 0.
        assert_eq!(jenkins_oaat(&[0x00]), 0);
        assert_eq!(jenkins_oaat(&[0x00]), jenkins_oaat(b""));
    }

    #[test]
    fn single_nonzero_byte_is_nonzero() {
        // Every non-zero single byte mixes to a non-zero digest.
        for b in 1u16..=255 {
            assert_ne!(jenkins_oaat(&[b as u8]), 0);
        }
    }

    #[test]
    fn nonzero_single_bytes_distinct_from_empty() {
        for b in 1u16..=255 {
            assert_ne!(jenkins_oaat(&[b as u8]), jenkins_oaat(b""));
        }
    }

    #[test]
    fn single_byte_values_mostly_unique() {
        // All 256 single-byte inputs must map to distinct digests.
        let mut digests = [0u32; 256];
        for b in 0u16..=255 {
            digests[b as usize] = jenkins_oaat(&[b as u8]);
        }
        for i in 0..256 {
            for j in (i + 1)..256 {
                assert_ne!(digests[i], digests[j], "collision at {i} vs {j}");
            }
        }
    }

    #[test]
    fn single_byte_0x61_matches_a() {
        assert_eq!(jenkins_oaat(&[0x61]), jenkins_oaat(b"a"));
    }

    // ----- Long input --------------------------------------------------

    #[test]
    fn long_zeros_input() {
        let data = [0u8; 1024];
        // Just has to compute without panicking and be deterministic.
        assert_eq!(jenkins_oaat(&data), jenkins_oaat(&data));
    }

    #[test]
    fn long_sequential_input() {
        let mut data = [0u8; 512];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = (i & 0xFF) as u8;
        }
        let h = jenkins_oaat(&data);
        assert_eq!(h, jenkins_oaat(&data));
    }

    #[test]
    fn long_input_differs_from_prefix() {
        let mut data = [0u8; 300];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = (i as u32 & 0xFF) as u8;
        }
        let full = jenkins_oaat(&data);
        let prefix = jenkins_oaat(&data[..299]);
        assert_ne!(full, prefix);
    }

    #[test]
    fn repeated_byte_lengths_differ() {
        let a = [0x5Au8; 16];
        let b = [0x5Au8; 17];
        assert_ne!(jenkins_oaat(&a), jenkins_oaat(&b));
    }

    // ----- Determinism -------------------------------------------------

    #[test]
    fn deterministic_repeated_calls() {
        let data = b"prism-particle-id-0007";
        let first = jenkins_oaat(data);
        for _ in 0..64 {
            assert_eq!(jenkins_oaat(data), first);
        }
    }

    #[test]
    fn deterministic_fox() {
        let data = b"The quick brown fox jumps over the lazy dog";
        assert_eq!(jenkins_oaat(data), jenkins_oaat(data));
    }

    #[test]
    fn deterministic_empty() {
        assert_eq!(jenkins_oaat(b""), jenkins_oaat(b""));
    }

    // ----- Incremental consistency -------------------------------------

    #[test]
    fn incremental_single_update_matches_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut acc = JenkinsOaat::new();
        acc.update(data);
        assert_eq!(acc.finalize(), jenkins_oaat(data));
    }

    #[test]
    fn incremental_default_matches_new() {
        let data = b"hash-me";
        let mut a = JenkinsOaat::new();
        a.update(data);
        let mut b = JenkinsOaat::default();
        b.update(data);
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn incremental_byte_by_byte() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut acc = JenkinsOaat::new();
        for &b in data {
            acc.update(&[b]);
        }
        assert_eq!(acc.finalize(), jenkins_oaat(data));
    }

    #[test]
    fn incremental_two_chunks() {
        let data = b"abcdefghijklmnopqrstuvwxyz0123456789";
        for split in 0..=data.len() {
            let mut acc = JenkinsOaat::new();
            acc.update(&data[..split]);
            acc.update(&data[split..]);
            assert_eq!(acc.finalize(), jenkins_oaat(data), "split at {split}");
        }
    }

    #[test]
    fn incremental_three_chunks() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut acc = JenkinsOaat::new();
        acc.update(&data[..10]);
        acc.update(&data[10..25]);
        acc.update(&data[25..]);
        assert_eq!(acc.finalize(), jenkins_oaat(data));
    }

    #[test]
    fn incremental_empty_updates_noop() {
        let data = b"interleaved-empties";
        let mut acc = JenkinsOaat::new();
        acc.update(&[]);
        acc.update(data);
        acc.update(&[]);
        assert_eq!(acc.finalize(), jenkins_oaat(data));
    }

    #[test]
    fn incremental_empty_total_is_zero() {
        let acc = JenkinsOaat::new();
        assert_eq!(acc.finalize(), 0);
    }

    #[test]
    fn incremental_single_byte_matches() {
        let mut acc = JenkinsOaat::new();
        acc.update(b"a");
        assert_eq!(acc.finalize(), 0xCA2E_9442);
    }

    #[test]
    fn incremental_many_random_splits() {
        let data = b"prism-render-architecture-particle-jenkins-oaat";
        let one = jenkins_oaat(data);
        // Deterministic pseudo-splits using the hash itself as a stepper.
        let mut step = 1usize;
        for _ in 0..8 {
            let mut acc = JenkinsOaat::new();
            let mut pos = 0usize;
            while pos < data.len() {
                let end = (pos + step).min(data.len());
                acc.update(&data[pos..end]);
                pos = end;
                step = (step % 7) + 1;
            }
            assert_eq!(acc.finalize(), one);
        }
    }

    #[test]
    fn finalize_does_not_require_updates() {
        let a = JenkinsOaat::new().finalize();
        assert_eq!(a, jenkins_oaat(b""));
    }

    // ----- Avalanche ---------------------------------------------------

    #[test]
    fn avalanche_single_bit_flip() {
        let base = jenkins_oaat(b"avalanche");
        let flipped = jenkins_oaat(b"bvalanche"); // 'a' -> 'b', one bit flip
        assert_ne!(base, flipped);
    }

    #[test]
    fn avalanche_last_byte_change() {
        let a = jenkins_oaat(b"message0");
        let b = jenkins_oaat(b"message1");
        assert_ne!(a, b);
    }

    #[test]
    fn avalanche_many_bits_change() {
        // Flipping one input bit should change many output bits (good diffusion).
        let base = jenkins_oaat(&[0x00]);
        let flip = jenkins_oaat(&[0x01]);
        let diff = (base ^ flip).count_ones();
        assert!(diff >= 8, "weak avalanche: only {diff} bits changed");
    }

    #[test]
    fn avalanche_byte_order_matters() {
        let ab = jenkins_oaat(b"ab");
        let ba = jenkins_oaat(b"ba");
        assert_ne!(ab, ba);
    }

    #[test]
    fn avalanche_length_extension_differs() {
        let short = jenkins_oaat(b"key");
        let long = jenkins_oaat(b"key\0");
        assert_ne!(short, long);
    }

    // ----- Collision sanity on a small corpus --------------------------

    #[test]
    fn small_corpus_distinct() {
        let corpus: [&[u8]; 10] = [
            b"particle",
            b"Particle",
            b"particles",
            b"atlas",
            b"atlas-0",
            b"atlas-1",
            b"gpu-buffer",
            b"gpu_buffer",
            b"",
            b" ",
        ];
        for i in 0..corpus.len() {
            for j in (i + 1)..corpus.len() {
                assert_ne!(
                    jenkins_oaat(corpus[i]),
                    jenkins_oaat(corpus[j]),
                    "collision between {i} and {j}"
                );
            }
        }
    }

    #[test]
    fn case_sensitivity() {
        assert_ne!(jenkins_oaat(b"ABC"), jenkins_oaat(b"abc"));
    }

    #[test]
    fn whitespace_sensitivity() {
        assert_ne!(jenkins_oaat(b"a b"), jenkins_oaat(b"ab"));
    }

    // ----- Internal helper properties ----------------------------------

    #[test]
    fn finish_zero_is_zero() {
        assert_eq!(finish(0), 0);
    }

    #[test]
    fn finish_nonzero_changes_value() {
        assert_ne!(finish(1), 1);
    }

    #[test]
    fn fold_then_finish_matches_oneshot_single() {
        let folded = fold_byte(0, b'a');
        assert_eq!(finish(folded), jenkins_oaat(b"a"));
    }

    #[test]
    fn fold_byte_is_pure() {
        assert_eq!(fold_byte(123, 45), fold_byte(123, 45));
    }

    #[test]
    fn digest_fits_u32() {
        // Trivially true by type, but guards against accidental widening.
        let h: u32 = jenkins_oaat(b"width-check");
        assert_eq!(h, h & 0xFFFF_FFFF);
    }

    #[test]
    fn two_long_distinct_inputs_differ() {
        let mut a = [0u8; 256];
        let mut b = [0u8; 256];
        for i in 0..256 {
            a[i] = (i & 0xFF) as u8;
            b[i] = ((i + 1) & 0xFF) as u8;
        }
        assert_ne!(jenkins_oaat(&a), jenkins_oaat(&b));
    }
}
