//! `CPU` golden twins for the `GPU` radix sort.
//!
//! [`cpu_radix_sort_keys`] and [`cpu_radix_sort_pairs`] run the same
//! least-significant-digit counting sort the `WGSL` kernels reproduce in
//! parallel: [`PASSES`](super::config::PASSES) stable passes over
//! [`RADIX_BITS`](super::config::RADIX_BITS)-bit digits. Because sorting `u32`
//! keys is a pure integer permutation, a passing real-device parity test is
//! direct bit-for-bit evidence that the ported kernels compute the same ordered
//! stream — not merely that the shaders compiled.
//!
//! # Stability
//!
//! Each pass is a stable counting sort, so equal keys keep their input order.
//! Stability is what makes the multi-pass `LSD` scheme correct — a lower digit
//! sorted earlier must survive intact when a higher digit ties — and it is also
//! the contract for the key/value variant, where payloads must follow their
//! keys in the original relative order.
//!
//! # Provenance
//!
//! The `LSD` radix sort is a classical, openly published algorithm. This module
//! contains no Unreal Engine source or derived code.

use super::config::{PASSES, RADIX, RADIX_BITS, RADIX_MASK};

/// Returns the `pass`-th [`RADIX_BITS`](super::config::RADIX_BITS)-bit digit of
/// `key`.
#[must_use]
fn digit(key: u32, pass: u32) -> usize {
    ((key >> (pass * RADIX_BITS)) & RADIX_MASK) as usize
}

/// Sorts `keys` ascending with a stable least-significant-digit radix sort.
///
/// Returns a new vector; the input is left untouched. Matches the ordering the
/// `GPU` kernels produce bit-for-bit.
#[must_use]
pub fn cpu_radix_sort_keys(keys: &[u32]) -> Vec<u32> {
    let mut src = keys.to_vec();
    let mut dst = vec![0u32; keys.len()];
    for pass in 0..PASSES {
        let mut counts = [0u32; RADIX as usize];
        for &k in &src {
            counts[digit(k, pass)] += 1;
        }
        let mut offsets = [0u32; RADIX as usize];
        let mut running = 0u32;
        for (offset, &count) in offsets.iter_mut().zip(counts.iter()) {
            *offset = running;
            running += count;
        }
        for &k in &src {
            let d = digit(k, pass);
            dst[offsets[d] as usize] = k;
            offsets[d] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }
    src
}

/// Sorts `keys` ascending with a stable `LSD` radix sort, carrying each entry of
/// `values` alongside its key.
///
/// Returns the sorted keys and their reordered values. Equal keys preserve
/// input order, so the paired values follow their keys stably.
///
/// # Panics
///
/// Panics if `keys` and `values` do not have the same length.
#[must_use]
pub fn cpu_radix_sort_pairs(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    assert!(
        keys.len() == values.len(),
        "keys and values must have equal length"
    );
    let mut src_k = keys.to_vec();
    let mut src_v = values.to_vec();
    let mut dst_k = vec![0u32; keys.len()];
    let mut dst_v = vec![0u32; keys.len()];
    for pass in 0..PASSES {
        let mut counts = [0u32; RADIX as usize];
        for &k in &src_k {
            counts[digit(k, pass)] += 1;
        }
        let mut offsets = [0u32; RADIX as usize];
        let mut running = 0u32;
        for (offset, &count) in offsets.iter_mut().zip(counts.iter()) {
            *offset = running;
            running += count;
        }
        for (&k, &v) in src_k.iter().zip(src_v.iter()) {
            let d = digit(k, pass);
            let pos = offsets[d] as usize;
            dst_k[pos] = k;
            dst_v[pos] = v;
            offsets[d] += 1;
        }
        std::mem::swap(&mut src_k, &mut dst_k);
        std::mem::swap(&mut src_v, &mut dst_v);
    }
    (src_k, src_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_sorts_to_empty() {
        assert!(cpu_radix_sort_keys(&[]).is_empty());
    }

    #[test]
    fn keys_sort_ascending_like_the_standard_sort() {
        let keys = [5u32, 1, 4, 1, 3, 9, 2, 6];
        let mut expected = keys.to_vec();
        expected.sort_unstable();
        assert_eq!(cpu_radix_sort_keys(&keys), expected);
    }

    #[test]
    fn boundary_key_values_sort() {
        let keys = [u32::MAX, 0, 1, u32::MAX - 1, 0x00FF_00FF, 0xFF00_FF00];
        let mut expected = keys.to_vec();
        expected.sort_unstable();
        assert_eq!(cpu_radix_sort_keys(&keys), expected);
    }

    #[test]
    fn pairs_carry_values_and_are_stable() {
        // Two entries share key 1; their values must keep input order (10, 40).
        let keys = [1u32, 3, 1, 2];
        let values = [10u32, 20, 40, 30];
        let (sk, sv) = cpu_radix_sort_pairs(&keys, &values);
        assert_eq!(sk, vec![1, 1, 2, 3]);
        assert_eq!(sv, vec![10, 40, 30, 20]);
    }

    #[test]
    fn pairs_match_a_stable_reference_sort() {
        let keys = [7u32, 7, 3, 3, 3, 1, 9, 7];
        let values: Vec<u32> = (0..keys.len() as u32).collect();
        let (sk, sv) = cpu_radix_sort_pairs(&keys, &values);

        let mut reference: Vec<(u32, u32)> = keys.iter().copied().zip(values).collect();
        reference.sort_by_key(|&(k, _)| k); // stable
        let want_k: Vec<u32> = reference.iter().map(|&(k, _)| k).collect();
        let want_v: Vec<u32> = reference.iter().map(|&(_, v)| v).collect();
        assert_eq!(sk, want_k);
        assert_eq!(sv, want_v);
    }
}
