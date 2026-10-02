//! `std430` ray/hit storage-buffer `ABI` for the `GPU` traversal kernel.
//!
//! The packed `BVH` buffers in [`super::gpu_layout`] and the escape array in
//! [`super::traversal_stackless_gpu_layout`] describe the *scene* a compute
//! kernel binds; this module pins the *query* `ABI` — the input ray records a
//! kernel reads and the output hit records it writes. One dispatch invocation
//! consumes one [`TRACE_RAY_WORDS`]-word ray record and produces one
//! [`TRACE_HIT_WORDS`]-word hit record, so the host uploads an `array<u32>` of
//! rays and reads back a parallel `array<u32>` of hits.
//!
//! Encoding is dependency-free and mirrors the rest of `ray_scene`: every field
//! is a `u32` word, with `f32` values stored as their `to_bits` pattern
//! (little-endian on upload), matching a `WESL` `array<u32>` bound as `std430`
//! storage. Both records are eight words (32 bytes), a multiple of four words,
//! so each record is 16-byte aligned.
//!
//! The companion kernel [`super::gpu_trace_kernel`] walks the scene buffers per
//! invocation and the `GPU`↔`CPU` parity test diffs the decoded hits against the
//! [`super::bvh::Bvh::closest_hit`] golden.

use super::traversal::{Hit, Ray};

/// `u32` words per packed ray record (32 bytes, 16-byte aligned).
///
/// Layout: `origin.xyz` (0..3) as `to_bits`, `t_min` (3), `direction.xyz`
/// (4..7), `t_max` (7).
pub const TRACE_RAY_WORDS: usize = 8;

/// `u32` words per packed hit record (32 bytes, 16-byte aligned).
///
/// Layout: `t` (0), `u` (1), `v` (2) as `to_bits`, `primitive` (3), hit flag
/// (4; [`HIT_FLAG_HIT`] or [`HIT_FLAG_MISS`]), padding (5..8).
pub const TRACE_HIT_WORDS: usize = 8;

/// Compute-kernel workgroup size along `x` (one invocation per ray record).
pub const WORKGROUP_SIZE: u32 = 64;

/// Hit-flag word value meaning "the ray missed every primitive".
pub const HIT_FLAG_MISS: u32 = 0;

/// Hit-flag word value meaning "the record carries a committed intersection".
pub const HIT_FLAG_HIT: u32 = 1;

/// Serializes `ray` into its packed [`TRACE_RAY_WORDS`]-word record.
#[must_use]
pub fn encode_ray(ray: &Ray) -> [u32; TRACE_RAY_WORDS] {
    let o = ray.origin();
    let d = ray.direction();
    [
        o[0].to_bits(),
        o[1].to_bits(),
        o[2].to_bits(),
        ray.t_min().to_bits(),
        d[0].to_bits(),
        d[1].to_bits(),
        d[2].to_bits(),
        ray.t_max().to_bits(),
    ]
}

/// Decodes a packed ray record (at least [`TRACE_RAY_WORDS`] words) back into a
/// [`Ray`].
///
/// Routes through [`Ray::new`] so the reciprocal direction is recomputed and the
/// `t` interval re-clamped exactly as a freshly built ray; because
/// [`encode_ray`] only stores already-valid fields the round trip is bit-exact.
#[must_use]
pub fn decode_ray(words: &[u32]) -> Ray {
    Ray::new(
        [
            f32::from_bits(words[0]),
            f32::from_bits(words[1]),
            f32::from_bits(words[2]),
        ],
        [
            f32::from_bits(words[4]),
            f32::from_bits(words[5]),
            f32::from_bits(words[6]),
        ],
        f32::from_bits(words[3]),
        f32::from_bits(words[7]),
    )
}

/// Serializes an optional [`Hit`] into its packed [`TRACE_HIT_WORDS`]-word
/// record.
///
/// A miss writes zeroed `t`/`u`/`v`/`primitive` words and [`HIT_FLAG_MISS`]; a
/// hit stores the `to_bits` fields and [`HIT_FLAG_HIT`]. Padding words are zero.
#[must_use]
pub fn encode_hit(hit: Option<Hit>) -> [u32; TRACE_HIT_WORDS] {
    match hit {
        Some(h) => [
            h.t.to_bits(),
            h.u.to_bits(),
            h.v.to_bits(),
            h.primitive,
            HIT_FLAG_HIT,
            0,
            0,
            0,
        ],
        None => [0, 0, 0, 0, HIT_FLAG_MISS, 0, 0, 0],
    }
}

/// Decodes a packed hit record (at least [`TRACE_HIT_WORDS`] words).
///
/// Returns [`None`] unless the flag word equals [`HIT_FLAG_HIT`], so any
/// non-hit flag (including [`HIT_FLAG_MISS`]) decodes to a miss.
#[must_use]
pub fn decode_hit(words: &[u32]) -> Option<Hit> {
    if words[4] == HIT_FLAG_HIT {
        Some(Hit {
            t: f32::from_bits(words[0]),
            u: f32::from_bits(words[1]),
            v: f32::from_bits(words[2]),
            primitive: words[3],
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_eight_words_and_sixteen_byte_aligned() {
        assert_eq!(TRACE_RAY_WORDS, 8);
        assert_eq!(TRACE_HIT_WORDS, 8);
        assert_eq!((TRACE_RAY_WORDS * 4) % 16, 0);
        assert_eq!((TRACE_HIT_WORDS * 4) % 16, 0);
    }

    #[test]
    fn hit_flags_are_distinct() {
        assert_ne!(HIT_FLAG_HIT, HIT_FLAG_MISS);
    }

    #[test]
    fn ray_record_round_trips_bit_for_bit() {
        let ray = Ray::new([1.0, -2.0, 3.5], [0.1, 0.2, -0.3], 0.25, 42.0);
        let words = encode_ray(&ray);
        // Re-encoding the decoded ray reproduces the exact words.
        assert_eq!(encode_ray(&decode_ray(&words)), words);
    }

    #[test]
    fn infinite_ray_record_round_trips_bit_for_bit() {
        let ray = Ray::infinite([0.0, 1.0, 2.0], [0.0, 0.0, -1.0]);
        let words = encode_ray(&ray);
        assert_eq!(words[7], f32::INFINITY.to_bits());
        assert_eq!(encode_ray(&decode_ray(&words)), words);
    }

    #[test]
    fn hit_record_round_trips_bit_for_bit() {
        let hit = Hit {
            t: 7.25,
            u: 0.125,
            v: 0.375,
            primitive: 11,
        };
        let words = encode_hit(Some(hit));
        assert_eq!(words[4], HIT_FLAG_HIT);
        let back = decode_hit(&words).expect("hit flag set");
        assert_eq!(back.t.to_bits(), hit.t.to_bits());
        assert_eq!(back.u.to_bits(), hit.u.to_bits());
        assert_eq!(back.v.to_bits(), hit.v.to_bits());
        assert_eq!(back.primitive, hit.primitive);
    }

    #[test]
    fn miss_record_decodes_to_none() {
        let words = encode_hit(None);
        assert_eq!(words[4], HIT_FLAG_MISS);
        assert!(decode_hit(&words).is_none());
    }
}
