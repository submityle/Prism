//! Octahedral encoding of unit normals.
//!
//! Octahedral mapping stores a unit [`Vec3`] direction in two numbers by
//! projecting the sphere onto an octahedron and unfolding it to the unit
//! square. It is the standard compact normal encoding for `GPU` g-buffers and
//! vertex streams: compared with storing three floats it halves (or, when
//! packed to fixed point, quarters) the footprint while keeping the direction
//! error negligible.
//!
//! [`encode`] / [`decode`] use full `f32` precision and round-trip a unit
//! vector to within about `1e-6` of angular error. [`pack_snorm`] /
//! [`unpack_snorm`] additionally quantize to a 16-bit-per-channel `u32`
//! suitable for direct upload; that path round-trips to within about `1e-3`.

use crate::float::f32 as mf;
use crate::vec::{Vec2, Vec3};

/// Branchless sign that returns `+1.0` for a value `>= 0` (including `+0.0`)
/// and `-1.0` otherwise. This matches the convention required for the fold:
/// zero components must not collapse the encoded coordinate to zero.
#[inline]
fn sign_nonzero(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// Encode a (not necessarily normalized) direction to octahedral coordinates
/// in the `[-1, 1]` square.
///
/// The input is normalized by its L1 norm internally, so any non-zero vector
/// along the same direction encodes identically.
#[inline]
#[must_use]
pub fn encode(n: Vec3) -> Vec2 {
    let inv_l1 = 1.0 / (mf::abs(n.x) + mf::abs(n.y) + mf::abs(n.z));
    let p = Vec2::new(n.x * inv_l1, n.y * inv_l1);
    if n.z >= 0.0 {
        p
    } else {
        // Fold the lower hemisphere out across the octahedron edges.
        Vec2::new(
            (1.0 - mf::abs(p.y)) * sign_nonzero(p.x),
            (1.0 - mf::abs(p.x)) * sign_nonzero(p.y),
        )
    }
}

/// Decode octahedral coordinates in the `[-1, 1]` square back to a unit
/// direction.
#[inline]
#[must_use]
pub fn decode(e: Vec2) -> Vec3 {
    let mut n = Vec3::new(e.x, e.y, 1.0 - mf::abs(e.x) - mf::abs(e.y));
    // Reconstruct the lower hemisphere by folding back.
    let t = (-n.z).max(0.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize()
}

/// Encode and quantize a direction to a packed `u32` holding two 16-bit signed
/// normalized (snorm) channels (`x` in the low 16 bits, `y` in the high 16).
#[inline]
#[must_use]
pub fn pack_snorm(n: Vec3) -> u32 {
    let e = encode(n);
    let x = snorm16(e.x) as u32;
    let y = snorm16(e.y) as u32;
    x | (y << 16)
}

/// Decode a packed `u32` produced by [`pack_snorm`] back to a unit direction.
#[inline]
#[must_use]
pub fn unpack_snorm(bits: u32) -> Vec3 {
    let x = unsnorm16((bits & 0xFFFF) as u16);
    let y = unsnorm16(((bits >> 16) & 0xFFFF) as u16);
    decode(Vec2::new(x, y))
}

/// Quantize a value in `[-1, 1]` to 16-bit snorm (round to nearest).
#[inline]
fn snorm16(v: f32) -> u16 {
    let c = v.clamp(-1.0, 1.0);
    let scaled = c * 32767.0;
    // Round to nearest, ties away from zero, then reinterpret as two's complement.
    let r = mf::round(scaled) as i32;
    (r as i16) as u16
}

/// Dequantize a 16-bit snorm value back to `[-1, 1]`.
#[inline]
fn unsnorm16(bits: u16) -> f32 {
    let v = (bits as i16) as f32 / 32767.0;
    v.clamp(-1.0, 1.0)
}
