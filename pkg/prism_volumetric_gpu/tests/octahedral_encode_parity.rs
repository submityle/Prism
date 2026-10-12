//! Real-device parity for the octahedral unit-normal codec twin:
//! [`GpuOctahedralEncode`](prism_volumetric_gpu::octahedral_encode::GpuOctahedralEncode)
//! must reproduce the `CPU` golden `prism_math::octahedral` for all four
//! stateless operations — `encode`, `decode`, `pack_snorm`, `unpack_snorm`.
//!
//! The oracle here is an independent `f32` re-implementation of those closed
//! forms — the `L1`-normalized octahedral fold, its inverse with the final
//! renormalization, and the 16-bit snorm quantization with round-ties-away and
//! two's-complement truncation — written out directly so the test never imports
//! `prism_math`, `glam`, `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover each operation against the oracle, an `encode` then
//! `decode` round-trip of unit directions, a `pack` then `unpack` round-trip
//! (with the packed `bits` word checked bit-exact), the axis directions, the
//! lower-hemisphere (`vz < 0`) fold, degenerate zero / non-finite vectors that
//! report `valid = 0`, an unknown `op_id` that reports `valid = 0`, a mixed
//! multi-operation batch that validates the `std430` stride, and an empty batch
//! the host short-circuits with no dispatch. A random sweep over all four
//! operations follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact.
//! Continuous outputs are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag and the packed `bits` word
//! are compared exactly. The sweep normalizes directions and rejects near-zero
//! `L1` norms so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_math::octahedral`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::octahedral_encode::{
    GpuOctahedralEncode, OctahedralEncodeQuery, OctahedralEncodeResult, OP_DECODE, OP_ENCODE,
    OP_PACK, OP_UNPACK,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Ordered finiteness limit mirroring the kernel's `abs(x) < 3.0e38` guard.
const FINITE_LIMIT: f32 = 3.0e38;

/// `L1`-norm degeneracy floor mirroring the kernel's `l1 > 1e-20` guard.
const TINY_L1: f32 = 1.0e-20;

/// Branchless sign: `+1.0` for `v >= 0` (including `+0.0`), else `-1.0`.
fn sign_nonzero(v: f32) -> f32 {
    if v >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent oracle for `encode`: direction to octahedral `(x, y)`.
fn encode(x: f32, y: f32, z: f32) -> (f32, f32) {
    let inv_l1 = 1.0 / (x.abs() + y.abs() + z.abs());
    let px = x * inv_l1;
    let py = y * inv_l1;
    if z >= 0.0 {
        (px, py)
    } else {
        (
            (1.0 - py.abs()) * sign_nonzero(px),
            (1.0 - px.abs()) * sign_nonzero(py),
        )
    }
}

/// Independent oracle for `decode`: octahedral `(x, y)` to a unit direction.
fn decode(ex: f32, ey: f32) -> (f32, f32, f32) {
    let nz = 1.0 - ex.abs() - ey.abs();
    let t = (-nz).max(0.0);
    let nx = ex + if ex >= 0.0 { -t } else { t };
    let ny = ey + if ey >= 0.0 { -t } else { t };
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    (nx / len, ny / len, nz / len)
}

/// Independent oracle for 16-bit snorm quantization, round ties away from zero,
/// returned in the low 16 bits of a `u32` as a two's-complement pattern.
fn snorm16(v: f32) -> u32 {
    let c = v.clamp(-1.0, 1.0);
    let scaled = c * 32767.0;
    // Rust `f32::round` is round-half-away-from-zero, matching libm `roundf`.
    let r = scaled.round() as i32;
    (r as i16) as u16 as u32
}

/// Independent oracle for 16-bit snorm dequantization back to `[-1, 1]`.
fn unsnorm16(b16: u32) -> f32 {
    let v = ((b16 & 0xFFFF) as u16 as i16) as f32 / 32767.0;
    v.clamp(-1.0, 1.0)
}

/// Independent oracle for `pack_snorm`: encode then quantize to a packed `u32`.
fn pack(x: f32, y: f32, z: f32) -> u32 {
    let (ex, ey) = encode(x, y, z);
    snorm16(ex) | (snorm16(ey) << 16)
}

/// Independent oracle for `unpack_snorm`: packed `u32` to a unit direction.
fn unpack(bits: u32) -> (f32, f32, f32) {
    let x = unsnorm16(bits & 0xFFFF);
    let y = unsnorm16((bits >> 16) & 0xFFFF);
    decode(x, y)
}

/// Validity decision for `encode`/`pack`, mirroring the kernel's ordered
/// finiteness and `L1`-floor guards.
fn vec_ok(x: f32, y: f32, z: f32) -> bool {
    let finite = x.abs() < FINITE_LIMIT && y.abs() < FINITE_LIMIT && z.abs() < FINITE_LIMIT;
    finite && (x.abs() + y.abs() + z.abs()) > TINY_L1
}

/// The independent oracle: resolves one query to the expected result exactly as
/// the kernel should, dispatched on the operation selector.
fn oracle(q: &OctahedralEncodeQuery) -> OctahedralEncodeResult {
    let mut out = OctahedralEncodeResult {
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
        bits: 0,
        valid: 0,
    };
    match q.op_id {
        OP_ENCODE => {
            if vec_ok(q.vx, q.vy, q.vz) {
                let (ex, ey) = encode(q.vx, q.vy, q.vz);
                out.ox = ex;
                out.oy = ey;
                out.valid = 1;
            }
        }
        OP_DECODE => {
            let (nx, ny, nz) = decode(q.vx, q.vy);
            out.ox = nx;
            out.oy = ny;
            out.oz = nz;
            out.valid = 1;
        }
        OP_PACK => {
            if vec_ok(q.vx, q.vy, q.vz) {
                out.bits = pack(q.vx, q.vy, q.vz);
                out.valid = 1;
            }
        }
        OP_UNPACK => {
            let (nx, ny, nz) = unpack(q.bits);
            out.ox = nx;
            out.oy = ny;
            out.oz = nz;
            out.valid = 1;
        }
        _ => {}
    }
    out
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag and the packed `bits` word exactly, continuous outputs to
/// tolerance when valid, dispatched on the query's operation.
fn assert_parity(gpu: &OctahedralEncodeResult, q: &OctahedralEncodeQuery, label: &str) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    if want.valid != 1 {
        return;
    }
    match q.op_id {
        OP_ENCODE => {
            assert!(
                close(gpu.ox, want.ox) && close(gpu.oy, want.oy),
                "{label}: encode mismatch gpu=({},{}) oracle=({},{})",
                gpu.ox,
                gpu.oy,
                want.ox,
                want.oy
            );
        }
        OP_DECODE | OP_UNPACK => {
            assert!(
                close(gpu.ox, want.ox) && close(gpu.oy, want.oy) && close(gpu.oz, want.oz),
                "{label}: direction mismatch gpu=({},{},{}) oracle=({},{},{})",
                gpu.ox,
                gpu.oy,
                gpu.oz,
                want.ox,
                want.oy,
                want.oz
            );
        }
        OP_PACK => {
            assert_eq!(
                gpu.bits, want.bits,
                "{label}: packed bits mismatch gpu={} oracle={}",
                gpu.bits, want.bits
            );
        }
        _ => {}
    }
}

#[test]
fn encode_of_up_axis_is_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    // +Z maps to the square origin.
    let q = OctahedralEncodeQuery::encode(0.0, 0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].ox, 0.0) && close(out[0].oy, 0.0));
    assert_parity(&out[0], &q, "encode_up");
}

#[test]
fn encode_folds_lower_hemisphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    // A clearly lower-hemisphere direction exercises the fold branch.
    let q = OctahedralEncodeQuery::encode(0.3, -0.4, -0.6);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "encode_fold");
}

#[test]
fn decode_is_always_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let q = OctahedralEncodeQuery::decode(0.25, -0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "decode");
}

#[test]
fn encode_decode_round_trip_recovers_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let axes = [
        [1.0_f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for (i, a) in axes.iter().enumerate() {
        let enc_q = OctahedralEncodeQuery::encode(a[0], a[1], a[2]);
        let enc = gpu.evaluate(&ctx, std::slice::from_ref(&enc_q));
        assert_eq!(enc[0].valid, 1);
        assert_parity(&enc[0], &enc_q, &format!("axis_encode[{i}]"));
        let dec_q = OctahedralEncodeQuery::decode(enc[0].ox, enc[0].oy);
        let dec = gpu.evaluate(&ctx, std::slice::from_ref(&dec_q));
        assert_eq!(dec[0].valid, 1);
        assert_parity(&dec[0], &dec_q, &format!("axis_decode[{i}]"));
        // The decoded direction is a unit vector along the original axis.
        assert!(
            close(dec[0].ox, a[0]) && close(dec[0].oy, a[1]) && close(dec[0].oz, a[2]),
            "axis[{i}] round-trip gpu=({},{},{}) want=({},{},{})",
            dec[0].ox,
            dec[0].oy,
            dec[0].oz,
            a[0],
            a[1],
            a[2]
        );
    }
}

#[test]
fn pack_unpack_round_trip_is_bit_exact_against_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let dirs = [
        [0.2_f32, 0.3, 0.9],
        [-0.5, 0.1, 0.7],
        [0.6, -0.4, -0.5],
        [-0.3, -0.7, -0.2],
    ];
    for (i, d) in dirs.iter().enumerate() {
        let pack_q = OctahedralEncodeQuery::pack(d[0], d[1], d[2]);
        let packed = gpu.evaluate(&ctx, std::slice::from_ref(&pack_q));
        assert_eq!(packed[0].valid, 1);
        // The packed word matches the oracle bit-for-bit.
        assert_parity(&packed[0], &pack_q, &format!("pack[{i}]"));
        let unpack_q = OctahedralEncodeQuery::unpack(packed[0].bits);
        let unpacked = gpu.evaluate(&ctx, std::slice::from_ref(&unpack_q));
        assert_eq!(unpacked[0].valid, 1);
        assert_parity(&unpacked[0], &unpack_q, &format!("unpack[{i}]"));
        // The unpacked direction stays close to the normalized input.
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (nx, ny, nz) = (d[0] / len, d[1] / len, d[2] / len);
        let dot = unpacked[0].ox * nx + unpacked[0].oy * ny + unpacked[0].oz * nz;
        assert!(dot > 0.999, "pack/unpack[{i}] drifted: dot={dot}");
    }
}

#[test]
fn degenerate_vectors_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let zero_encode = OctahedralEncodeQuery::encode(0.0, 0.0, 0.0);
    let zero_pack = OctahedralEncodeQuery::pack(0.0, 0.0, 0.0);
    let nan_encode = OctahedralEncodeQuery::encode(f32::NAN, 0.0, 1.0);
    let inf_pack = OctahedralEncodeQuery::pack(f32::INFINITY, 0.0, 0.0);
    let queries = [zero_encode, zero_pack, nan_encode, inf_pack];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 4);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "degenerate[{i}] should be invalid");
        assert_eq!(res.ox, 0.0);
        assert_eq!(res.oy, 0.0);
        assert_eq!(res.oz, 0.0);
        assert_eq!(res.bits, 0);
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn unknown_op_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    // An out-of-range operation selector reports an invalid, zeroed result.
    let q = OctahedralEncodeQuery {
        op_id: 7,
        bits: 0,
        vx: 0.3,
        vy: 0.4,
        vz: 0.5,
    };
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "unknown_op");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let queries = vec![
        OctahedralEncodeQuery::encode(0.1, 0.2, 0.95),
        OctahedralEncodeQuery::decode(0.3, -0.4),
        OctahedralEncodeQuery::pack(-0.6, 0.3, 0.7),
        OctahedralEncodeQuery::unpack(0x1234_ABCD),
        OctahedralEncodeQuery::encode(0.0, 0.0, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 1);
    assert_eq!(out[4].valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralEncode::new(&ctx);
    let mut lcg = Lcg::new(0x0C7A_15ED);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let op = lcg.next_u32() & 3;
        let q = match op {
            0 => {
                // A random direction, rejected if its L1 norm is near zero.
                let x = lcg.next_range(-1.0, 1.0);
                let y = lcg.next_range(-1.0, 1.0);
                let z = lcg.next_range(-1.0, 1.0);
                if x.abs() + y.abs() + z.abs() < 0.1 {
                    continue;
                }
                let len = (x * x + y * y + z * z).sqrt();
                OctahedralEncodeQuery::encode(x / len, y / len, z / len)
            }
            1 => {
                // Octahedral coordinates inside the valid square.
                let ex = lcg.next_range(-0.95, 0.95);
                let ey = lcg.next_range(-0.95, 0.95);
                OctahedralEncodeQuery::decode(ex, ey)
            }
            2 => {
                let x = lcg.next_range(-1.0, 1.0);
                let y = lcg.next_range(-1.0, 1.0);
                let z = lcg.next_range(-1.0, 1.0);
                if x.abs() + y.abs() + z.abs() < 0.1 {
                    continue;
                }
                let len = (x * x + y * y + z * z).sqrt();
                let (nx, ny, nz) = (x / len, y / len, z / len);
                // Reject directions whose snorm scaling lands near an X.5 tie,
                // where a 1-ULP difference between host and device could flip
                // the rounded integer and break the bit-exact comparison.
                if near_snorm_tie(nx, ny, nz) {
                    continue;
                }
                OctahedralEncodeQuery::pack(nx, ny, nz)
            }
            _ => OctahedralEncodeQuery::unpack(lcg.next_u32()),
        };
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// True when either octahedral channel's 16-bit snorm scaling lands within a
/// small band of an `X.5` rounding tie, where a 1-ULP host/device difference
/// could flip the rounded integer. The sweep rejects such directions so the
/// packed `bits` word stays bit-exact.
fn near_snorm_tie(x: f32, y: f32, z: f32) -> bool {
    let (ex, ey) = encode(x, y, z);
    [ex, ey].iter().any(|&e| {
        let scaled = e.clamp(-1.0, 1.0) * 32767.0;
        let frac = scaled.abs().fract();
        (frac - 0.5).abs() < 0.01
    })
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
