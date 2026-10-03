//! Unit tests for the device-free `DMM` baker.

use alloc::vec;
use alloc::vec::Vec;

use crate::dmm::{
    bake_triangle, barycentric_f32, dequantize_unorm11, displaced_position, micro_vertex_at,
    micro_vertices, pack_unorm11, packed_len, quantize_unorm11, unpack_unorm11, DisplacementMap,
    DisplacementScaleBias, DmmBakeInput, DmmBuilder, DmmSubdivisionLevel, ScaleBiasMode,
    TextureDisplacementMap, WrapMode, UNORM11_MAX,
};

/// Approximate float equality for deterministic scalar checks.
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

/// A canonical unit base triangle covering the lower-left `UV` corner.
const BASE_TRI: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

fn level(l: u8) -> DmmSubdivisionLevel {
    DmmSubdivisionLevel::new(l).expect("valid level")
}

#[test]
fn micro_vertex_count_matches_closed_form() {
    for l in 0..=5u8 {
        let lvl = level(l);
        let n = lvl.segments();
        let expected = (n + 1) * (n + 2) / 2;
        assert_eq!(lvl.micro_vertex_count(), expected);
        assert_eq!(lvl.micro_triangle_count(), 1u32 << (2 * u32::from(l)));
    }
}

#[test]
fn micro_vertex_at_matches_iteration_and_sums_to_n() {
    for l in 0..=5u8 {
        let lvl = level(l);
        let n = lvl.segments();
        let all = micro_vertices(lvl);
        assert_eq!(all.len() as u32, lvl.micro_vertex_count());
        for (i, &v) in all.iter().enumerate() {
            let via_at = micro_vertex_at(lvl, i as u32).expect("in range");
            assert_eq!(via_at, v);
            assert_eq!(v[0] + v[1] + v[2], n);
        }
        assert!(micro_vertex_at(lvl, lvl.micro_vertex_count()).is_none());
    }
}

#[test]
fn barycentric_weights_sum_to_one() {
    let lvl = level(3);
    for v in micro_vertices(lvl) {
        let b = barycentric_f32(lvl, v);
        assert!(approx(b[0] + b[1] + b[2], 1.0));
    }
}

#[test]
fn quantize_endpoints() {
    assert_eq!(quantize_unorm11(0.0), 0);
    assert_eq!(quantize_unorm11(1.0), UNORM11_MAX);
    // Clamping outside the domain.
    assert_eq!(quantize_unorm11(-0.5), 0);
    assert_eq!(quantize_unorm11(2.0), UNORM11_MAX);
}

#[test]
fn quantize_round_trip_within_half_step() {
    let half_step = 0.5 / f32::from(UNORM11_MAX);
    let mut x = 0.0f32;
    while x <= 1.0 {
        let back = dequantize_unorm11(quantize_unorm11(x));
        assert!((back - x).abs() <= half_step + 1e-7, "x = {x}");
        x += 1.0 / 97.0;
    }
}

#[test]
fn pack_unpack_round_trip_various_counts() {
    for &count in &[0usize, 1, 2, 3, 7, 8, 15, 100] {
        let codes: Vec<u16> = (0..count).map(|i| (i as u16 * 37) & 0x07FF).collect();
        let packed = pack_unorm11(&codes);
        assert_eq!(packed.len(), packed_len(count));
        let back = unpack_unorm11(&packed, count).expect("enough bytes");
        assert_eq!(back, codes);
    }
}

#[test]
fn packed_len_values() {
    assert_eq!(packed_len(0), 0);
    assert_eq!(packed_len(1), 2); // 11 bits -> 2 bytes
    assert_eq!(packed_len(8), 11); // 88 bits -> 11 bytes
    assert_eq!(packed_len(16), 22);
}

#[test]
fn unpack_rejects_short_buffer() {
    let packed = pack_unorm11(&[2047, 1, 0]);
    assert!(unpack_unorm11(&packed, 3).is_some());
    assert!(unpack_unorm11(&packed[..1], 3).is_none());
}

/// A displacement map returning a single constant height everywhere.
struct FlatMap(f32);
impl DisplacementMap for FlatMap {
    fn sample_height(&self, _u: f32, _v: f32) -> f32 {
        self.0
    }
}

#[test]
fn flat_map_bakes_all_equal_codes() {
    let input = DmmBakeInput {
        uv: BASE_TRI,
        level: level(3),
        scale_bias_mode: ScaleBiasMode::PerTriangle,
    };
    let baked = bake_triangle(&input, &FlatMap(4.2));
    assert!(baked.scale_bias().is_degenerate());
    assert!(baked.codes().iter().all(|&c| c == 0));
}

#[test]
fn ramp_map_bakes_monotonic_codes_along_u() {
    // Height increases with U. A wide ramp texture so bilinear sampling of the
    // interior stays monotonic.
    let width = 16u32;
    let data: Vec<f32> = (0..width).map(|x| x as f32).collect();
    let map = TextureDisplacementMap::new(width, 1, data, WrapMode::Clamp);
    let input = DmmBakeInput {
        uv: BASE_TRI,
        level: level(3),
        scale_bias_mode: ScaleBiasMode::PerTriangle,
    };
    let baked = bake_triangle(&input, &map);

    // The micro-vertex with w1 == n (full weight on UV vertex 1 = (1,0)) is the
    // largest-U vertex and must carry the max code; w0 == n carries code 0.
    let n = baked.level().segments();
    let verts = micro_vertices(baked.level());
    let codes = baked.codes();
    let max_idx = verts.iter().position(|v| v[1] == n).unwrap();
    let min_idx = verts.iter().position(|v| v[0] == n).unwrap();
    assert_eq!(codes[max_idx], UNORM11_MAX);
    assert_eq!(codes[min_idx], 0);
    // Some genuine variation was produced.
    assert!(codes.iter().any(|&c| c > 0 && c < UNORM11_MAX));
}

#[test]
fn scale_bias_maps_min_and_max() {
    let sb = DisplacementScaleBias::new(5.0, 10.0);
    assert!(approx(sb.normalize(5.0), 0.0));
    assert!(approx(sb.normalize(10.0), 1.0));
    assert_eq!(sb.quantize(5.0), 0);
    assert_eq!(sb.quantize(10.0), UNORM11_MAX);
    assert!(approx(sb.denormalize(0.5), 7.5));
}

#[test]
fn scale_bias_degenerate_is_zero() {
    let sb = DisplacementScaleBias::new(3.0, 3.0);
    assert!(sb.is_degenerate());
    assert!(approx(sb.normalize(3.0), 0.0));
    assert_eq!(sb.quantize(3.0), 0);
}

#[test]
fn builder_dedups_identical_and_separates_distinct() {
    let flat = FlatMap(1.0);
    let ramp_data: Vec<f32> = (0..8u32).map(|x| x as f32).collect();
    let ramp = TextureDisplacementMap::new(8, 1, ramp_data, WrapMode::Clamp);

    let flat_input = DmmBakeInput {
        uv: BASE_TRI,
        level: level(2),
        scale_bias_mode: ScaleBiasMode::PerTriangle,
    };
    let ramp_input = DmmBakeInput {
        uv: BASE_TRI,
        level: level(2),
        scale_bias_mode: ScaleBiasMode::PerTriangle,
    };

    let mut builder = DmmBuilder::new();
    builder.add_triangle(&flat_input, &flat);
    builder.add_triangle(&flat_input, &flat); // identical -> dedup
    builder.add_triangle(&ramp_input, &ramp); // distinct
    let out = builder.finish();

    assert_eq!(out.triangle_count(), 3);
    assert_eq!(out.unique_count(), 2);
    assert_eq!(out.indices, vec![0, 0, 1]);
}

#[test]
fn displaced_position_zero_height_is_barycentric_base() {
    let base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let dirs = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    let bary = [0.25, 0.5, 0.25];
    let p = displaced_position(base, dirs, bary, 0.0);
    assert!(approx(p[0], 0.5));
    assert!(approx(p[1], 0.25));
    assert!(approx(p[2], 0.0));

    // Non-zero height offsets along the (constant) direction.
    let q = displaced_position(base, dirs, bary, 2.0);
    assert!(approx(q[2], 2.0));
}
