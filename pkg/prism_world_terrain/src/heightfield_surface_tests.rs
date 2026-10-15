//! Exact planar surface queries over genuinely baked heightfield snapshots.

use alloc::sync::Arc;
use core::convert::Infallible;

use crate::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldQueryError,
    HeightfieldSpec, HeightfieldSurfaceSample, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey,
};

struct GridSampler<'a> {
    side: u32,
    heights_mm: &'a [i64],
}

impl HeightSampler for GridSampler<'_> {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        // Invert the baker's normalized, rounded UV coordinates for our small
        // grids. This is a real sampler: no private tile payload is fabricated.
        let node = |uv| ((u64::from(uv) * u64::from(self.side - 1) + 32_767) / 65_535) as usize;
        let x = node(sample.u_q16());
        let z = node(sample.v_q16());
        Ok(HeightSampleValue::from_millimetres(
            self.heights_mm[z * self.side as usize + x],
        ))
    }
}

fn bake_at(tile: TileKey, side: u32, scale_mm: i64, heights_mm: &[i64]) -> Arc<HeightfieldTile> {
    let count = side as usize * side as usize;
    assert_eq!(heights_mm.len(), count);
    Arc::new(
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(side, scale_mm).expect("valid test grid"),
            &GridSampler { side, heights_mm },
            BakeLimits {
                max_samples: count,
                max_bytes: count * 2,
            },
        )
        .expect("real bounded heightfield bake"),
    )
}

fn bake(side: u32, scale_mm: i64, heights_mm: &[i64]) -> Arc<HeightfieldTile> {
    bake_at(
        TileKey::root(CubeFace::PositiveY),
        side,
        scale_mm,
        heights_mm,
    )
}

fn embedding(origin: [i64; 3], spacing: u32) -> PlanarHeightfieldEmbedding {
    PlanarHeightfieldEmbedding {
        sample_origin_mm: origin,
        sample_spacing_mm: spacing,
    }
}

fn surface(tile: Arc<HeightfieldTile>, origin: [i64; 3], spacing: u32) -> PlanarHeightfieldSurface {
    PlanarHeightfieldSurface::new(tile, embedding(origin, spacing)).expect("valid planar surface")
}

fn check_sample(
    surface: &PlanarHeightfieldSurface,
    xz_mm: [i64; 2],
    numerator: i128,
    denominator: u32,
    normal: [i128; 3],
) -> HeightfieldSurfaceSample {
    let sample = surface.sample_xz_mm(xz_mm).expect("query inside surface");
    assert_eq!(sample.height_numerator_mm(), numerator, "query={xz_mm:?}");
    assert_eq!(sample.height_denominator(), denominator, "query={xz_mm:?}");
    assert_eq!(sample.normal_mm(), normal, "query={xz_mm:?}");
    sample
}

fn check_outside(surface: &PlanarHeightfieldSurface, xz_mm: [i64; 2]) {
    assert!(
        matches!(
            surface.sample_xz_mm(xz_mm),
            Err(HeightfieldQueryError::OutsideSurface)
        ),
        "query={xz_mm:?} must be outside"
    );
}

#[test]
fn flat_baked_surface_preserves_snapshot_embedding_and_exact_bounds() {
    let tile = bake(3, 1, &[-37; 9]);
    assert_eq!(tile.samples(), &[-37; 9]);
    let surface = surface(Arc::clone(&tile), [-13, 101, -23], 7);
    assert!(Arc::ptr_eq(surface.tile(), &tile));
    assert_eq!(surface.embedding().sample_origin_mm, [-13, 101, -23]);
    assert_eq!(surface.embedding().sample_spacing_mm, 7);
    assert_eq!(surface.min_xz_mm(), [-13, -23]);
    assert_eq!(surface.max_xz_mm(), [1, -9]);
    for z in -23..-9 {
        for x in -13..1 {
            check_sample(&surface, [x, z], 64 * 7, 7, [0, 7, 0]);
        }
    }
    assert_eq!(tile.revision(), 0);
    assert_eq!(tile.samples(), &[-37; 9]);
}

#[test]
fn flat_negative_height_includes_negative_y_origin_without_rounding() {
    let surface = surface(bake(2, 1, &[-37; 4]), [4, -101, 8], 3);
    check_sample(&surface, [5, 9], -138 * 3, 3, [0, 3, 0]);
}

#[test]
fn ramp_matches_exact_affine_height_in_both_triangles_and_all_cells() {
    // Encoded nodes: -11 + 3*x - 5*z. The scale is two millimetres.
    let tile = bake(3, 2, &[-22, -16, -10, -32, -26, -20, -42, -36, -30]);
    assert_eq!(tile.samples(), &[-11, -8, -5, -16, -13, -10, -21, -18, -15]);
    let surface = surface(tile, [-30, 17, -70], 7);
    for rz in 0..14_i64 {
        for rx in 0..14_i64 {
            check_sample(
                &surface,
                [-30 + rx, -70 + rz],
                -5 * 7 + 6 * i128::from(rx) - 10 * i128::from(rz),
                7,
                [-6, 7, 10],
            );
        }
    }
    // A genuinely fractional negative height must not be truncated or rounded.
    check_sample(&surface, [-29, -69], -39, 7, [-6, 7, 10]);
}

#[test]
fn noncoplanar_quad_uses_a_c_b_and_b_c_d_not_bilinear_interpolation() {
    // Row-major a=0, b=10, c=20, d=100. The world Y origin is seven.
    let surface = surface(bake(2, 1, &[0, 10, 20, 100]), [-20, 7, -30], 10);
    let first = check_sample(&surface, [-18, -27], 150, 10, [-10, 10, -20]);
    let second = check_sample(&surface, [-12, -24], 550, 10, [-80, 10, -90]);
    // Independent bilinear numerators over denominator 100 would be 1920
    // and 6060, respectively (including the world Y origin). They differ.
    assert_ne!(first.height_numerator_mm() * 10, 1_920);
    assert_ne!(second.height_numerator_mm() * 10, 6_060);
}

#[test]
fn diagonal_ties_choose_first_triangle_normal_and_neighbors_choose_each_side() {
    let surface = surface(bake(2, 1, &[0, 10, 20, 100]), [-20, 7, -30], 10);
    for rx in 1..10_i64 {
        let rz = 10 - rx;
        check_sample(
            &surface,
            [-20 + rx, -30 + rz],
            70 + 10 * i128::from(rx) + 20 * i128::from(rz),
            10,
            [-10, 10, -20],
        );
    }
    check_sample(&surface, [-16, -25], 210, 10, [-10, 10, -20]);
    check_sample(&surface, [-16, -24], 230, 10, [-10, 10, -20]);
    check_sample(&surface, [-15, -24], 310, 10, [-80, 10, -90]);
}

#[test]
fn small_quads_exhaustively_match_integer_barycentric_oracle() {
    for heights in [
        [0_i64, 10, 20, 100],
        [-32_768, 32_767, 12_345, -23_456],
        [29, -7, -101, 83],
    ] {
        let surface = surface(bake(2, 1, &heights), [-11, -19, -31], 9);
        let [a, b, c, d] = heights.map(i128::from);
        for rz in 0..9_i64 {
            for rx in 0..9_i64 {
                let x = i128::from(rx);
                let z = i128::from(rz);
                // Triangle vertex weights sum to spacing, not spacing^2.
                // Keep this oracle in barycentric form rather than computing
                // the implementation's height-at-origin plus gradient form.
                let (height, normal) = if rx + rz <= 9 {
                    (a * (9 - x - z) + b * x + c * z, [a - b, 9, a - c])
                } else {
                    (
                        b * (9 - z) + c * (9 - x) + d * (x + z - 9),
                        [c - d, 9, b - d],
                    )
                };
                check_sample(&surface, [-11 + rx, -31 + rz], -19 * 9 + height, 9, normal);
            }
        }
    }
}

#[test]
fn interior_grid_lines_belong_to_cell_at_that_grid_coordinate() {
    // h(x,z)=x^2+10*z^2 at the NODES, giving distinct adjacent gradients.
    let surface = surface(bake(3, 1, &[0, 1, 4, 10, 11, 14, 40, 41, 44]), [0; 3], 5);
    check_sample(&surface, [5, 2], 25, 5, [-3, 5, -10]);
    check_sample(&surface, [2, 5], 52, 5, [-1, 5, -30]);
    check_sample(&surface, [5, 5], 55, 5, [-3, 5, -30]);
    check_sample(&surface, [9, 9], 187, 5, [-3, 5, -30]);
}

#[test]
fn surface_owns_half_open_xz_and_rejects_all_four_outer_edges_and_corners() {
    let surface = surface(bake(3, 1, &[0; 9]), [-13, 77, -23], 7);
    for xz in [[-13, -23], [-13, -10], [0, -23], [0, -10]] {
        check_sample(&surface, xz, 77 * 7, 7, [0, 7, 0]);
    }
    for xz in [
        [-14, -23],
        [-14, -10],
        [1, -23],
        [1, -10],
        [-13, -24],
        [0, -24],
        [-13, -9],
        [0, -9],
        [-14, -24],
        [1, -24],
        [-14, -9],
        [1, -9],
        [i64::MIN, i64::MIN],
        [i64::MAX, i64::MAX],
    ] {
        check_outside(&surface, xz);
    }
}

#[test]
fn negative_coordinates_beyond_f64_integer_precision_preserve_single_millimetres() {
    let origin = [
        -(1_i64 << 53) - 101,
        -(1_i64 << 53) - 39,
        -(1_i64 << 54) - 7,
    ];
    let surface = surface(bake(2, 1, &[0, 2, 5, 7]), origin, 17);
    for rz in 0..17_i64 {
        for rx in 0..17_i64 {
            check_sample(
                &surface,
                [origin[0] + rx, origin[2] + rz],
                i128::from(origin[1]) * 17 + 2 * i128::from(rx) + 5 * i128::from(rz),
                17,
                [-2, 17, -5],
            );
        }
    }
    let a = surface.sample_xz_mm([origin[0], origin[2]]).unwrap();
    let b = surface.sample_xz_mm([origin[0] + 1, origin[2]]).unwrap();
    assert_eq!(b.height_numerator_mm() - a.height_numerator_mm(), 2);
}

#[test]
fn zero_spacing_is_invalid_even_when_origin_is_extreme() {
    let tile = bake(2, 1, &[0; 4]);
    for origin in [[0; 3], [i64::MIN; 3], [i64::MAX; 3]] {
        assert!(matches!(
            PlanarHeightfieldSurface::new(Arc::clone(&tile), embedding(origin, 0)),
            Err(HeightfieldQueryError::InvalidSpacing)
        ));
    }
}

#[test]
fn constructor_checks_x_and_z_extent_without_clamping_or_wrapping() {
    let tile = bake(2, 1, &[0; 4]);
    for origin in [
        [i64::MAX - 6, 0, 0],
        [0, 0, i64::MAX - 6],
        [i64::MAX, i64::MIN, i64::MAX],
    ] {
        assert!(matches!(
            PlanarHeightfieldSurface::new(Arc::clone(&tile), embedding(origin, 7)),
            Err(HeightfieldQueryError::CoordinateOverflow)
        ));
    }
    // Two cells at the largest u32 spacing: extent itself must not use u32.
    let tile = bake(3, 1, &[0; 9]);
    let extent = 2 * i64::from(u32::MAX);
    assert!(matches!(
        PlanarHeightfieldSurface::new(tile, embedding([i64::MAX - extent + 1, 0, 0], u32::MAX)),
        Err(HeightfieldQueryError::CoordinateOverflow)
    ));
}

#[test]
fn world_edges_and_largest_spacing_have_exact_bounds_and_safe_outside_queries() {
    let extent = 2 * i64::from(u32::MAX);
    let origin = [i64::MIN, 0, i64::MAX - extent];
    let surface = surface(bake(3, 1, &[0; 9]), origin, u32::MAX);
    assert_eq!(surface.min_xz_mm(), [i64::MIN, i64::MAX - extent]);
    assert_eq!(surface.max_xz_mm(), [i64::MIN + extent, i64::MAX]);
    for xz in [
        [origin[0], origin[2]],
        [origin[0] + i64::from(u32::MAX), origin[2]],
        [i64::MIN + extent - 1, i64::MAX - 1],
    ] {
        check_sample(&surface, xz, 0, u32::MAX, [0, i128::from(u32::MAX), 0]);
    }
    for xz in [
        [i64::MIN + extent, origin[2]],
        [origin[0], i64::MAX],
        [i64::MAX, origin[2]],
        [origin[0], i64::MIN],
    ] {
        // Some out-of-domain differences exceed i64::MAX; classify before
        // narrowing or subtract in widened arithmetic, never overflow first.
        check_outside(&surface, xz);
    }
}

#[test]
fn maximum_height_scale_keeps_widened_height_origin_and_gradient() {
    let tile = bake(2, i64::MAX, &[i64::MAX, i64::MIN, i64::MAX, i64::MIN]);
    assert_eq!(tile.samples(), &[1, -1, 1, -1]);
    let surface = surface(tile, [0, i64::MAX, 0], u32::MAX);
    let height = i128::from(i64::MAX);
    let spacing = i128::from(u32::MAX);
    let normal = [2 * height, spacing, 0];
    check_sample(&surface, [0, 0], 2 * height * spacing, u32::MAX, normal);
    check_sample(
        &surface,
        [i64::from(u32::MAX) - 1, 1],
        2 * height,
        u32::MAX,
        normal,
    );
}

#[test]
fn real_bake_can_decode_above_i64_and_query_remains_exact() {
    // MAX / scale rounds to two, although 2*scale no longer fits i64.
    // Build through the public sampler/baker rather than forging encoded data.
    let scale = i64::MAX / 2 + 2;
    let tile = bake(2, scale, &[i64::MAX; 4]);
    assert_eq!(tile.samples(), &[2; 4]);
    let decoded = 2 * i128::from(scale);
    assert!(decoded > i128::from(i64::MAX));
    assert_eq!(tile.height_mm([0, 0]), Some(decoded));
    let surface = surface(tile, [0, i64::MAX, 0], u32::MAX);
    check_sample(
        &surface,
        [i64::from(u32::MAX) - 1, i64::from(u32::MAX) - 1],
        (decoded + i128::from(i64::MAX)) * i128::from(u32::MAX),
        u32::MAX,
        [0, i128::from(u32::MAX), 0],
    );
}

#[test]
fn real_bake_can_decode_below_i64_and_query_remains_exact() {
    let scale = i64::MAX / 2 + 2;
    let tile = bake(2, scale, &[i64::MIN; 4]);
    assert_eq!(tile.samples(), &[-2; 4]);
    let decoded = -2 * i128::from(scale);
    assert!(decoded < i128::from(i64::MIN));
    assert_eq!(tile.height_mm([1, 1]), Some(decoded));
    let surface = surface(tile, [0, i64::MIN, 0], u32::MAX);
    check_sample(
        &surface,
        [1, 1],
        (decoded + i128::from(i64::MIN)) * i128::from(u32::MAX),
        u32::MAX,
        [0, i128::from(u32::MAX), 0],
    );
}

#[test]
fn decoded_extreme_ramp_uses_i128_differences_and_exact_fraction() {
    let scale = i64::MAX / 2 + 2;
    let tile = bake(2, scale, &[i64::MIN, i64::MAX, i64::MIN, i64::MAX]);
    assert_eq!(tile.samples(), &[-2, 2, -2, 2]);
    let surface = surface(tile, [0, i64::MIN, 0], 3);
    let scale = i128::from(scale);
    check_sample(
        &surface,
        [1, 1],
        i128::from(i64::MIN) * 3 - 2 * scale,
        3,
        [-4 * scale, 3, 0],
    );
    check_sample(
        &surface,
        [2, 2],
        i128::from(i64::MIN) * 3 + 2 * scale,
        3,
        [-4 * scale, 3, 0],
    );
}

#[test]
fn equality_compares_snapshot_content_not_arc_identity_or_tile_revision_alone() {
    let first = bake(2, 1, &[12; 4]);
    let identical = bake(2, 1, &[12; 4]);
    let different = bake(2, 1, &[13; 4]);
    assert!(!Arc::ptr_eq(&first, &identical));
    assert!(!Arc::ptr_eq(&first, &different));
    assert_eq!(first.tile(), different.tile());
    assert_eq!(first.revision(), different.revision());
    assert_eq!(first.revision(), 0);
    let a = surface(Arc::clone(&first), [-7, 23, -11], 5);
    let shared = surface(Arc::clone(&first), [-7, 23, -11], 5);
    let b = surface(identical, [-7, 23, -11], 5);
    let c = surface(different, [-7, 23, -11], 5);
    assert!(a == shared);
    assert!(a == b, "independent identical snapshots compare equal");
    assert!(
        a != c,
        "same tile/revision does not identify sample content"
    );
    check_sample(&a, [-6, -10], 35 * 5, 5, [0, 5, 0]);
    check_sample(&c, [-6, -10], 36 * 5, 5, [0, 5, 0]);
}

#[test]
fn equality_includes_embedding_tile_key_and_encoded_height_specification() {
    let tile = bake(2, 1, &[12; 4]);
    let a = surface(Arc::clone(&tile), [0; 3], 5);
    for (origin, spacing) in [([1, 0, 0], 5), ([0, 1, 0], 5), ([0, 0, 1], 5), ([0; 3], 6)] {
        assert!(a != surface(Arc::clone(&tile), origin, spacing));
    }
    let other_tile = bake_at(TileKey::root(CubeFace::NegativeY), 2, 1, &[12; 4]);
    assert!(a != surface(other_tile, [0; 3], 5));
    // Identical decoded millimetres do not erase the snapshot's encoded spec.
    let other_scale = bake(2, 2, &[12; 4]);
    assert_eq!(other_scale.height_mm([0, 0]), tile.height_mm([0, 0]));
    assert!(a != surface(other_scale, [0; 3], 5));
}

#[test]
fn repeated_queries_are_deterministic_and_do_not_mutate_baked_snapshot() {
    let tile = bake(2, 1, &[0, 10, 20, 100]);
    let surface = surface(Arc::clone(&tile), [0, -7, 0], 10);
    let revision = tile.revision();
    for _ in 0..64 {
        check_sample(&surface, [2, 3], 10, 10, [-10, 10, -20]);
        check_sample(&surface, [8, 6], 410, 10, [-80, 10, -90]);
        check_outside(&surface, [10, 0]);
    }
    assert_eq!(tile.revision(), revision);
    assert_eq!(tile.samples(), &[0, 10, 20, 100]);
    assert!(Arc::ptr_eq(surface.tile(), &tile));
}
