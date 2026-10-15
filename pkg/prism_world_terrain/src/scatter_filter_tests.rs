//! End-to-end terrain thinning over real baked tiles and the residency kernel.

use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::{convert::Infallible, mem::size_of};

use super::*;
use crate::scatter_filter::{
    scatter_with_heightfield, HeightfieldScatterBudget, HeightfieldScatterError,
    HeightfieldScatterFilters, HeightfieldScatterOutput, HEIGHTFIELD_SCATTER_FILTER_VERSION,
};
use prism_world_residency::{
    estimate_scatter_work, scatter, scatter_with_density_field, DensityBrush, DensityBrushOp,
    DensityNoise, FieldScatterBudget, FieldScatterOutput, ScatterBudget, ScatterCandidateKey,
    ScatterConfig, ScatterDensityField, ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig,
    ScatterFieldError, ScatterFieldSample, ScatterPoint, ScatterRegion,
};

// Successful fixtures are small. The only enormous regions are preflight-only
// overflow cases, with zero source budgets as a second enumeration guard.
const POINT_LIMIT: usize = 8_192;

struct GridHeights<'a> {
    side: u32,
    heights: &'a [i64],
}

impl HeightSampler for GridHeights<'_> {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |q: u16| ((u64::from(q) * u64::from(self.side - 1) + 32_767) / 65_535) as usize;
        Ok(HeightSampleValue::from_millimetres(
            self.heights[index(sample.v_q16()) * self.side as usize + index(sample.u_q16())],
        ))
    }
}

fn surface(
    side: u32,
    spacing: u32,
    origin: [i64; 3],
    scale: i64,
    heights: &[i64],
) -> PlanarHeightfieldSurface {
    assert_eq!(heights.len(), side as usize * side as usize);
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(side, scale).unwrap(),
        &GridHeights { side, heights },
        BakeLimits {
            max_samples: heights.len(),
            max_bytes: heights.len() * size_of::<i16>(),
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: origin,
            sample_spacing_mm: spacing,
        },
    )
    .unwrap()
}

fn config(spacing: u32, distance: u32) -> ScatterConfig {
    ScatterConfig {
        seed: 0x3a65_c807_9742_1eab,
        version: 7,
        salt: 0xf09c_1172,
        grid_spacing_mm: spacing,
        min_distance_mm: distance,
    }
}

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn source_budget() -> ScatterBudget {
    ScatterBudget {
        max_candidate_evaluations: 1_000_000,
        max_output_points: POINT_LIMIT,
        max_output_bytes: POINT_LIMIT * size_of::<ScatterPoint>(),
    }
}

fn field_budget() -> FieldScatterBudget {
    FieldScatterBudget {
        max_field_work: 1_000_000,
        max_output_bytes: POINT_LIMIT
            * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
    }
}

fn make_field(
    density: u16,
    noise: Option<DensityNoise>,
    brushes: Vec<DensityBrush>,
) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 91,
            version: 3,
            base_density: density,
            base_species: 11,
            noise,
        },
        brushes,
        budget,
    )
    .unwrap()
}

fn full_field() -> ScatterDensityField {
    make_field(u16::MAX, None, Vec::new())
}

fn unrestricted() -> HeightfieldScatterFilters {
    HeightfieldScatterFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        clearance_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

fn filtered(
    config: ScatterConfig,
    region: ScatterRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> HeightfieldScatterOutput {
    scatter_with_heightfield(
        config,
        region,
        source_budget(),
        field,
        field_budget(),
        surface,
        filters,
        HeightfieldScatterBudget {
            max_sample_reads: 4 * POINT_LIMIT as u64,
        },
    )
    .unwrap()
}

fn density_source(
    config: ScatterConfig,
    region: ScatterRegion,
    field: &ScatterDensityField,
) -> FieldScatterOutput {
    scatter_with_density_field(config, region, source_budget(), field, field_budget()).unwrap()
}

fn positions(output: &HeightfieldScatterOutput) -> Vec<[i64; 3]> {
    output
        .points
        .iter()
        .map(|point| point.position_mm)
        .collect()
}

#[test]
fn closed_height_and_signed_clearance_endpoints_do_not_project_points() {
    // h(x,z) = x - 2. Spacing-one scatter puts a candidate at every integer.
    let surface = surface(2, 4, [0, -2, 0], 1, &[0, 4, 0, 4]);
    let config = config(1, 0);
    let region = region([0, -3, 0], [3, 4, 3]);
    let field = full_field();
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-1, 0],
        clearance_mm: [-1, 1],
        max_slope_rise: 1,
        max_slope_run: 1,
    };
    let source = density_source(config, region, &field);
    let output = filtered(config, region, &field, &surface, filters);
    let expected: Vec<_> = source
        .points
        .iter()
        .copied()
        .filter(|point| {
            let [x, y, _] = point.position_mm;
            (1..=2).contains(&x) && (x - 3..=x - 1).contains(&y)
        })
        .collect();
    assert_eq!(output.points, expected);
    assert_eq!(output.points.len(), 18);
    for x in [1, 2] {
        for clearance in [-1, 1] {
            assert!(positions(&output).contains(&[x, x - 2 + clearance, 0]));
        }
    }
    assert!(output
        .points
        .iter()
        .all(|point| point.key.rebuild().unwrap() == *point));
    assert_eq!(output.attributes.len(), output.points.len());
    assert_eq!(output.candidate_evaluations, source.candidate_evaluations);
    assert_eq!(output.field_work, source.field_work);
    assert_eq!(output.surface_sample_reads, source.points.len() as u64 * 4);
    assert_eq!(output.points.capacity(), source.points.capacity());
    assert_eq!(output.attributes.capacity(), source.attributes.capacity());
}

#[test]
fn fractional_height_and_clearance_are_compared_without_integer_rounding() {
    // h(x,z) = (-3 + 2*x)/3: -1, -1/3, +1/3 at the requested X positions.
    let surface = surface(2, 3, [0, 0, 0], 1, &[-1, 1, -1, 1]);
    let config = config(1, 0);
    let region = region([0, -2, 0], [3, 3, 3]);
    let field = full_field();
    let source = density_source(config, region, &field);
    for (height_bounds, clearance_bounds) in [
        ([0, 0], [i64::MIN, i64::MAX]),
        ([-1, 0], [i64::MIN, i64::MAX]),
        ([i64::MIN, i64::MAX], [0, 0]),
        ([i64::MIN, i64::MAX], [-1, 0]),
        ([i64::MIN, i64::MAX], [0, 1]),
    ] {
        let filters = HeightfieldScatterFilters {
            surface_height_mm: height_bounds,
            clearance_mm: clearance_bounds,
            ..unrestricted()
        };
        let expected: Vec<_> = source
            .points
            .iter()
            .copied()
            .filter(|point| {
                let [x, y, _] = point.position_mm;
                let height = -3_i128 + 2 * i128::from(x);
                let clearance = 3 * i128::from(y) - height;
                let within = |value, bounds: [i64; 2]| {
                    i128::from(bounds[0]) * 3 <= value && value <= i128::from(bounds[1]) * 3
                };
                within(height, height_bounds) && within(clearance, clearance_bounds)
            })
            .collect();
        let output = filtered(config, region, &field, &surface, filters);
        assert_eq!(
            output.points, expected,
            "height={height_bounds:?}, clearance={clearance_bounds:?}"
        );
        if height_bounds == [0, 0] {
            assert!(
                output.points.is_empty(),
                "neither fractional value equals zero"
            );
        } else {
            assert!(!output.points.is_empty());
            assert!(output.points.len() < source.points.len());
        }
    }
}

#[test]
fn geometric_slope_uses_both_components_exact_rationals_and_closed_equality() {
    let field = full_field();
    for (spacing, heights, rise, run, accept) in [
        (5, [0, 3, 4, 7], 1, 1, true), // 3-4-5 equality.
        (5, [0, 3, 4, 7], u32::MAX - 1, u32::MAX, false),
        (5, [0, 3, 4, 7], 4, 5, false), // Each component fits; their norm does not.
        (10, [0, 3, 4, 7], 1, 2, true),
        (10, [0, 3, 4, 7], 2_147_483_647, u32::MAX, false),
        (3, [0, 1, 1, 2], 1, 2, true), // sqrt(2) <= 3/2, not floor(3/2).
        (3, [0, 1, 1, 2], 1, 3, false),
        (3, [2, 1, 1, 0], 1, 2, true), // Signs do not change slope.
        (3, [2, 1, 1, 0], 0, u32::MAX, false),
        (3, [2, 2, 2, 2], 0, u32::MAX, true),
    ] {
        let surface = surface(2, spacing, [0, 0, 0], 1, &heights);
        let region = region([0, 0, 0], [i64::from(spacing), 1, i64::from(spacing)]);
        let output = filtered(
            config(1, 0),
            region,
            &field,
            &surface,
            HeightfieldScatterFilters {
                max_slope_rise: rise,
                max_slope_run: run,
                ..unrestricted()
            },
        );
        assert_eq!(
            output.points.len(),
            if accept {
                spacing as usize * spacing as usize
            } else {
                0
            },
            "spacing={spacing}, heights={heights:?}, slope={rise}/{run}"
        );
    }
}

#[test]
fn face_slope_and_diagonal_ownership_follow_real_tile_topology() {
    // Only the upper b-c-d triangle rises. The shared diagonal belongs to a-c-b.
    let surface = surface(2, 4, [-2, 0, -2], 1, &[0, 0, 0, 4]);
    let region = region([-2, 0, -2], [2, 1, 2]);
    let field = full_field();
    let source = density_source(config(1, 0), region, &field);
    let output = filtered(
        config(1, 0),
        region,
        &field,
        &surface,
        HeightfieldScatterFilters {
            max_slope_rise: 0,
            ..unrestricted()
        },
    );
    let expected: Vec<_> = source
        .points
        .into_iter()
        .filter(|point| point.position_mm[0] + point.position_mm[2] <= 0)
        .collect();
    assert_eq!(output.points, expected);
    assert_eq!(output.points.len(), 13);
    assert!(positions(&output).contains(&[0, 0, 0]));
    assert!(!positions(&output).contains(&[1, 0, 1]));
}

fn painted_field() -> ScatterDensityField {
    make_field(
        46_000,
        Some(DensityNoise {
            cell_size_mm: 7,
            amplitude: 12_000,
            warp_cell_size_mm: 11,
            warp_amplitude_mm: 3,
        }),
        vec![
            DensityBrush {
                id: 9,
                center_mm: [5, 0, 0],
                inner_radius_mm: 7,
                outer_radius_mm: 12,
                opacity: u16::MAX,
                density: DensityBrushOp::Add(-20_000),
                species: Some(97),
            },
            DensityBrush {
                id: 2,
                center_mm: [-5, 0, 0],
                inner_radius_mm: 7,
                outer_radius_mm: 12,
                opacity: u16::MAX,
                density: DensityBrushOp::Add(4_000),
                species: Some(23),
            },
        ],
    )
}

#[test]
fn density_attributes_compact_in_order_without_repacking_or_losing_source_distance() {
    let surface = surface(3, 8, [-8, 0, -8], 1, &[0, 2, 4, -2, 0, 2, -4, -2, 0]);
    // This tile is h=(x-z)/4; the source is independently distance-thinned.
    let config = config(1, 2);
    let region = region([-6, -4, -6], [7, 5, 7]);
    let field = painted_field();
    let raw = scatter(config, region, source_budget()).unwrap();
    let source = density_source(config, region, &field);
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-2, 2],
        clearance_mm: [-2, 2],
        max_slope_rise: 1,
        max_slope_run: 2,
    };
    let output = filtered(config, region, &field, &surface, filters);
    let expected: Vec<_> = source
        .points
        .iter()
        .copied()
        .zip(source.attributes.iter().copied())
        .filter(|(point, _)| {
            let [x, y, z] = point.position_mm;
            let height = x - z;
            (-8..=8).contains(&height) && (-8..=8).contains(&(4 * y - height))
        })
        .collect();
    assert!(!expected.is_empty());
    assert!(output.points.len() < source.points.len());
    assert!(
        source.points.len() < raw.points.len(),
        "density must actually reject source points"
    );
    assert_eq!(
        output
            .points
            .iter()
            .copied()
            .zip(output.attributes.iter().copied())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(output.points.capacity(), source.points.capacity());
    assert_eq!(output.attributes.capacity(), source.attributes.capacity());
    assert_eq!(output.candidate_evaluations, raw.candidate_evaluations);
    assert_eq!(
        output.field_work,
        raw.points.len() as u64 * field.work_per_sample()
    );
    assert_eq!(output.surface_sample_reads, source.points.len() as u64 * 4);
    for (index, (point, attributes)) in output.points.iter().zip(&output.attributes).enumerate() {
        assert_eq!(*attributes, field.sample(point.position_mm));
        assert!(field.accepts(point.key, attributes.density));
        assert_eq!(point.key.rebuild().unwrap(), *point);
        assert!(region.contains_mm(point.position_mm));
        assert!(raw.points.contains(point));
        for other in &output.points[..index] {
            let distance_squared: i128 = point
                .position_mm
                .into_iter()
                .zip(other.position_mm)
                .map(|(a, b)| (i128::from(a) - i128::from(b)).pow(2))
                .sum();
            assert!(distance_squared >= i128::from(config.min_distance_mm).pow(2));
        }
    }
    assert_eq!(output.cache_key.source, source.cache_key);
}

fn identities(
    output: &HeightfieldScatterOutput,
) -> BTreeMap<ScatterCandidateKey, (ScatterPoint, ScatterFieldSample)> {
    let mut result = BTreeMap::new();
    assert_eq!(output.points.len(), output.attributes.len());
    for (&point, &attributes) in output.points.iter().zip(&output.attributes) {
        assert!(
            result.insert(point.key, (point, attributes)).is_none(),
            "duplicate candidate identity"
        );
    }
    result
}

#[test]
fn reversed_partitions_on_all_three_axes_preserve_the_entire_nonempty_set() {
    let surface = surface(2, 20, [-10, -2, -10], 1, &[0, 10, -10, 0]);
    let config = config(2, 3);
    let region = region([-8, -6, -8], [9, 7, 9]);
    let field = painted_field();
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-6, 6],
        clearance_mm: [-4, 6],
        max_slope_rise: 1,
        max_slope_run: 1,
    };
    let whole = filtered(config, region, &field, &surface, filters);
    let expected = identities(&whole);
    assert!(
        expected.len() >= 2,
        "a vacuous empty partition comparison is not useful"
    );
    let cuts = [[-8, -1, 3, 9], [-6, -1, 2, 7], [-8, 0, 5, 9]];
    // Separately split X, Y and Z, then split all three into 27 boxes.
    for split_axis in 0..4 {
        let mut partitions = Vec::new();
        for ix in 0..if split_axis == 0 || split_axis == 3 {
            3
        } else {
            1
        } {
            for iy in 0..if split_axis == 1 || split_axis == 3 {
                3
            } else {
                1
            } {
                for iz in 0..if split_axis == 2 || split_axis == 3 {
                    3
                } else {
                    1
                } {
                    let mut min = region.min_mm;
                    let mut max = region.max_mm;
                    for (axis, part) in [ix, iy, iz].into_iter().enumerate() {
                        if split_axis == axis || split_axis == 3 {
                            min[axis] = cuts[axis][part];
                            max[axis] = cuts[axis][part + 1];
                        }
                    }
                    partitions.push(ScatterRegion::new(min, max).unwrap());
                }
            }
        }
        let mut combined = BTreeMap::new();
        for partition in partitions.into_iter().rev() {
            let output = filtered(config, partition, &field, &surface, filters);
            assert!(output
                .points
                .iter()
                .all(|point| partition.contains_mm(point.position_mm)));
            for (key, value) in identities(&output) {
                assert!(
                    combined.insert(key, value).is_none(),
                    "overlap at a partition boundary"
                );
            }
        }
        assert_eq!(
            combined, expected,
            "split axis {split_axis} (3 means all axes)"
        );
    }
}

#[test]
fn cache_key_uses_sample_contents_embedding_filters_and_complete_source_recipe() {
    let a = surface(2, 4, [0, 0, 0], 1, &[0, 0, 0, 0]);
    let equal = surface(2, 4, [0, 0, 0], 1, &[0, 0, 0, 0]);
    let changed = surface(2, 4, [0, 0, 0], 1, &[0, 0, 0, 1]);
    assert_eq!(a.tile().tile(), changed.tile().tile());
    assert_eq!(a.tile().revision(), changed.tile().revision());
    assert!(!Arc::ptr_eq(a.tile(), equal.tile()));
    let config = config(1, 0);
    let region = region([0, 0, 0], [2, 1, 2]);
    let field = full_field();
    let base = filtered(config, region, &field, &a, unrestricted());
    let equal_output = filtered(config, region, &field, &equal, unrestricted());
    let changed_output = filtered(config, region, &field, &changed, unrestricted());
    assert_eq!(base.cache_key, equal_output.cache_key);
    assert_ne!(base.cache_key, changed_output.cache_key);
    assert_eq!(
        base.points, changed_output.points,
        "even an inactive corner invalidates the full key"
    );
    assert_eq!(base.cache_key.filter_version, 1);
    assert_eq!(
        base.cache_key.filter_version,
        HEIGHTFIELD_SCATTER_FILTER_VERSION
    );
    assert_eq!(base.cache_key.surface_version, HEIGHTFIELD_SURFACE_VERSION);
    assert!(
        Arc::ptr_eq(base.cache_key.surface.tile(), a.tile()),
        "key clones must share the tile snapshot"
    );
    for embedding in [
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, 1, 0],
            sample_spacing_mm: 4,
        },
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-1, 0, -1],
            sample_spacing_mm: 4,
        },
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, 0, 0],
            sample_spacing_mm: 5,
        },
    ] {
        let moved = PlanarHeightfieldSurface::new(Arc::clone(a.tile()), embedding).unwrap();
        let output = filtered(config, region, &field, &moved, unrestricted());
        assert_ne!(base.cache_key, output.cache_key);
    }
    for filters in [
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN + 1, i64::MAX],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN, i64::MAX - 1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            clearance_mm: [i64::MIN + 1, i64::MAX],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            clearance_mm: [i64::MIN, i64::MAX - 1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            max_slope_rise: u32::MAX - 1,
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            max_slope_run: 2,
            ..unrestricted()
        },
    ] {
        let output = filtered(config, region, &field, &a, filters);
        assert_eq!(base.points, output.points);
        assert_ne!(base.cache_key, output.cache_key);
    }
    for changed_config in [
        ScatterConfig {
            seed: config.seed + 1,
            ..config
        },
        ScatterConfig {
            salt: config.salt + 1,
            ..config
        },
        ScatterConfig {
            version: config.version + 1,
            ..config
        },
        ScatterConfig {
            grid_spacing_mm: 2,
            ..config
        },
        ScatterConfig {
            min_distance_mm: 1,
            ..config
        },
    ] {
        assert_ne!(
            base.cache_key,
            filtered(changed_config, region, &field, &a, unrestricted()).cache_key
        );
    }
    let different_field = make_field(u16::MAX - 1, None, Vec::new());
    assert_ne!(
        base.cache_key,
        filtered(config, region, &different_field, &a, unrestricted()).cache_key
    );
    assert_ne!(
        base.cache_key,
        filtered(
            config,
            ScatterRegion::new([0, 0, 0], [3, 1, 2]).unwrap(),
            &field,
            &a,
            unrestricted()
        )
        .cache_key
    );
    assert_eq!(base.cache_key.surface.tile().samples(), &[0, 0, 0, 0]);
}

#[test]
fn public_rebake_publishes_a_new_snapshot_without_changing_retained_output() {
    let old_surface = surface(2, 2, [0; 3], 1, &[0; 4]);
    let config = config(1, 0);
    let region = region([0; 3], [2; 3]);
    let field = full_field();
    let filters = HeightfieldScatterFilters {
        clearance_mm: [0, 0],
        max_slope_rise: 0,
        ..unrestricted()
    };
    let old_output = filtered(config, region, &field, &old_surface, filters);
    let old_key = old_output.cache_key.clone();
    let old_points = old_output.points.clone();
    let old_attributes = old_output.attributes.clone();
    let old_revision = old_surface.tile().revision();
    let embedding = old_surface.embedding();
    assert_eq!(old_points.len(), 4);
    assert!(old_points.iter().all(|point| point.position_mm[1] == 0));

    // Deliberately deep-clone the tile into a distinct mutable owner. Neither
    // Arc::clone nor Arc::make_mut is the snapshot publication operation here.
    let mut rebaked_owner = old_surface.tile().as_ref().clone();
    assert_ne!(
        rebaked_owner.samples().as_ptr(),
        old_surface.tile().samples().as_ptr()
    );
    let full_bounds = DirtyRegion::new(rebaked_owner.tile(), 0, 0, u16::MAX, u16::MAX).unwrap();
    let limits = BakeLimits {
        max_samples: 4,
        max_bytes: 4 * size_of::<i16>(),
    };
    let change = rebaked_owner
        .rebake(
            full_bounds,
            &GridHeights {
                side: 2,
                heights: &[1; 4],
            },
            limits,
        )
        .unwrap();
    assert_eq!(change.revision(), old_revision + 1);
    assert_eq!(rebaked_owner.revision(), old_revision + 1);
    assert_eq!(rebaked_owner.samples(), &[1; 4]);

    // The final sample fails encoding after earlier samples were staged. No
    // prefix may leak into either samples or revision on this real failure.
    let before_failure = rebaked_owner.clone();
    assert_eq!(
        rebaked_owner.rebake(
            full_bounds,
            &GridHeights {
                side: 2,
                heights: &[2, 2, 2, i64::MAX],
            },
            limits,
        ),
        Err(BakeError::HeightOutOfRange)
    );
    assert_eq!(rebaked_owner, before_failure);
    assert_eq!(rebaked_owner.samples(), &[1; 4]);
    assert_eq!(rebaked_owner.revision(), old_revision + 1);

    let new_surface = PlanarHeightfieldSurface::new(Arc::new(rebaked_owner), embedding).unwrap();
    assert_eq!(new_surface.embedding(), old_surface.embedding());
    assert_eq!(new_surface.tile().tile(), old_surface.tile().tile());
    assert_eq!(new_surface.tile().spec(), old_surface.tile().spec());
    assert!(!Arc::ptr_eq(new_surface.tile(), old_surface.tile()));
    let new_output = filtered(config, region, &field, &new_surface, filters);
    assert_eq!(new_output.points.len(), 4);
    assert!(new_output
        .points
        .iter()
        .all(|point| point.position_mm[1] == 1));
    assert_ne!(new_output.points, old_points);
    assert_ne!(new_output.cache_key, old_key);
    assert_eq!(new_output.cache_key.source, old_key.source);
    assert!(Arc::ptr_eq(
        new_output.cache_key.surface.tile(),
        new_surface.tile()
    ));

    // Drop the original surface handle: the previously returned cache key is
    // itself a retained immutable snapshot and still reproduces its selection.
    drop(old_surface);
    assert_eq!(old_output.points, old_points);
    assert_eq!(old_output.attributes, old_attributes);
    assert_eq!(old_output.cache_key, old_key);
    assert_eq!(old_output.cache_key.surface.tile().revision(), old_revision);
    assert_eq!(old_output.cache_key.surface.tile().samples(), &[0; 4]);
    assert_eq!(new_output.cache_key.surface.tile().samples(), &[1; 4]);
    assert_eq!(
        filtered(
            config,
            region,
            &field,
            &old_output.cache_key.surface,
            filters
        ),
        old_output
    );
    assert_eq!(
        filtered(
            config,
            region,
            &field,
            &new_output.cache_key.surface,
            filters
        ),
        new_output
    );
}

#[test]
fn surface_budget_admits_primary_candidates_before_any_density_or_terrain_rejection() {
    let surface = surface(2, 2, [0, 0, 0], 1, &[0; 4]);
    let config = config(1, 0);
    let region = region([0; 3], [2; 3]);
    assert_eq!(
        estimate_scatter_work(config, region)
            .unwrap()
            .primary_candidates,
        8
    );
    for density in [0, u16::MAX] {
        let field = make_field(density, None, Vec::new());
        for filters in [
            unrestricted(),
            HeightfieldScatterFilters {
                surface_height_mm: [1, 1],
                ..unrestricted()
            },
        ] {
            let error = scatter_with_heightfield(
                config,
                region,
                ScatterBudget {
                    max_candidate_evaluations: 0,
                    max_output_points: 0,
                    max_output_bytes: 0,
                },
                &field,
                FieldScatterBudget {
                    max_field_work: 0,
                    max_output_bytes: 0,
                },
                &surface,
                filters,
                HeightfieldScatterBudget {
                    max_sample_reads: 31,
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                    required: 32,
                    limit: 31
                }
            );
            let output = scatter_with_heightfield(
                config,
                region,
                source_budget(),
                &field,
                field_budget(),
                &surface,
                filters,
                HeightfieldScatterBudget {
                    max_sample_reads: 32,
                },
            )
            .unwrap();
            assert_eq!(
                output.surface_sample_reads,
                if density == 0 { 0 } else { 32 }
            );
            assert_eq!(output.field_work, 8 * field.work_per_sample());
            if density == 0 || filters.surface_height_mm == [1, 1] {
                assert!(output.points.is_empty());
                assert!(output.attributes.is_empty());
                // Thinning does not shrink away capacity charged to the field budget.
                assert_eq!(output.points.capacity(), 8);
                assert_eq!(output.attributes.capacity(), 8);
            } else {
                assert_eq!(output.points.len(), 8);
            }
        }
    }
}

#[test]
fn source_and_field_work_and_allocation_errors_survive_terrain_thinning() {
    let surface = surface(2, 2, [0, 0, 0], 1, &[0; 4]);
    let config = config(1, 0);
    let region = region([0; 3], [2; 3]);
    let field = full_field();
    let reject_all = HeightfieldScatterFilters {
        surface_height_mm: [1, 1],
        ..unrestricted()
    };
    let run = |source_budget, field_budget, filters| {
        scatter_with_heightfield(
            config,
            region,
            source_budget,
            &field,
            field_budget,
            &surface,
            filters,
            HeightfieldScatterBudget {
                max_sample_reads: 32,
            },
        )
    };
    assert_eq!(
        run(
            source_budget(),
            FieldScatterBudget {
                max_field_work: 7,
                ..field_budget()
            },
            reject_all
        )
        .unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::WorkBudgetExceeded {
            required: 8,
            limit: 7
        })
    );
    assert_eq!(
        run(
            ScatterBudget {
                max_candidate_evaluations: 7,
                ..source_budget()
            },
            field_budget(),
            reject_all
        )
        .unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::Scatter(
            ScatterError::CandidateBudgetExceeded {
                required: 8,
                limit: 7
            }
        ))
    );
    assert_eq!(
        run(
            ScatterBudget {
                max_output_points: 0,
                ..source_budget()
            },
            field_budget(),
            reject_all
        )
        .unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::Scatter(
            ScatterError::OutputCountBudgetExceeded {
                required: 1,
                limit: 0
            }
        ))
    );
    assert_eq!(
        run(
            ScatterBudget {
                max_output_bytes: 0,
                ..source_budget()
            },
            field_budget(),
            reject_all
        )
        .unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::Scatter(
            ScatterError::OutputBytesBudgetExceeded {
                required: size_of::<ScatterPoint>(),
                limit: 0
            }
        ))
    );
    let baseline = run(source_budget(), field_budget(), unrestricted()).unwrap();
    let bytes = baseline.points.capacity() * size_of::<ScatterPoint>()
        + baseline.attributes.capacity() * size_of::<ScatterFieldSample>();
    assert_eq!(
        run(
            source_budget(),
            FieldScatterBudget {
                max_output_bytes: bytes - 1,
                ..field_budget()
            },
            reject_all
        )
        .unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::OutputBytesBudgetExceeded {
            required: bytes,
            limit: bytes - 1
        })
    );
    let exact = run(
        ScatterBudget {
            max_candidate_evaluations: 8,
            max_output_points: 8,
            max_output_bytes: 8 * size_of::<ScatterPoint>(),
        },
        FieldScatterBudget {
            max_field_work: 8,
            max_output_bytes: bytes,
        },
        unrestricted(),
    )
    .unwrap();
    assert_eq!(
        exact, baseline,
        "admission limits are not part of result identity"
    );
}

#[test]
fn enormous_surface_work_is_rejected_without_enumeration_or_large_allocations() {
    let field = full_field();
    for (edge, expected) in [
        (
            1_u32 << 20,
            HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                required: 1_u64 << 62,
                limit: 0,
            },
        ),
        (1_u32 << 21, HeightfieldScatterError::SurfaceWorkOverflow),
    ] {
        // Only four real samples; coverage is large, allocation is not.
        let surface = surface(2, edge, [0; 3], 1, &[0; 4]);
        let region = region([0; 3], [i64::from(edge); 3]);
        let error = scatter_with_heightfield(
            config(1, 0),
            region,
            ScatterBudget {
                max_candidate_evaluations: 0,
                max_output_points: 0,
                max_output_bytes: 0,
            },
            &field,
            FieldScatterBudget {
                max_field_work: 0,
                max_output_bytes: 0,
            },
            &surface,
            unrestricted(),
            HeightfieldScatterBudget {
                max_sample_reads: 0,
            },
        )
        .unwrap_err();
        assert_eq!(error, expected);
    }
}

#[test]
fn complete_nonempty_request_coverage_is_required_even_when_density_rejects_everything() {
    let surface = surface(2, 4, [0; 3], 1, &[0; 4]);
    let field = make_field(0, None, Vec::new());
    let inside = filtered(
        config(1, 0),
        region([0; 3], [4, 1, 4]),
        &field,
        &surface,
        unrestricted(),
    );
    assert!(inside.points.is_empty());
    assert_eq!(inside.surface_sample_reads, 0);
    for region in [
        region([-1, 0, 0], [1, 1, 1]),
        region([0, 0, -1], [1, 1, 1]),
        region([0; 3], [5, 1, 4]),
        region([0; 3], [4, 1, 5]),
        region([4, 0, 0], [5, 1, 1]),
        region([0, 0, 4], [1, 1, 5]),
    ] {
        let error = scatter_with_heightfield(
            config(1, 0),
            region,
            source_budget(),
            &field,
            field_budget(),
            &surface,
            unrestricted(),
            HeightfieldScatterBudget {
                max_sample_reads: 1_000,
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface)
        );
    }
}

#[test]
fn any_empty_axis_needs_no_coverage_or_work_or_output_capacity() {
    let surface = surface(2, 1, [0; 3], 1, &[0; 4]);
    let field = full_field();
    for axis in 0..3 {
        let min = [100, -100, 100];
        let mut max = [101, -99, 101];
        max[axis] = min[axis];
        let output = scatter_with_heightfield(
            config(1, 0),
            region(min, max),
            ScatterBudget {
                max_candidate_evaluations: 0,
                max_output_points: 0,
                max_output_bytes: 0,
            },
            &field,
            FieldScatterBudget {
                max_field_work: 0,
                max_output_bytes: 0,
            },
            &surface,
            unrestricted(),
            HeightfieldScatterBudget {
                max_sample_reads: 0,
            },
        )
        .unwrap();
        assert!(output.points.is_empty());
        assert!(output.attributes.is_empty());
        assert_eq!(output.candidate_evaluations, 0);
        assert_eq!(output.field_work, 0);
        assert_eq!(output.surface_sample_reads, 0);
    }
}

#[test]
fn invalid_filters_and_source_inputs_return_the_public_error_variants() {
    let surface = surface(2, 2, [0; 3], 1, &[0; 4]);
    let field = full_field();
    let region = region([0; 3], [1; 3]);
    for filters in [
        HeightfieldScatterFilters {
            surface_height_mm: [1, -1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            clearance_mm: [1, -1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            max_slope_run: 0,
            ..unrestricted()
        },
    ] {
        assert_eq!(
            scatter_with_heightfield(
                config(1, 0),
                region,
                source_budget(),
                &field,
                field_budget(),
                &surface,
                filters,
                HeightfieldScatterBudget {
                    max_sample_reads: 4
                }
            )
            .unwrap_err(),
            HeightfieldScatterError::InvalidFilters
        );
    }
    for (config, region, expected) in [
        (config(0, 0), region, ScatterError::InvalidGridSpacing),
        (
            config(1, 0),
            ScatterRegion {
                min_mm: [1, 0, 0],
                max_mm: [0, 1, 1],
            },
            ScatterError::InvalidRegion,
        ),
    ] {
        assert_eq!(
            scatter_with_heightfield(
                config,
                region,
                source_budget(),
                &field,
                field_budget(),
                &surface,
                unrestricted(),
                HeightfieldScatterBudget {
                    max_sample_reads: 4
                }
            )
            .unwrap_err(),
            HeightfieldScatterError::Source(ScatterFieldError::Scatter(expected))
        );
    }
}

#[test]
fn extreme_world_height_clearance_slope_and_u32_spacing_do_not_wrap() {
    let field = full_field();
    // Heights may exceed i64 after adding the embedding Y; broad i64 filters
    // must still reject them, rather than truncate or saturate into the range.
    for (origin_y, tile_height) in [(i64::MAX, 1), (i64::MIN, -1)] {
        let surface = surface(
            2,
            u32::MAX,
            [i64::MIN, origin_y, i64::MIN],
            1,
            &[tile_height; 4],
        );
        let output = filtered(
            config(1, 0),
            region([i64::MIN; 3], [i64::MIN + 1; 3]),
            &field,
            &surface,
            unrestricted(),
        );
        assert!(output.points.is_empty());
        assert_eq!(output.surface_sample_reads, 4);
    }
    for (height, point_y) in [(i64::MAX, i64::MIN), (i64::MIN, i64::MAX - 1)] {
        let surface = surface(2, 1, [0, height, 0], 1, &[0; 4]);
        let output = filtered(
            config(1, 0),
            region([0, point_y, 0], [1, point_y + 1, 1]),
            &field,
            &surface,
            unrestricted(),
        );
        assert!(
            output.points.is_empty(),
            "clearance outside i64 is not in the full i64 interval"
        );
        assert_eq!(output.surface_sample_reads, 4);
    }
    let surface = surface(
        2,
        u32::MAX,
        [0; 3],
        i64::MAX,
        &[-i64::MAX, i64::MAX, i64::MAX, -i64::MAX],
    );
    let output = filtered(
        config(1, 0),
        region([0; 3], [1; 3]),
        &field,
        &surface,
        HeightfieldScatterFilters {
            max_slope_rise: u32::MAX,
            max_slope_run: u32::MAX,
            ..unrestricted()
        },
    );
    assert!(
        output.points.is_empty(),
        "oversized squared slope components must reject safely"
    );
    assert_eq!(output.surface_sample_reads, 4);
}
