//! Public-contract tests with independently baked terrain and exact geometry/rank oracles.
#![cfg(feature = "scatter")]

use std::{collections::BTreeMap, convert::Infallible, error::Error, mem::size_of, sync::Arc};

use prism_world_residency::{
    DensityBrush, DensityBrushOp, DensityNoise, ScatterCandidateKey, ScatterDensityField,
    ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample, SectorCoord,
    SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace, DirtyRegion,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementCacheKey, HeightfieldPlacementConfig, HeightfieldPlacementError,
    HeightfieldPlacementFilters, HeightfieldPlacementKey, HeightfieldPlacementOutput,
    HeightfieldPlacementPoint, HeightfieldPlacementRegion, HeightfieldPlacementWork,
    HeightfieldQueryError, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_PLACEMENT_VERSION, HEIGHTFIELD_SURFACE_VERSION,
};

const ROW_BYTES: usize = size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>();
const HEIGHTS: [i64; 9] = [-7, 2, 9, 4, -6, 12, 13, 5, -3];

// This sampler owns the source data contract; it does not call the surface/placement sampler.
struct AuthoredGrid<'a> {
    side: usize,
    heights: &'a [i64],
}

impl HeightSampler for AuthoredGrid<'_> {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |q: u16| (usize::from(q) * (self.side - 1) + 32_767) / 65_535;
        Ok(HeightSampleValue::from_millimetres(
            self.heights[index(sample.v_q16()) * self.side + index(sample.u_q16())],
        ))
    }
}

fn bake(heights: &[i64], side: usize, scale: i64, key: TileKey) -> HeightfieldTile {
    assert_eq!(heights.len(), side * side);
    HeightfieldTile::bake(
        key,
        HeightfieldSpec::new(side as u32, scale).unwrap(),
        &AuthoredGrid { side, heights },
        BakeLimits {
            max_samples: heights.len(),
            max_bytes: heights.len() * size_of::<i16>(),
        },
    )
    .unwrap()
}

struct Terrain {
    surface: PlanarHeightfieldSurface,
    origin: [i64; 3],
    spacing: u32,
    heights: [i64; 9],
}

impl Terrain {
    fn new(origin: [i64; 3], spacing: u32, heights: [i64; 9]) -> Self {
        let tile = bake(&heights, 3, 1, TileKey::root(CubeFace::PositiveY));
        assert_eq!(tile.samples(), heights.map(|h| h as i16));
        Self {
            surface: PlanarHeightfieldSurface::new(
                Arc::new(tile),
                PlanarHeightfieldEmbedding {
                    sample_origin_mm: origin,
                    sample_spacing_mm: spacing,
                },
            )
            .unwrap(),
            origin,
            spacing,
            heights,
        }
    }

    fn region(&self) -> HeightfieldPlacementRegion {
        region(self.surface.min_xz_mm(), self.surface.max_xz_mm())
    }

    // Solve the plane using an oriented triangle cross product. No production
    // sample, interpolation, rounding or placement rebuild is an oracle here.
    fn geometry(&self, xz: [i64; 2], offset: i64) -> ([i64; 3], i128, [i128; 3]) {
        let s = i128::from(self.spacing);
        let x = i128::from(xz[0]) - i128::from(self.origin[0]);
        let z = i128::from(xz[1]) - i128::from(self.origin[2]);
        assert!(x >= 0 && x < 2 * s && z >= 0 && z < 2 * s);
        let cell_x = (x / s) as usize;
        let cell_z = (z / s) as usize;
        let rx = x % s;
        let rz = z % s;
        let h = |dx, dz| {
            i128::from(self.origin[1]) + i128::from(self.heights[(cell_z + dz) * 3 + cell_x + dx])
        };
        let a = [0, h(0, 0), 0];
        let b = [s, h(1, 0), 0];
        let c = [0, h(0, 1), s];
        let d = [s, h(1, 1), s];
        let [v, w, t] = if rx + rz <= s { [a, c, b] } else { [b, c, d] };
        let u = std::array::from_fn::<_, 3, _>(|i| w[i] - v[i]);
        let q = std::array::from_fn::<_, 3, _>(|i| t[i] - v[i]);
        let normal = [
            (u[1] * q[2] - u[2] * q[1]) / s,
            (u[2] * q[0] - u[0] * q[2]) / s,
            (u[0] * q[1] - u[1] * q[0]) / s,
        ];
        assert_eq!(normal[1], s);
        let numerator = v[1] * s - normal[0] * (rx - v[0]) - normal[2] * (rz - v[2]);
        let total = numerator + i128::from(offset) * s;
        let floor = total.div_euclid(s);
        let remainder = total.rem_euclid(s);
        let round_up = 2 * remainder > s || (2 * remainder == s && total >= 0);
        let rounded = floor + i128::from(round_up);
        assert!((rounded * s - total).abs() * 2 <= s);
        (
            [xz[0], i64::try_from(rounded).unwrap(), xz[1]],
            numerator,
            normal,
        )
    }
}

fn config() -> HeightfieldPlacementConfig {
    HeightfieldPlacementConfig {
        seed: 0x3a65_c807_9742_1eab,
        version: 7,
        salt: 0xf09c_1172_d538_208a,
        grid_spacing_mm: 3,
        min_distance_mm: 0,
        vertical_offset_mm: 0,
    }
}

fn region(min: [i64; 2], max: [i64; 2]) -> HeightfieldPlacementRegion {
    HeightfieldPlacementRegion::new(min, max).unwrap()
}

fn filters() -> HeightfieldPlacementFilters {
    HeightfieldPlacementFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

fn budget() -> HeightfieldPlacementBudget {
    HeightfieldPlacementBudget {
        max_candidate_evaluations: 10_000_000,
        max_sample_reads: 100_000,
        max_field_work: 1_000_000,
        max_output_points: 16_384,
        max_output_bytes: 16_384 * ROW_BYTES,
    }
}

fn zero_budget() -> HeightfieldPlacementBudget {
    HeightfieldPlacementBudget {
        max_candidate_evaluations: 0,
        max_sample_reads: 0,
        max_field_work: 0,
        max_output_points: 0,
        max_output_bytes: 0,
    }
}

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 91,
        version: 3,
        base_density: u16::MAX,
        base_species: 11,
        noise: None,
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let limits = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, limits).unwrap()
}

fn full_field() -> ScatterDensityField {
    field(field_config(), vec![])
}

fn brush() -> DensityBrush {
    DensityBrush {
        id: 42,
        center_mm: [0, 0, 0],
        inner_radius_mm: 8,
        outer_radius_mm: 18,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(u16::MAX),
        species: Some(77),
    }
}

fn noise() -> DensityNoise {
    DensityNoise {
        cell_size_mm: 7,
        amplitude: 12_000,
        warp_cell_size_mm: 11,
        warp_amplitude_mm: 5,
    }
}

fn painted_field() -> ScatterDensityField {
    field(
        ScatterFieldConfig {
            base_density: 40_000,
            noise: Some(noise()),
            ..field_config()
        },
        vec![
            DensityBrush {
                id: 80,
                density: DensityBrushOp::Add(-25_000),
                ..brush()
            },
            DensityBrush {
                id: 10,
                center_mm: [-5, 7, 4],
                density: DensityBrushOp::Replace(55_000),
                ..brush()
            },
        ],
    )
}

fn key(config: HeightfieldPlacementConfig, x: i64, z: i64) -> HeightfieldPlacementKey {
    HeightfieldPlacementKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source: ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: config.seed,
            version: config.version,
            salt: config.salt,
            grid_spacing_mm: config.grid_spacing_mm,
            grid: SectorCoord::new(x, 0, z),
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct Raw {
    key: HeightfieldPlacementKey,
    position: [i64; 3],
    priority: u64,
}

// Version-1 public priority specification, deliberately independent of the
// production placement helpers (including its PCG helper implementation).
fn priority(key: HeightfieldPlacementKey) -> u64 {
    let pcg = |input: u64| {
        let state = input
            .wrapping_add(0x853c_49e6_748f_ea9b)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((((state >> 18) ^ state) >> 27) as u32).rotate_right((state >> 59) as u32)
    };
    let mix = |v| (u64::from(pcg(v)) << 32) | u64::from(pcg(v ^ 0x9e37_79b9_7f4a_7c15));
    let source = key.source;
    [
        u64::from(source.kernel_version),
        source.seed,
        u64::from(source.version),
        source.salt,
        u64::from(source.grid_spacing_mm),
        source.grid.x as u64,
        source.grid.z as u64,
    ]
    .into_iter()
    .fold(
        mix(0x706c_6163_655f_787a ^ u64::from(key.placement_version)),
        |rank, word| mix(rank ^ word),
    )
}

fn raw_candidates(
    config: HeightfieldPlacementConfig,
    area: HeightfieldPlacementRegion,
    padding: i64,
) -> Vec<Raw> {
    let s = i64::from(config.grid_spacing_mm);
    let bounds = |axis: usize| {
        let lo = i128::from(area.min_xz_mm[axis].div_euclid(s)) - i128::from(padding);
        let hi = i128::from((area.max_xz_mm[axis] - 1).div_euclid(s)) + i128::from(padding);
        (
            lo.max(i128::from(i64::MIN)) as i64,
            hi.min(i128::from(i64::MAX)) as i64,
        )
    };
    let (xmin, xmax) = bounds(0);
    let (zmin, zmax) = bounds(1);
    let mut result = vec![];
    for x in xmin..=xmax {
        for z in zmin..=zmax {
            let key = key(config, x, z);
            if let Ok(point) = key.source.rebuild() {
                result.push(Raw {
                    key,
                    position: point.position_mm,
                    priority: priority(key),
                });
            }
        }
    }
    result
}

fn owns(area: HeightfieldPlacementRegion, raw: &Raw) -> bool {
    [raw.position[0], raw.position[2]]
        .into_iter()
        .enumerate()
        .all(|(i, p)| area.min_xz_mm[i] <= p && p < area.max_xz_mm[i])
}

fn blocks(other: &Raw, current: &Raw, radius: u32) -> bool {
    let r = u128::from(radius);
    let dx = u128::from(other.position[0].abs_diff(current.position[0]));
    let dz = u128::from(other.position[2].abs_diff(current.position[2]));
    (other.priority, other.key) < (current.priority, current.key)
        && dx < r
        && dz < r
        && dx * dx + dz * dz < r * r
}

fn oracle_survivors(
    config: HeightfieldPlacementConfig,
    area: HeightfieldPlacementRegion,
) -> Vec<Raw> {
    // Enumerate a superset of the full expanded footprint, then perform an
    // all-pairs test. Not a production-style halo walk, early exit, or greedy fill.
    let padding = i64::from(config.min_distance_mm.div_ceil(config.grid_spacing_mm)) + 1;
    let pool = raw_candidates(config, area, padding);
    pool.iter()
        .filter(|current| {
            owns(area, current)
                && !pool
                    .iter()
                    .any(|other| blocks(other, current, config.min_distance_mm))
        })
        .copied()
        .collect()
}

type Rows = BTreeMap<HeightfieldPlacementKey, ([i64; 3], ScatterFieldSample)>;

fn rows(output: &HeightfieldPlacementOutput) -> Rows {
    assert_eq!(output.points.len(), output.attributes.len());
    assert!(output.points.windows(2).all(|p| p[0].key < p[1].key));
    let result: Rows = output
        .points
        .iter()
        .zip(&output.attributes)
        .map(|(point, sample)| (point.key, (point.position_mm, *sample)))
        .collect();
    assert_eq!(
        result.len(),
        output.points.len(),
        "full placement IDs must be unique"
    );
    result
}

fn oracle_rows(
    config: HeightfieldPlacementConfig,
    area: HeightfieldPlacementRegion,
    terrain: &Terrain,
    author: &ScatterDensityField,
    limits: HeightfieldPlacementFilters,
) -> Rows {
    oracle_survivors(config, area)
        .into_iter()
        .filter_map(|raw| {
            let (position, numerator, normal) = terrain.geometry(
                [raw.position[0], raw.position[2]],
                config.vertical_offset_mm,
            );
            let d = i128::from(terrain.spacing);
            let slope_lhs = (normal[0] * normal[0] + normal[2] * normal[2])
                * i128::from(limits.max_slope_run).pow(2);
            let slope_rhs = normal[1].pow(2) * i128::from(limits.max_slope_rise).pow(2);
            if numerator < i128::from(limits.surface_height_mm[0]) * d
                || numerator > i128::from(limits.surface_height_mm[1]) * d
                || slope_lhs > slope_rhs
            {
                return None;
            }
            let sample = author.sample(position);
            author
                .accepts(raw.key.source, sample.density)
                .then_some((raw.key, (position, sample)))
        })
        .collect()
}

fn run(
    config: HeightfieldPlacementConfig,
    area: HeightfieldPlacementRegion,
    terrain: &Terrain,
    author: &ScatterDensityField,
) -> HeightfieldPlacementOutput {
    scatter_on_heightfield(config, area, author, &terrain.surface, filters(), budget()).unwrap()
}

#[test]
fn exact_triangle_planes_normals_and_rounding_match_authored_geometry_exhaustively() {
    let author = full_field();
    for spacing in [7, 8] {
        let terrain = Terrain::new([-9, -3, -11], spacing, HEIGHTS);
        for offset in [-9, -1, 0, 1, 9] {
            let config = HeightfieldPlacementConfig {
                grid_spacing_mm: 1,
                vertical_offset_mm: offset,
                ..config()
            };
            let output = run(config, terrain.region(), &terrain, &author);
            assert_eq!(output.points.len(), (2 * spacing).pow(2) as usize);
            assert_eq!(
                rows(&output),
                oracle_rows(config, terrain.region(), &terrain, &author, filters())
            );
            for point in &output.points {
                let (position, numerator, normal) =
                    terrain.geometry([point.position_mm[0], point.position_mm[2]], offset);
                assert_eq!(point.position_mm, position);
                assert_eq!(point.surface_sample.height_numerator_mm(), numerator);
                assert_eq!(point.surface_sample.height_denominator(), spacing);
                assert_eq!(point.surface_sample.normal_mm(), normal);
                assert_eq!(point.key.rebuild(&terrain.surface, offset).unwrap(), *point);
            }
            assert_eq!(output.candidate_evaluations, output.points.len() as u64);
            assert_eq!(output.surface_sample_reads, 4 * output.points.len() as u64);
            assert_eq!(output.field_work, output.points.len() as u64);
        }
    }
}

#[test]
fn negative_half_ties_and_offsets_crossing_zero_round_the_sum_not_the_surface() {
    let terrain = Terrain::new([-2, 0, -2], 2, [-1, 0, 1, -1, 0, 1, -1, 0, 1]);
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    for (x, offset, expected_y) in [
        (-1, 0, -1),
        (-1, 1, 1),
        (-1, -1, -2),
        (1, 0, 1),
        (1, -1, -1),
        (1, 1, 2),
        (-2, 1, 0),
        (0, -1, -1),
    ] {
        let key = key(config, x, -1);
        let point = key.rebuild(&terrain.surface, offset).unwrap();
        assert_eq!(
            point.position_mm,
            [x, expected_y, -1],
            "x={x}, offset={offset}"
        );
        assert_eq!(point.surface_sample.height_numerator_mm(), i128::from(x));
        let output = run(
            HeightfieldPlacementConfig {
                vertical_offset_mm: offset,
                ..config
            },
            region([x, -1], [x + 1, 0]),
            &terrain,
            &full_field(),
        );
        assert_eq!(output.points, vec![point]);
    }
}

#[test]
fn author_density_and_species_are_sampled_at_final_offset_projected_y_not_raw_y() {
    let terrain = Terrain::new([-12, 1_200, -12], 12, [0; 9]);
    let config = HeightfieldPlacementConfig {
        vertical_offset_mm: 37,
        ..config()
    };
    let area = region([-10, -10], [11, 11]);
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            center_mm: [0, 1_237, 0],
            inner_radius_mm: 40,
            outer_radius_mm: 41,
            ..brush()
        }],
    );
    let expected: Rows = raw_candidates(config, area, 0)
        .iter()
        .filter(|raw| owns(area, raw))
        .map(|raw| {
            assert_eq!(author.sample(raw.position).density, 0);
            (
                raw.key,
                (
                    [raw.position[0], 1_237, raw.position[2]],
                    ScatterFieldSample {
                        density: u16::MAX,
                        species: 77,
                    },
                ),
            )
        })
        .collect();
    assert!(!expected.is_empty());
    let output = run(config, area, &terrain, &author);
    assert_eq!(rows(&output), expected);
    // A field that accepts the raw-Y slab must not leak those placements through.
    let raw_only = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            center_mm: [0, 0, 0],
            inner_radius_mm: 40,
            outer_radius_mm: 41,
            ..brush()
        }],
    );
    let rejected = run(config, area, &terrain, &raw_only);
    assert!(rejected.points.is_empty());
    assert!(rejected.field_work > 0);
    assert_eq!(rejected.field_work, output.field_work);
}

#[test]
fn mixed_author_field_membership_and_attribute_alignment_match_projected_oracle() {
    let terrain = Terrain::new([-16, 0, -16], 16, HEIGHTS);
    let author = painted_field();
    for offset in [-17, 0, 13] {
        let config = HeightfieldPlacementConfig {
            vertical_offset_mm: offset,
            ..config()
        };
        let area = region([-15, -13], [14, 15]);
        let output = run(config, area, &terrain, &author);
        let expected = oracle_rows(config, area, &terrain, &author, filters());
        assert!(!expected.is_empty());
        assert!(
            expected.len()
                < raw_candidates(config, area, 0)
                    .iter()
                    .filter(|raw| owns(area, raw))
                    .count()
        );
        assert_eq!(rows(&output), expected);
    }
}

#[test]
fn full_structural_keys_rebuild_geometry_without_proving_selection_or_ownership() {
    let terrain = Terrain::new([-16, 100, -16], 16, HEIGHTS);
    let author = full_field();
    let area = region([-10, -10], [11, 11]);
    let base = config();
    for config in [
        base,
        HeightfieldPlacementConfig {
            seed: base.seed ^ (1 << 63),
            ..base
        },
        HeightfieldPlacementConfig {
            version: base.version + 1,
            ..base
        },
        HeightfieldPlacementConfig {
            salt: base.salt ^ (1 << 61),
            ..base
        },
        HeightfieldPlacementConfig {
            grid_spacing_mm: 4,
            ..base
        },
    ] {
        let output = run(config, area, &terrain, &author);
        assert!(!output.points.is_empty());
        for point in &output.points {
            assert_eq!(
                point.key,
                key(config, point.key.source.grid.x, point.key.source.grid.z)
            );
            let raw = point.key.source.rebuild().unwrap();
            assert_eq!(
                [point.position_mm[0], point.position_mm[2]],
                [raw.position_mm[0], raw.position_mm[2]]
            );
            assert_ne!(point.position_mm[1], raw.position_mm[1]);
            assert_eq!(
                point
                    .key
                    .rebuild(&terrain.surface, config.vertical_offset_mm)
                    .unwrap(),
                *point
            );
        }
    }
    let original = run(base, area, &terrain, &author);
    let selected = original.points[0];
    let changed = Terrain::new([-16, -500, -16], 16, [3; 9]);
    let rebuilt = selected.key.rebuild(&changed.surface, 91).unwrap();
    assert_eq!(rebuilt.key, selected.key);
    assert_eq!(rebuilt.position_mm[1], -406);
    // Rebuild neither receives nor checks the tiny request, restrictive filter,
    // zero-density field, or exclusion radius that could reject this identity.
    let thinned = run(
        HeightfieldPlacementConfig {
            min_distance_mm: 8,
            ..base
        },
        area,
        &terrain,
        &author,
    );
    assert!(thinned.points.len() < original.points.len());
    for point in &thinned.points {
        assert_eq!(
            point,
            original.points.iter().find(|p| p.key == point.key).unwrap()
        );
    }
}

#[test]
fn invalid_keys_and_uncovered_rebuilds_return_typed_errors() {
    let terrain = Terrain::new([-8, 0, -8], 8, [0; 9]);
    let valid = key(config(), 0, 0);
    for placement_version in [0, HEIGHTFIELD_PLACEMENT_VERSION + 1] {
        assert_eq!(
            HeightfieldPlacementKey {
                placement_version,
                ..valid
            }
            .rebuild(&terrain.surface, 0),
            Err(HeightfieldPlacementError::InvalidKey)
        );
    }
    for y in [i64::MIN, -1, 1, i64::MAX] {
        let invalid = HeightfieldPlacementKey {
            source: ScatterCandidateKey {
                grid: SectorCoord::new(0, y, 0),
                ..valid.source
            },
            ..valid
        };
        assert_eq!(
            invalid.rebuild(&terrain.surface, 0),
            Err(HeightfieldPlacementError::InvalidKey)
        );
    }
    for (source, error) in [
        (
            ScatterCandidateKey {
                kernel_version: SCATTER_KERNEL_VERSION + 1,
                ..valid.source
            },
            ScatterError::UnsupportedKernelVersion,
        ),
        (
            ScatterCandidateKey {
                grid_spacing_mm: 0,
                ..valid.source
            },
            ScatterError::InvalidGridSpacing,
        ),
        (
            ScatterCandidateKey {
                grid: SectorCoord::new(i64::MAX, 0, 0),
                ..valid.source
            },
            ScatterError::CoordinateOverflow,
        ),
    ] {
        let error = HeightfieldPlacementError::Source(error);
        assert_eq!(
            HeightfieldPlacementKey { source, ..valid }.rebuild(&terrain.surface, 0),
            Err(error)
        );
        assert!(error.source().is_some());
    }
    let error = key(config(), 100, 100)
        .rebuild(&terrain.surface, 0)
        .unwrap_err();
    assert_eq!(
        error,
        HeightfieldPlacementError::Surface(HeightfieldQueryError::OutsideSurface)
    );
    assert!(error.source().is_some());
}

#[test]
fn density_is_two_dimensional_and_independent_of_surface_y_and_vertical_offset() {
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 4,
        ..config()
    };
    let area = region([-16, -16], [16, 16]);
    for density in [u16::MAX, 31_337] {
        let author = field(
            ScatterFieldConfig {
                base_density: density,
                ..field_config()
            },
            vec![],
        );
        let mut identities = None;
        for y in [-1_000_000_000_000, 0, 1_000_000_000_000] {
            let terrain = Terrain::new([-16, y, -16], 16, [0; 9]);
            for offset in [-23, 0, 41] {
                let output = run(
                    HeightfieldPlacementConfig {
                        vertical_offset_mm: offset,
                        ..config
                    },
                    area,
                    &terrain,
                    &author,
                );
                assert_eq!(output.candidate_evaluations, 64);
                assert_eq!(output.surface_sample_reads, 256);
                assert_eq!(output.field_work, 64);
                if density == u16::MAX {
                    assert_eq!(output.points.len(), 64);
                }
                let keys = output.points.iter().map(|p| p.key).collect::<Vec<_>>();
                assert!(!keys.is_empty());
                if let Some(expected) = &identities {
                    assert_eq!(&keys, expected);
                } else {
                    identities = Some(keys);
                }
                assert!(output.points.iter().all(|p| p.position_mm[1] == y + offset));
            }
        }
    }
}

#[test]
fn matern_ii_matches_independent_all_pairs_oracle_and_enforces_horizontal_spacing() {
    let terrain = Terrain::new([-16, 0, -16], 16, HEIGHTS);
    let area = region([-11, -9], [10, 12]);
    let author = full_field();
    for spacing in [1, 3, 5] {
        for radius in [0, 1, 4, 9] {
            let config = HeightfieldPlacementConfig {
                grid_spacing_mm: spacing,
                min_distance_mm: radius,
                vertical_offset_mm: 17,
                ..config()
            };
            let expected = oracle_survivors(config, area);
            let output = run(config, area, &terrain, &author);
            assert!(!expected.is_empty());
            assert_eq!(
                rows(&output),
                oracle_rows(config, area, &terrain, &author, filters()),
                "spacing={spacing}, radius={radius}"
            );
            assert_eq!(
                output.points.iter().map(|p| p.key).collect::<Vec<_>>(),
                expected.iter().map(|p| p.key).collect::<Vec<_>>()
            );
            assert_eq!(output.surface_sample_reads, 4 * expected.len() as u64);
            assert_eq!(output.field_work, expected.len() as u64);
            let work = estimate_heightfield_placement_work(config, area, &author).unwrap();
            assert!(output.candidate_evaluations <= work.max_candidate_evaluations);
            assert!(output.candidate_evaluations >= work.primary_candidates);
            for (i, a) in output.points.iter().enumerate() {
                for b in &output.points[i + 1..] {
                    let dx = i128::from(a.position_mm[0]) - i128::from(b.position_mm[0]);
                    let dz = i128::from(a.position_mm[2]) - i128::from(b.position_mm[2]);
                    assert!(dx * dx + dz * dz >= i128::from(radius).pow(2));
                }
            }
            // At a radius of exactly one, a unit integer lattice has equality
            // neighbours and no strictly closer neighbours: none may be removed.
            if spacing == 1 && radius == 1 {
                assert_eq!(output.points.len(), 21 * 21);
            }
        }
    }
}

#[test]
fn outside_request_and_uncovered_or_field_rejected_candidates_still_block() {
    let config = HeightfieldPlacementConfig {
        min_distance_mm: 8,
        ..config()
    };
    let area = region([-20, -20], [20, 20]);
    let pool = raw_candidates(config, area, 4);
    let current = pool
        .iter()
        .find(|current| {
            owns(area, current)
                && pool.iter().any(|other| {
                    blocks(other, current, config.min_distance_mm)
                        && (other.position[0] < current.position[0]
                            || other.position[2] < current.position[2]
                            || other.position[0] >= current.position[0] + 2
                            || other.position[2] >= current.position[2] + 2)
                })
        })
        .expect("fixture must contain an outside-surface blocker");
    let [x, _, z] = current.position;
    let terrain = Terrain::new([x, 100, z], 1, [0; 9]);
    let singleton = region([x, z], [x + 1, z + 1]);
    assert_eq!(
        raw_candidates(config, singleton, 0)
            .iter()
            .filter(|r| owns(singleton, r))
            .count(),
        1
    );
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            center_mm: [x, 100, z],
            inner_radius_mm: 1,
            outer_radius_mm: 1,
            ..brush()
        }],
    );
    assert_eq!(author.sample([x, 100, z]).density, u16::MAX);
    for other in pool
        .iter()
        .filter(|other| blocks(other, current, config.min_distance_mm))
    {
        assert!(!owns(singleton, other));
        assert_eq!(
            author
                .sample([other.position[0], 100, other.position[2]])
                .density,
            0
        );
    }
    let output = run(config, singleton, &terrain, &author);
    assert!(output.points.is_empty());
    assert!(output.attributes.is_empty());
    assert_eq!(output.surface_sample_reads, 0);
    assert_eq!(output.field_work, 0);
    let unthinned = run(
        HeightfieldPlacementConfig {
            min_distance_mm: 0,
            ..config
        },
        singleton,
        &terrain,
        &author,
    );
    assert_eq!(unthinned.points.len(), 1);
    assert_eq!(unthinned.points[0].key, current.key);
}

#[test]
fn terrain_rejected_and_themselves_suppressed_raw_candidates_are_not_removed_as_blockers() {
    let config = HeightfieldPlacementConfig {
        min_distance_mm: 8,
        ..config()
    };
    let terrain = Terrain::new([-32, 0, -32], 32, [-32, 0, 32, -32, 0, 32, -32, 0, 32]);
    let area = region([-24, -24], [24, 24]);
    let pool = raw_candidates(config, area, 4);
    let expected = oracle_survivors(config, area);
    let current = pool
        .iter()
        .find(|current| {
            let blockers: Vec<_> = pool
                .iter()
                .filter(|other| blocks(other, current, config.min_distance_mm))
                .collect();
            owns(area, current)
                && !blockers.is_empty()
                && blockers
                    .iter()
                    .all(|other| other.position[0] != current.position[0])
        })
        .expect("fixture must contain a blocker at a different exact terrain height");
    let limits = HeightfieldPlacementFilters {
        surface_height_mm: [current.position[0]; 2],
        ..filters()
    };
    let author = full_field();
    let output =
        scatter_on_heightfield(config, area, &author, &terrain.surface, limits, budget()).unwrap();
    assert_eq!(
        rows(&output),
        oracle_rows(config, area, &terrain, &author, limits)
    );
    assert!(!output.points.iter().any(|p| p.key == current.key));
    assert!(current.key.rebuild(&terrain.surface, 0).is_ok());
    // Matérn-II is not greedy sequential inhibition: a rejected original input
    // still suppresses higher ranks, even when no accepted lower neighbour does.
    let chain = pool
        .iter()
        .find(|current| {
            owns(area, current)
                && pool
                    .iter()
                    .any(|other| blocks(other, current, config.min_distance_mm))
                && pool
                    .iter()
                    .filter(|other| blocks(other, current, config.min_distance_mm))
                    .all(|other| {
                        owns(area, other)
                            && !expected.iter().any(|accepted| accepted.key == other.key)
                    })
        })
        .expect("fixture must distinguish Matérn-II from greedy filling");
    let full = run(config, area, &terrain, &author);
    assert!(!full.points.iter().any(|p| p.key == chain.key));
    assert_eq!(
        rows(&full),
        oracle_rows(config, area, &terrain, &author, filters())
    );
}

#[test]
fn arbitrary_unaligned_xz_partitions_in_reversed_request_order_preserve_full_rows() {
    let terrain = Terrain::new([-16, 0, -16], 16, HEIGHTS);
    let author = painted_field();
    let xs = [-15, -11, -2, 1, 8, 14];
    let zs = [-13, -12, -4, 3, 11, 15];
    let area = region([xs[0], zs[0]], [xs[xs.len() - 1], zs[zs.len() - 1]]);
    for radius in [0, 7] {
        let config = HeightfieldPlacementConfig {
            min_distance_mm: radius,
            vertical_offset_mm: 5,
            ..config()
        };
        let limits = HeightfieldPlacementFilters {
            surface_height_mm: [-4, 9],
            max_slope_rise: 1,
            max_slope_run: 1,
        };
        let whole =
            scatter_on_heightfield(config, area, &author, &terrain.surface, limits, budget())
                .unwrap();
        assert!(!whole.points.is_empty());
        let expected = oracle_rows(config, area, &terrain, &author, limits);
        assert_eq!(rows(&whole), expected);
        let mut parts = vec![];
        for x in xs.windows(2) {
            for z in zs.windows(2) {
                parts.push(region([x[0], z[0]], [x[1], z[1]]));
            }
        }
        parts.reverse();
        let mut merged = BTreeMap::new();
        for part in parts {
            let output =
                scatter_on_heightfield(config, part, &author, &terrain.surface, limits, budget())
                    .unwrap();
            assert_eq!(
                rows(&output),
                oracle_rows(config, part, &terrain, &author, limits)
            );
            for (key, row) in rows(&output) {
                assert!(
                    merged.insert(key, row).is_none(),
                    "partition overlap or duplicate identity"
                );
            }
        }
        assert_eq!(merged, expected);
        assert_eq!(
            run(config, area, &terrain, &author),
            run(config, area, &terrain, &author)
        );
    }
}

#[test]
fn exact_closed_height_filters_ignore_offset_and_rounded_y() {
    let terrain = Terrain::new([-2, 0, -2], 2, [-1, 0, 1, -1, 0, 1, -1, 0, 1]);
    let author = full_field();
    for (x, range, count) in [
        (-1, [-1, 0], 1),
        (-1, [-1, -1], 0),
        (-1, [0, 0], 0),
        (1, [0, 1], 1),
        (1, [0, 0], 0),
        (1, [1, 1], 0),
        (-2, [-1, -1], 1),
        (0, [0, 0], 1),
    ] {
        for offset in [-100, 0, 100] {
            let config = HeightfieldPlacementConfig {
                grid_spacing_mm: 1,
                vertical_offset_mm: offset,
                ..config()
            };
            let limits = HeightfieldPlacementFilters {
                surface_height_mm: range,
                ..filters()
            };
            let output = scatter_on_heightfield(
                config,
                region([x, -1], [x + 1, 0]),
                &author,
                &terrain.surface,
                limits,
                budget(),
            )
            .unwrap();
            assert_eq!(
                output.points.len(),
                count,
                "x={x}, interval={range:?}, offset={offset}"
            );
            assert_eq!(output.surface_sample_reads, 4);
            assert_eq!(output.field_work, count as u64);
        }
    }
}

#[test]
fn exact_rational_slope_boundary_uses_both_horizontal_components() {
    let terrain = Terrain::new([-5, 0, -5], 5, [0, 3, 6, 4, 7, 10, 8, 11, 14]);
    let author = full_field();
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    for (rise, run, accepted) in [
        (1, 1, true),
        (2, 2, true),
        (1_000, 1_000, true),
        (999, 1_000, false),
        (0, 1, false),
    ] {
        let limits = HeightfieldPlacementFilters {
            max_slope_rise: rise,
            max_slope_run: run,
            ..filters()
        };
        let output = scatter_on_heightfield(
            config,
            terrain.region(),
            &author,
            &terrain.surface,
            limits,
            budget(),
        )
        .unwrap();
        assert_eq!(output.points.len(), if accepted { 100 } else { 0 });
        assert_eq!(output.surface_sample_reads, 400);
        assert_eq!(output.field_work, if accepted { 100 } else { 0 });
        assert!(output
            .points
            .iter()
            .all(|p| p.surface_sample.normal_mm() == [-3, 5, -4]));
    }
    let flat = Terrain::new([-5, 0, -5], 5, [4; 9]);
    let output = scatter_on_heightfield(
        config,
        flat.region(),
        &author,
        &flat.surface,
        HeightfieldPlacementFilters {
            max_slope_rise: 0,
            ..filters()
        },
        budget(),
    )
    .unwrap();
    assert_eq!(output.points.len(), 100);
}

#[test]
fn estimator_matches_independent_global_grid_and_linear_field_work_formula() {
    let area = region([-8, -7], [8, 9]);
    for author in [
        full_field(),
        painted_field(),
        field(
            ScatterFieldConfig {
                noise: Some(DensityNoise {
                    amplitude: 0,
                    ..noise()
                }),
                ..field_config()
            },
            vec![brush()],
        ),
    ] {
        for radius in [0, 1, 4, 7] {
            let config = HeightfieldPlacementConfig {
                min_distance_mm: radius,
                ..config()
            };
            let primary = 6 * 6;
            let halo = u64::from(radius).div_ceil(3);
            let noise_work = author
                .config()
                .noise
                .filter(|n| n.amplitude != 0)
                .map_or(0, |n| 8 + if n.warp_amplitude_mm == 0 { 0 } else { 24 });
            let field_work = 1 + noise_work + author.brushes().len() as u64;
            assert_eq!(
                estimate_heightfield_placement_work(config, area, &author).unwrap(),
                HeightfieldPlacementWork {
                    primary_candidates: primary,
                    max_candidate_evaluations: primary * (2 * halo + 1).pow(2),
                    max_sample_reads: primary * 4,
                    max_field_work: primary * field_work,
                }
            );
        }
    }
}

#[test]
fn every_work_limit_is_strict_and_preflighted_even_when_all_output_would_be_rejected() {
    let terrain = Terrain::new([-16, 0, -16], 16, [0; 9]);
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            noise: Some(noise()),
            ..field_config()
        },
        vec![DensityBrush {
            opacity: 0,
            ..brush()
        }],
    );
    let config = HeightfieldPlacementConfig {
        min_distance_mm: 4,
        ..config()
    };
    let area = region([-8, -7], [8, 9]);
    let work = estimate_heightfield_placement_work(config, area, &author).unwrap();
    let exact = HeightfieldPlacementBudget {
        max_candidate_evaluations: work.max_candidate_evaluations,
        max_sample_reads: work.max_sample_reads,
        max_field_work: work.max_field_work,
        ..zero_budget()
    };
    let reject_all = HeightfieldPlacementFilters {
        surface_height_mm: [1, 1],
        ..filters()
    };
    let output =
        scatter_on_heightfield(config, area, &author, &terrain.surface, reject_all, exact).unwrap();
    assert!(output.points.is_empty());
    assert_eq!(output.field_work, 0);
    assert_eq!(output.points.capacity(), 0);
    assert_eq!(output.attributes.capacity(), 0);
    for (limits, error) in [
        (
            HeightfieldPlacementBudget {
                max_candidate_evaluations: work.max_candidate_evaluations - 1,
                ..exact
            },
            HeightfieldPlacementError::CandidateBudgetExceeded {
                required: work.max_candidate_evaluations,
                limit: work.max_candidate_evaluations - 1,
            },
        ),
        (
            HeightfieldPlacementBudget {
                max_sample_reads: work.max_sample_reads - 1,
                ..exact
            },
            HeightfieldPlacementError::SurfaceBudgetExceeded {
                required: work.max_sample_reads,
                limit: work.max_sample_reads - 1,
            },
        ),
        (
            HeightfieldPlacementBudget {
                max_field_work: work.max_field_work - 1,
                ..exact
            },
            HeightfieldPlacementError::FieldBudgetExceeded {
                required: work.max_field_work,
                limit: work.max_field_work - 1,
            },
        ),
    ] {
        assert_eq!(
            scatter_on_heightfield(config, area, &author, &terrain.surface, reject_all, limits),
            Err(error)
        );
    }
}

#[test]
fn output_count_and_combined_capacity_byte_limits_are_strict_without_partial_results() {
    let terrain = Terrain::new([-4, 0, -4], 4, [0; 9]);
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let area = region([-2, -2], [2, 2]);
    let author = full_field();
    let exact = HeightfieldPlacementBudget {
        max_output_points: 16,
        max_output_bytes: 16 * ROW_BYTES,
        ..budget()
    };
    let output =
        scatter_on_heightfield(config, area, &author, &terrain.surface, filters(), exact).unwrap();
    assert_eq!(output.points.len(), 16);
    assert_eq!(
        output.points.capacity() * size_of::<HeightfieldPlacementPoint>()
            + output.attributes.capacity() * size_of::<ScatterFieldSample>(),
        exact.max_output_bytes
    );
    for limit in [0, 1, 15] {
        let limited = HeightfieldPlacementBudget {
            max_output_points: limit,
            ..exact
        };
        assert_eq!(
            scatter_on_heightfield(config, area, &author, &terrain.surface, filters(), limited),
            Err(HeightfieldPlacementError::OutputCountBudgetExceeded {
                required: limit + 1,
                limit
            })
        );
    }
    for limit in [0, ROW_BYTES - 1, ROW_BYTES, 16 * ROW_BYTES - 1] {
        let limited = HeightfieldPlacementBudget {
            max_output_bytes: limit,
            ..exact
        };
        assert_eq!(
            scatter_on_heightfield(config, area, &author, &terrain.surface, filters(), limited),
            Err(HeightfieldPlacementError::OutputBytesBudgetExceeded {
                required: (limit / ROW_BYTES + 1) * ROW_BYTES,
                limit
            })
        );
    }
}

#[test]
fn rejected_output_still_charges_reserved_spare_capacity_but_only_accepted_rows_use_count() {
    let terrain = Terrain::new([-4, 0, -4], 4, [0; 9]);
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let area = region([-2, -2], [2, 2]);
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    );
    for (count, bytes, capacity) in [
        (16, 16 * ROW_BYTES, 16),
        (0, 0, 0),
        (16, 3 * ROW_BYTES, 3),
        (2, 16 * ROW_BYTES, 2),
    ] {
        let output = scatter_on_heightfield(
            config,
            area,
            &author,
            &terrain.surface,
            filters(),
            HeightfieldPlacementBudget {
                max_output_points: count,
                max_output_bytes: bytes,
                ..budget()
            },
        )
        .unwrap();
        assert!(output.points.is_empty());
        assert!(output.attributes.is_empty());
        assert_eq!(output.points.capacity(), capacity);
        assert_eq!(output.attributes.capacity(), capacity);
        assert_eq!(
            output.field_work, 16,
            "density-rejected candidates still cost field work"
        );
        assert_eq!(output.surface_sample_reads, 64);
        assert!(
            output.points.capacity() * size_of::<HeightfieldPlacementPoint>()
                + output.attributes.capacity() * size_of::<ScatterFieldSample>()
                <= bytes
        );
    }
    // A sparse accepted output fits one row even though sixteen primaries run.
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            inner_radius_mm: 1,
            outer_radius_mm: 1,
            ..brush()
        }],
    );
    let output = scatter_on_heightfield(
        config,
        area,
        &author,
        &terrain.surface,
        filters(),
        HeightfieldPlacementBudget {
            max_output_points: 1,
            max_output_bytes: ROW_BYTES,
            ..budget()
        },
    )
    .unwrap();
    assert_eq!(output.points.len(), 1);
    assert_eq!(output.points[0].position_mm, [0, 0, 0]);
}

#[test]
fn empty_axes_require_zero_work_and_capacity_but_do_not_bypass_input_validation() {
    let terrain = Terrain::new([-4, 0, -4], 4, [0; 9]);
    let author = full_field();
    let config = HeightfieldPlacementConfig {
        min_distance_mm: u32::MAX,
        ..config()
    };
    for area in [
        region([i64::MIN, i64::MIN], [i64::MIN, i64::MAX]),
        region([i64::MIN, i64::MAX], [i64::MAX, i64::MAX]),
        region([99, 99], [99, 99]),
    ] {
        assert_eq!(
            estimate_heightfield_placement_work(config, area, &author).unwrap(),
            HeightfieldPlacementWork {
                primary_candidates: 0,
                max_candidate_evaluations: 0,
                max_sample_reads: 0,
                max_field_work: 0,
            }
        );
        let output = scatter_on_heightfield(
            config,
            area,
            &author,
            &terrain.surface,
            filters(),
            zero_budget(),
        )
        .unwrap();
        assert!(output.points.is_empty());
        assert!(output.attributes.is_empty());
        assert_eq!(
            (output.points.capacity(), output.attributes.capacity()),
            (0, 0)
        );
        assert_eq!(
            (
                output.candidate_evaluations,
                output.surface_sample_reads,
                output.field_work
            ),
            (0, 0, 0)
        );
        assert!(!area.contains(area.min_xz_mm));
        let invalid = HeightfieldPlacementConfig {
            grid_spacing_mm: 0,
            ..config
        };
        assert_eq!(
            estimate_heightfield_placement_work(invalid, area, &author),
            Err(HeightfieldPlacementError::Source(
                ScatterError::InvalidGridSpacing
            ))
        );
        assert_eq!(
            scatter_on_heightfield(
                invalid,
                area,
                &author,
                &terrain.surface,
                filters(),
                zero_budget()
            ),
            Err(HeightfieldPlacementError::Source(
                ScatterError::InvalidGridSpacing
            ))
        );
        for limits in [
            HeightfieldPlacementFilters {
                surface_height_mm: [1, 0],
                ..filters()
            },
            HeightfieldPlacementFilters {
                max_slope_run: 0,
                ..filters()
            },
        ] {
            assert_eq!(
                scatter_on_heightfield(
                    config,
                    area,
                    &author,
                    &terrain.surface,
                    limits,
                    zero_budget()
                ),
                Err(HeightfieldPlacementError::InvalidFilters)
            );
        }
    }
}

#[test]
fn inverted_axes_and_every_uncovered_footprint_edge_fail_even_with_zero_density() {
    let terrain = Terrain::new([-4, 0, -4], 4, [0; 9]);
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    );
    for (min, max) in [([1, 0], [0, 0]), ([0, 1], [0, 0]), ([1, 1], [0, 0])] {
        assert_eq!(
            HeightfieldPlacementRegion::new(min, max),
            Err(HeightfieldPlacementError::InvalidRegion)
        );
        let area = HeightfieldPlacementRegion {
            min_xz_mm: min,
            max_xz_mm: max,
        };
        assert_eq!(
            estimate_heightfield_placement_work(config(), area, &author),
            Err(HeightfieldPlacementError::InvalidRegion)
        );
        assert_eq!(
            scatter_on_heightfield(
                config(),
                area,
                &author,
                &terrain.surface,
                filters(),
                zero_budget()
            ),
            Err(HeightfieldPlacementError::InvalidRegion)
        );
    }
    for area in [
        region([-5, -4], [4, 4]),
        region([-4, -5], [4, 4]),
        region([-4, -4], [5, 4]),
        region([-4, -4], [4, 5]),
    ] {
        assert_eq!(
            scatter_on_heightfield(
                config(),
                area,
                &author,
                &terrain.surface,
                filters(),
                zero_budget()
            ),
            Err(HeightfieldPlacementError::Surface(
                HeightfieldQueryError::OutsideSurface
            ))
        );
    }
    let area = region([-4, -4], [4, 4]);
    assert!(area.contains([-4, -4]));
    assert!(area.contains([3, 3]));
    for xz in [[4, 0], [0, 4], [-5, 0], [0, -5]] {
        assert!(!area.contains(xz));
    }
    assert!(run(config(), area, &terrain, &author).points.is_empty());
}

#[test]
fn checked_preflight_rejects_primary_halo_surface_and_field_work_overflow() {
    let terrain = Terrain::new([-4, 0, -4], 4, [0; 9]);
    let full = full_field();
    let noisy = field(
        ScatterFieldConfig {
            noise: Some(noise()),
            ..field_config()
        },
        vec![],
    );
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    for (config, area, author) in [
        (
            config,
            region([i64::MIN, i64::MIN], [i64::MAX, i64::MAX]),
            &full,
        ),
        (config, region([i64::MIN, 0], [i64::MAX, 1]), &full),
        (
            HeightfieldPlacementConfig {
                min_distance_mm: u32::MAX,
                ..config
            },
            region([0, 0], [1, 1]),
            &full,
        ),
        (config, region([0, 0], [1_i64 << 59, 1]), &noisy),
    ] {
        assert_eq!(
            estimate_heightfield_placement_work(config, area, author),
            Err(HeightfieldPlacementError::ArithmeticOverflow)
        );
        // Layout must fail before coverage, reservations, or expensive enumeration.
        assert_eq!(
            scatter_on_heightfield(
                config,
                area,
                author,
                &terrain.surface,
                filters(),
                zero_budget()
            ),
            Err(HeightfieldPlacementError::ArithmeticOverflow)
        );
    }
}

#[test]
fn work_admission_precedes_projection_and_projection_overflow_precedes_later_rejection() {
    let terrain = Terrain::new([0, i64::MAX, 0], 1, [0; 9]);
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        vertical_offset_mm: 1,
        ..config()
    };
    let area = region([0, 0], [1, 1]);
    let author = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    );
    let reject = HeightfieldPlacementFilters {
        surface_height_mm: [0, 0],
        ..filters()
    };
    for (limits, error) in [
        (
            HeightfieldPlacementBudget {
                max_candidate_evaluations: 0,
                ..budget()
            },
            HeightfieldPlacementError::CandidateBudgetExceeded {
                required: 1,
                limit: 0,
            },
        ),
        (
            HeightfieldPlacementBudget {
                max_sample_reads: 3,
                ..budget()
            },
            HeightfieldPlacementError::SurfaceBudgetExceeded {
                required: 4,
                limit: 3,
            },
        ),
        (
            HeightfieldPlacementBudget {
                max_field_work: 0,
                ..budget()
            },
            HeightfieldPlacementError::FieldBudgetExceeded {
                required: 1,
                limit: 0,
            },
        ),
        (
            HeightfieldPlacementBudget {
                max_output_points: 0,
                max_output_bytes: 0,
                ..budget()
            },
            HeightfieldPlacementError::ProjectionOverflow,
        ),
    ] {
        assert_eq!(
            scatter_on_heightfield(config, area, &author, &terrain.surface, reject, limits),
            Err(error)
        );
    }
    assert_eq!(
        key(config, 0, 0).rebuild(&terrain.surface, 1),
        Err(HeightfieldPlacementError::ProjectionOverflow)
    );
    let low = Terrain::new([0, i64::MIN, 0], 1, [0; 9]);
    assert_eq!(
        key(config, 0, 0).rebuild(&low.surface, -1),
        Err(HeightfieldPlacementError::ProjectionOverflow)
    );
    for offset in [i64::MIN, i64::MAX] {
        let flat = Terrain::new([0, 0, 0], 1, [0; 9]);
        assert_eq!(
            key(config, 0, 0)
                .rebuild(&flat.surface, offset)
                .unwrap()
                .position_mm[1],
            offset
        );
    }
}

#[test]
fn cache_records_all_versions_config_region_filters_and_excludes_every_budget() {
    let terrain = Terrain::new([-8, 0, -8], 8, [0; 9]);
    let author = full_field();
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let area = region([-1, -1], [2, 2]);
    let limits = filters();
    let baseline = run(config, area, &terrain, &author);
    assert_eq!(
        baseline.cache_key,
        HeightfieldPlacementCacheKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source_kernel_version: SCATTER_KERNEL_VERSION,
            surface_version: HEIGHTFIELD_SURFACE_VERSION,
            field_version: SCATTER_FIELD_VERSION,
            config,
            region: area,
            filters: limits,
            surface: terrain.surface.clone(),
            field: author.clone(),
        }
    );
    assert!(Arc::ptr_eq(
        baseline.cache_key.surface.tile(),
        terrain.surface.tile()
    ));
    for changed in [
        HeightfieldPlacementConfig {
            seed: config.seed ^ (1 << 63),
            ..config
        },
        HeightfieldPlacementConfig {
            version: config.version + 1,
            ..config
        },
        HeightfieldPlacementConfig {
            salt: config.salt ^ (1 << 63),
            ..config
        },
        HeightfieldPlacementConfig {
            grid_spacing_mm: 2,
            ..config
        },
        HeightfieldPlacementConfig {
            min_distance_mm: 1,
            ..config
        },
        HeightfieldPlacementConfig {
            vertical_offset_mm: 1,
            ..config
        },
    ] {
        assert_ne!(
            run(changed, area, &terrain, &author).cache_key,
            baseline.cache_key,
            "config={changed:?}"
        );
    }
    for changed in [
        region([-2, -1], [2, 2]),
        region([-1, -2], [2, 2]),
        region([-1, -1], [3, 2]),
        region([-1, -1], [2, 3]),
    ] {
        assert_ne!(
            run(config, changed, &terrain, &author).cache_key,
            baseline.cache_key
        );
    }
    // These limits are inactive on the flat surface, but still invalidate.
    for changed in [
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN + 1, i64::MAX],
            ..limits
        },
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX - 1],
            ..limits
        },
        HeightfieldPlacementFilters {
            max_slope_rise: limits.max_slope_rise - 1,
            ..limits
        },
        HeightfieldPlacementFilters {
            max_slope_run: 2,
            ..limits
        },
    ] {
        let output =
            scatter_on_heightfield(config, area, &author, &terrain.surface, changed, budget())
                .unwrap();
        assert_eq!(rows(&output), rows(&baseline));
        assert_ne!(output.cache_key, baseline.cache_key);
    }
    for revision in 0..4 {
        let mut changed = baseline.cache_key.clone();
        match revision {
            0 => changed.placement_version += 1,
            1 => changed.source_kernel_version += 1,
            2 => changed.surface_version += 1,
            _ => changed.field_version += 1,
        }
        assert_ne!(changed, baseline.cache_key);
    }
    let work = estimate_heightfield_placement_work(config, area, &author).unwrap();
    let tight = HeightfieldPlacementBudget {
        max_candidate_evaluations: work.max_candidate_evaluations,
        max_sample_reads: work.max_sample_reads,
        max_field_work: work.max_field_work,
        max_output_points: baseline.points.len(),
        max_output_bytes: baseline.points.len() * ROW_BYTES,
    };
    for changed in [
        tight,
        HeightfieldPlacementBudget {
            max_candidate_evaluations: tight.max_candidate_evaluations + 1,
            ..tight
        },
        HeightfieldPlacementBudget {
            max_sample_reads: tight.max_sample_reads + 1,
            ..tight
        },
        HeightfieldPlacementBudget {
            max_field_work: tight.max_field_work + 1,
            ..tight
        },
        HeightfieldPlacementBudget {
            max_output_points: tight.max_output_points + 1,
            ..tight
        },
        HeightfieldPlacementBudget {
            max_output_bytes: tight.max_output_bytes + 1,
            ..tight
        },
    ] {
        assert_eq!(
            scatter_on_heightfield(config, area, &author, &terrain.surface, limits, changed)
                .unwrap(),
            baseline
        );
    }
}

#[test]
fn cache_compares_complete_terrain_payload_revision_tile_spec_and_embedding_by_value() {
    let terrain = Terrain::new([-8, 0, -8], 8, [0; 9]);
    let author = full_field();
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let area = region([0, 0], [2, 2]);
    let baseline = run(config, area, &terrain, &author);
    let equal = Terrain::new(terrain.origin, terrain.spacing, terrain.heights);
    assert!(!Arc::ptr_eq(equal.surface.tile(), terrain.surface.tile()));
    assert_eq!(
        run(config, area, &equal, &author).cache_key,
        baseline.cache_key
    );
    let embedding = terrain.surface.embedding();
    let mut variants = vec![];
    for axis in 0..3 {
        let mut changed = embedding;
        changed.sample_origin_mm[axis] += 1;
        variants
            .push(PlanarHeightfieldSurface::new(terrain.surface.tile().clone(), changed).unwrap());
    }
    variants.push(
        PlanarHeightfieldSurface::new(
            terrain.surface.tile().clone(),
            PlanarHeightfieldEmbedding {
                sample_spacing_mm: 9,
                ..embedding
            },
        )
        .unwrap(),
    );
    let mut samples = [0; 9];
    samples[8] = 1;
    let changed_payload = bake(&samples, 3, 1, terrain.surface.tile().tile());
    assert_eq!(
        changed_payload.revision(),
        terrain.surface.tile().revision()
    );
    assert_eq!(changed_payload.tile(), terrain.surface.tile().tile());
    variants.push(PlanarHeightfieldSurface::new(Arc::new(changed_payload), embedding).unwrap());
    for tile in [
        bake(&[0; 9], 3, 1, TileKey::root(CubeFace::NegativeY)),
        bake(
            &[0; 9],
            3,
            1,
            TileKey::new(CubeFace::PositiveY, 1, 0, 0).unwrap(),
        ),
        bake(&[0; 9], 3, 2, terrain.surface.tile().tile()),
        bake(&[0; 16], 4, 1, terrain.surface.tile().tile()),
    ] {
        variants.push(PlanarHeightfieldSurface::new(Arc::new(tile), embedding).unwrap());
    }
    let mut rebaked = terrain.surface.tile().as_ref().clone();
    rebaked
        .rebake(
            DirtyRegion::new(rebaked.tile(), 0, 0, u16::MAX, u16::MAX).unwrap(),
            &AuthoredGrid {
                side: 3,
                heights: &[0; 9],
            },
            BakeLimits {
                max_samples: 9,
                max_bytes: 18,
            },
        )
        .unwrap();
    assert_eq!(rebaked.samples(), terrain.surface.tile().samples());
    assert_ne!(rebaked.revision(), terrain.surface.tile().revision());
    variants.push(PlanarHeightfieldSurface::new(Arc::new(rebaked), embedding).unwrap());
    for changed in variants {
        let output =
            scatter_on_heightfield(config, area, &author, &changed, filters(), budget()).unwrap();
        assert_ne!(output.cache_key, baseline.cache_key);
        assert!(Arc::ptr_eq(output.cache_key.surface.tile(), changed.tile()));
    }
}

#[test]
fn cache_retains_every_field_input_including_disabled_noise_and_zero_opacity_brushes() {
    let terrain = Terrain::new([-8, 0, -8], 8, [0; 9]);
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let area = region([-1, -1], [2, 2]);
    let inactive = DensityBrush {
        center_mm: [1_000; 3],
        opacity: 0,
        ..brush()
    };
    let disabled = DensityNoise {
        amplitude: 0,
        warp_amplitude_mm: 0,
        ..noise()
    };
    let recipe = ScatterFieldConfig {
        noise: Some(disabled),
        ..field_config()
    };
    let author = field(recipe, vec![inactive]);
    let baseline = run(config, area, &terrain, &author);
    assert_eq!(baseline.cache_key.field, author);
    assert_eq!(
        baseline.cache_key.field.brushes().as_ptr(),
        author.brushes().as_ptr()
    );
    let mut recipes = vec![
        ScatterFieldConfig {
            seed: recipe.seed ^ (1 << 63),
            ..recipe
        },
        ScatterFieldConfig {
            version: recipe.version + 1,
            ..recipe
        },
        ScatterFieldConfig {
            base_density: recipe.base_density - 1,
            ..recipe
        },
        ScatterFieldConfig {
            base_species: recipe.base_species + 1,
            ..recipe
        },
        ScatterFieldConfig {
            noise: None,
            ..recipe
        },
    ];
    for changed in [
        DensityNoise {
            cell_size_mm: disabled.cell_size_mm + 1,
            ..disabled
        },
        DensityNoise {
            amplitude: 1,
            ..disabled
        },
        DensityNoise {
            warp_cell_size_mm: disabled.warp_cell_size_mm + 1,
            ..disabled
        },
        DensityNoise {
            warp_amplitude_mm: 1,
            ..disabled
        },
    ] {
        recipes.push(ScatterFieldConfig {
            noise: Some(changed),
            ..recipe
        });
    }
    for changed in recipes {
        assert_ne!(
            run(config, area, &terrain, &field(changed, vec![inactive])).cache_key,
            baseline.cache_key
        );
    }
    let mut brushes = vec![
        DensityBrush {
            id: inactive.id + 1,
            ..inactive
        },
        DensityBrush {
            inner_radius_mm: inactive.inner_radius_mm + 1,
            ..inactive
        },
        DensityBrush {
            outer_radius_mm: inactive.outer_radius_mm + 1,
            ..inactive
        },
        DensityBrush {
            opacity: 1,
            ..inactive
        },
        DensityBrush {
            density: DensityBrushOp::Add(0),
            ..inactive
        },
        DensityBrush {
            density: DensityBrushOp::Replace(0),
            ..inactive
        },
        DensityBrush {
            species: None,
            ..inactive
        },
        DensityBrush {
            species: Some(78),
            ..inactive
        },
    ];
    for axis in 0..3 {
        let mut changed = inactive;
        changed.center_mm[axis] += 1;
        brushes.push(changed);
    }
    for changed in brushes {
        let output = run(config, area, &terrain, &field(recipe, vec![changed]));
        assert_eq!(
            rows(&output),
            rows(&baseline),
            "inactive field edit changed geometry"
        );
        assert_ne!(output.cache_key, baseline.cache_key);
    }
    assert_ne!(
        run(config, area, &terrain, &field(recipe, vec![])).cache_key,
        baseline.cache_key
    );
    let other = DensityBrush { id: 2, ..inactive };
    let forward = field(recipe, vec![other, inactive]);
    let reversed = field(recipe, vec![inactive, other]);
    assert_eq!(
        run(config, area, &terrain, &forward).cache_key,
        run(config, area, &terrain, &reversed).cache_key
    );
    assert_eq!(
        baseline.cache_key,
        run(config, area, &terrain, &field(recipe, vec![inactive])).cache_key
    );
}

#[test]
fn i64_horizontal_edges_use_widened_coordinates_and_global_halo_without_wraparound() {
    let author = full_field();
    for origin in [
        [i64::MIN, 0, i64::MIN],
        [i64::MAX - 32, 0, i64::MAX - 32],
        [i64::MIN, 0, i64::MAX - 32],
    ] {
        let terrain = Terrain::new(origin, 16, HEIGHTS);
        for spacing in [1, 3, u32::MAX] {
            for radius in [0, 4] {
                let config = HeightfieldPlacementConfig {
                    grid_spacing_mm: spacing,
                    min_distance_mm: radius,
                    ..config()
                };
                let area = terrain.region();
                let output = run(config, area, &terrain, &author);
                assert_eq!(
                    rows(&output),
                    oracle_rows(config, area, &terrain, &author, filters()),
                    "origin={origin:?}, spacing={spacing}, radius={radius}"
                );
                let work = estimate_heightfield_placement_work(config, area, &author).unwrap();
                assert!(output.candidate_evaluations <= work.max_candidate_evaluations);
                for point in &output.points {
                    assert_eq!(point.key.rebuild(&terrain.surface, 0).unwrap(), *point);
                    assert!(area.contains([point.position_mm[0], point.position_mm[2]]));
                }
            }
        }
    }
}

#[test]
fn exact_projection_can_cancel_out_of_i64_surface_height_and_extreme_normals_fail_slope() {
    let author = full_field();
    let config = HeightfieldPlacementConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    for (origin_y, height, offset, expected) in
        [(i64::MAX, 1, -1, i64::MAX), (i64::MIN, -1, 1, i64::MIN)]
    {
        let terrain = Terrain::new([0, origin_y, 0], 1, [height; 9]);
        let point = key(config, 0, 0).rebuild(&terrain.surface, offset).unwrap();
        assert_eq!(point.position_mm, [0, expected, 0]);
        assert_eq!(
            point.surface_sample.height_numerator_mm(),
            i128::from(origin_y) + i128::from(height)
        );
        assert_eq!(
            key(config, 0, 0).rebuild(&terrain.surface, 0),
            Err(HeightfieldPlacementError::ProjectionOverflow)
        );
        // Geometry fits after cancellation; exact unoffset height is still out
        // of the i64-valued closed filter interval and must be rejected.
        let output = run(
            HeightfieldPlacementConfig {
                vertical_offset_mm: offset,
                ..config
            },
            region([0, 0], [1, 1]),
            &terrain,
            &author,
        );
        assert!(output.points.is_empty());
        assert_eq!(output.surface_sample_reads, 4);
        assert_eq!(output.field_work, 0);
    }
    let tile = bake(
        &[-i64::MAX, i64::MAX, i64::MAX, 0],
        2,
        i64::MAX,
        TileKey::root(CubeFace::PositiveY),
    );
    let surface = PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, i64::MAX, 0],
            sample_spacing_mm: 1,
        },
    )
    .unwrap();
    let point = key(config, 0, 0).rebuild(&surface, 0).unwrap();
    assert_eq!(point.position_mm, [0, 0, 0]);
    assert_eq!(
        point.surface_sample.normal_mm(),
        [-2 * i128::from(i64::MAX), 1, -2 * i128::from(i64::MAX)]
    );
    let output = scatter_on_heightfield(
        config,
        region([0, 0], [1, 1]),
        &author,
        &surface,
        filters(),
        budget(),
    )
    .unwrap();
    assert!(output.points.is_empty());
    assert_eq!(output.surface_sample_reads, 4);
    assert_eq!(output.field_work, 0);
}
