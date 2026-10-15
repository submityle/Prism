//! Content-neutral deletion thinning after CPU heightfield scatter selection.
//!
//! The complete source cache key, deletion set (including absent keys), and
//! [`prism_world_residency::SCATTER_DELETION_VERSION`] are cache dependencies.
//! Neither a per-point hash nor a shortened terrain/field descriptor replaces
//! them. This adapter owns no runtime, persistence, or GPU policy.

use prism_world_residency::{ScatterDeletionError, ScatterDeletionSet};

use crate::HeightfieldScatterOutput;

/// Heightfield survivors and the complete additional deletion cache dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct DeletionFilteredHeightfieldScatter {
    /// Original output with deleted rows removed, retaining its full cache key,
    /// upstream work counters, and both Vec allocations (including capacity).
    pub source: HeightfieldScatterOutput,
    /// Complete immutable deletion input, including keys absent from this output.
    pub deletions: ScatterDeletionSet,
    /// Actual number of source rows removed, not the deletion-set cardinality.
    pub removed: usize,
}

/// Stably removes exact full-key matches without allocating point/attribute Vecs.
///
/// Points and their author-field attributes compact together in source order;
/// survivor positions, minimum distances, and XYZ partition ownership are
/// unchanged. Neither Vec's pointer or capacity changes, even if all rows are
/// removed. All upstream work counters and the complete source cache key remain
/// unchanged: deletion inputs are an additional cache dependency, not a rewrite
/// of the source key. Cloning the deletion set shares its immutable allocation.
///
/// Unequal column lengths return [`ScatterDeletionError::MisalignedAttributes`]
/// before filtering, including for empty deletion sets or one empty column.
/// The source is consumed on either success or failure. Filtering costs
/// O(P log(D + 1)) time and O(1) additional space for P rows and D deletion keys;
/// it does not revalidate manually constructed source keys or rebake terrain.
pub fn filter_heightfield_scatter(
    mut source: HeightfieldScatterOutput,
    deletions: &ScatterDeletionSet,
) -> Result<DeletionFilteredHeightfieldScatter, ScatterDeletionError> {
    if source.points.len() != source.attributes.len() {
        return Err(ScatterDeletionError::MisalignedAttributes);
    }
    let original_len = source.points.len();
    let mut write = 0;
    for read in 0..original_len {
        if !deletions.is_deleted(source.points[read].key) {
            if read != write {
                source.points.swap(write, read);
                source.attributes.swap(write, read);
            }
            write += 1;
        }
    }
    source.points.truncate(write);
    source.attributes.truncate(write);
    Ok(DeletionFilteredHeightfieldScatter {
        source,
        deletions: deletions.clone(),
        removed: original_len - write,
    })
}

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
    use core::{convert::Infallible, mem::size_of};

    use super::*;
    use crate::{
        scatter_with_heightfield, BakeLimits, CubeFace, DirtyRegion, HeightSample,
        HeightSampleValue, HeightSampler, HeightfieldScatterBudget, HeightfieldScatterFilters,
        HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding, PlanarHeightfieldSurface,
        TileKey,
    };
    use prism_world_residency::{
        DensityBrush, DensityBrushOp, DensityNoise, FieldScatterBudget, ScatterBudget,
        ScatterCandidateKey, ScatterConfig, ScatterDeletionBudget, ScatterDensityField,
        ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
        ScatterPoint, ScatterRegion,
    };

    const LIMIT: usize = 8_192;
    const FLAT: [i64; 9] = [0; 9];
    const SLOPE: [i64; 9] = [0, 2, 4, -2, 0, 2, -4, -2, 0];

    struct GridHeights<'a>(&'a [i64; 9]);

    impl HeightSampler for GridHeights<'_> {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let index = |q: u16| ((u64::from(q) * 2 + 32_767) / 65_535) as usize;
            Ok(HeightSampleValue::from_millimetres(
                self.0[index(sample.v_q16()) * 3 + index(sample.u_q16())],
            ))
        }
    }

    fn surface(heights: &[i64; 9]) -> PlanarHeightfieldSurface {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(3, 1).unwrap(),
            &GridHeights(heights),
            bake_limits(),
        )
        .unwrap();
        PlanarHeightfieldSurface::new(
            Arc::new(tile),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-8, 0, -8],
                sample_spacing_mm: 8,
            },
        )
        .unwrap()
    }

    fn bake_limits() -> BakeLimits {
        BakeLimits {
            max_samples: 9,
            max_bytes: 9 * size_of::<i16>(),
        }
    }

    fn config() -> ScatterConfig {
        ScatterConfig {
            seed: 0x3a65_c807_9742_1eab,
            version: 7,
            salt: 0xf09c_1172,
            grid_spacing_mm: 1,
            min_distance_mm: 0,
        }
    }

    fn region() -> ScatterRegion {
        ScatterRegion::new([-6, -3, -6], [7, 4, 7]).unwrap()
    }

    fn unrestricted() -> HeightfieldScatterFilters {
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            clearance_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        }
    }

    fn field(painted: bool) -> ScatterDensityField {
        let brushes = if painted {
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
            ]
        } else {
            Vec::new()
        };
        let budget = ScatterFieldBuildBudget {
            max_brushes: brushes.len(),
            max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
        };
        ScatterDensityField::new(
            ScatterFieldConfig {
                seed: 91,
                version: 3,
                base_density: if painted { 46_000 } else { u16::MAX },
                base_species: 11,
                noise: painted.then_some(DensityNoise {
                    cell_size_mm: 7,
                    amplitude: 12_000,
                    warp_cell_size_mm: 11,
                    warp_amplitude_mm: 3,
                }),
            },
            brushes,
            budget,
        )
        .unwrap()
    }

    fn generate(
        config: ScatterConfig,
        region: ScatterRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldScatterFilters,
    ) -> HeightfieldScatterOutput {
        scatter_with_heightfield(
            config,
            region,
            ScatterBudget {
                max_candidate_evaluations: 1_000_000,
                max_output_points: LIMIT,
                max_output_bytes: LIMIT * size_of::<ScatterPoint>(),
            },
            field,
            FieldScatterBudget {
                max_field_work: 1_000_000,
                max_output_bytes: LIMIT
                    * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
            },
            surface,
            filters,
            HeightfieldScatterBudget {
                max_sample_reads: 4 * LIMIT as u64,
            },
        )
        .unwrap()
    }

    fn source(painted: bool) -> HeightfieldScatterOutput {
        generate(
            config(),
            region(),
            &field(painted),
            &surface(&SLOPE),
            unrestricted(),
        )
    }

    fn deletion_budget() -> ScatterDeletionBudget {
        ScatterDeletionBudget {
            max_keys: LIMIT,
            max_key_bytes: LIMIT * size_of::<ScatterCandidateKey>(),
        }
    }

    fn deletions(keys: Vec<ScatterCandidateKey>) -> ScatterDeletionSet {
        ScatterDeletionSet::new(keys, deletion_budget()).unwrap()
    }

    fn outside_key() -> ScatterCandidateKey {
        let mut key = source(false).points[0].key;
        key.grid.x = 1_000;
        key
    }

    fn layout(
        source: &HeightfieldScatterOutput,
    ) -> (*const ScatterPoint, usize, *const ScatterFieldSample, usize) {
        (
            source.points.as_ptr(),
            source.points.capacity(),
            source.attributes.as_ptr(),
            source.attributes.capacity(),
        )
    }

    fn rows(source: &HeightfieldScatterOutput) -> Vec<(ScatterPoint, ScatterFieldSample)> {
        source
            .points
            .iter()
            .copied()
            .zip(source.attributes.iter().copied())
            .collect()
    }

    #[test]
    fn empty_deletions_preserve_entire_real_source_and_allocations() {
        let original = source(true);
        let expected = source(true);
        let before = layout(&original);
        let tile = original.cache_key.surface.tile().clone();
        let result = filter_heightfield_scatter(original, &ScatterDeletionSet::default()).unwrap();
        assert_eq!(result.source, expected);
        assert_eq!(result.removed, 0);
        assert_eq!(layout(&result.source), before);
        assert!(Arc::ptr_eq(result.source.cache_key.surface.tile(), &tile));
    }

    #[test]
    fn empty_generated_source_retains_spare_capacity_and_full_deletions() {
        let mut original = generate(
            config(),
            ScatterRegion::new([-6, -3, -6], [-6, 4, 7]).unwrap(),
            &field(false),
            &surface(&FLAT),
            unrestricted(),
        );
        assert!(original.points.is_empty());
        original.points.reserve(17);
        original.attributes.reserve(31);
        let before = layout(&original);
        let key = original.cache_key.clone();
        let set = deletions(vec![outside_key()]);
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert_eq!(result.removed, 0);
        assert_eq!(result.source.cache_key, key);
        assert_eq!(layout(&result.source), before);
        assert_eq!(result.deletions, set);
        assert_eq!(result.deletions.keys().as_ptr(), set.keys().as_ptr());
    }

    #[test]
    fn deleting_every_row_retains_both_nonempty_allocations() {
        let mut original = source(false);
        original.points.reserve(19);
        original.attributes.reserve(41);
        let before = layout(&original);
        let count = original.points.len();
        assert!(count > 0);
        let set = deletions(original.points.iter().map(|p| p.key).collect());
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert_eq!(result.removed, count);
        assert!(result.source.points.is_empty());
        assert!(result.source.attributes.is_empty());
        assert_eq!(layout(&result.source), before);
    }

    #[test]
    fn first_middle_last_and_alternating_deletions_are_stable() {
        for alternating in [false, true] {
            let original = source(false);
            let last = original.points.len() - 1;
            let deleted = |i: usize| {
                if alternating {
                    i.is_multiple_of(2)
                } else {
                    [0, last / 2, last].contains(&i)
                }
            };
            let expected: Vec<_> = rows(&original)
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !deleted(*i))
                .map(|(_, row)| row)
                .collect();
            let set = deletions(
                original
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| deleted(*i))
                    .map(|(_, p)| p.key)
                    .collect(),
            );
            let count = original.points.len();
            let result = filter_heightfield_scatter(original, &set).unwrap();
            assert_eq!(rows(&result.source), expected);
            assert_eq!(result.removed, count - expected.len());
        }
    }

    #[test]
    fn real_density_and_species_remain_paired_with_original_points() {
        let original = source(true);
        assert!(original
            .attributes
            .windows(2)
            .any(|w| w[0].density != w[1].density));
        assert!(original
            .attributes
            .windows(2)
            .any(|w| w[0].species != w[1].species));
        let expected: Vec<_> = rows(&original)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % 3 != 1)
            .map(|(_, row)| row)
            .collect();
        let set = deletions(
            original
                .points
                .iter()
                .skip(1)
                .step_by(3)
                .map(|p| p.key)
                .collect(),
        );
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert!(result.removed > 0);
        assert_eq!(rows(&result.source), expected);
    }

    #[test]
    fn partial_compaction_preserves_pointers_and_distinct_spare_capacities() {
        let mut original = source(true);
        original.points.reserve(97);
        original.attributes.reserve(313);
        assert!(original.points.capacity() > original.points.len());
        assert!(original.attributes.capacity() > original.attributes.len());
        let before = layout(&original);
        let set = deletions(vec![original.points[0].key, original.points[2].key]);
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert_eq!(result.removed, 2);
        assert!(!result.source.points.is_empty());
        assert_eq!(layout(&result.source), before);
    }

    #[test]
    fn misalignment_fails_before_any_empty_or_full_set_shortcut() {
        for mode in 0..4 {
            for populated in [false, true] {
                let mut original = source(false);
                let set = if populated {
                    deletions(original.points.iter().map(|p| p.key).collect())
                } else {
                    ScatterDeletionSet::default()
                };
                match mode {
                    0 => {
                        original.points.pop();
                    }
                    1 => {
                        original.attributes.pop();
                    }
                    2 => original.points.clear(),
                    _ => original.attributes.clear(),
                }
                assert_eq!(
                    filter_heightfield_scatter(original, &set),
                    Err(ScatterDeletionError::MisalignedAttributes)
                );
            }
        }
    }

    #[test]
    fn every_candidate_key_field_participates_in_exact_membership() {
        let key = source(false).points[0].key;
        let mut alternatives = [key; 7];
        alternatives[0].seed ^= 1;
        alternatives[1].version += 1;
        alternatives[2].salt ^= 1;
        alternatives[3].grid_spacing_mm += 1;
        alternatives[4].grid.x += 1;
        alternatives[5].grid.y += 1;
        alternatives[6].grid.z += 1;
        for alternative in alternatives {
            let mut original = source(false);
            original.points.truncate(1);
            original.attributes.truncate(1);
            let result =
                filter_heightfield_scatter(original, &deletions(vec![alternative])).unwrap();
            assert_eq!(result.removed, 0);
            assert_eq!(result.source.points[0].key, key);
        }
        let mut foreign = key;
        foreign.kernel_version += 1;
        assert_eq!(
            ScatterDeletionSet::new(vec![foreign], deletion_budget()),
            Err(ScatterDeletionError::InvalidKey(
                ScatterError::UnsupportedKernelVersion
            ))
        );
        // Public output fields can contain a synthetic foreign-kernel row. The
        // adapter compares its complete key rather than projecting known fields.
        let mut original = source(false);
        original.points.truncate(1);
        original.attributes.truncate(1);
        original.points[0].key = foreign;
        let set = deletions(vec![key]);
        assert_eq!(
            filter_heightfield_scatter(original, &set).unwrap().removed,
            0
        );
        let result = filter_heightfield_scatter(source(false), &set).unwrap();
        assert_eq!(result.removed, 1);
    }

    #[test]
    fn regenerated_seed_version_salt_and_spacing_namespaces_do_not_alias() {
        let original = source(false);
        let set = deletions(original.points.iter().map(|p| p.key).collect());
        let mut configs = [config(); 4];
        configs[0].seed += 1;
        configs[1].version += 1;
        configs[2].salt += 1;
        configs[3].grid_spacing_mm = 2;
        for config in configs {
            let regenerated = generate(
                config,
                region(),
                &field(false),
                &surface(&SLOPE),
                unrestricted(),
            );
            assert!(!regenerated.points.is_empty());
            let expected = rows(&regenerated);
            let result = filter_heightfield_scatter(regenerated, &set).unwrap();
            assert_eq!(result.removed, 0);
            assert_eq!(rows(&result.source), expected);
        }
    }

    #[test]
    fn reversed_xyz_partitions_equal_whole_request_with_terrain_and_density() {
        let config = ScatterConfig {
            min_distance_mm: 2,
            ..config()
        };
        let surface = surface(&SLOPE);
        let field = field(true);
        let filters = HeightfieldScatterFilters {
            surface_height_mm: [-2, 2],
            clearance_mm: [-2, 2],
            max_slope_rise: 1,
            max_slope_run: 2,
        };
        let original = generate(config, region(), &field, &surface, filters);
        assert!(original.points.len() > 3);
        let mut keys: Vec<_> = original.points.iter().step_by(3).map(|p| p.key).collect();
        keys.push(outside_key());
        let set = deletions(keys);
        let whole = filter_heightfield_scatter(original, &set).unwrap();
        let expected: BTreeMap<_, _> = rows(&whole.source)
            .into_iter()
            .map(|row| (row.0.key, row))
            .collect();
        let mut union = BTreeMap::new();
        let mut removed = 0;
        for x in (0..2).rev() {
            for y in (0..2).rev() {
                for z in (0..2).rev() {
                    let xs = [-6, 0, 7];
                    let ys = [-3, 0, 4];
                    let zs = [-6, 0, 7];
                    let region = ScatterRegion::new(
                        [xs[x], ys[y], zs[z]],
                        [xs[x + 1], ys[y + 1], zs[z + 1]],
                    )
                    .unwrap();
                    let part = filter_heightfield_scatter(
                        generate(config, region, &field, &surface, filters),
                        &set,
                    )
                    .unwrap();
                    assert_eq!(part.deletions, set);
                    removed += part.removed;
                    for row in rows(&part.source) {
                        assert!(union.insert(row.0.key, row).is_none());
                    }
                }
            }
        }
        assert_eq!(removed, whole.removed);
        assert_eq!(union, expected);
    }

    #[test]
    fn surviving_coordinates_rebuild_and_source_minimum_distance_is_preserved() {
        let config = ScatterConfig {
            min_distance_mm: 2,
            ..config()
        };
        let original = generate(
            config,
            region(),
            &field(true),
            &surface(&SLOPE),
            unrestricted(),
        );
        let set = deletions(original.points.iter().step_by(2).map(|p| p.key).collect());
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert!(result.source.points.len() > 2);
        assert!(result.removed > 0);
        for (i, point) in result.source.points.iter().enumerate() {
            assert_eq!(point.key.rebuild().unwrap(), *point);
            assert!(region().contains_mm(point.position_mm));
            for other in &result.source.points[i + 1..] {
                let squared: i128 = point
                    .position_mm
                    .iter()
                    .zip(other.position_mm)
                    .map(|(&a, b)| (i128::from(a) - i128::from(b)).pow(2))
                    .sum();
                assert!(squared >= i128::from(config.min_distance_mm).pow(2));
            }
        }
    }

    #[test]
    fn full_source_key_positive_work_counters_and_shared_snapshots_are_unchanged() {
        let original = source(true);
        let key = original.cache_key.clone();
        let counts = (
            original.candidate_evaluations,
            original.field_work,
            original.surface_sample_reads,
        );
        assert!(counts.0 > 0 && counts.1 > 0 && counts.2 > 0);
        let set = deletions(original.points.iter().step_by(3).map(|p| p.key).collect());
        let result = filter_heightfield_scatter(original, &set).unwrap();
        assert!(result.removed > 0);
        assert_eq!(result.source.cache_key, key);
        assert!(Arc::ptr_eq(
            result.source.cache_key.surface.tile(),
            key.surface.tile()
        ));
        assert_eq!(result.deletions.keys().as_ptr(), set.keys().as_ptr());
        assert_eq!(
            (
                result.source.candidate_evaluations,
                result.source.field_work,
                result.source.surface_sample_reads
            ),
            counts
        );
    }

    #[test]
    fn equal_bakes_share_value_identity_but_inactive_samples_still_invalidate() {
        let a = surface(&FLAT);
        let equal = surface(&FLAT);
        assert!(!Arc::ptr_eq(a.tile(), equal.tile()));
        let mut heights = FLAT;
        heights[8] = 9; // Outside the queried top-left cell, but part of full identity.
        let changed = surface(&heights);
        let small = ScatterRegion::new([-7, -1, -7], [-5, 2, -5]).unwrap();
        let set = ScatterDeletionSet::default();
        let run = |surface| {
            filter_heightfield_scatter(
                generate(config(), small, &field(false), surface, unrestricted()),
                &set,
            )
            .unwrap()
        };
        let first = run(&a);
        let same = run(&equal);
        let different = run(&changed);
        assert_eq!(first, same);
        assert_eq!(rows(&first.source), rows(&different.source));
        assert_eq!(a.tile().revision(), changed.tile().revision());
        assert_eq!(a.tile().tile(), changed.tile().tile());
        assert_ne!(first.source.cache_key, different.source.cache_key);
    }

    #[test]
    fn embedding_filters_field_source_and_all_algorithm_revisions_remain_dependencies() {
        let terrain = surface(&FLAT);
        let original = generate(config(), region(), &field(false), &terrain, unrestricted());
        let key = original.cache_key.clone();
        let shifted = PlanarHeightfieldSurface::new(
            terrain.tile().clone(),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-8, 1, -8],
                ..terrain.embedding()
            },
        )
        .unwrap();
        let changed_filters = HeightfieldScatterFilters {
            clearance_mm: [-100, 100],
            ..unrestricted()
        };
        let altered_field = ScatterDensityField::new(
            ScatterFieldConfig {
                version: 4,
                ..field(false).config()
            },
            Vec::new(),
            ScatterFieldBuildBudget {
                max_brushes: 0,
                max_brush_bytes: 0,
            },
        )
        .unwrap();
        let variants = [
            generate(config(), region(), &field(false), &shifted, unrestricted()),
            generate(config(), region(), &field(false), &terrain, changed_filters),
            generate(config(), region(), &altered_field, &terrain, unrestricted()),
            generate(
                ScatterConfig {
                    min_distance_mm: 1,
                    ..config()
                },
                region(),
                &field(false),
                &terrain,
                unrestricted(),
            ),
        ];
        for source in variants {
            assert_ne!(source.cache_key, key);
            let expected = source.cache_key.clone();
            let result =
                filter_heightfield_scatter(source, &ScatterDeletionSet::default()).unwrap();
            assert_eq!(result.source.cache_key, expected);
        }
        // Public descriptors may carry future revisions: filtering must retain,
        // not normalize or reconstruct, each layer's full algorithm identity.
        for revision in 0..4 {
            let mut original = source(false);
            match revision {
                0 => original.cache_key.surface_version += 1,
                1 => original.cache_key.filter_version += 1,
                2 => original.cache_key.source.field_version += 1,
                _ => original.cache_key.source.scatter.kernel_version += 1,
            }
            let expected = original.cache_key.clone();
            let result =
                filter_heightfield_scatter(original, &ScatterDeletionSet::default()).unwrap();
            assert_eq!(result.source.cache_key, expected);
            assert_ne!(result.source.cache_key, source(false).cache_key);
        }
    }

    #[test]
    fn absent_deletions_are_complete_additional_cache_dependencies() {
        let first =
            filter_heightfield_scatter(source(true), &ScatterDeletionSet::default()).unwrap();
        let set = deletions(vec![outside_key()]);
        let second = filter_heightfield_scatter(source(true), &set).unwrap();
        assert_eq!(first.source, second.source);
        assert_eq!(first.removed, second.removed);
        assert_ne!(first.deletions, second.deletions);
        assert_ne!(first, second);
    }

    #[test]
    fn retained_wrapper_survives_public_rebake_and_immutable_deletion_edits() {
        let old_surface = surface(&FLAT);
        let original = generate(
            config(),
            region(),
            &field(true),
            &old_surface,
            unrestricted(),
        );
        let set = deletions(vec![original.points[0].key]);
        let key = original.cache_key.clone();
        let expected = rows(&original)[1..].to_vec();
        let retained = filter_heightfield_scatter(original, &set).unwrap();
        let mut owner = old_surface.tile().as_ref().clone();
        let dirty = DirtyRegion::new(owner.tile(), 0, 0, u16::MAX, u16::MAX).unwrap();
        owner
            .rebake(dirty, &GridHeights(&[1; 9]), bake_limits())
            .unwrap();
        let published =
            PlanarHeightfieldSurface::new(Arc::new(owner), old_surface.embedding()).unwrap();
        let edited = set
            .with_deleted(outside_key(), true, deletion_budget())
            .unwrap();
        let regenerated = generate(config(), region(), &field(true), &published, unrestricted());
        let next = filter_heightfield_scatter(regenerated, &edited).unwrap();
        assert_ne!(next.source.cache_key, retained.source.cache_key);
        assert_ne!(next.deletions, retained.deletions);
        assert_eq!(next.source.cache_key.surface.tile().samples(), &[1; 9]);
        assert_eq!(
            next.source.cache_key.surface.tile().revision(),
            key.surface.tile().revision() + 1
        );
        drop(old_surface);
        drop(set);
        assert_eq!(retained.source.cache_key, key);
        assert_eq!(retained.source.cache_key.surface.tile().samples(), &[0; 9]);
        assert_eq!(rows(&retained.source), expected);
        assert_eq!(retained.deletions.keys().len(), 1);
        assert_eq!(retained.removed, 1);
    }
}
