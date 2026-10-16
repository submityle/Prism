use super::*;
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetPlacementKey, TileKey,
};
use std::convert::Infallible;

struct Constant(i64);

impl HeightSampler for Constant {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(self.0))
    }
}

fn tile() -> std::sync::Arc<prism_world_terrain::HeightfieldTile> {
    std::sync::Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveZ),
            HeightfieldSpec::new(4, 1).unwrap(),
            &Constant(0),
            BakeLimits {
                max_samples: 16,
                max_bytes: 32,
            },
        )
        .unwrap(),
    )
}

fn config() -> PlanetPlacementConfig {
    PlanetPlacementConfig {
        seed: 7,
        version: 3,
        salt: 11,
        radius_mm: 6_371_000_000,
        origin_mm: [0; 3],
        vertical_offset_mm: 0,
        acceptance_threshold: u16::MAX,
    }
}

fn key(cell: [u32; 2]) -> PlanetPlacementKey {
    PlanetPlacementKey {
        placement_version: PLANET_PLACEMENT_VERSION,
        seed: 7,
        version: 3,
        salt: 11,
        tile: TileKey::root(CubeFace::PositiveZ),
        cell,
    }
}

fn deletion_set(keys: Vec<PlanetPlacementKey>) -> PlanetPlacementDeletionSet {
    PlanetPlacementDeletionSet::new(
        keys,
        prism_world_terrain::PlanetPlacementDeletionBudget {
            max_keys: usize::MAX,
            max_key_bytes: usize::MAX,
        },
    )
    .unwrap()
}

#[test]
fn shader_uses_complete_planetary_key_abi() {
    let source = include_str!("shaders/planet_placement_deletion.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).expect("planet deletion shader parses");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("planet deletion shader uses core capabilities");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    assert!(source.contains("array<u32, 12>"));
    assert!(source.contains("48 bytes"));
}

#[test]
fn key_packing_preserves_all_identity_fields_and_order() {
    let low = key([0, 9]);
    let high = key([1, 0]);
    let low_words = key_words(&low);
    let high_words = key_words(&high);
    assert_eq!(low_words[0], PLANET_PLACEMENT_VERSION);
    assert_eq!(low_words[1..6], [7, 0, 3, 11, 0]);
    assert_eq!(low_words[6..10], [4, 0, 0, 0]);
    assert_eq!(low_words[10..12], [0, 9]);
    assert!(low_words < high_words);
}

#[test]
fn preflight_rejects_budget_without_mutating_existing_request_shape() {
    let source = tile();
    let region = PlanetPlacementRegion::new(source.tile(), [0, 0], [2, 2]).unwrap();
    let deletions = deletion_set(vec![key([0, 0])]);
    let placement_budget = GpuPlanetPlacementWorkBudget {
        max_candidate_evaluations: 4,
        max_sample_reads: 16,
    };
    let error = validate_work(
        4,
        deletions.keys().len(),
        1,
        GpuPlanetPlacementDeletionWorkBudget {
            max_input_slots: 3,
            max_keys: 1,
            max_key_comparisons: 4,
        },
        1024,
    )
    .unwrap_err();
    assert_eq!(
        error,
        GpuPlanetPlacementDeletionError::InputBudgetExceeded {
            required: 4,
            limit: 3,
        }
    );
    let _ = (config(), region, placement_budget);
}

#[test]
fn comparison_budget_is_checked_with_binary_search_bound() {
    let budget = GpuPlanetPlacementDeletionWorkBudget {
        max_input_slots: 8,
        max_keys: 3,
        max_key_comparisons: 16,
    };
    assert_eq!(validate_work(8, 3, 3, budget, 1024).unwrap(), [1, 1]);
    let error = validate_work(
        8,
        3,
        3,
        GpuPlanetPlacementDeletionWorkBudget {
            max_key_comparisons: 7,
            ..budget
        },
        1024,
    )
    .unwrap_err();
    assert_eq!(
        error,
        GpuPlanetPlacementDeletionError::ComparisonBudgetExceeded {
            required: 16,
            limit: 7,
        }
    );
}

#[test]
fn capacity_math_accounts_for_upstream_and_deletion_buffers() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        max_bindings_per_bind_group: 7,
        ..wgpu::Limits::downlevel_defaults()
    };
    assert_eq!(
        validate_capacity(
            0,
            100,
            GpuPlanetPlacementDeletionCapacity {
                max_keys: 0,
                max_buffer_bytes: 100 + 48 + 4 + 64 + 16,
                max_scratch_bytes: 0,
            },
            &limits
        ),
        Ok((48 + 4 + 64 + 16, 100 + 48 + 4 + 64 + 16))
    );
    assert!(matches!(
        validate_capacity(
            2,
            100,
            GpuPlanetPlacementDeletionCapacity {
                max_keys: 2,
                max_buffer_bytes: 100,
                max_scratch_bytes: 96,
            },
            &limits
        ),
        Err(GpuPlanetPlacementDeletionError::Placement(
            GpuPlanetPlacementError::Gpu(crate::GpuScatterError::BufferBudgetExceeded { .. })
        ))
    ));
}
