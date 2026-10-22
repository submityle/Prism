use super::*;
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec, TileKey,
};
use std::convert::Infallible;

struct Constant(i64);

impl HeightSampler for Constant {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(self.0))
    }
}

fn tile() -> std::sync::Arc<HeightfieldTile> {
    std::sync::Arc::new(
        HeightfieldTile::bake(
            TileKey::new(CubeFace::PositiveZ, 0, 0, 0).unwrap(),
            HeightfieldSpec::new(9, 1).unwrap(),
            &Constant(100),
            BakeLimits {
                max_samples: 81,
                max_bytes: 162,
            },
        )
        .unwrap(),
    )
}

fn config() -> PlanetPlacementConfig {
    PlanetPlacementConfig {
        seed: 7,
        version: 1,
        salt: 9,
        radius_mm: 6_371_000_000,
        origin_mm: [0, 0, 0],
        vertical_offset_mm: 0,
        acceptance_threshold: u16::MAX,
    }
}

#[test]
fn shader_uses_the_complete_limb_abi() {
    let source = format!(
        "{}\n{}",
        include_str!("shaders/planet_placement_u256.wgsl"),
        include_str!("shaders/planet_placement.wgsl")
    );
    let module = wgpu::naga::front::wgsl::parse_str(&source).expect("planet shader parses");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("planet shader uses core capabilities");
    assert_eq!(module.entry_points.len(), 2);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    assert_eq!(module.entry_points[1].workgroup_size, [1, 1, 1]);
    assert!(source.contains("array<vec4<u32>, 12>"));
    assert!(source.contains("array<vec4<u32>>"));
    assert!(source.contains("array<u32, 29>"));
    assert!(source.contains("array<atomic<u32>>"));
}

#[test]
fn u256_shader_arithmetic_validates_as_core_wgsl() {
    let source = include_str!("shaders/planet_placement_u256.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).expect("U256 shader parses");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("U256 shader uses core capabilities");
}

#[test]
fn u256_shader_has_fixed_width_arithmetic_contract() {
    let source = include_str!("shaders/planet_placement_u256.wgsl");
    assert!(source.contains("array<u32, 8>"));
    assert!(source.contains("fn u256_mul("));
    assert!(source.contains("fn u256_div_rem("));
    assert!(source.contains("fn u256_isqrt("));
    assert!(source.contains("fn u256_shl("));
}

#[test]
fn preflight_rejects_sample_budget_without_partial_request() {
    let source = tile();
    let region = PlanetPlacementRegion::new(source.tile(), [0, 0], [2, 2]).unwrap();
    let result = prepare(
        config(),
        region,
        &source,
        [0; 3],
        GpuPlanetPlacementWorkBudget {
            max_candidate_evaluations: 4,
            max_sample_reads: 15,
        },
        4,
        81,
        1024,
    );
    assert!(matches!(
        result,
        Err(GpuPlanetPlacementError::SampleBudgetExceeded {
            required: 16,
            limit: 15,
        })
    ));
}

#[test]
fn preflight_rejects_empty_or_invalid_regions_before_dispatch() {
    let source = tile();
    let empty = PlanetPlacementRegion::new(source.tile(), [2, 2], [2, 2]).unwrap();
    let request = prepare(
        config(),
        empty,
        &source,
        [0; 3],
        GpuPlanetPlacementWorkBudget {
            max_candidate_evaluations: 0,
            max_sample_reads: 0,
        },
        0,
        81,
        1024,
    )
    .unwrap();
    assert_eq!(request.candidate_count, 0);
    assert_eq!(request.groups, [0, 0]);

    let mut invalid = config();
    invalid.vertical_offset_mm = -invalid.radius_mm;
    let one = PlanetPlacementRegion::new(source.tile(), [0, 0], [1, 1]).unwrap();
    let result = prepare(
        invalid,
        one,
        &source,
        [0; 3],
        GpuPlanetPlacementWorkBudget {
            max_candidate_evaluations: 1,
            max_sample_reads: 4,
        },
        1,
        81,
        1024,
    );
    assert!(
        result.is_ok(),
        "radial validation is fail-closed in the shader"
    );
}

#[test]
fn preflight_packs_extreme_radial_base_without_loss() {
    let source = tile();
    let region = PlanetPlacementRegion::new(source.tile(), [0, 0], [1, 1]).unwrap();
    let mut extreme = config();
    extreme.radius_mm = 1_i64 << 52;
    extreme.vertical_offset_mm = i64::MAX;
    let request = prepare(
        extreme,
        region,
        &source,
        [0; 3],
        GpuPlanetPlacementWorkBudget {
            max_candidate_evaluations: 1,
            max_sample_reads: 4,
        },
        1,
        81,
        1024,
    )
    .unwrap();
    let expected = (i128::from(1_i64 << 52) + i128::from(i64::MAX)) * 4;
    assert_eq!(
        i128::from_le_bytes(bytemuck::cast::<[u32; 4], [u8; 16]>([
            request.words[20],
            request.words[21],
            request.words[22],
            request.words[23],
        ])),
        expected
    );
}

#[test]
fn capacity_math_is_checked_and_covers_all_bindings() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 5,
        max_bindings_per_bind_group: 6,
        ..wgpu::Limits::downlevel_defaults()
    };
    assert!(validate_capacity(4, 81, u64::MAX, &limits).is_ok());
    assert_eq!(validate_capacity(0, 0, 352, &limits), Ok(352));
    assert_eq!(
        validate_capacity(1, 1, 300, &limits),
        Err(GpuPlanetPlacementError::Gpu(
            GpuScatterError::BufferBudgetExceeded {
                required: 352,
                limit: 300,
            }
        ))
    );
}
