//! No-`GPU` validation for the surfel-allocation producer slice.
//!
//! Two complementary checks stand in for a real dispatch (the sandbox has no
//! `GPU`): `naga` compiles the `WESL` kernel, and the `CPU` mirror
//! [`allocate_slot`] is cross-checked bit-for-bit against the
//! [`SurfelAtlas`] golden over a direction/id sweep.

use bevy_math::Vec3;

use crate::gi::surface_cache::atlas::SurfelAtlas;
use crate::gi::surface_cache::gpu::abi::{
    GpuSurfelAllocParams, GpuSurfelAllocRequest, GpuSurfelAllocSlot, SURFEL_ALLOC_PARAMS_SIZE,
    SURFEL_ALLOC_REQUEST_STRIDE, SURFEL_ALLOC_SLOT_STRIDE, SURFEL_ALLOC_WORKGROUP_SIZE,
};
use crate::gi::surface_cache::gpu::alloc::allocate_slot;

/// The `WESL` source compiled and validated by [`alloc_wesl_compiles`].
const SURFEL_ALLOC_WESL: &str = include_str!("shaders/surfel_alloc.wesl");

/// `naga` parses and type-checks the kernel, proving it compiles exactly as it
/// will on device (the sandbox cannot dispatch it).
#[test]
fn alloc_wesl_compiles() {
    let module = naga::front::wgsl::parse_str(SURFEL_ALLOC_WESL)
        .unwrap_or_else(|error| panic!("surfel_alloc.wesl failed to parse: {error:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("surfel_alloc.wesl failed to validate: {error:?}"));
}

/// The `repr(C)` `ABI` strides match the constants the kernel and host
/// allocator assume (the layout itself is additionally pinned by the
/// `const` size assertions in `abi.rs`).
#[test]
fn abi_matches_shader_layout() {
    assert_eq!(size_of::<GpuSurfelAllocParams>(), SURFEL_ALLOC_PARAMS_SIZE);
    assert_eq!(
        size_of::<GpuSurfelAllocRequest>(),
        SURFEL_ALLOC_REQUEST_STRIDE
    );
    assert_eq!(size_of::<GpuSurfelAllocSlot>(), SURFEL_ALLOC_SLOT_STRIDE);
    assert_eq!(align_of::<GpuSurfelAllocParams>(), 4);
    assert_eq!(align_of::<GpuSurfelAllocRequest>(), 4);
    assert_eq!(align_of::<GpuSurfelAllocSlot>(), 4);
}

/// [`GpuSurfelAllocParams::workgroup_count`] ceil-divides the request count by
/// the workgroup size, covering every request with no empty trailing group.
#[test]
fn workgroup_count_covers_every_request() {
    let atlas = SurfelAtlas::new(4, 4, 8);
    let wg = SURFEL_ALLOC_WORKGROUP_SIZE;
    for count in [0, 1, wg - 1, wg, wg + 1, 3 * wg, 3 * wg + 7] {
        let params = GpuSurfelAllocParams::from_atlas(&atlas, count);
        let groups = params.workgroup_count();
        assert!(groups * wg >= count, "under-covered count {count}");
        if count > 0 {
            assert!((groups - 1) * wg < count, "over-covered count {count}");
        } else {
            assert_eq!(groups, 0, "empty dispatch for zero requests");
        }
    }
}

/// [`GpuSurfelAllocParams::from_atlas`] copies the atlas's already-clamped
/// dimensions verbatim.
#[test]
fn params_from_atlas_copies_clamped_dimensions() {
    let atlas = SurfelAtlas::new(0, 0, 0);
    let params = GpuSurfelAllocParams::from_atlas(&atlas, 5);
    assert_eq!(params.count, 5);
    assert_eq!(params.tiles_per_row, 1);
    assert_eq!(params.tile_rows, 1);
    assert_eq!(params.tile_resolution, 1);
}

/// A representative direction sweep: the axes, both poles, the diagonal fold
/// seams (`z = 0`), interior diagonals and the degenerate zero vector.
fn sweep_directions() -> [Vec3; 12] {
    [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
        Vec3::new(0.4, 0.5, 0.76).normalize(),
        Vec3::new(-0.3, 0.2, -0.93).normalize(),
        Vec3::new(0.7, 0.7, 0.0).normalize(),
        Vec3::new(-0.6, 0.0, -0.8).normalize(),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
    ]
}

/// The `CPU` mirror reproduces the [`SurfelAtlas`] golden bit-for-bit for every
/// in-range id across the direction sweep and several atlas shapes: the global
/// texel equals `dir_to_texel`, and the tile equals `tile_coord`.
#[test]
fn allocate_slot_matches_atlas_golden() {
    let atlases = [
        SurfelAtlas::new(1, 1, 8),
        SurfelAtlas::new(4, 3, 16),
        SurfelAtlas::new(2, 2, 64),
        SurfelAtlas::new(7, 5, 1),
    ];
    for atlas in atlases {
        let capacity = atlas.capacity();
        for id in 0..capacity {
            let tile = atlas.tile_coord(id).expect("in range");
            for dir in sweep_directions() {
                let request = GpuSurfelAllocRequest::new(id, dir.x, dir.y, dir.z);
                let slot = allocate_slot(&atlas, &request);
                let golden = atlas.dir_to_texel(id, dir).expect("in range");
                assert!(slot.valid(), "id {id} dir {dir:?} should be valid");
                assert_eq!(slot.texel_x, golden.x, "texel x drift id {id} dir {dir:?}");
                assert_eq!(slot.texel_y, golden.y, "texel y drift id {id} dir {dir:?}");
                assert_eq!(slot.tile_col, tile.col, "tile col drift id {id}");
                assert_eq!(slot.tile_row, tile.row, "tile row drift id {id}");
            }
        }
    }
}

/// Out-of-range ids yield the all-zero invalid slot, exactly where the golden
/// `dir_to_texel` returns `None`.
#[test]
fn allocate_slot_rejects_out_of_range_ids() {
    let atlas = SurfelAtlas::new(2, 2, 8);
    let capacity = atlas.capacity();
    for id in [capacity, capacity + 1, capacity + 99, u32::MAX] {
        let request = GpuSurfelAllocRequest::new(id, 0.0, 0.0, 1.0);
        let slot = allocate_slot(&atlas, &request);
        assert_eq!(
            slot,
            GpuSurfelAllocSlot::invalid(),
            "id {id} should be invalid"
        );
        assert!(!slot.valid());
        assert!(atlas.dir_to_texel(id, Vec3::Z).is_none());
    }
}

/// The resolved texel always lands inside the surfel's own tile, matching the
/// golden `tile_origin`.
#[test]
fn allocate_slot_texel_stays_inside_tile() {
    let atlas = SurfelAtlas::new(4, 4, 16);
    for id in 0..atlas.capacity() {
        let origin = atlas.tile_origin(id).expect("in range");
        for dir in sweep_directions() {
            let request = GpuSurfelAllocRequest::new(id, dir.x, dir.y, dir.z);
            let slot = allocate_slot(&atlas, &request);
            assert!(slot.texel_x >= origin.x && slot.texel_x < origin.x + atlas.tile_resolution);
            assert!(slot.texel_y >= origin.y && slot.texel_y < origin.y + atlas.tile_resolution);
        }
    }
}

/// The mirror is deterministic: identical inputs yield an identical slot.
#[test]
fn allocate_slot_is_deterministic() {
    let atlas = SurfelAtlas::new(4, 4, 16);
    let request = GpuSurfelAllocRequest::new(7, 0.2, 0.3, 0.9);
    assert_eq!(
        allocate_slot(&atlas, &request),
        allocate_slot(&atlas, &request)
    );
}

use crate::gi::surface_cache::gpu::abi::{
    GpuSurfelUpdateInput, GpuSurfelUpdateParams, GpuSurfelUpdateResult, SURFEL_UPDATE_INPUT_STRIDE,
    SURFEL_UPDATE_PARAMS_SIZE, SURFEL_UPDATE_RESULT_STRIDE, SURFEL_UPDATE_WORKGROUP_SIZE,
};
use crate::gi::surface_cache::gpu::update::update_entry;
use crate::gi::surface_cache::integration::{integrate_radiance, SurfelCacheEntry, TemporalParams};
use crate::gi::surface_cache::surfel::Surfel;

/// The `WESL` source compiled and validated by [`update_wesl_compiles`].
const SURFEL_UPDATE_WESL: &str = include_str!("shaders/surfel_update.wesl");

/// `naga` parses and type-checks the update kernel, proving it compiles exactly
/// as it will on device (the sandbox cannot dispatch it).
#[test]
fn update_wesl_compiles() {
    let module = naga::front::wgsl::parse_str(SURFEL_UPDATE_WESL)
        .unwrap_or_else(|error| panic!("surfel_update.wesl failed to parse: {error:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("surfel_update.wesl failed to validate: {error:?}"));
}

/// The update `repr(C)` `ABI` strides match the constants the kernel and host
/// assume (the layout itself is additionally pinned by the `const` size
/// assertions in `abi.rs`).
#[test]
fn update_abi_matches_shader_layout() {
    assert_eq!(
        size_of::<GpuSurfelUpdateParams>(),
        SURFEL_UPDATE_PARAMS_SIZE
    );
    assert_eq!(
        size_of::<GpuSurfelUpdateInput>(),
        SURFEL_UPDATE_INPUT_STRIDE
    );
    assert_eq!(
        size_of::<GpuSurfelUpdateResult>(),
        SURFEL_UPDATE_RESULT_STRIDE
    );
    assert_eq!(align_of::<GpuSurfelUpdateParams>(), 4);
    assert_eq!(align_of::<GpuSurfelUpdateInput>(), 4);
    assert_eq!(align_of::<GpuSurfelUpdateResult>(), 4);
}

/// [`GpuSurfelUpdateParams::workgroup_count`] ceil-divides the input count by
/// the workgroup size, covering every input with no empty trailing group.
#[test]
fn update_workgroup_count_covers_every_input() {
    let params = TemporalParams::default();
    let wg = SURFEL_UPDATE_WORKGROUP_SIZE;
    for count in [0, 1, wg - 1, wg, wg + 1, 5 * wg, 5 * wg + 3] {
        let p = GpuSurfelUpdateParams::from_temporal(&params, count);
        let groups = p.workgroup_count();
        assert!(groups * wg >= count, "under-covered count {count}");
        if count > 0 {
            assert!((groups - 1) * wg < count, "over-covered count {count}");
        } else {
            assert_eq!(groups, 0, "empty dispatch for zero inputs");
        }
    }
}

/// [`GpuSurfelUpdateParams::from_temporal`] copies the temporal tunables
/// verbatim.
#[test]
fn update_params_from_temporal_copies_fields() {
    let params = TemporalParams {
        max_samples: 24,
        position_tolerance: 0.75,
        normal_tolerance: 0.25,
    };
    let p = GpuSurfelUpdateParams::from_temporal(&params, 9);
    assert_eq!(p.count, 9);
    assert_eq!(p.max_samples, 24);
    assert!((p.position_tolerance - 0.75).abs() < 1e-7);
    assert!((p.normal_tolerance - 0.25).abs() < 1e-7);
}

/// Run both the `CPU` mirror and the host golden on one `(prev, curr, sample)`
/// state and assert the advanced entry agrees bit-for-bit.
fn assert_update_parity(
    prev: Option<(SurfelCacheEntry, Surfel)>,
    curr: &Surfel,
    new_radiance: Vec3,
    params: &TemporalParams,
) -> GpuSurfelUpdateResult {
    let golden = integrate_radiance(prev, curr, new_radiance, params);
    let gpu_params = GpuSurfelUpdateParams::from_temporal(params, 1);
    let input = GpuSurfelUpdateInput::from_states(
        prev.as_ref().map(|(entry, surfel)| (entry, surfel)),
        curr,
        new_radiance,
    );
    let mirror = update_entry(&gpu_params, &input);
    assert_eq!(
        mirror.radiance(),
        golden.radiance,
        "radiance drift (prev {prev:?}, sample {new_radiance:?})"
    );
    assert_eq!(
        mirror.sample_count, golden.sample_count,
        "count drift (prev {prev:?}, sample {new_radiance:?})"
    );
    mirror
}

/// The `CPU` mirror reproduces the `integrate_radiance` golden bit-for-bit
/// across first-sight, steady blend, both disocclusion resets, confidence
/// saturation and `NaN` sanitisation.
#[test]
fn update_entry_matches_integration_golden() {
    let params = TemporalParams::default();
    let base = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);

    // First sight: no history, reseed from the (sanitised) sample.
    let seeded = assert_update_parity(None, &base, Vec3::new(0.2, 0.4, 0.8), &params);
    assert_eq!(seeded.sample_count, 1);

    // Steady blend: same surface, history present.
    let entry = SurfelCacheEntry {
        radiance: Vec3::splat(4.0),
        sample_count: 7,
    };
    let blended = assert_update_parity(Some((entry, base)), &base, Vec3::splat(1.0), &params);
    assert_eq!(blended.sample_count, 8);
    assert!(blended.radiance().x < 4.0 && blended.radiance().x > 1.0);

    // Disocclusion by a large anchor jump: reset to the sample.
    let jumped = Surfel::new(Vec3::new(3.0, 0.0, 0.0), Vec3::Z, 1.0);
    let reset_pos = assert_update_parity(Some((entry, base)), &jumped, Vec3::splat(2.0), &params);
    assert_eq!(reset_pos.sample_count, 1);

    // Disocclusion by a flipped normal: reset to the sample.
    let flipped = Surfel::new(Vec3::ZERO, Vec3::NEG_Z, 1.0);
    let reset_normal =
        assert_update_parity(Some((entry, base)), &flipped, Vec3::splat(0.5), &params);
    assert_eq!(reset_normal.sample_count, 1);

    // Confidence saturation: a prev_count at the cap stays capped.
    let saturated_entry = SurfelCacheEntry {
        radiance: Vec3::splat(3.0),
        sample_count: params.max_samples,
    };
    let saturated = assert_update_parity(
        Some((saturated_entry, base)),
        &base,
        Vec3::splat(1.0),
        &params,
    );
    assert_eq!(saturated.sample_count, params.max_samples);

    // NaN / negative sample: sanitised finite and non-negative.
    let nan = assert_update_parity(
        Some((entry, base)),
        &base,
        Vec3::new(f32::NAN, -5.0, f32::INFINITY),
        &params,
    );
    assert!(nan.radiance().is_finite());
    assert!(nan.radiance().x >= 0.0 && nan.radiance().y >= 0.0 && nan.radiance().z >= 0.0);
}

/// A multi-frame accumulation sweep stays bit-for-bit with the golden as the
/// `EMA` converges and the confidence saturates.
#[test]
fn update_entry_converges_with_golden() {
    let params = TemporalParams::default();
    let surfel = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    let target = Vec3::new(0.3, 0.9, 1.5);
    let mut entry = SurfelCacheEntry::from_sample(Vec3::ZERO);
    for _ in 0..128 {
        let mirror = assert_update_parity(Some((entry, surfel)), &surfel, target, &params);
        // Advance the golden in lock-step for the next frame's history.
        entry = integrate_radiance(Some((entry, surfel)), &surfel, target, &params);
        assert_eq!(mirror.radiance(), entry.radiance);
        assert_eq!(mirror.sample_count, entry.sample_count);
    }
    assert_eq!(entry.sample_count, params.max_samples);
    assert!(
        (entry.radiance - target).length() < 1e-2,
        "{:?}",
        entry.radiance
    );
}

/// The mirror is deterministic: identical inputs yield an identical result.
#[test]
fn update_entry_is_deterministic() {
    let params = GpuSurfelUpdateParams::from_temporal(&TemporalParams::default(), 1);
    let surfel = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    let entry = SurfelCacheEntry {
        radiance: Vec3::splat(0.5),
        sample_count: 4,
    };
    let input =
        GpuSurfelUpdateInput::from_states(Some((&entry, &surfel)), &surfel, Vec3::splat(1.0));
    assert_eq!(update_entry(&params, &input), update_entry(&params, &input));
}

use crate::gi::surface_cache::gpu::abi::{
    GpuSpatialCenter, GpuSpatialFilterParams, GpuSpatialNeighbor, GpuSpatialResult,
    SURFEL_SPATIAL_CENTER_STRIDE, SURFEL_SPATIAL_NEIGHBOR_STRIDE, SURFEL_SPATIAL_PARAMS_SIZE,
    SURFEL_SPATIAL_RESULT_STRIDE, SURFEL_SPATIAL_WORKGROUP_SIZE,
};
use crate::gi::surface_cache::gpu::filter::filter_center;
use crate::gi::surface_cache::integration::spatial_filter;
use crate::gi::surface_cache::surfel::CoverageParams;

/// The `WESL` source compiled and validated by [`spatial_filter_wesl_compiles`].
const SURFEL_SPATIAL_WESL: &str = include_str!("shaders/surfel_spatial_filter.wesl");

/// `naga` parses and type-checks the spatial-filter kernel, proving it compiles
/// exactly as it will on device (the sandbox cannot dispatch it).
#[test]
fn spatial_filter_wesl_compiles() {
    let module = naga::front::wgsl::parse_str(SURFEL_SPATIAL_WESL)
        .unwrap_or_else(|error| panic!("surfel_spatial_filter.wesl failed to parse: {error:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("surfel_spatial_filter.wesl failed to validate: {error:?}"));
}

/// The spatial-filter `repr(C)` `ABI` strides match the constants the kernel and
/// host assume (the layout itself is additionally pinned by the `const` size
/// assertions in `abi.rs`).
#[test]
fn spatial_abi_matches_shader_layout() {
    assert_eq!(
        size_of::<GpuSpatialFilterParams>(),
        SURFEL_SPATIAL_PARAMS_SIZE
    );
    assert_eq!(size_of::<GpuSpatialCenter>(), SURFEL_SPATIAL_CENTER_STRIDE);
    assert_eq!(
        size_of::<GpuSpatialNeighbor>(),
        SURFEL_SPATIAL_NEIGHBOR_STRIDE
    );
    assert_eq!(size_of::<GpuSpatialResult>(), SURFEL_SPATIAL_RESULT_STRIDE);
    assert_eq!(align_of::<GpuSpatialFilterParams>(), 4);
    assert_eq!(align_of::<GpuSpatialCenter>(), 4);
    assert_eq!(align_of::<GpuSpatialNeighbor>(), 4);
    assert_eq!(align_of::<GpuSpatialResult>(), 4);
}

/// [`GpuSpatialFilterParams::workgroup_count`] ceil-divides the centre count by
/// the workgroup size, covering every centre with no empty trailing group.
#[test]
fn spatial_workgroup_count_covers_every_center() {
    let params = CoverageParams::default();
    let wg = SURFEL_SPATIAL_WORKGROUP_SIZE;
    for count in [0, 1, wg - 1, wg, wg + 1, 4 * wg, 4 * wg + 5] {
        let p = GpuSpatialFilterParams::from_coverage(&params, count);
        let groups = p.workgroup_count();
        assert!(groups * wg >= count, "under-covered count {count}");
        if count > 0 {
            assert!((groups - 1) * wg < count, "over-covered count {count}");
        } else {
            assert_eq!(groups, 0, "empty dispatch for zero centres");
        }
    }
}

/// [`GpuSpatialFilterParams::from_coverage`] copies the coverage tunables
/// verbatim and zeroes the reserved padding.
#[test]
fn spatial_params_from_coverage_copies_fields() {
    let params = CoverageParams {
        normal_sharpness: 12.0,
        axial_tolerance: 0.4,
    };
    let p = GpuSpatialFilterParams::from_coverage(&params, 11);
    assert_eq!(p.count, 11);
    assert!((p.normal_sharpness - 12.0).abs() < 1e-7);
    assert!((p.axial_tolerance - 0.4).abs() < 1e-7);
    assert_eq!(p.reserved, 0);
}

/// Run both the `CPU` mirror and the host golden on one centre and its
/// neighbour list, asserting the filtered radiance agrees bit-for-bit.
fn assert_filter_parity(
    center: &Surfel,
    center_radiance: Vec3,
    neighbours: &[(Surfel, Vec3)],
    params: &CoverageParams,
) -> GpuSpatialResult {
    let golden = spatial_filter(center, center_radiance, neighbours, params);

    let gpu_params = GpuSpatialFilterParams::from_coverage(params, 1);
    let gpu_center = GpuSpatialCenter::new(center, center_radiance, 0, neighbours.len() as u32);
    let gpu_neighbours: Vec<GpuSpatialNeighbor> = neighbours
        .iter()
        .map(|(surfel, radiance)| GpuSpatialNeighbor::new(surfel, *radiance))
        .collect();
    let mirror = filter_center(&gpu_params, &gpu_center, &gpu_neighbours);
    assert_eq!(
        mirror.radiance(),
        golden,
        "radiance drift (centre {center_radiance:?}, neighbours {neighbours:?})"
    );
    mirror
}

/// The `CPU` mirror reproduces the `spatial_filter` golden bit-for-bit across a
/// constant signal, the empty / incompatible identity cases, a genuine pull
/// toward a compatible neighbour, and `NaN` sanitisation.
#[test]
fn filter_center_matches_spatial_golden() {
    let params = CoverageParams::default();
    let center = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);

    // Locally constant signal: reproduced exactly, weight accumulates.
    let signal = Vec3::new(0.3, 0.6, 0.9);
    let constant = [
        (Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0), signal),
        (Surfel::new(Vec3::new(0.0, 0.2, 0.0), Vec3::Z, 1.0), signal),
    ];
    let out = assert_filter_parity(&center, signal, &constant, &params);
    assert!(out.total_weight > 1.0);

    // No neighbours: identity, unit weight.
    let empty = assert_filter_parity(&center, signal, &[], &params);
    assert!((empty.total_weight - 1.0).abs() < 1e-7);

    // Incompatible neighbour (far off the disc): dropped, weight stays unit.
    let far = [(
        Surfel::new(Vec3::new(50.0, 0.0, 0.0), Vec3::Z, 1.0),
        Vec3::splat(9.0),
    )];
    let dropped = assert_filter_parity(&center, signal, &far, &params);
    assert!((dropped.total_weight - 1.0).abs() < 1e-7);

    // Back-facing neighbour: normal agreement kills the weight.
    let flipped = [(
        Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::NEG_Z, 1.0),
        Vec3::splat(9.0),
    )];
    let back = assert_filter_parity(&center, signal, &flipped, &params);
    assert!((back.total_weight - 1.0).abs() < 1e-7);

    // Genuine pull: a compatible brighter neighbour raises the centre value.
    let pull = [(
        Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0),
        Vec3::splat(4.0),
    )];
    let pulled = assert_filter_parity(&center, Vec3::ZERO, &pull, &params);
    assert!(pulled.radiance().x > 0.0);
    assert!(pulled.total_weight > 1.0);

    // NaN / negative radiance: sanitised finite and non-negative.
    let nan = [(
        Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0),
        Vec3::new(f32::NAN, -3.0, f32::INFINITY),
    )];
    let cleaned = assert_filter_parity(&center, Vec3::splat(1.0), &nan, &params);
    assert!(cleaned.radiance().is_finite());
    assert!(
        cleaned.radiance().x >= 0.0 && cleaned.radiance().y >= 0.0 && cleaned.radiance().z >= 0.0
    );
}

/// A longer neighbour slice (mixing kept and dropped neighbours) stays
/// bit-for-bit with the golden, exercising the in-order running sum.
#[test]
fn filter_center_matches_golden_over_mixed_slice() {
    let params = CoverageParams::default();
    let center = Surfel::new(Vec3::ZERO, Vec3::Z, 2.0);
    let neighbours = [
        (
            Surfel::new(Vec3::new(0.3, 0.0, 0.0), Vec3::Z, 1.0),
            Vec3::new(1.0, 0.5, 0.25),
        ),
        (
            Surfel::new(Vec3::new(99.0, 0.0, 0.0), Vec3::Z, 1.0),
            Vec3::splat(5.0),
        ),
        (
            Surfel::new(Vec3::new(0.0, 0.4, 0.1), Vec3::new(0.1, 0.1, 0.98), 1.0),
            Vec3::new(0.2, 0.8, 1.3),
        ),
        (
            Surfel::new(Vec3::new(0.0, 0.0, 0.0), Vec3::NEG_Z, 1.0),
            Vec3::splat(7.0),
        ),
    ];
    assert_filter_parity(&center, Vec3::new(0.1, 0.2, 0.3), &neighbours, &params);
}

/// The mirror is deterministic: identical inputs yield an identical result.
#[test]
fn filter_center_is_deterministic() {
    let params = GpuSpatialFilterParams::from_coverage(&CoverageParams::default(), 1);
    let center = GpuSpatialCenter::new(
        &Surfel::new(Vec3::ZERO, Vec3::Z, 1.0),
        Vec3::splat(0.5),
        0,
        1,
    );
    let neighbours = [GpuSpatialNeighbor::new(
        &Surfel::new(Vec3::new(0.2, 0.0, 0.0), Vec3::Z, 1.0),
        Vec3::splat(1.0),
    )];
    assert_eq!(
        filter_center(&params, &center, &neighbours),
        filter_center(&params, &center, &neighbours)
    );
}

use crate::gi::surface_cache::atlas::AtlasTexel;
use crate::gi::surface_cache::gpu::abi::{
    GpuSurfelDecodeParams, GpuSurfelDecodeRequest, GpuSurfelDecodeResult,
    SURFEL_DECODE_PARAMS_SIZE, SURFEL_DECODE_REQUEST_STRIDE, SURFEL_DECODE_RESULT_STRIDE,
    SURFEL_DECODE_WORKGROUP_SIZE,
};
use crate::gi::surface_cache::gpu::decode::decode_slot;

/// The `WESL` source compiled and validated by [`decode_wesl_compiles`].
const SURFEL_DECODE_WESL: &str = include_str!("shaders/surfel_decode.wesl");

/// `naga` parses and type-checks the decode kernel, proving it compiles exactly
/// as it will on device (the sandbox cannot dispatch it).
#[test]
fn decode_wesl_compiles() {
    let module = naga::front::wgsl::parse_str(SURFEL_DECODE_WESL)
        .unwrap_or_else(|error| panic!("surfel_decode.wesl failed to parse: {error:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("surfel_decode.wesl failed to validate: {error:?}"));
}

/// The decode `repr(C)` `ABI` strides match the constants the kernel and host
/// assume (the layout itself is additionally pinned by the `const` size
/// assertions in `abi.rs`).
#[test]
fn decode_abi_matches_shader_layout() {
    assert_eq!(
        size_of::<GpuSurfelDecodeParams>(),
        SURFEL_DECODE_PARAMS_SIZE
    );
    assert_eq!(
        size_of::<GpuSurfelDecodeRequest>(),
        SURFEL_DECODE_REQUEST_STRIDE
    );
    assert_eq!(
        size_of::<GpuSurfelDecodeResult>(),
        SURFEL_DECODE_RESULT_STRIDE
    );
    assert_eq!(align_of::<GpuSurfelDecodeParams>(), 4);
    assert_eq!(align_of::<GpuSurfelDecodeRequest>(), 4);
    assert_eq!(align_of::<GpuSurfelDecodeResult>(), 4);
}

/// [`GpuSurfelDecodeParams::workgroup_count`] ceil-divides the request count by
/// the workgroup size, covering every request with no empty trailing group.
#[test]
fn decode_workgroup_count_covers_every_request() {
    let atlas = SurfelAtlas::new(4, 4, 8);
    let wg = SURFEL_DECODE_WORKGROUP_SIZE;
    for count in [0, 1, wg - 1, wg, wg + 1, 3 * wg, 3 * wg + 7] {
        let params = GpuSurfelDecodeParams::from_atlas(&atlas, count);
        let groups = params.workgroup_count();
        assert!(groups * wg >= count, "under-covered count {count}");
        if count > 0 {
            assert!((groups - 1) * wg < count, "over-covered count {count}");
        } else {
            assert_eq!(groups, 0, "empty dispatch for zero requests");
        }
    }
}

/// [`GpuSurfelDecodeParams::from_atlas`] copies the atlas's already-clamped
/// dimensions verbatim.
#[test]
fn decode_params_from_atlas_copies_clamped_dimensions() {
    let atlas = SurfelAtlas::new(0, 0, 0);
    let params = GpuSurfelDecodeParams::from_atlas(&atlas, 5);
    assert_eq!(params.count, 5);
    assert_eq!(params.tiles_per_row, 1);
    assert_eq!(params.tile_rows, 1);
    assert_eq!(params.tile_resolution, 1);
}

/// The `CPU` mirror reproduces the [`SurfelAtlas::texel_to_dir`] golden
/// bit-for-bit for every in-tile texel across several atlas shapes: a valid
/// decode matches the golden direction exactly, and the decode is the exact
/// inverse of the encode (`dir_to_texel` of the decoded direction returns the
/// same texel).
#[test]
fn decode_slot_matches_atlas_golden() {
    let atlases = [
        SurfelAtlas::new(1, 1, 8),
        SurfelAtlas::new(4, 3, 16),
        SurfelAtlas::new(2, 2, 32),
    ];
    for atlas in atlases {
        for id in 0..atlas.capacity() {
            let origin = atlas.tile_origin(id).expect("in range");
            for ly in 0..atlas.tile_resolution {
                for lx in 0..atlas.tile_resolution {
                    let texel = AtlasTexel {
                        x: origin.x + lx,
                        y: origin.y + ly,
                    };
                    let request = GpuSurfelDecodeRequest::new(id, texel.x, texel.y);
                    let slot = decode_slot(&atlas, &request);
                    let golden = atlas.texel_to_dir(id, texel).expect("in-tile texel");
                    assert!(slot.valid(), "id {id} texel {texel:?} should decode");
                    assert_eq!(slot.direction(), golden, "id {id} texel {texel:?}");
                    // Encode of the decoded direction returns the same texel:
                    // the decode truly inverts the atlas addressing.
                    assert_eq!(atlas.dir_to_texel(id, slot.direction()), Some(texel));
                }
            }
        }
    }
}

/// Out-of-range ids and texels that fall outside the surfel's own tile decode
/// to the all-zero invalid result, matching the golden `None`.
#[test]
fn decode_slot_rejects_out_of_range_and_foreign_texels() {
    let atlas = SurfelAtlas::new(3, 3, 8);

    // Out-of-range id: invalid, golden `None`.
    for id in [atlas.capacity(), atlas.capacity() + 5, u32::MAX] {
        let request = GpuSurfelDecodeRequest::new(id, 0, 0);
        let slot = decode_slot(&atlas, &request);
        assert_eq!(slot, GpuSurfelDecodeResult::invalid(), "id {id}");
        assert!(!slot.valid());
        assert!(atlas.texel_to_dir(id, AtlasTexel { x: 0, y: 0 }).is_none());
    }

    // A texel from a neighbouring tile is outside this surfel's tile: invalid.
    let id = 0;
    let foreign = AtlasTexel {
        x: atlas.tile_resolution,
        y: 0,
    };
    let request = GpuSurfelDecodeRequest::new(id, foreign.x, foreign.y);
    let slot = decode_slot(&atlas, &request);
    assert!(!slot.valid());
    assert_eq!(slot, GpuSurfelDecodeResult::invalid());
    assert!(atlas.texel_to_dir(id, foreign).is_none());
}

/// The mirror is deterministic: identical inputs yield an identical result.
#[test]
fn decode_slot_is_deterministic() {
    let atlas = SurfelAtlas::new(4, 4, 16);
    let origin = atlas.tile_origin(7).expect("in range");
    let request = GpuSurfelDecodeRequest::new(7, origin.x + 3, origin.y + 5);
    assert_eq!(decode_slot(&atlas, &request), decode_slot(&atlas, &request));
}
