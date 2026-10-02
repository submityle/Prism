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
