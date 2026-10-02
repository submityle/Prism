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
