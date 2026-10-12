//! Verification for the GPU-driven culling slice.
//!
//! The CPU-only tests pin the reference behaviour and the byte layouts. The
//! end-to-end test brings up a real device, culls on the GPU, and asserts the
//! GPU's surviving set and its rasterized coverage both match the CPU
//! reference. Without an adapter it returns early, so the suite still passes on
//! a GPU-less host (the base CI sandbox).

use prism_render_driver::RenderQueue;
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};

use crate::cull::{CullParams, CullStage};
use crate::draw::{DrawStage, CELL_PIXELS, INDICES_PER_INSTANCE};
use crate::encoding::{instance_storage_bytes, IndirectArgs, AABB_STRIDE};
use crate::frustum::{Aabb, Frustum};
use crate::readback::count_lit_pixels;
use crate::reference::cull_cpu;
use crate::scene;

/// One instance draws a four-index triangle strip (the quad), matching the
/// draw stage's geometry.
const INDEX_COUNT: u32 = INDICES_PER_INSTANCE;

/// The square offscreen target's side, in pixels. Sixty-four survivors, the
/// default lattice's count, each cover a 4x4-pixel quad in a distinct cell.
const RESOLUTION: u32 = 256;

/// Brings up a device/queue pair, or `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

/// When `PRISM_REQUIRE_GPU` is set, a missing adapter fails the test instead of
/// skipping it, so a run meant to exercise hardware cannot silently degrade
/// into a no-op.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// The default lattice must exercise both outcomes: some instances inside the
/// view volume and some outside, with a known survivor count.
#[test]
fn cpu_reference_culls_the_expected_lattice_subset() {
    let instances = scene::lattice_instances(
        scene::DEFAULT_GRID,
        scene::DEFAULT_SPACING,
        scene::DEFAULT_HALF_EXTENT,
    );
    let reference = cull_cpu(&scene::view_box_frustum(), &instances);

    assert_eq!(instances.len(), 1000, "a 10^3 lattice");
    assert_eq!(
        reference.len(),
        64,
        "the view box keeps the 4^3 interior lattice sites"
    );
    assert!(
        reference.windows(2).all(|pair| pair[0] < pair[1]),
        "the reference is in ascending id order"
    );
}

/// A box fully behind a plane is rejected; a box straddling one is kept, and
/// no lattice site lands near a plane, so the GPU and CPU cannot disagree on a
/// floating-point tie.
#[test]
fn cpu_reference_is_conservative_and_boundary_clear() {
    let frustum = Frustum::from_box([-0.5; 3], [0.5; 3]);

    assert!(!crate::frustum::aabb_inside(
        &frustum,
        &Aabb::from_center([0.6, 0.0, 0.0], [0.05; 3])
    ));
    assert!(crate::frustum::aabb_inside(
        &frustum,
        &Aabb::from_center([0.45, 0.0, 0.0], [0.05; 3])
    ));

    for aabb in scene::lattice_instances(
        scene::DEFAULT_GRID,
        scene::DEFAULT_SPACING,
        scene::DEFAULT_HALF_EXTENT,
    ) {
        let margin = crate::frustum::signed_distance(&frustum, &aabb);
        assert!(
            margin.abs() >= 0.04,
            "a lattice site sits {margin} from a plane, too close to a tie"
        );
    }
}

/// The instance storage upload is padded to the stride the shader reads.
#[test]
fn instance_storage_is_padded_to_the_shader_stride() {
    let instances = [Aabb::from_center([1.0, 2.0, 3.0], [0.5; 3])];
    let bytes = instance_storage_bytes(&instances);
    assert_eq!(bytes.len() as u64, AABB_STRIDE);
    assert_eq!(
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0.5
    );
    assert_eq!(
        f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        0.0
    );
    assert_eq!(
        f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        1.5
    );
}

/// The indirect record survives a byte round trip unchanged.
#[test]
fn indirect_args_round_trip_through_bytes() {
    let args = IndirectArgs::new(4, 64, 0, -3, 7);
    let bytes = args.to_bytes();
    assert_eq!(bytes.len(), IndirectArgs::BYTES as usize);
    assert_eq!(IndirectArgs::from_bytes(&bytes), args);
}

/// The end-to-end slice: compute cull on the GPU, compare against the CPU
/// reference, then rasterize through both indirect-draw primitives and check
/// the coverage.
#[test]
fn gpu_cull_matches_the_cpu_reference_and_drives_indirect_draws() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    let instances = scene::lattice_instances(
        scene::DEFAULT_GRID,
        scene::DEFAULT_SPACING,
        scene::DEFAULT_HALF_EXTENT,
    );
    let frustum = scene::view_box_frustum();
    let reference = cull_cpu(&frustum, &instances);
    assert!(
        !reference.is_empty() && reference.len() < instances.len(),
        "the scene must both keep and reject instances"
    );
    assert!(
        instances.len() <= (RESOLUTION / CELL_PIXELS).pow(2) as usize,
        "every instance id must fit a distinct draw cell"
    );

    // --- cull on the GPU ---
    let cull = CullStage::new(
        &device,
        instances.len() as u32,
        CullParams {
            index_count: INDEX_COUNT,
            first_index: 0,
            base_vertex: 0,
        },
    );
    cull.set_scene(&queue, &frustum, &instances);
    let output = cull.run(&queue);

    // The GPU-authored count is the surviving count the CPU never computed.
    assert_eq!(
        output.args.instance_count,
        reference.len() as u32,
        "GPU instance count equals the CPU reference count"
    );
    assert_eq!(output.args.index_count, INDEX_COUNT);
    assert_eq!(output.args.first_index, 0);
    assert_eq!(output.args.base_vertex, 0);
    assert_eq!(output.args.first_instance, 0);

    // The atomic append order is not specified, so the comparison is on the
    // set: sorting the GPU list must reproduce the ascending CPU reference
    // exactly, which proves the same count and the same survivors.
    assert_eq!(output.indices.len(), reference.len());
    let mut gpu_sorted = output.indices.clone();
    gpu_sorted.sort_unstable();
    assert_eq!(
        gpu_sorted, reference,
        "the GPU survivors equal the CPU reference as a set"
    );

    // Every survivor occupied a distinct slot, and each packed multi-draw
    // record names that slot through its own index block base.
    assert_eq!(output.multi_args.len(), reference.len());
    for (slot, record) in output.multi_args.iter().enumerate() {
        assert_eq!(
            record.first_index,
            slot as u32 * INDICES_PER_INSTANCE,
            "record {slot} points at its own index block"
        );
        assert_eq!(record.first_instance, 0);
        assert_eq!(record.instance_count, 1);
        assert_eq!(record.index_count, INDEX_COUNT);
    }

    // --- draw the GPU-authored result ---
    let draw = DrawStage::new(&device, RESOLUTION, cull.visible(), instances.len() as u32);
    draw.upload(&queue);
    let expected = DrawStage::expected_coverage(reference.len());

    queue.submit(&[draw.encode_indexed_indirect(cull.draw_args(), 0)]);
    assert!(
        device.poll_wait(),
        "the device stays healthy after the indirect draw"
    );
    let pixels = queue.read_texture_rgba8(draw.target(), RESOLUTION, RESOLUTION);
    assert_eq!(
        count_lit_pixels(&pixels),
        expected,
        "one indexed indirect draw covers one quad per GPU-counted survivor"
    );

    queue.submit(&[draw.encode_multi_indirect(
        cull.multi_args(),
        0,
        reference.len() as u32,
        IndirectArgs::BYTES as u32,
    )]);
    assert!(
        device.poll_wait(),
        "the device stays healthy after the multi-draw"
    );
    let pixels = queue.read_texture_rgba8(draw.target(), RESOLUTION, RESOLUTION);
    assert_eq!(
        count_lit_pixels(&pixels),
        expected,
        "the packed multi-draw covers the same survivors"
    );

    draw.destroy(&device);
    cull.destroy(&device);
}
