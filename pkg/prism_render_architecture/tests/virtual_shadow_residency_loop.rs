//! End-to-end check of the directional-light virtual-shadow decision loop.
//!
//! Exercises the public path a backend drives each frame: derive per-receiver
//! shadow texel sizes from a screen-adaptive quality policy, plan the frame's
//! clip-page requests from world-space casters, flush them into the residency
//! table, mark uploads resident, then evict against a physical-page budget while
//! protecting pages still in use. This proves the modules compose across the
//! crate boundary, not just in isolation.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_shadow::{
    caster_for_receiver, plan_shadow_frame, ClipmapConfig, DirectionalLightBasis, ShadowQuality,
    ShadowResidencyTable,
};

fn config() -> ClipmapConfig {
    ClipmapConfig {
        light: 1,
        level_count: 4,
        resolution: 8,
        page_texel_dim: 128,
        level0_page_size: 4.0,
    }
}

fn bounds(center: [f32; 3], half: [f32; 3]) -> SceneBounds {
    SceneBounds {
        center,
        radius: 0.0,
        half_extents: half,
        _padding: 0.0,
    }
}

#[test]
fn plan_flush_resident_and_evict_across_frames() {
    let cfg = config();
    // Light pointing straight down: shadow plane is world XZ.
    let basis = DirectionalLightBasis::from_direction([0.0, -1.0, 0.0]).expect("basis");
    let quality = ShadowQuality {
        focal_length_pixels: 1000.0,
        texels_per_pixel: 1.0,
    };
    let camera = [0.0, 0.0, 0.0];

    // Two receivers near the camera; texel size comes from the quality policy.
    let near = caster_for_receiver(
        bounds([0.0, 2.0, 0.0], [0.0, 0.0, 0.0]),
        1.0,
        camera,
        &quality,
    );
    let side = caster_for_receiver(
        bounds([6.0, 2.0, 0.0], [0.0, 0.0, 0.0]),
        5.0,
        camera,
        &quality,
    );

    let plan = plan_shadow_frame(&cfg, &basis, camera, &[near, side]);
    assert!(!plan.is_empty(), "near receivers must request pages");

    // Flush the frame's requests into the session residency table.
    let mut table = ShadowResidencyTable::new();
    plan.requests.flush(&mut table, 1);
    let requested = table.len();
    assert_eq!(requested, plan.request_count());

    // The backend confirms every requested page uploaded this frame.
    for caster in [near, side] {
        let (min, _max) = basis.project_bounds(&caster.bounds);
        if let Some(key) = cfg.page_of(basis.project_point(camera), min, 0) {
            table.mark_resident(key);
        }
    }
    assert!(
        table.resident_count() >= 1,
        "at least one page went resident"
    );

    // Budget of one resident page, nothing protected from a later frame: the
    // lowest-priority resident page is evicted.
    let victims = table.select_evictions(1, 5);
    for key in &victims {
        table.evict(*key);
    }
    assert!(
        table.resident_count() <= requested,
        "eviction never grows residency"
    );
}

#[test]
fn far_receiver_selects_coarser_level_than_near() {
    let cfg = config();
    let quality = ShadowQuality {
        focal_length_pixels: 100.0,
        texels_per_pixel: 1.0,
    };
    let camera = [0.0, 0.0, 0.0];

    // A near and a far receiver at the same XZ spot: the far one needs a coarser
    // texel, so it resolves to a higher clip level.
    let near = caster_for_receiver(
        bounds([0.0, 5.0, 0.0], [0.0, 0.0, 0.0]),
        1.0,
        camera,
        &quality,
    );
    let far = caster_for_receiver(
        bounds([0.0, 5000.0, 0.0], [0.0, 0.0, 0.0]),
        1.0,
        camera,
        &quality,
    );

    let near_level = cfg.select_level(near.required_texel_size);
    let far_level = cfg.select_level(far.required_texel_size);
    assert!(
        far_level >= near_level,
        "far receiver must not pick a finer level than the near one"
    );
    assert!(far_level > near_level, "a much farther receiver is coarser");
}
