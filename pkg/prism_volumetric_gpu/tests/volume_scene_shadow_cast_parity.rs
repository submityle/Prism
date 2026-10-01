//! Real-device parity for the particle-volume soft-shadow caster twin:
//! [`GpuVolumeShadowCast`] must reproduce the `CPU` golden
//! [`VolumeShadowCaster`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster)
//! (design section 20) for the final attenuation (`transmittance_along` /
//! `cast`), the baked deep opacity curve (`bake_light_ray`) and the deep-opacity
//! lookup (`attenuation_from_map`), across several densities, extinctions, step
//! counts, lights and receivers.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both sides run the identical march — the same step midpoints, the same
//! `floor`-based froxel cell lookup, and the same algebraic step-opacity
//! recurrence `transmittance *= 1 - clamp(density * sigma * step_length, 0, 1)`
//! over the identical density field — with no reorderable summation. Each value
//! is asserted to within `abs_diff < 1e-5` or `rel_diff < 1e-4`, tight enough to
//! fail a wrong port (a dropped clamp, a swapped step offset, a misread cell)
//! yet loose enough to admit legal fused multiply-add contraction over the
//! survival product chain. The scenes also assert the physical shape — an empty
//! medium stays fully lit, a denser or more extinctive volume attenuates more,
//! and the baked curve is monotone non-increasing — so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: standard Nelson-Max front-to-back `1 - alpha` volume
//! compositing; no Unreal Engine source or derived code.

use prism_render_architecture::particle::volume_scene_shadow_cast::{
    SceneLight, VolumeShadowCaster,
};
use prism_render_architecture::particle::volumetrics::{
    sample_deep_transmittance, DeepOpacityLayer, FroxelDensityField, FroxelGrid,
};
use prism_render_architecture::particle::{ParticleSystemHandle, Vec3};
use prism_volumetric_gpu::volume_scene_shadow_cast::{GpuVolumeShadowCast, VolumeShadowRay};
use prism_volumetric_gpu::GpuContext;

/// Shared absolute tolerance floor for the relative-difference guard.
const REL_FLOOR: f32 = 1e-6;

/// Converts a [`Vec3`] into the plain component array a [`VolumeShadowRay`] carries.
fn arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Asserts `got` matches `want` to within the documented tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs_diff = (got - want).abs();
    let rel_diff = abs_diff / want.abs().max(REL_FLOOR);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-4,
        "{what} mismatch: gpu {got}, cpu {want} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts the `gpu` curve matches the `cpu` golden curve layer-for-layer.
fn assert_curve_parity(cpu: &[DeepOpacityLayer], gpu: &[DeepOpacityLayer]) {
    assert_eq!(gpu.len(), cpu.len(), "one layer per recorded step");
    for (i, (got, want)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(got.depth, want.depth, &format!("layer {i} depth"));
        assert_close(
            got.transmittance,
            want.transmittance,
            &format!("layer {i} transmittance"),
        );
    }
}

/// A `4 x 1 x 1` grid with a single occluding cell at `x` in `[1, 2)`, matching
/// the `CPU` golden's own fixture.
fn single_slab_field(density: f32) -> FroxelDensityField {
    let grid = FroxelGrid::new([4, 1, 1], Vec3::ZERO, Vec3::splat(1.0));
    let mut field = FroxelDensityField::new(grid);
    // Center of cell [1, 0, 0].
    let _ = field.inject(Vec3::new(1.5, 0.5, 0.5), density);
    field
}

fn backlit_receiver() -> Vec3 {
    Vec3::new(3.5, 0.5, 0.5)
}

fn lit_receiver() -> Vec3 {
    Vec3::new(0.5, 0.5, 0.5)
}

fn travel_plus_x() -> SceneLight {
    SceneLight::Directional {
        travel: Vec3::new(1.0, 0.0, 0.0),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_cast_matches_cpu_golden_across_densities() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volume-scene-shadow-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);

    let receiver = backlit_receiver();
    let light = travel_plus_x();
    // A battery of densities marched in one batched dispatch (one thread each).
    let densities = [0.0, 0.1, 0.3, 0.5, 0.8, 1.5];
    let fields: Vec<FroxelDensityField> = densities.iter().map(|&d| single_slab_field(d)).collect();

    for field in &fields {
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(0), field, 1.0, 0.25, 16);
        // Resolve the light direction on the host exactly as `cast` does, then
        // march the identical ray on the device.
        let dir = light.toward_light(receiver);
        let ray = VolumeShadowRay {
            start: arr(receiver),
            direction: arr(dir),
        };
        let out = gpu.march(&ctx, field, 1.0, 0.25, 16, &[ray]);
        let cpu_cast = caster.cast(receiver, light);
        assert_close(out[0].transmittance, cpu_cast, "cast attenuation");
    }
}

#[test]
fn gpu_transmittance_matches_cpu_across_extinctions_and_steps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.5);
    let start = backlit_receiver();
    // The raw (un-normalized) direction the kernel normalizes itself.
    let direction = Vec3::new(-1.0, 0.0, 0.0);

    // Vary extinction, step length and step count together in one dispatch.
    let configs = [
        (0.5_f32, 0.25_f32, 16_u32),
        (1.0, 0.25, 16),
        (1.5, 0.5, 8),
        (2.0, 0.1, 40),
    ];
    let rays: Vec<VolumeShadowRay> = configs
        .iter()
        .map(|_| VolumeShadowRay {
            start: arr(start),
            direction: arr(direction),
        })
        .collect();

    for ((ext, step, steps), ray) in configs.iter().zip(rays.iter()) {
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(1), &field, *ext, *step, *steps);
        let out = gpu.march(&ctx, &field, *ext, *step, *steps, std::slice::from_ref(ray));
        let cpu = caster.transmittance_along(start, direction);
        assert_close(out[0].transmittance, cpu, "transmittance_along");
    }
}

#[test]
fn gpu_bake_matches_cpu_golden_curve() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.5);
    let caster = VolumeShadowCaster::new(ParticleSystemHandle(2), &field, 1.0, 0.25, 16);

    let entry = Vec3::new(0.0, 0.5, 0.5);
    let travel = Vec3::new(1.0, 0.0, 0.0);
    let ray = VolumeShadowRay {
        start: arr(entry),
        direction: arr(travel),
    };
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[ray]);
    let cpu = caster.bake_light_ray(entry, travel);
    assert_curve_parity(&cpu, &out[0].layers);

    // The baked curve is monotone non-increasing in transmittance.
    for pair in out[0].layers.windows(2) {
        assert!(
            pair[1].transmittance <= pair[0].transmittance + 1e-5,
            "the baked curve must not brighten with depth"
        );
    }
}

#[test]
fn gpu_baked_map_lookup_matches_cpu_cast() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.5);
    let caster = VolumeShadowCaster::new(ParticleSystemHandle(3), &field, 1.0, 0.25, 16);

    let entry = Vec3::new(0.0, 0.5, 0.5);
    let travel = Vec3::new(1.0, 0.0, 0.0);
    let ray = VolumeShadowRay {
        start: arr(entry),
        direction: arr(travel),
    };
    let baked = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[ray]);

    // Sample the GPU-baked deep opacity map at the receiver's projected depth
    // and confirm it agrees with a direct CPU `cast`, proving the curves are
    // interchangeable with the reference's `attenuation_from_map` path.
    let receiver = backlit_receiver();
    let dir = travel.normalize_or_zero();
    let along = receiver.sub(entry).dot(dir);
    let depth = if along > 0.0 { along } else { 0.0 };
    let mapped = sample_deep_transmittance(&baked[0].layers, depth);
    let marched = caster.cast(receiver, travel_plus_x());
    assert_close(mapped, marched, "deep-opacity-map lookup");
}

#[test]
fn gpu_point_light_behind_volume_shadows_receiver() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.7);
    let caster = VolumeShadowCaster::new(ParticleSystemHandle(4), &field, 1.0, 0.25, 16);

    // Light on the -X side; the receiver at +X marches back through the slab.
    let light = SceneLight::Point {
        position: Vec3::new(-2.0, 0.5, 0.5),
    };
    let receiver = backlit_receiver();
    let dir = light.toward_light(receiver);
    let ray = VolumeShadowRay {
        start: arr(receiver),
        direction: arr(dir),
    };
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[ray]);
    let cpu = caster.cast(receiver, light);
    assert_close(out[0].transmittance, cpu, "point-light cast");
    assert!(out[0].transmittance < 1.0, "a backlit receiver is shadowed");
}

#[test]
fn gpu_empty_volume_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.0);
    let light = travel_plus_x();
    let receiver = backlit_receiver();
    let ray = VolumeShadowRay {
        start: arr(receiver),
        direction: arr(light.toward_light(receiver)),
    };
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[ray]);
    // Zero density never attenuates.
    assert_close(out[0].transmittance, 1.0, "empty-volume attenuation");
}

#[test]
fn gpu_denser_and_backlit_scenes_follow_physics() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let light = travel_plus_x();
    let receiver = backlit_receiver();

    let sparse = single_slab_field(0.2);
    let dense = single_slab_field(0.8);
    let sparse_ray = VolumeShadowRay {
        start: arr(receiver),
        direction: arr(light.toward_light(receiver)),
    };
    let att_sparse = gpu.march(&ctx, &sparse, 1.0, 0.25, 16, &[sparse_ray])[0].transmittance;
    let att_dense = gpu.march(&ctx, &dense, 1.0, 0.25, 16, &[sparse_ray])[0].transmittance;
    // Monotonic: more density -> less light survives.
    assert!(
        att_dense < att_sparse,
        "a denser volume must attenuate more"
    );
    assert!(
        att_sparse < 1.0,
        "a non-empty volume attenuates the backlit side"
    );

    // The lit side marches away from the slab and stays fully lit.
    let field = single_slab_field(0.6);
    let lit_ray = VolumeShadowRay {
        start: arr(lit_receiver()),
        direction: arr(light.toward_light(lit_receiver())),
    };
    let backlit_ray = VolumeShadowRay {
        start: arr(receiver),
        direction: arr(light.toward_light(receiver)),
    };
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[lit_ray, backlit_ray]);
    assert_close(out[0].transmittance, 1.0, "lit-side attenuation");
    assert!(
        out[1].transmittance < out[0].transmittance,
        "the backlit receiver is darker than the lit receiver"
    );
}

#[test]
fn gpu_degenerate_direction_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.9);
    let caster = VolumeShadowCaster::new(ParticleSystemHandle(5), &field, 1.0, 0.25, 16);

    // A zero direction (a degenerate light) must stay fully lit with a single
    // baked layer, exactly like the CPU early return.
    let receiver = backlit_receiver();
    let ray = VolumeShadowRay {
        start: arr(receiver),
        direction: [0.0, 0.0, 0.0],
    };
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[ray]);
    let dead = SceneLight::Directional { travel: Vec3::ZERO };
    assert_close(
        out[0].transmittance,
        caster.cast(receiver, dead),
        "degenerate cast",
    );
    assert_eq!(out[0].layers.len(), 1, "a degenerate ray bakes one layer");
    assert_close(
        out[0].layers[0].depth,
        0.0,
        "the lone layer sits at depth 0",
    );
    assert_close(
        out[0].layers[0].transmittance,
        1.0,
        "the lone layer is fully lit",
    );
}

#[test]
fn gpu_empty_rays_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeShadowCast::new(&ctx);
    let field = single_slab_field(0.5);
    let out = gpu.march(&ctx, &field, 1.0, 0.25, 16, &[]);
    assert!(out.is_empty(), "an empty ray slice yields no marches");
}
