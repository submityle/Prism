//! Scene-level CPU parity between the realtime clearcoat response and the
//! independent path-traced reference.
//!
//! The fixture intentionally uses the supported `ClearcoatDiffuse` slice. It
//! exercises real triangle intersection, camera rays, point-light visibility,
//! the reference path integrator, and the realtime CPU direct evaluator in one
//! deterministic image gate. It does not claim parity for unsupported
//! realtime/reference lobes or for the GPU path.

use prism_render_architecture::ray_scene::triangle_mesh::{TriangleMesh, TriangleMeshBvh};
use prism_render_architecture::reference_pt::{
    camera::PinholeCamera,
    compare::compare_films,
    estimator::Light,
    film::Film,
    integrator::{Material, PathIntegrator, Scene},
    sampler::{Rng, Sample2},
    Vec3,
};
use prism_render_shading::{
    evaluate_clearcoat_direct, reference_pt_adapter::ReferencePtMaterialAdapter, DirectLightSample,
    ShadingFrame, SurfaceSample,
};

const WIDTH: u32 = 24;
const HEIGHT: u32 = 16;
const LIGHT_POSITION: Vec3 = Vec3::new(1.25, 4.5, -1.0);
const LIGHT_INTENSITY: Vec3 = Vec3::new(18.0, 16.0, 14.0);

fn fixture_surface() -> SurfaceSample {
    SurfaceSample {
        base_color: [0.29, 0.51, 0.73],
        base_weight: 0.87,
        diffuse_roughness: 0.23,
        specular_weight: 0.0,
        clearcoat: 0.63,
        clearcoat_roughness: 0.24,
        coat_ior: 1.61,
        coat_color: [0.93, 0.81, 0.68],
        coat_darkening: 0.42,
        emissive: [0.0; 3],
        ..SurfaceSample::default()
    }
}

fn fixture_scene(surface: SurfaceSample) -> Scene {
    // Counter-clockwise winding as viewed from +Y gives the floor a +Y normal.
    let mesh = TriangleMesh::new(
        vec![
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            [-10.0, 0.0, 10.0],
        ],
        vec![],
        vec![],
        vec![[0, 2, 1], [0, 3, 2]],
    )
    .expect("fixture floor mesh is valid");
    let geometry = TriangleMeshBvh::build(mesh);
    let reference = ReferencePtMaterialAdapter::from_surface(surface)
        .expect("the fixture is inside the ClearcoatDiffuse reference slice");
    Scene::new(
        geometry,
        vec![Material::new(reference.bsdf()); 2],
        vec![Light::Point {
            position: LIGHT_POSITION,
            intensity: LIGHT_INTENSITY,
        }],
        Vec3::ZERO,
    )
    .expect("material count matches the two floor triangles")
}

fn camera() -> PinholeCamera {
    PinholeCamera::look_at(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0),
        0.42,
        WIDTH as f32 / HEIGHT as f32,
    )
    .expect("fixture camera is valid")
}

fn center_ray(
    camera: &PinholeCamera,
    x: u32,
    y: u32,
) -> prism_render_architecture::ray_scene::traversal::Ray {
    camera.primary_ray(x, y, WIDTH, HEIGHT, Sample2 { x: 0.5, y: 0.5 })
}

fn realtime_pixel(
    scene: &Scene,
    ray: prism_render_architecture::ray_scene::traversal::Ray,
    surface: SurfaceSample,
) -> Vec3 {
    let Some(hit) = scene.intersect(&ray) else {
        return Vec3::ZERO;
    };
    let view = Vec3::from_array(ray.direction())
        .negate()
        .normalize_or_zero();
    let normal = hit.normal.faced_toward(view);
    let to_light = LIGHT_POSITION.sub(hit.position);
    let distance_squared = to_light.length_squared();
    let direction = to_light.scale(distance_squared.sqrt().recip());
    let frame = ShadingFrame {
        normal: normal.to_array(),
        view: view.to_array(),
        tangent: [1.0, 0.0, 0.0],
        bitangent: [0.0, 0.0, -1.0],
    };
    let direct = evaluate_clearcoat_direct(
        surface,
        frame,
        DirectLightSample {
            direction: direction.to_array(),
            illuminance: LIGHT_INTENSITY.scale(distance_squared.recip()).to_array(),
            visibility: 1.0,
        },
    );
    Vec3::from_array(direct)
}

#[test]
fn path_traced_reference_matches_realtime_cpu_scene_image_for_clearcoat_slice() {
    let surface = fixture_surface();
    let scene = fixture_scene(surface);
    let camera = camera();
    let integrator = PathIntegrator::new(1, 1);
    let mut reference_pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    let mut realtime_pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let ray = center_ray(&camera, x, y);
            let pixel_index = u64::from(y * WIDTH + x);
            let mut rng = Rng::with_stream(0x52_4546_5041_5249, pixel_index + 1);
            reference_pixels.push(integrator.radiance(&scene, ray, &mut rng));
            realtime_pixels.push(realtime_pixel(&scene, ray, surface));
        }
    }

    let reference = Film::from_pixels(WIDTH, HEIGHT, reference_pixels);
    let realtime = Film::from_pixels(WIDTH, HEIGHT, realtime_pixels);
    let metrics = compare_films(&reference, &realtime).expect("image dimensions match");
    eprintln!("clearcoat scene CPU parity metrics: {metrics:?}");
    assert!(
        metrics.max_abs <= 2.0e-4 && metrics.rmse <= 5.0e-5,
        "scene-level realtime/reference parity drifted: metrics={metrics:?}"
    );
}

#[test]
fn path_traced_reference_scene_gate_is_deterministic() {
    let surface = fixture_surface();
    let scene = fixture_scene(surface);
    let camera = camera();
    let integrator = PathIntegrator::new(1, 1);
    let mut first = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    let mut second = Vec::with_capacity((WIDTH * HEIGHT) as usize);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let ray = center_ray(&camera, x, y);
            let pixel_index = u64::from(y * WIDTH + x);
            let mut first_rng = Rng::with_stream(0x52_4546_5041_5249, pixel_index + 1);
            let mut second_rng = Rng::with_stream(0x52_4546_5041_5249, pixel_index + 1);
            first.push(integrator.radiance(&scene, ray, &mut first_rng));
            second.push(integrator.radiance(&scene, ray, &mut second_rng));
        }
    }

    assert_eq!(
        Film::from_pixels(WIDTH, HEIGHT, first).pixels(),
        Film::from_pixels(WIDTH, HEIGHT, second).pixels()
    );
}
