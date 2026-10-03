//! End-to-end integration tests for the Prism visibility plugin running inside
//! a minimal Bevy `App` (CPU only, no GPU device required).

use bevy_app::prelude::*;
use bevy_camera::primitives::{Aabb, Frustum};
use bevy_math::{Vec3, Vec4};
use bevy_shape::{HalfSpace, ViewFrustum};
use bevy_transform::components::GlobalTransform;

use prism_bevy::{
    GeometryHandle, GeometryLod, GeometryLodChain, MaterialRecord, PrismCamera, PrismRenderScene,
    PrismRenderable, PrismViewVisibility, PrismVisibilityPlugin, SceneMaterialHandle,
};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    GpuSurfaceParameters, Illumination, MaterialDomain, MaterialFeatureFlags, MaterialRenderClass,
};

fn handle(index: u32) -> GenerationalHandle {
    GenerationalHandle::new(index, 0)
}

/// A camera looking down `-Z`: anything in front (negative Z) is inside, behind
/// (positive Z) is culled. Mirrors the convention proven in the visibility crate.
fn forward_frustum() -> Frustum {
    Frustum(ViewFrustum {
        half_spaces: [
            HalfSpace::new(Vec4::new(1.0, 0.0, 0.0, 10.0)),
            HalfSpace::new(Vec4::new(-1.0, 0.0, 0.0, 10.0)),
            HalfSpace::new(Vec4::new(0.0, -1.0, 0.0, 10.0)),
            HalfSpace::new(Vec4::new(0.0, 1.0, 0.0, 10.0)),
            HalfSpace::new(Vec4::new(0.0, 0.0, -1.0, -1.0)),
            HalfSpace::new(Vec4::new(0.0, 0.0, 1.0, 100.0)),
        ],
    })
}

fn geometry_handle() -> GeometryHandle {
    handle(7)
}

fn material_handle() -> SceneMaterialHandle {
    handle(1)
}

fn lod_chain() -> GeometryLodChain {
    GeometryLodChain {
        geometry: geometry_handle(),
        lods: vec![GeometryLod {
            level: 0,
            screen_error: 0.0,
            resident: true,
            fallback: true,
        }],
    }
}

fn material() -> MaterialRecord {
    MaterialRecord {
        handle: material_handle(),
        revision: 1,
        domain: MaterialDomain::default(),
        render_class: MaterialRenderClass::Opaque,
        illumination: Illumination::default(),
        features: MaterialFeatureFlags::default(),
        closure_mask: 0,
        surface: GpuSurfaceParameters::default(),
        textures: Vec::new(),
        custom_program: None,
    }
}

fn unit_cube() -> Aabb {
    Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0))
}

/// Builds an app with the plugin and the backend geometry/material registered.
fn make_app() -> App {
    let mut app = App::new();
    app.add_plugins(PrismVisibilityPlugin);
    {
        let mut scene = app.world_mut().resource_mut::<PrismRenderScene>();
        scene.insert_geometry(lod_chain());
        scene.insert_material(material());
    }
    app
}

#[test]
fn in_frustum_entity_is_marked_visible_and_behind_is_culled() {
    let mut app = make_app();

    // Camera looking down -Z from the origin.
    app.world_mut().spawn((
        forward_frustum(),
        GlobalTransform::IDENTITY,
        PrismCamera::default(),
    ));

    // One renderable in front of the camera, one behind it.
    let front = app
        .world_mut()
        .spawn((
            PrismRenderable::new(geometry_handle(), material_handle()),
            unit_cube(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -5.0)),
            PrismViewVisibility::default(),
        ))
        .id();
    let behind = app
        .world_mut()
        .spawn((
            PrismRenderable::new(geometry_handle(), material_handle()),
            unit_cube(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 50.0)),
            PrismViewVisibility::default(),
        ))
        .id();

    app.update();

    assert!(app.world().get::<PrismViewVisibility>(front).unwrap().visible);
    assert!(!app.world().get::<PrismViewVisibility>(behind).unwrap().visible);
    assert_eq!(app.world().resource::<PrismRenderScene>().instance_count(), 2);
}

#[test]
fn moving_an_entity_into_the_frustum_flips_visibility() {
    let mut app = make_app();
    app.world_mut().spawn((
        forward_frustum(),
        GlobalTransform::IDENTITY,
        PrismCamera::default(),
    ));

    let entity = app
        .world_mut()
        .spawn((
            PrismRenderable::new(geometry_handle(), material_handle()),
            unit_cube(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 50.0)),
            PrismViewVisibility::default(),
        ))
        .id();

    app.update();
    assert!(!app.world().get::<PrismViewVisibility>(entity).unwrap().visible);

    // Move it in front of the camera.
    *app.world_mut().get_mut::<GlobalTransform>(entity).unwrap() =
        GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -5.0));
    app.update();
    assert!(app.world().get::<PrismViewVisibility>(entity).unwrap().visible);
}

#[test]
fn despawning_an_entity_removes_it_from_the_scene() {
    let mut app = make_app();
    app.world_mut().spawn((
        forward_frustum(),
        GlobalTransform::IDENTITY,
        PrismCamera::default(),
    ));

    let entity = app
        .world_mut()
        .spawn((
            PrismRenderable::new(geometry_handle(), material_handle()),
            unit_cube(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -5.0)),
            PrismViewVisibility::default(),
        ))
        .id();

    app.update();
    assert_eq!(app.world().resource::<PrismRenderScene>().instance_count(), 1);

    app.world_mut().despawn(entity);
    app.update();
    assert_eq!(app.world().resource::<PrismRenderScene>().instance_count(), 0);
}
