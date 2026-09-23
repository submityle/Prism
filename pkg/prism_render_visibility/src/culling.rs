use crate::{
    GeometryLodChain, GpuRenderWorkItem, GpuViewRecord, RenderPassMask, VisibilityDiagnostics,
    WorkSortKey,
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use prism_render_architecture::gpu_scene::{CpuRenderScene, SceneHandle};
use prism_render_material::{MaterialRecord, MaterialRenderClass};

pub struct VisibilityInput<'a> {
    pub scene: &'a CpuRenderScene,
    pub handles: &'a [SceneHandle],
    pub geometry:
        &'a BTreeMap<prism_render_architecture::gpu_scene::GeometryHandle, GeometryLodChain>,
    pub materials:
        &'a BTreeMap<prism_render_architecture::gpu_scene::SceneMaterialHandle, MaterialRecord>,
    pub previous_lods: &'a BTreeMap<(crate::ViewHandle, SceneHandle), u16>,
    pub occluded: &'a BTreeSet<(crate::ViewHandle, SceneHandle)>,
    pub capacity: u32,
    pub previous_history_epoch: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CullReason {
    StaleHandle,
    LayerMask,
    Frustum,
    Occluded,
    MissingGeometry,
    MissingMaterial,
    LodUnavailable,
    Overflow,
}

pub fn cull_view(
    view: &GpuViewRecord,
    input: VisibilityInput<'_>,
) -> (Vec<GpuRenderWorkItem>, VisibilityDiagnostics) {
    let mut work = Vec::new();
    let mut stats = VisibilityDiagnostics {
        input_instances: input.handles.len() as u32,
        ..Default::default()
    };
    let use_occlusion =
        view.history_policy(input.previous_history_epoch) == crate::HistoryPolicy::Reuse;
    for &handle in input.handles {
        let Some(instance) = input.scene.get(handle) else {
            stats.stale_handles += 1;
            continue;
        };
        if instance.render_layers & view.layer_mask == 0 {
            stats.layer_rejected += 1;
            continue;
        }
        if !sphere_visible(
            &view.frustum_planes,
            instance.bounds.center,
            instance.bounds.radius,
        ) {
            stats.frustum_rejected += 1;
            continue;
        }
        if use_occlusion && input.occluded.contains(&(view.handle, handle)) {
            stats.occlusion_rejected += 1;
            continue;
        }
        let Some(geometry) = input.geometry.get(&instance.geometry) else {
            stats.missing_geometry += 1;
            continue;
        };
        let Some(material) = input.materials.get(&instance.material) else {
            stats.missing_material += 1;
            continue;
        };
        let distance = squared_distance(view.world_position, instance.bounds.center)
            .sqrt()
            .max(1.0e-4);
        let projected_radius = instance.bounds.radius / distance;
        let Some(lod) = geometry.select(
            projected_radius,
            view.lod_scale,
            input.previous_lods.get(&(view.handle, handle)).copied(),
        ) else {
            stats.missing_geometry += 1;
            continue;
        };
        if work.len() >= input.capacity as usize {
            stats.overflowed = true;
            break;
        }
        if lod.used_fallback {
            stats.lod_fallbacks += 1;
        }
        let pass_mask = pass_mask(material.render_class, instance.flags, view.flags);
        stats.shadow_casters += u32::from(pass_mask.0 & RenderPassMask::SHADOW.0 != 0);
        stats.ray_scene_instances += u32::from(pass_mask.0 & RenderPassMask::RAY_SCENE.0 != 0);
        work.push(GpuRenderWorkItem {
            scene: handle,
            geometry: instance.geometry,
            material: instance.material,
            lod_or_cluster: lod.level as u32,
            pass_mask,
            sort_key: WorkSortKey::new(
                pass_class(material.render_class),
                material.shading_model as u8,
                material.render_class as u8,
                instance.geometry.index as u16,
                depth_bucket(distance),
            ),
        });
    }
    stats.visible_instances = work.len() as u32;
    stats.material_bins = work
        .iter()
        .map(|item| item.material)
        .collect::<BTreeSet<_>>()
        .len() as u32;
    (work, stats)
}

fn sphere_visible(planes: &[[f32; 4]; 6], center: [f32; 3], radius: f32) -> bool {
    planes
        .iter()
        .all(|p| p[0] * center[0] + p[1] * center[1] + p[2] * center[2] + p[3] >= -radius)
}
fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
fn depth_bucket(distance: f32) -> u16 {
    distance.min(u16::MAX as f32) as u16
}
fn pass_class(class: MaterialRenderClass) -> u8 {
    match class {
        MaterialRenderClass::Opaque
        | MaterialRenderClass::OpaqueTwoSided
        | MaterialRenderClass::NprOpaque
        | MaterialRenderClass::CustomOpaque => 0,
        MaterialRenderClass::Masked | MaterialRenderClass::MaskedTwoSided => 1,
        _ => 2,
    }
}
fn pass_mask(class: MaterialRenderClass, flags: u32, view: crate::ViewFlags) -> RenderPassMask {
    let mut mask = match pass_class(class) {
        0 => RenderPassMask::OPAQUE,
        1 => RenderPassMask::MASKED,
        _ => RenderPassMask::TRANSPARENT,
    };
    if flags & 1 == 0 {
        mask |= RenderPassMask::SHADOW;
    }
    mask |= RenderPassMask::GI | RenderPassMask::RAY_SCENE | RenderPassMask::PICKING;
    if view.contains(crate::ViewFlags::OFFLINE) {
        mask |= RenderPassMask::OFFLINE;
    }
    mask
}
