//! Camera-local scene and particle targets composed before frame-wide Ember work.

use super::*;
use alloc::collections::BTreeSet;
use prism_ecs::prelude::Entity;

/// One camera's local attachments and its placement in the output.
/// Queues retain their camera's frame-uniform slot and binding sidecars.
pub(crate) struct SurfaceViewGraphInput<'a> {
    pub camera: Entity,
    pub order: i32,
    pub clear: ClearMode,
    pub viewport: [u32; 4],
    pub target: &'a RenderTarget,
    pub queues: &'a RenderQueues,
    pub lighting: Option<LightingGpuPass>,
}

/// Rejected before any camera is appended; no accepted draws are discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceViewGraphError {
    DuplicateCamera(Entity),
    InvalidViewport(Entity),
    AliasedTarget(TextureId),
    MissingExternalTextureImport(TextureViewId),
    MissingCompositeBinding(Option<Entity>),
    InconsistentLighting(Option<Entity>),
}

/// Aggregate accounting is deliberately separate from the single-target receipt.
/// Every member still satisfies `FrameGraphStages::composite_passes <= 1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SurfaceViewGraphStages {
    pub manual: FrameGraphStages,
    pub cameras: Vec<(Entity, FrameGraphStages)>,
    pub scene_passes: u32,
    pub trace_dispatches: u32,
    pub candidate_build_dispatches: u32,
    pub direct_recipes: u32,
    pub gi_recipes: u32,
    pub composite_passes: u32,
    pub present_passes: u32,
    pub particle_passes: u32,
    pub ember_compute_passes: u32,
    pub lighting_dispatches: u32,
    pub copy_passes: u32,
    pub present_source: TextureHandle,
}

impl SurfaceViewGraphStages {
    fn new(manual: FrameGraphStages) -> Self {
        Self {
            manual,
            cameras: Vec::new(),
            scene_passes: manual.scene_passes,
            trace_dispatches: manual.trace_dispatches,
            candidate_build_dispatches: manual.candidate_build_dispatches,
            direct_recipes: manual.direct_recipes,
            gi_recipes: manual.gi_recipes,
            composite_passes: manual.composite_passes,
            present_passes: 0,
            particle_passes: 0,
            ember_compute_passes: 0,
            lighting_dispatches: manual.lighting_dispatches,
            copy_passes: 0,
            present_source: manual.present_source,
        }
    }

    fn push(&mut self, camera: Entity, stages: FrameGraphStages) {
        self.scene_passes += stages.scene_passes;
        self.trace_dispatches += stages.trace_dispatches;
        self.candidate_build_dispatches += stages.candidate_build_dispatches;
        self.direct_recipes += stages.direct_recipes;
        self.gi_recipes += stages.gi_recipes;
        self.composite_passes += stages.composite_passes;
        self.lighting_dispatches += stages.lighting_dispatches;
        self.copy_passes += 1;
        self.cameras.push((camera, stages));
    }

    pub(crate) fn assert_contract(&self) {
        self.manual.assert_contract();
        for (_, stages) in &self.cameras {
            stages.assert_contract();
            assert_eq!(stages.ember_compute_passes, 0, "Ember is a frame-wide tail");
        }
        assert_eq!(
            self.particle_passes,
            self.manual.particle_passes
                + self
                    .cameras
                    .iter()
                    .map(|(_, stages)| stages.particle_passes)
                    .sum::<u32>()
        );
        assert!(self.trace_dispatches <= 1);
        assert!(self.candidate_build_dispatches <= self.trace_dispatches);
        assert!(self.candidate_build_dispatches <= self.gi_recipes);
        assert!(self.gi_recipes <= 1);
        assert!(self.composite_passes as usize <= self.cameras.len() + 1);
        assert_eq!(self.copy_passes as usize, self.cameras.len());
        assert_eq!(
            self.lighting_dispatches,
            self.gi_recipes + self.composite_passes
        );
        assert!(self.present_passes <= 1);
    }
}

/// Move-only preparation packet. Consume it once with `finalize_surface_view_graph`.
/// It owns declarations only, not the targets or their completion lifetime.
pub(crate) struct PreparedSurfaceViewGraph {
    graph: RenderGraph,
    color: TextureHandle,
    depth: TextureHandle,
    output_view: TextureViewId,
    output_depth_view: TextureViewId,
    textures: BTreeMap<TextureViewId, TextureHandle>,
    stages: SurfaceViewGraphStages,
    camera_outputs: Vec<PreparedCameraOutput>,
}

struct PreparedCameraOutput {
    camera: Entity,
    viewport: [u32; 4],
    color: TextureHandle,
    depth: TextureHandle,
    color_view: TextureViewId,
    depth_view: TextureViewId,
}

impl PreparedSurfaceViewGraph {
    pub(crate) fn stages(&self) -> &SurfaceViewGraphStages {
        &self.stages
    }
}

/// Draw explicit queues over the whole output first, then overwrite camera rects
/// in ascending `(camera.order, camera.index(), camera.generation())` order. Empty cameras still
/// clear/composite and copy their entire local HDR image, including background.
/// Trace, candidate construction and GI run only for the first lit target.
/// The caller supplies target-aware lighting bindings and owns all GPU lifetimes.
pub(crate) fn prepare_surface_view_graph(
    output: &RenderTarget,
    manual: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    views: &[SurfaceViewGraphInput<'_>],
    external_views: &FrameGraphTextureImports,
) -> Result<PreparedSurfaceViewGraph, SurfaceViewGraphError> {
    let mut ordered: Vec<_> = views.iter().collect();
    ordered.sort_by_key(|view| (view.order, view.camera.index(), view.camera.generation()));
    let mut identities = BTreeSet::new();
    let mut targets = BTreeSet::new();
    reserve_target(output, &mut targets)?;
    let mut first_lighting = lighting;
    validate_lighting(None, lighting, &mut first_lighting)?;
    let mut inherited_clear = ClearMode::ColorAndDepth;
    let mut resolved_clears = Vec::with_capacity(ordered.len());
    for view in &ordered {
        if !identities.insert((view.camera.index(), view.camera.generation())) {
            return Err(SurfaceViewGraphError::DuplicateCamera(view.camera));
        }
        let [x, y, width, height] = view.viewport;
        if width == 0
            || height == 0
            || x.checked_add(width)
                .is_none_or(|end| end > output.viewport().width)
            || y.checked_add(height)
                .is_none_or(|end| end > output.viewport().height)
            || width != view.target.viewport().width
            || height != view.target.viewport().height
            || view.target.format() != output.format()
        {
            return Err(SurfaceViewGraphError::InvalidViewport(view.camera));
        }
        let clear = match view.clear {
            ClearMode::Inherit => inherited_clear,
            explicit => explicit,
        };
        inherited_clear = ClearMode::None;
        reserve_target(view.target, &mut targets)?;
        validate_lighting(Some(view.camera), view.lighting, &mut first_lighting)?;
        resolved_clears.push((view.camera, clear));
    }

    validate_sampled_texture_imports(manual, functions, external_views)?;
    for view in &ordered {
        validate_sampled_texture_imports(view.queues, functions, external_views)?;
    }

    let mut graph = RenderGraph::new(output.viewport().extent());
    let mut buffers = FrameGraphBuffers::default();
    buffers.reserve_target(manual, functions, lighting, trace);
    for view in &ordered {
        buffers.reserve_target(view.queues, functions, view.lighting, None);
    }
    let mut textures = BTreeMap::new();
    let mut used_lighting = lighting.is_some();
    // A camera-only lighting plan must see its trace on that first camera,
    // rather than consuming the trace in an unrelated unlit manual target.
    let manual_trace = if lighting.is_some() || first_lighting.is_none() {
        trace
    } else {
        None
    };
    let appended = append_target_graph(
        &mut graph,
        &mut buffers,
        &mut textures,
        output,
        manual,
        functions,
        lighting,
        manual_trace,
        None,
        ClearMode::ColorAndDepth,
        external_views,
    );
    let color = appended.color;
    let depth = appended.depth;
    let mut stages = SurfaceViewGraphStages::new(appended.stages);
    let mut camera_outputs = Vec::with_capacity(ordered.len());
    for (view, (resolved_camera, clear)) in ordered.into_iter().zip(resolved_clears) {
        debug_assert_eq!(view.camera, resolved_camera);
        let (view_lighting, view_trace) = match view.lighting {
            Some(mut pass) if used_lighting => {
                pass.run_gi = false;
                (Some(pass), None)
            }
            Some(pass) => {
                used_lighting = true;
                (Some(pass), trace)
            }
            None => (None, None),
        };
        let appended = append_target_graph(
            &mut graph,
            &mut buffers,
            &mut textures,
            view.target,
            view.queues,
            functions,
            view_lighting,
            view_trace,
            None,
            clear,
            external_views,
        );
        // Delay composition until the camera's own particles have sampled its
        // own scene depth. Output placement is not the local raster viewport.
        camera_outputs.push(PreparedCameraOutput {
            camera: view.camera,
            viewport: view.viewport,
            color: appended.color,
            depth: appended.depth,
            color_view: view.target.view(),
            depth_view: view.target.depth_view(),
        });
        stages.push(view.camera, appended.stages);
    }
    textures.insert(output.view(), color);
    stages.present_source = color;
    stages.assert_contract();
    // Do not put aggregated composite counts into the single-target receipt.
    graph.blackboard_mut().set(stages.manual);
    graph.blackboard_mut().set(stages.clone());
    Ok(PreparedSurfaceViewGraph {
        graph,
        color,
        depth,
        output_view: output.view(),
        output_depth_view: output.depth_view(),
        textures,
        stages,
        camera_outputs,
    })
}

/// Append particles to each camera's local HDR/depth, then compose camera rects
/// and append frame-wide Ember work once. Hosts present this same HDR graph.
pub(crate) fn finalize_surface_view_graph(
    prepared: PreparedSurfaceViewGraph,
    particle: Option<&ParticleGpuState>,
    ember: Option<&EmberGpuAssetRegistry>,
    external_imports: &BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>,
    frame_index: u64,
) -> Result<(RenderGraph, TextureHandle), DisplayTransformGpuError> {
    let PreparedSurfaceViewGraph {
        mut graph,
        mut color,
        mut depth,
        output_view,
        output_depth_view,
        mut textures,
        mut stages,
        camera_outputs,
    } = prepared;
    if let Some(particle) = particle {
        (color, depth) = particle
            .append_gpu_to_graph(&mut graph, color, depth)
            .map_err(|_| DisplayTransformGpuError::ParticleGraph)?;
        stages.manual.particle_passes = particle.fabric_gpu_frame().map_or(0, |fabric| {
            u32::try_from(fabric.draws().len()).expect("particle passes fit graph capacity")
        });
        stages.particle_passes = stages.manual.particle_passes;
        stages.manual.present_source = color;
    }
    for (output, (camera, camera_stages)) in camera_outputs.into_iter().zip(&mut stages.cameras) {
        debug_assert_eq!(output.camera, *camera);
        let mut local_color = output.color;
        let mut local_depth = output.depth;
        if let Some(particle) = particle {
            (local_color, local_depth) = particle
                .append_view_gpu_to_graph(output.camera, &mut graph, local_color, local_depth)
                .map_err(|_| DisplayTransformGpuError::ParticleGraph)?;
            camera_stages.particle_passes = particle
                .fabric_gpu_views()
                .find(|(entity, _)| *entity == output.camera)
                .map_or(0, |(_, fabric)| {
                    u32::try_from(fabric.draws().len()).expect("particle passes fit graph capacity")
                });
            stages.particle_passes += camera_stages.particle_passes;
        }
        camera_stages.present_source = local_color;
        textures.insert(output.color_view, local_color);
        textures.insert(output.depth_view, local_depth);
        let [x, y, width, height] = output.viewport;
        graph.add_transfer_pass(
            "prism-surface-view-copy",
            PassFlags::SIDE_EFFECT,
            |builder| {
                let source = builder.copy_src_texture(local_color);
                color = builder.copy_dst_texture(color);
                let destination = color;
                move |context| {
                    context.copy_texture_to_texture(
                        source,
                        0,
                        [0, 0, 0],
                        destination,
                        0,
                        [x, y, 0],
                        prism_render_driver::Extent3d::new_2d(width, height),
                    );
                }
            },
        );
    }
    textures.insert(output_view, color);
    textures.insert(output_depth_view, depth);
    if let Some(ember) = ember {
        ember
            .append_to_graph(&mut graph, frame_index, external_imports, &textures)
            .map_err(|_| DisplayTransformGpuError::EmberGraph)?;
        stages.ember_compute_passes =
            u32::try_from(ember.compute_pass_count()).expect("Ember passes fit graph capacity");
    }
    stages.present_source = color;
    stages.assert_contract();
    graph.blackboard_mut().set(stages);
    Ok((graph, color))
}

fn reserve_target(
    target: &RenderTarget,
    textures: &mut BTreeSet<TextureId>,
) -> Result<(), SurfaceViewGraphError> {
    for texture in [
        target.texture(),
        target.position_x_texture(),
        target.position_y_texture(),
        target.position_z_texture(),
        target.normal_texture(),
        target.albedo_texture(),
        target.surface_id_texture(),
        target.depth_texture(),
    ] {
        if !textures.insert(texture) {
            return Err(SurfaceViewGraphError::AliasedTarget(texture));
        }
    }
    Ok(())
}

fn validate_lighting(
    camera: Option<Entity>,
    pass: Option<LightingGpuPass>,
    first: &mut Option<LightingGpuPass>,
) -> Result<(), SurfaceViewGraphError> {
    let Some(pass) = pass else {
        return Ok(());
    };
    if pass.composite_target_bind_group.is_none() {
        return Err(SurfaceViewGraphError::MissingCompositeBinding(camera));
    }
    if let Some(shared) = *first {
        if [
            pass.gi_output,
            pass.params,
            pass.directionals,
            pass.punctuals,
            pass.areas,
            pass.candidates,
            pass.surfaces,
            pass.radiance,
        ] != [
            shared.gi_output,
            shared.params,
            shared.directionals,
            shared.punctuals,
            shared.areas,
            shared.candidates,
            shared.surfaces,
            shared.radiance,
        ] || pass.run_gi != shared.run_gi
            || pass.run_direct != shared.run_direct
            || pass.candidate_source != shared.candidate_source
            || pass.gi_pipeline != shared.gi_pipeline
            || pass.gi_bind_group != shared.gi_bind_group
            || pass.composite_pipeline != shared.composite_pipeline
            || pass.composite_static_layout != shared.composite_static_layout
            || pass.composite_target_layout != shared.composite_target_layout
            || pass.composite_bind_group != shared.composite_bind_group
            || pass.candidate_surface_count != shared.candidate_surface_count
            || pass.candidate_rays_per_surface != shared.candidate_rays_per_surface
            || pass.workgroups != shared.workgroups
        {
            return Err(SurfaceViewGraphError::InconsistentLighting(camera));
        }
    } else {
        *first = Some(pass);
    }
    Ok(())
}

fn validate_sampled_texture_imports(
    queues: &RenderQueues,
    functions: &DrawFunctions,
    external: &FrameGraphTextureImports,
) -> Result<(), SurfaceViewGraphError> {
    for phase in RenderPhase::ORDER {
        for (index, item) in queues.phase(phase).iter().enumerate() {
            for binding in item
                .bind_groups(functions)
                .iter()
                .chain(queues.dynamic_bind_groups(phase, index).iter())
            {
                for &view in &binding.sampled_textures {
                    if !external.iter().any(|import| import.view == view) {
                        return Err(SurfaceViewGraphError::MissingExternalTextureImport(view));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
