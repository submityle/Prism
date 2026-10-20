//! Render-world ownership of the unified lighting plan.

use bytemuck::{Pod, Zeroable};
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_render_architecture::lighting::contract::{
    BudgetCompiler, ChangeFields, ChangeJournal, GiCandidateAvailability, GiCandidateRejection,
    LightingPlan, PlanCaps, RecipeExecution, RecipeId, RejectionExplanation, SceneStats, Tier,
};
use prism_render_architecture::lighting::{DebugMeta, DebugView};
use prism_render_architecture::ray_scene::GpuBvhBuffers;
use prism_render_driver::RenderDevice;
use std::sync::Arc;

use crate::config::RenderAppConfig;
use crate::record::RenderRecord;
use crate::stats::RenderFrameStats;

/// The frame's compiled lighting/GI decision and its explainable rejection set.
#[derive(Clone, Debug, Resource)]
pub struct LightingPlanState {
    /// The current frame's immutable compiled plan.
    pub plan: LightingPlan,
    /// Extracted frame index for which `plan` was compiled.
    pub compiled_frame: u64,
    /// Number of successful plan compilations in this render world.
    pub compile_count: u64,
    /// Scene statistics retained for deterministic recompilation.
    pub scene_stats: SceneStats,
    /// Capability policy retained for deterministic recompilation.
    pub caps: PlanCaps,
    /// Quality tier retained for deterministic recompilation.
    pub tier: Tier,
    /// Stable recipe execution declarations consumed by the frame graph.
    pub executions: Vec<RecipeExecution>,
    /// Explain output retained with the frame so diagnostics do not recompile.
    pub explain: String,
    /// Last extracted scene-change checkpoint used by this plan.
    pub journal_snapshot: prism_render_architecture::lighting::contract::JournalSnapshot,
}

/// Render-world debug snapshot consumed by diagnostic views.
///
/// The metadata, view registrations, rejection explanations, and EXPLAIN text
/// are published as one snapshot so consumers cannot observe mixed-frame data.
#[derive(Clone, Debug, Default, PartialEq, Resource)]
pub struct LightingDebugViewState {
    /// Frame for which this diagnostic snapshot was compiled.
    pub compiled_frame: u64,
    /// Plan compilation count paired with [`compiled_frame`].
    pub plan_compile_count: u64,
    /// Selected recipes and their contract-driven display metadata.
    pub selected_recipes: Vec<LightingDebugRecipe>,
    /// Structured reasons for every recipe rejected by the plan.
    pub rejection_explanations: Vec<RejectionExplanation>,
    /// The planner explanation retained for consumers without recompilation.
    pub explain: String,
}

/// One selected recipe exposed to the DebugView consumer chain.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingDebugRecipe {
    /// Recipe identity from the lighting contract.
    pub recipe: RecipeId,
    /// Display semantics declared by the recipe contract.
    pub meta: DebugMeta,
    /// Related views registered by the recipe metadata.
    pub views: Vec<DebugView>,
}

impl LightingDebugViewState {
    /// Creates a debug snapshot immediately consistent with a plan.
    #[must_use]
    pub fn from_plan(state: &LightingPlanState) -> Self {
        let mut snapshot = Self::default();
        snapshot.update_from_plan(state);
        snapshot
    }

    /// Replaces the snapshot from one compiled plan without recompiling it.
    ///
    /// Returns whether the paired frame/compile identity changed.
    pub(crate) fn update_from_plan(&mut self, state: &LightingPlanState) -> bool {
        let changed = self.compiled_frame != state.compiled_frame
            || self.plan_compile_count != state.compile_count;
        self.compiled_frame = state.compiled_frame;
        self.plan_compile_count = state.compile_count;
        self.selected_recipes = state
            .plan
            .debug_metadata()
            .into_iter()
            .map(|(recipe, meta)| LightingDebugRecipe {
                recipe,
                meta,
                views: meta.combine_with.to_vec(),
            })
            .collect();
        self.rejection_explanations = state.plan.rejection_explanations();
        self.explain.clone_from(&state.explain);
        changed
    }
}

/// One analytic directional light consumed by the deterministic direct-light
/// recipe. Values are linear and already expressed in world space.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingDirectionalInput {
    pub direction_to_light: [f32; 3],
    pub visibility: f32,
    pub illuminance: [f32; 3],
    pub _padding: f32,
}

impl Default for LightingDirectionalInput {
    fn default() -> Self {
        Self {
            direction_to_light: [0.0, 1.0, 0.0],
            visibility: 1.0,
            illuminance: [0.0; 3],
            _padding: 0.0,
        }
    }
}

/// One analytic point or spot light consumed by the deterministic direct-light
/// recipe. `spot_scale == 0` denotes an omnidirectional point light.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingPunctualInput {
    pub position: [f32; 3],
    pub range: f32,
    pub intensity: [f32; 3],
    pub spot_scale: f32,
    pub direction: [f32; 3],
    pub spot_offset: f32,
    pub visibility: f32,
    pub _padding: [f32; 7],
}

impl Default for LightingPunctualInput {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            range: 0.0,
            intensity: [0.0; 3],
            spot_scale: 0.0,
            direction: [0.0, 0.0, -1.0],
            spot_offset: 1.0,
            visibility: 1.0,
            _padding: [0.0; 7],
        }
    }
}

/// The surface sample used by the direct-light compute recipe.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingSurfaceInput {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub albedo: [f32; 3],
    pub ambient: [f32; 3],
}

/// One deterministic indirect-light candidate for the initial GI reservoir.
/// `source_pdf` and `target_pdf` use solid-angle measure. `target_pdf` is the
/// non-normalized luminance of the Lambertian directional integrand at the
/// owning surface, including `albedo / PI`, the incident cosine, and
/// visibility. Misses remain proposals with positive `source_pdf` and zero
/// `target_pdf`, so the RIS denominator counts every attempted sample.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingGiCandidateInput {
    pub position: [f32; 3],
    pub source_pdf: f32,
    pub normal: [f32; 3],
    pub target_pdf: f32,
    pub radiance: [f32; 3],
    pub selection_u01: f32,
    pub visibility: f32,
    pub _padding: [f32; 3],
}

impl Default for LightingGiCandidateInput {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, 0.0],
            source_pdf: 1.0,
            normal: [0.0, 1.0, 0.0],
            target_pdf: 0.0,
            radiance: [0.0; 3],
            selection_u01: 0.5,
            visibility: 1.0,
            _padding: [0.0; 3],
        }
    }
}

impl Default for LightingSurfaceInput {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            albedo: [1.0; 3],
            ambient: [0.0; 3],
        }
    }
}

/// Main-world lighting inputs. The host publishes the actual analytic lights,
/// a deterministic batch of surface samples, and appends invalidation fields;
/// extraction copies only this immutable snapshot into the render world.
#[derive(Clone, Debug, Resource)]
#[allow(missing_docs)]
pub struct LightingSceneInputs {
    pub stats: SceneStats,
    pub journal: ChangeJournal,
    pub directionals: Vec<LightingDirectionalInput>,
    pub punctuals: Vec<LightingPunctualInput>,
    /// Per-surface GI candidates. Each surface owns its own ordered RIS batch;
    /// candidates are never implicitly broadcast across unrelated surfaces.
    pub gi_candidates: Vec<Vec<LightingGiCandidateInput>>,
    pub surfaces: Vec<LightingSurfaceInput>,
}

/// Immutable outgoing radiance authored for one stable BVH primitive.
///
/// `primitive_id` is the stable ID stored in the packed BVH triangle record;
/// it is deliberately not a packed-array index because BVH construction may
/// reorder triangles into leaf-contiguous order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightingTriangleRadiance {
    /// Stable primitive ID from the corresponding packed BVH triangle.
    pub primitive_id: u32,
    /// Linear, non-negative outgoing RGB radiance.
    pub rgb: [f32; 3],
}

/// Optional host-owned packed scene used by the software-BVH GI trace pass.
/// Surface IDs are stable identities aligned one-to-one with `surfaces` in
/// [`LightingSceneInputs`]; malformed alignment is rejected by host preparation.
#[derive(Clone, Debug, Resource)]
pub struct LightingTraceSceneInput {
    /// Packed BVH in the architecture crate's frozen GPU ABI.
    pub bvh: Arc<GpuBvhBuffers>,
    /// Stable identities for the current lighting surfaces, in matching order.
    pub surface_ids: Vec<u64>,
    /// Deterministic cosine rays generated for each surface each frame.
    pub rays_per_surface: u32,
    /// Positive minimum trace distance used for self-intersection rejection.
    pub ray_epsilon: f32,
    /// Maximum world-space ray distance.
    pub ray_tmax: f32,
    /// Per-primitive outgoing radiance for trace-hit GI evaluation. Entries
    /// may arrive in any order, but IDs must exactly cover the packed BVH.
    /// Sharing this immutable table and advancing its revision only when
    /// material/texture evaluation changes lets the renderer avoid per-frame
    /// allocation and upload.
    pub outgoing_radiance: Arc<[LightingTriangleRadiance]>,
    /// Monotonic content revision for the outgoing-radiance table.
    pub radiance_revision: u64,
    packed_radiance: Arc<std::sync::Mutex<Option<PackedRadianceCache>>>,
}

#[derive(Debug)]
struct PackedRadianceCache {
    // Retain owners, not raw pointer keys: this prevents allocator ABA and
    // forces Arc::make_mut to copy before changing a cached input snapshot.
    bvh: Arc<GpuBvhBuffers>,
    radiance: Arc<[LightingTriangleRadiance]>,
    revision: u64,
    result: Result<Arc<[[f32; 4]]>, crate::lighting_trace_gpu::TracePreparationError>,
}

impl LightingTraceSceneInput {
    /// Creates an immutable trace snapshot and its revision-scoped upload cache.
    #[must_use]
    pub fn new(
        bvh: Arc<GpuBvhBuffers>,
        surface_ids: Vec<u64>,
        rays_per_surface: u32,
        ray_epsilon: f32,
        ray_tmax: f32,
        outgoing_radiance: Arc<[LightingTriangleRadiance]>,
        radiance_revision: u64,
    ) -> Self {
        Self {
            bvh,
            surface_ids,
            rays_per_surface,
            ray_epsilon,
            ray_tmax,
            outgoing_radiance,
            radiance_revision,
            packed_radiance: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Produces GPU upload order matching packed BVH triangle indices.
    pub(crate) fn packed_outgoing_radiance(
        &self,
    ) -> Result<Arc<[[f32; 4]]>, crate::lighting_trace_gpu::TracePreparationError> {
        let mut cache = self
            .packed_radiance
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.as_ref() {
            if Arc::ptr_eq(&cached.bvh, &self.bvh)
                && Arc::ptr_eq(&cached.radiance, &self.outgoing_radiance)
                && cached.revision == self.radiance_revision
            {
                return cached.result.clone();
            }
        }
        let result = self.build_packed_outgoing_radiance().map(Arc::from);
        *cache = Some(PackedRadianceCache {
            bvh: Arc::clone(&self.bvh),
            radiance: Arc::clone(&self.outgoing_radiance),
            revision: self.radiance_revision,
            result: result.clone(),
        });
        result
    }

    fn build_packed_outgoing_radiance(
        &self,
    ) -> Result<Vec<[f32; 4]>, crate::lighting_trace_gpu::TracePreparationError> {
        crate::lighting_trace_gpu::BorrowedPackedTraceScene::from(self.bvh.as_ref())
            .validate()
            .map_err(crate::lighting_trace_gpu::TracePreparationError::Trace)?;
        let triangles = self.bvh.triangles.chunks_exact(12);
        let triangle_count = self.bvh.triangles.len() / 12;
        if self.outgoing_radiance.len() != triangle_count {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::RadianceCountMismatch {
                    triangles: triangle_count,
                    radiance: self.outgoing_radiance.len(),
                },
            );
        }
        let mut sorted = self.outgoing_radiance.iter().copied().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|entry| entry.primitive_id);
        if let Some(pair) = sorted
            .windows(2)
            .find(|pair| pair[0].primitive_id == pair[1].primitive_id)
        {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::DuplicateRadiancePrimitive(
                    pair[0].primitive_id,
                ),
            );
        }
        let mut packed_ids = triangles
            .clone()
            .map(|triangle| triangle[9])
            .collect::<Vec<_>>();
        packed_ids.sort_unstable();
        if let Some(pair) = packed_ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::DuplicatePackedPrimitive(pair[0]),
            );
        }
        if let Some((radiance_id, packed_id)) = sorted
            .iter()
            .map(|entry| entry.primitive_id)
            .zip(packed_ids.iter().copied())
            .find(|(radiance_id, packed_id)| radiance_id != packed_id)
        {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::RadiancePrimitiveSetMismatch {
                    radiance_id,
                    packed_id,
                },
            );
        }
        let mut packed = Vec::with_capacity(triangle_count);
        for (packed_index, triangle) in triangles.enumerate() {
            let primitive_id = triangle[9];
            let entry = sorted
                .binary_search_by_key(&primitive_id, |entry| entry.primitive_id)
                .ok()
                .map(|index| sorted[index])
                .ok_or(
                    crate::lighting_trace_gpu::TracePreparationError::MissingRadiancePrimitive {
                        packed_index,
                        primitive_id,
                    },
                )?;
            if entry
                .rgb
                .iter()
                .any(|channel| !channel.is_finite() || *channel < 0.0)
            {
                return Err(
                    crate::lighting_trace_gpu::TracePreparationError::InvalidRadiance(packed_index),
                );
            }
            packed.push([entry.rgb[0], entry.rgb[1], entry.rgb[2], 0.0]);
        }
        Ok(packed)
    }
}

impl LightingSceneInputs {
    /// Creates inputs from the host's configured scene estimate.
    #[must_use]
    pub fn with_stats(stats: SceneStats) -> Self {
        Self {
            stats,
            journal: ChangeJournal::new(),
            directionals: Vec::new(),
            punctuals: Vec::new(),
            gi_candidates: Vec::new(),
            surfaces: vec![LightingSurfaceInput::default()],
        }
    }
}

impl Default for LightingSceneInputs {
    fn default() -> Self {
        Self {
            stats: SceneStats::default(),
            journal: ChangeJournal::new(),
            directionals: Vec::new(),
            punctuals: Vec::new(),
            gi_candidates: Vec::new(),
            surfaces: vec![LightingSurfaceInput::default()],
        }
    }
}

impl LightingSceneInputs {
    /// Appends a field-level invalidation record for the next extracted frame.
    pub fn mark_changed(&mut self, frame: u64, fields: ChangeFields) {
        self.journal.append(frame, fields);
    }
}

/// Render-world copy of the current lighting inputs.
#[derive(Clone, Debug, Default, Resource)]
#[allow(missing_docs)]
pub struct ExtractedLightingInputs {
    pub stats: SceneStats,
    pub journal_snapshot: prism_render_architecture::lighting::contract::JournalSnapshot,
    pub directionals: Vec<LightingDirectionalInput>,
    pub punctuals: Vec<LightingPunctualInput>,
    /// Per-surface GI candidates, aligned with `surfaces` when present.
    pub gi_candidates: Vec<Vec<LightingGiCandidateInput>>,
    pub surfaces: Vec<LightingSurfaceInput>,
    /// Optional packed scene used to generate and trace this frame's GI rays.
    pub trace_scene: Option<LightingTraceSceneInput>,
}

impl ExtractedLightingInputs {
    /// Validates the candidate producer against this frame and the actual
    /// device. A supplied trace scene is an explicit source request: invalid
    /// trace data never silently selects the host-candidate route.
    pub(crate) fn candidate_availability(
        &self,
        capabilities: &prism_render_driver::DeviceCapabilities,
    ) -> GiCandidateAvailability {
        use crate::lighting_gpu::{
            GI_CANDIDATES_PER_SURFACE, GI_CANDIDATE_BUFFER_BYTES, LIGHTING_PARAMS_BYTES,
            LIGHTING_SURFACE_BUFFER_BYTES, MAX_LIGHTING_SURFACES, OUTPUT_BYTES,
            TRACE_RADIANCE_BYTES,
        };
        use GiCandidateAvailability::{HostProvided, Missing, Rejected, SoftwareBvhGpu};
        use GiCandidateRejection::{DeviceLimits, InvalidInputs, TraceBackendUnavailable};

        if self.surfaces.is_empty() {
            return Missing;
        }
        if self.surfaces.len() > MAX_LIGHTING_SURFACES
            || self.surfaces.iter().any(|surface| {
                let normal_len = dot(surface.normal, surface.normal);
                surface.position.iter().any(|value| !value.is_finite())
                    || surface.normal.iter().any(|value| !value.is_finite())
                    || !normal_len.is_finite()
                    || normal_len <= 1.0e-20
                    || surface
                        .albedo
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Rejected(InvalidInputs);
        }
        let limits = capabilities.limits;
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_bind_groups == 0
            || limits.max_bindings_per_bind_group < 7
        {
            return Rejected(TraceBackendUnavailable);
        }
        if [
            GI_CANDIDATE_BUFFER_BYTES,
            LIGHTING_SURFACE_BUFFER_BYTES,
            OUTPUT_BYTES,
        ]
        .iter()
        .any(|bytes| *bytes > limits.max_storage_buffer_binding_size)
            || LIGHTING_PARAMS_BYTES > limits.max_uniform_buffer_binding_size
            || (self.surfaces.len() as u32).div_ceil(64)
                > limits.max_compute_workgroups_per_dimension
        {
            return Rejected(DeviceLimits);
        }
        if let Some(trace) = self.trace_scene.as_ref() {
            if trace.surface_ids.len() != self.surfaces.len()
                || trace.rays_per_surface == 0
                || trace.rays_per_surface as usize > GI_CANDIDATES_PER_SURFACE
                || !trace.ray_epsilon.is_finite()
                || trace.ray_epsilon <= 0.0
                || !trace.ray_tmax.is_finite()
                || trace.ray_tmax <= trace.ray_epsilon
                || trace.packed_outgoing_radiance().is_err()
            {
                return Rejected(InvalidInputs);
            }
            // Bounded stack scratch avoids a per-frame allocation for ID
            // uniqueness checks; surface validation bounds the copied prefix.
            let mut ids = [0_u64; MAX_LIGHTING_SURFACES];
            let ids = &mut ids[..trace.surface_ids.len()];
            ids.copy_from_slice(&trace.surface_ids);
            ids.sort_unstable();
            if ids.windows(2).any(|pair| pair[0] == pair[1]) {
                return Rejected(InvalidInputs);
            }
            let ray_count = self.surfaces.len() as u64 * u64::from(trace.rays_per_surface);
            let node_bytes = trace.bvh.nodes.len() as u64 * 4;
            let triangle_bytes = trace.bvh.triangles.len() as u64 * 4;
            if [
                node_bytes,
                triangle_bytes,
                ray_count * 48,
                crate::lighting_trace_gpu::HOST_TRACE_OUTPUT_BYTES,
                TRACE_RADIANCE_BYTES,
            ]
            .iter()
            .any(|bytes| *bytes > limits.max_storage_buffer_binding_size)
                || ray_count.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            {
                return Rejected(DeviceLimits);
            }
            return SoftwareBvhGpu;
        }
        if self.gi_candidates.is_empty() {
            return Missing;
        }
        if self.gi_candidates.len() != self.surfaces.len()
            || self.gi_candidates.iter().any(|batch| {
                batch.len() > GI_CANDIDATES_PER_SURFACE
                    || batch.iter().any(|candidate| {
                        candidate
                            .position
                            .iter()
                            .chain(&candidate.normal)
                            .any(|value| !value.is_finite())
                            || candidate
                                .radiance
                                .iter()
                                .any(|value| !value.is_finite() || *value < 0.0)
                            || !candidate.source_pdf.is_finite()
                            || candidate.source_pdf <= 0.0
                            || !candidate.target_pdf.is_finite()
                            || candidate.target_pdf < 0.0
                            || !candidate.selection_u01.is_finite()
                            || !(0.0..=1.0).contains(&candidate.selection_u01)
                            || !candidate.visibility.is_finite()
                            || !(0.0..=1.0).contains(&candidate.visibility)
                    })
            })
        {
            return Rejected(InvalidInputs);
        }
        HostProvided
    }
}

impl Default for LightingPlanState {
    fn default() -> Self {
        Self::from_config(RenderAppConfig::DEFAULT)
    }
}

impl LightingPlanState {
    /// Builds a render-world plan state from host configuration.
    #[must_use]
    pub fn from_config(config: RenderAppConfig) -> Self {
        let plan = BudgetCompiler::compile(
            config.lighting_scene_stats,
            config.lighting_budget,
            config.lighting_caps,
            config.lighting_tier,
            GiCandidateAvailability::Missing,
        );
        Self {
            executions: plan.executions(),
            explain: plan.explain(),
            plan,
            compiled_frame: 0,
            compile_count: 0,
            scene_stats: config.lighting_scene_stats,
            caps: config.lighting_caps,
            tier: config.lighting_tier,
            journal_snapshot: Default::default(),
        }
    }
}

/// Copies the host-owned lighting snapshot across the one-way app boundary.
pub(crate) fn extract_lighting_inputs(main: &mut World, render: &mut World) {
    let inputs = main
        .get_resource::<LightingSceneInputs>()
        .cloned()
        .unwrap_or_default();
    let mut stats = inputs.stats;
    if !inputs.directionals.is_empty() || !inputs.punctuals.is_empty() {
        stats.lights = (inputs.directionals.len() + inputs.punctuals.len()) as u32;
    }
    render.insert_resource(ExtractedLightingInputs {
        stats,
        journal_snapshot: inputs.journal.snapshot(),
        directionals: inputs.directionals,
        punctuals: inputs.punctuals,
        gi_candidates: inputs.gi_candidates,
        surfaces: inputs.surfaces,
        trace_scene: main.get_resource::<LightingTraceSceneInput>().cloned(),
    });
}

/// Compiles the single plan once for each extracted frame.
pub(crate) fn compile_lighting_plan<I>(
    record: Res<RenderRecord>,
    inputs: Res<ExtractedLightingInputs>,
    runtime: Res<crate::RenderRuntimeResource<I>>,
    mut state: ResMut<LightingPlanState>,
    mut stats: ResMut<RenderFrameStats>,
) where
    I: prism_render_driver::Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + prism_render_driver::FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if state.compiled_frame == record.frame_index {
        return;
    }
    let previous_epoch = state.journal_snapshot.epoch;
    state.scene_stats = inputs.stats;
    state.journal_snapshot = inputs.journal_snapshot;
    let availability = inputs.candidate_availability(runtime.runtime().device().capabilities());
    state.plan = BudgetCompiler::compile(
        state.scene_stats,
        state.plan.budget,
        state.caps,
        state.tier,
        availability,
    );
    state.executions = state.plan.executions();
    state.explain = state.plan.explain();
    state.compiled_frame = record.frame_index;
    state.compile_count = state.compile_count.saturating_add(1);
    stats.lighting_plan_compiles = state.compile_count;
    stats.lighting_recipes_selected = state.plan.recipes.len() as u64;
    stats.lighting_recipes_rejected = state.plan.rejected.len() as u64;
    stats.lighting_executions_planned = state.executions.len() as u64;
    stats.lighting_cache_invalidations =
        changed_epoch_fields(previous_epoch, state.journal_snapshot.epoch) as u64;
}

/// Synchronizes plan diagnostics after compilation and before queue/execution.
pub(crate) fn sync_lighting_debug_view(
    plan: Res<LightingPlanState>,
    mut debug: ResMut<LightingDebugViewState>,
    mut stats: ResMut<RenderFrameStats>,
) {
    if debug.update_from_plan(&plan) {
        stats.lighting_debug_updates = stats.lighting_debug_updates.saturating_add(1);
        stats.lighting_debug_selected_recipes = debug.selected_recipes.len() as u64;
        stats.lighting_debug_rejections = debug.rejection_explanations.len() as u64;
    }
}

fn changed_epoch_fields(
    previous: prism_render_architecture::lighting::contract::FieldEpoch,
    current: prism_render_architecture::lighting::contract::FieldEpoch,
) -> u32 {
    [
        current.geometry != previous.geometry,
        current.material != previous.material,
        current.lighting != previous.lighting,
        current.environment != previous.environment,
        current.layout != previous.layout,
    ]
    .into_iter()
    .filter(|changed| *changed)
    .count() as u32
}

/// Evaluates the deterministic direct-light recipe on the CPU.
///
/// This is the numeric oracle for the GPU kernel. It intentionally covers one
/// stable Lambert contract: ambient irradiance, directional lights, and point
/// or spot lights with inverse-square, range, cone, and visibility terms.
#[must_use]
pub(crate) fn evaluate_deterministic_direct(
    surface: LightingSurfaceInput,
    directionals: &[LightingDirectionalInput],
    punctuals: &[LightingPunctualInput],
) -> [f32; 3] {
    let normal = normalize_or(surface.normal, [0.0, 1.0, 0.0]);
    let mut radiance = mul(surface.albedo, surface.ambient);
    for light in directionals {
        let direction = normalize_or(light.direction_to_light, [0.0, 1.0, 0.0]);
        let cosine = dot(normal, direction).max(0.0) * light.visibility.clamp(0.0, 1.0);
        radiance = add(
            radiance,
            mul(surface.albedo, scale(light.illuminance, cosine)),
        );
    }
    for light in punctuals {
        let to_light = sub(light.position, surface.position);
        let distance_squared = dot(to_light, to_light);
        if !distance_squared.is_finite() || distance_squared <= 0.0 {
            continue;
        }
        let distance = distance_squared.sqrt();
        let direction = scale(to_light, distance.recip());
        let mut attenuation = 1.0 / distance_squared.max(1.0e-4);
        if light.range.is_finite() && light.range > 0.0 {
            let ratio = (distance_squared / (light.range * light.range)).min(1.0);
            let window = (1.0 - ratio * ratio).max(0.0);
            attenuation *= window * window;
        }
        if light.spot_scale != 0.0 {
            let cone = (dot(light.direction, scale(direction, -1.0)) * light.spot_scale
                + light.spot_offset)
                .clamp(0.0, 1.0);
            attenuation *= cone * cone;
        }
        let cosine =
            dot(normal, direction).max(0.0) * attenuation * light.visibility.clamp(0.0, 1.0);
        radiance = add(
            radiance,
            mul(surface.albedo, scale(light.intensity, cosine)),
        );
    }
    radiance
}

/// Evaluates the deterministic initial GI reservoir used by the GPU twin.
/// This is a bounded one-bounce estimator: temporal/spatial reuse and scene
/// traversal remain separate contracts, while the selected sample and its RIS
/// normalization are fully executable and parity-testable here.
#[must_use]
pub(crate) fn evaluate_deterministic_gi(
    surface: LightingSurfaceInput,
    candidates: &[LightingGiCandidateInput],
) -> [f32; 3] {
    const MAX_CANDIDATES: usize = 4;
    let primary_normal = normalize_or(surface.normal, [0.0, 1.0, 0.0]);
    let mut weight_sum = 0.0;
    let proposal_count = candidates.len().min(MAX_CANDIDATES) as u32;
    let mut selected = None;
    for candidate in candidates.iter().take(MAX_CANDIDATES) {
        if !candidate.source_pdf.is_finite()
            || candidate.source_pdf <= 0.0
            || !candidate.target_pdf.is_finite()
            || candidate.target_pdf < 0.0
        {
            continue;
        }
        let weight = candidate.target_pdf / candidate.source_pdf;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        weight_sum += weight;
        if candidate.target_pdf == 0.0 {
            continue;
        }
        let u = candidate.selection_u01.clamp(0.0, 1.0);
        if u * weight_sum < weight {
            selected = Some(*candidate);
        }
    }
    let Some(candidate) = selected else {
        return [0.0; 3];
    };
    if proposal_count == 0 || weight_sum <= 0.0 || candidate.target_pdf <= 0.0 {
        return [0.0; 3];
    }
    let to_sample = sub(candidate.position, surface.position);
    let distance_squared = dot(to_sample, to_sample);
    if !distance_squared.is_finite() || distance_squared <= 1.0e-8 {
        return [0.0; 3];
    }
    let direction = scale(to_sample, distance_squared.sqrt().recip());
    let cosine_primary = dot(primary_normal, direction).max(0.0);
    let visibility = candidate.visibility.clamp(0.0, 1.0);
    let reservoir_weight = weight_sum / (proposal_count as f32 * candidate.target_pdf);
    scale(
        mul(surface.albedo, candidate.radiance),
        cosine_primary / core::f32::consts::PI * visibility * reservoir_weight,
    )
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn scale(a: [f32; 3], value: f32) -> [f32; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if length_squared.is_finite() && length_squared > 1.0e-12 {
        scale(value, length_squared.sqrt().recip())
    } else {
        fallback
    }
}

#[cfg(test)]
mod radiance_tests {
    use super::*;
    use prism_render_architecture::ray_scene::{Bvh, Triangle, TRIANGLE_WORDS};

    #[test]
    fn radiance_table_is_reordered_by_stable_primitive_id() {
        let bvh = Bvh::build(&[
            Triangle::new(
                [-2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.5, 1.0, -1.0],
                73,
            ),
            Triangle::new([1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [1.5, 1.0, -1.0], 11),
        ]);
        let bvh = Arc::new(GpuBvhBuffers::from_bvh(&bvh));
        let input = LightingTraceSceneInput::new(
            Arc::clone(&bvh),
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 11,
                    rgb: [11.0, 1.0, 0.0],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [73.0, 1.0, 0.0],
                },
            ]),
            4,
        );
        let packed = input.packed_outgoing_radiance().unwrap();
        assert!(Arc::ptr_eq(
            &packed,
            &input.clone().packed_outgoing_radiance().unwrap()
        ));
        let mut changed = input.clone();
        Arc::make_mut(&mut changed.outgoing_radiance)[0].rgb[0] = 99.0;
        let changed_packed = changed.packed_outgoing_radiance().unwrap();
        assert!(!Arc::ptr_eq(&packed, &changed_packed));
        assert!(changed_packed.iter().any(|rgb| rgb[0] == 99.0));
        assert_eq!(&*input.packed_outgoing_radiance().unwrap(), &*packed);
        changed.radiance_revision += 1;
        assert!(!Arc::ptr_eq(
            &changed_packed,
            &changed.packed_outgoing_radiance().unwrap()
        ));
        let primitive_ids = bvh
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| triangle[9])
            .collect::<Vec<_>>();
        assert_ne!(
            primitive_ids,
            [11, 73],
            "fixture must exercise BVH reordering"
        );
        assert_eq!(
            packed.iter().map(|entry| entry[0]).collect::<Vec<_>>(),
            primitive_ids
                .iter()
                .map(|primitive_id| *primitive_id as f32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn radiance_table_rejects_duplicate_and_mismatched_primitive_ids() {
        let bvh = Arc::new(GpuBvhBuffers::from_bvh(&Bvh::build(&[
            Triangle::new(
                [-2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.5, 1.0, -1.0],
                73,
            ),
            Triangle::new([1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [1.5, 1.0, -1.0], 12),
        ])));
        let duplicate = LightingTraceSceneInput::new(
            Arc::clone(&bvh),
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [1.0; 3],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [2.0; 3],
                },
            ]),
            0,
        );
        assert_eq!(
            duplicate.packed_outgoing_radiance(),
            Err(crate::lighting_trace_gpu::TracePreparationError::DuplicateRadiancePrimitive(73))
        );
        let mismatch = LightingTraceSceneInput::new(
            bvh,
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 11,
                    rgb: [1.0; 3],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [1.0; 3],
                },
            ]),
            0,
        );
        assert_eq!(
            mismatch.packed_outgoing_radiance(),
            Err(
                crate::lighting_trace_gpu::TracePreparationError::RadiancePrimitiveSetMismatch {
                    radiance_id: 11,
                    packed_id: 12,
                }
            )
        );
    }
}
