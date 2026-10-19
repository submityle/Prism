//! Render-world ownership of the unified lighting plan.

use bytemuck::{Pod, Zeroable};
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_render_architecture::lighting::contract::{
    BudgetCompiler, ChangeFields, ChangeJournal, LightingPlan, PlanCaps, RecipeExecution,
    SceneStats, Tier,
};

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
/// one surface sample for the compute slice, and appends invalidation fields;
/// extraction copies only this immutable snapshot into the render world.
#[derive(Clone, Debug, Resource)]
#[allow(missing_docs)]
pub struct LightingSceneInputs {
    pub stats: SceneStats,
    pub journal: ChangeJournal,
    pub directionals: Vec<LightingDirectionalInput>,
    pub punctuals: Vec<LightingPunctualInput>,
    pub surface: LightingSurfaceInput,
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
            surface: LightingSurfaceInput::default(),
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
            surface: LightingSurfaceInput::default(),
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
    pub surface: LightingSurfaceInput,
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
        surface: inputs.surface,
    });
}

/// Compiles the single plan once for each extracted frame.
pub(crate) fn compile_lighting_plan(
    record: Res<RenderRecord>,
    inputs: Res<ExtractedLightingInputs>,
    mut state: ResMut<LightingPlanState>,
    mut stats: ResMut<RenderFrameStats>,
) {
    if state.compiled_frame == record.frame_index {
        return;
    }
    let previous_epoch = state.journal_snapshot.epoch;
    state.scene_stats = inputs.stats;
    state.journal_snapshot = inputs.journal_snapshot;
    state.plan =
        BudgetCompiler::compile(state.scene_stats, state.plan.budget, state.caps, state.tier);
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
