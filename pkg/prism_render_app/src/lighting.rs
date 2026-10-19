//! Render-world ownership of the unified lighting plan.

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

/// Main-world lighting inputs. The host publishes scene cardinality and appends
/// invalidation fields here; extraction copies only the immutable snapshot into
/// the render world.
#[derive(Clone, Debug, Resource)]
#[allow(missing_docs)]
pub struct LightingSceneInputs {
    pub stats: SceneStats,
    pub journal: ChangeJournal,
}

impl LightingSceneInputs {
    /// Creates inputs from the host's configured scene estimate.
    #[must_use]
    pub fn with_stats(stats: SceneStats) -> Self {
        Self {
            stats,
            journal: ChangeJournal::new(),
        }
    }
}

impl Default for LightingSceneInputs {
    fn default() -> Self {
        Self {
            stats: SceneStats::default(),
            journal: ChangeJournal::new(),
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
#[derive(Clone, Copy, Debug, Default, Resource)]
#[allow(missing_docs)]
pub struct ExtractedLightingInputs {
    pub stats: SceneStats,
    pub journal_snapshot: prism_render_architecture::lighting::contract::JournalSnapshot,
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
    render.insert_resource(ExtractedLightingInputs {
        stats: inputs.stats,
        journal_snapshot: inputs.journal.snapshot(),
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
