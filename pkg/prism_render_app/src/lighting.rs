//! Render-world ownership of the unified lighting plan.

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_render_architecture::lighting::contract::{
    BudgetCompiler, LightingPlan, PlanCaps, SceneStats, Tier,
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
        Self {
            plan: BudgetCompiler::compile(
                config.lighting_scene_stats,
                config.lighting_budget,
                config.lighting_caps,
                config.lighting_tier,
            ),
            compiled_frame: 0,
            compile_count: 0,
            scene_stats: config.lighting_scene_stats,
            caps: config.lighting_caps,
            tier: config.lighting_tier,
        }
    }
}

/// Compiles the single plan once for each extracted frame.
pub(crate) fn compile_lighting_plan(
    record: Res<RenderRecord>,
    mut state: ResMut<LightingPlanState>,
    mut stats: ResMut<RenderFrameStats>,
) {
    if state.compiled_frame == record.frame_index {
        return;
    }
    state.plan =
        BudgetCompiler::compile(state.scene_stats, state.plan.budget, state.caps, state.tier);
    state.compiled_frame = record.frame_index;
    state.compile_count = state.compile_count.saturating_add(1);
    stats.lighting_plan_compiles = state.compile_count;
    stats.lighting_recipes_selected = state.plan.recipes.len() as u64;
    stats.lighting_recipes_rejected = state.plan.rejected.len() as u64;
}
