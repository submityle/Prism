//! Render-world ownership of the unified lighting plan.

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_render_architecture::lighting::contract::{
    BudgetCompiler, FrameResourceBudget, LightingPlan, PlanCaps, SceneStats, Tier,
};

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
}

impl Default for LightingPlanState {
    fn default() -> Self {
        Self {
            plan: BudgetCompiler::compile(
                SceneStats::default(),
                FrameResourceBudget {
                    gpu_ms: 16.666,
                    bandwidth: 512 * 1024 * 1024,
                    resident_bytes: 512 * 1024 * 1024,
                    io_ops: 4096,
                    shading_rate_budget: 1.0,
                },
                PlanCaps {
                    ray_tracing: false,
                    virtual_shadow: false,
                    offline: false,
                    deterministic_npr: false,
                },
                Tier::Balanced,
            ),
            compiled_frame: 0,
            compile_count: 0,
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
    state.plan = BudgetCompiler::compile(
        SceneStats::default(),
        state.plan.budget,
        PlanCaps {
            ray_tracing: false,
            virtual_shadow: false,
            offline: false,
            deterministic_npr: false,
        },
        state.plan.tier,
    );
    state.compiled_frame = record.frame_index;
    state.compile_count = state.compile_count.saturating_add(1);
    stats.lighting_plan_compiles = state.compile_count;
    stats.lighting_recipes_selected = state.plan.recipes.len() as u64;
    stats.lighting_recipes_rejected = state.plan.rejected.len() as u64;
}
