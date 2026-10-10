//! WESL link coverage for the shared expensive-lobe sample-budget module.
//!
//! Like the thin-film and specular-AA link tests, this builds a tiny compute
//! entry point that references every exported constant, the `LobeSampleBudget`
//! struct, and every validator, so import visibility, signatures, and type
//! checking are exercised rather than merely parsing the source. Numeric
//! fidelity is guaranteed by exact mirroring of the CPU golden
//! (`prism_render_shading::quality`): the floors, the odd-`>= 3` tap rule, and
//! the clamp/total helpers all follow the same operation order and branches.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("lobe-budget shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::lobe_budget::{
    LB_MIN_SSS_TAPS, LB_MIN_PREINTEGRATION_STEPS, LB_MIN_TRANSLUCENCY_SAMPLES,
    LB_MIN_MULTISCATTER_DIR, LB_MIN_MULTISCATTER_MARCH,
    LobeSampleBudget,
    lb_clamp_min,
    lb_force_odd_at_least_three,
    lb_clamped,
    lb_total_samples,
};

@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn lobe_budget_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    var raw: LobeSampleBudget;
    raw.sss_separable_taps = 4u;
    raw.sss_preintegration_steps = 2u;
    raw.translucency_profile_samples = 1u;
    raw.multiscatter_dir_samples = 1u;
    raw.multiscatter_march_samples = 1u;

    let fixed = lb_clamped(raw);

    output[0] = fixed.sss_separable_taps;
    output[1] = fixed.sss_preintegration_steps;
    output[2] = fixed.translucency_profile_samples;
    output[3] = fixed.multiscatter_dir_samples;
    output[4] = fixed.multiscatter_march_samples;
    output[5] = lb_total_samples(fixed);
    output[6] = lb_force_odd_at_least_three(8u);
    output[7] = lb_clamp_min(0u, LB_MIN_PREINTEGRATION_STEPS);
    output[8] = LB_MIN_SSS_TAPS;
    output[9] = LB_MIN_TRANSLUCENCY_SAMPLES;
    output[10] = LB_MIN_MULTISCATTER_DIR;
    output[11] = LB_MIN_MULTISCATTER_MARCH;
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported item so none can disappear behind dead-code elimination.
#[test]
fn lobe_budget_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_4c42_5544_4745_5f00_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/lobe_budget.wesl"),
            "embedded://prism_render_scene/shaders/lobe_budget.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_4c42_5544_4745_5f00_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/lobe_budget/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("lobe_budget.wesl failed to compile/link every helper: {error}")
    });
}
