//! WESL compile coverage for `shaders/transparent.wesl`.
//!
//! No GPU is available in CI, so - exactly like the opaque path - the shader is
//! stripped of its `import` block (the imported symbols are stubbed instead) and
//! compiled through [`bevy_shader::ShaderCache`] for every combination of the
//! optional vertex shader defs (`VERTEX_POSITIONS_COMPRESSED`, `VERTEX_NORMALS`,
//! `VERTEX_UVS`). This proves the vertex/fragment source stays syntactically and
//! type-valid under specialization without needing a device.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &bevy_shader::ValidateShader,
) -> Result<String, bevy_shader::ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("transparent shader is WESL"),
    }
}

#[test]
fn transparent_wesl_compiles_for_every_specialization() {
    let shader_id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5452_414e_5350_4152_454e),
    };
    let mut cache = ShaderCache::new((), load_source);
    let source = include_str!("../../shaders/transparent.wesl");
    // Drop the `import` block; everything the imports provide is stubbed below.
    let source = source[source.find("struct GpuSceneInstance").unwrap()..].replace(
        "bevy_render::utils::decompress_vertex_position",
        "decompress_vertex_position",
    );
    let stubs = r#"
struct TestView { world_position: vec3<f32> }
var<private> view: TestView;
fn affine3_to_square(value: mat3x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn position_world_to_clip(position: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(position, 1.0);
}
fn position_world_to_view(position: vec3<f32>) -> vec3<f32> {
    return position;
}
fn decompress_vertex_position(
    position: vec4<f32>,
    center: vec3<f32>,
    half_extents: vec3<f32>,
) -> vec3<f32> {
    return position.xyz;
}
struct OitTargets {
    accum: vec4<f32>,
    revealage: f32,
}
fn oit_accumulate(color: vec3<f32>, alpha: f32, view_depth: f32) -> OitTargets {
    var targets: OitTargets;
    targets.accum = vec4<f32>(color * alpha, alpha);
    targets.revealage = 1.0 - alpha;
    return targets;
}
"#;
    cache.set_shader(
        shader_id,
        Shader::from_wesl(
            format!("{stubs}{source}"),
            "shaders/prism_transparent.wesl",
        ),
    );
    for compressed in [false, true] {
        for normals in [false, true] {
            for uvs in [false, true] {
                let mut defs = Vec::new();
                if compressed {
                    defs.push("VERTEX_POSITIONS_COMPRESSED".into());
                }
                if normals {
                    defs.push("VERTEX_NORMALS".into());
                }
                if uvs {
                    defs.push("VERTEX_UVS".into());
                }
                cache
                    .get(0, shader_id, &defs)
                    .unwrap_or_else(|error| panic!("transparent specialization failed: {error}"));
            }
        }
    }
}
