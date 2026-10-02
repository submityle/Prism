//! `WESL` compilation coverage for the water subsystem shaders.
//!
//! These tests compile the water `WESL` sources through the same
//! [`ShaderCache`] / `wesl` pipeline the render world uses, so a green result
//! proves every source parses and type-checks exactly as it will on device and
//! that the intra-crate `import prism_render_scene::shaders::...` statements
//! resolve against the crate's embedded-asset module paths
//! (`embedded://prism_render_scene/shaders/<name>.wesl`, byte-identical to what
//! `load_shader_library!` produces from `lib.rs`).
//!
//! The five compute shaders (`water_ocean`, `water_flip`, `water_pbf`,
//! `water_surface`, `water_render_fx`) are self-contained, so compiling them
//! also guards the solver maths - the `Tessendorf` spectrum `IFFT`, the analytic
//! `Gerstner` fan, the `FLIP`/`APIC` `P2G`/`G2P` transfer and pressure
//! projection, the `PBF` density constraint and crest-spray emitter, the `SWE`
//! step, the foam advection, the waterline mask, and the caustics / dispersion /
//! underwater / wetness / coupling render passes - against drift from their
//! `CPU` golden twins in `prism_render_architecture::water`. The
//! `water.wesl` single-layer BSDF lobe additionally resolves the shared
//! `prism_render_scene::shaders::{brdf, lighting}` helpers, matching the render
//! world's `SingleLayerWater` closure.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("water shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles a single self-contained water compute shader, panicking with the
/// shader name on any parse / type-check failure.
fn compile_standalone(source: &'static str, path: &'static str, tag: u128) {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(tag);
    cache.set_shader(id, Shader::from_wesl(source, path));
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
}

/// The spectral inverse-`FFT` (`Tessendorf`) and analytic `Gerstner`
/// superposition kernels, sharing one nine-binding `@group(0)`.
#[test]
fn water_ocean_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_ocean.wesl"),
        "embedded://prism_render_scene/shaders/water_ocean.wesl",
        0x5052_4953_4d5f_5741_5445_524f_4345_4e01,
    );
}

/// The three `FLIP`/`APIC` passes (`P2G` scatter, pressure projection, `G2P`
/// gather) and the screen-space surface reconstruction.
#[test]
fn water_flip_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_flip.wesl"),
        "embedded://prism_render_scene/shaders/water_flip.wesl",
        0x5052_4953_4d5f_5741_5445_5246_4c49_5001,
    );
}

/// The `PBF` density solve and the crest-spray emitter (two distinct
/// `@group(0)` resource sets in one file).
#[test]
fn water_pbf_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_pbf.wesl"),
        "embedded://prism_render_scene/shaders/water_pbf.wesl",
        0x5052_4953_4d5f_5741_5445_5250_4246_0001,
    );
}

/// The `SWE` step (`@group(0)`), foam advection (`@group(1)`) and waterline
/// mask (`@group(2)`), each on its own group index.
#[test]
fn water_surface_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_surface.wesl"),
        "embedded://prism_render_scene/shaders/water_surface.wesl",
        0x5052_4953_4d5f_5741_5445_5253_5552_0001,
    );
}

/// The caustics projection, spectral dispersion refract, underwater volume,
/// wetness step and coupling readback render passes (`@group(0..=4)`).
#[test]
fn water_render_fx_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_render_fx.wesl"),
        "embedded://prism_render_scene/shaders/water_render_fx.wesl",
        0x5052_4953_4d5f_5741_5445_5246_5800_0001,
    );
}

/// The standalone ping-pong butterfly `FFT` (`water_fft_bitrev` /
/// `water_fft_stage` / `water_fft_normalize`), the real-device twin of the
/// `CPU` golden `prism_render_architecture::water::fft::ifft2`.
#[test]
fn water_butterfly_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_butterfly.wesl"),
        "embedded://prism_render_scene/shaders/water_butterfly.wesl",
        0x5052_4953_4d5f_5741_5445_5242_5546_0001,
    );
}

/// The packed spectral butterfly path (`water_spectrum_evolve` /
/// `water_spectrum_assemble`), the O(N log N) production replacement for the
/// direct-sum `water_spectrum_ifft`. Two disjoint-binding entry points in one
/// module; naga type-checks the whole module in a single compile.
#[test]
fn water_spectrum_fft_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_spectrum_fft.wesl"),
        "embedded://prism_render_scene/shaders/water_spectrum_fft.wesl",
        0x5052_4953_4d5f_5741_5445_5246_4654_0001,
    );
}

/// The surface *meshing* compute pass (`water_surface_mesh`): one invocation
/// per surface vertex samples the assembled displacement/normal textures and
/// scatters the result into the four per-vertex storage arrays the raster draw
/// reads. Self-contained (no imports), so a green compile guards the lattice
/// `uv` maths and the four `storage, read_write` output bindings against the
/// `CPU` twin `prism_render_architecture::water::gpu::surface_mesh`.
#[test]
fn water_surface_mesh_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_surface_mesh.wesl"),
        "embedded://prism_render_scene/shaders/water_surface_mesh.wesl",
        0x5052_4953_4d5f_5741_5445_524d_4553_0001,
    );
}

/// Registers `lighting.wesl` and `brdf.wesl` under their canonical module paths
/// and compiles `water.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::{brdf, lighting}::{...}` imports the
/// single-layer water BSDF lobe depends on.
#[test]
fn water_bsdf_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_00a1);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_00a1);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let water = shader_id(0x5052_4953_4d5f_5741_5445_5242_5344_0001);
    cache.set_shader(
        water,
        Shader::from_wesl(
            include_str!("../shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
    );

    cache
        .get(0, water, &[])
        .unwrap_or_else(|error| panic!("water.wesl failed to compile/resolve imports: {error}"));
}

/// The water-surface raster shader (`@vertex` + the four per-frontend
/// `@fragment` entries) must compile and resolve its `brdf`/`water` imports,
/// mirroring the `prism_render_architecture::water::gpu::surface_pass` contract.
#[test]
fn water_surface_raster_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_00a1);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_00a1);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let water = shader_id(0x5052_4953_4d5f_5741_5445_5242_5344_0001);
    cache.set_shader(
        water,
        Shader::from_wesl(
            include_str!("../shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
    );

    let raster = shader_id(0x5052_4953_4d5f_5741_5445_5253_5552_00f1);
    cache.set_shader(
        raster,
        Shader::from_wesl(
            include_str!("../shaders/water_surface_raster.wesl"),
            "embedded://prism_render_scene/shaders/water_surface_raster.wesl",
        ),
    );

    cache.get(0, raster, &[]).unwrap_or_else(|error| {
        panic!("water_surface_raster.wesl failed to compile/resolve imports: {error}")
    });
}

/// Regression guard for the water-surface `SSGI` wiring: the `@group(7)`
/// uniform must be *consumed*, not merely declared. Before this slice the
/// `ssgi_cfg` block was bound but inert (no gather, no call site), so this test
/// pins both the gather definition and its call from `water_ibl`, preventing a
/// refactor from silently regressing the near-field one-bounce back to a dead
/// binding.
#[test]
fn water_surface_raster_wesl_consumes_the_ssgi_binding() {
    let src = include_str!("../shaders/water_surface_raster.wesl");
    assert!(
        src.contains("fn water_ssgi_gather("),
        "water_surface_raster.wesl must define the SSGI gather",
    );
    assert!(
        src.contains("water_ssgi_gather(world_position, bent_normal_view"),
        "water_ibl must call the SSGI gather so the @group(7) uniform is consumed",
    );
}

/// Removes `//` line comments from a `WESL` source so a doc mention of a
/// binding identifier can never be mistaken for a real read when counting its
/// uses. The water shaders carry no block comments, so this is exhaustive.
fn strip_line_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let keep = line.find("//").map_or(line, |idx| &line[..idx]);
        out.push_str(keep);
        out.push('\n');
    }
    out
}

/// Parses the unsigned integer immediately following `marker` (e.g. the `3` in
/// `@group(3)`), returning `None` when the marker is absent or not numeric.
fn paren_u32(line: &str, marker: &str) -> Option<u32> {
    let start = line.find(marker)? + marker.len();
    let digits: String = line[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Extracts the identifier bound by a `var[<address-space>] NAME` declaration on
/// a single `@binding` line, skipping the optional `<...>` template, returning
/// `None` when no identifier follows.
fn var_identifier(line: &str) -> Option<String> {
    let after_binding = &line[line.find("@binding(")?..];
    let var_pos = after_binding.find("var")?;
    let mut rest = &after_binding[var_pos + "var".len()..];
    if let Some(stripped) = rest.trim_start().strip_prefix('<') {
        let close = stripped.find('>')?;
        rest = &stripped[close + 1..];
    }
    let ident: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() {
        None
    } else {
        Some(ident)
    }
}

/// Collects every `@group(G) @binding(B) var[<...>] NAME` declaration in a
/// source as `(group, binding, name)`, skipping commented-out lines. The water
/// shaders declare each binding on one line, matching this line-oriented parse.
fn binding_declarations(src: &str) -> Vec<(u32, u32, String)> {
    let mut out = Vec::new();
    for line in src.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        if !(line.contains("@group(") && line.contains("@binding(")) {
            continue;
        }
        let (Some(group), Some(binding), Some(name)) = (
            paren_u32(line, "@group("),
            paren_u32(line, "@binding("),
            var_identifier(line),
        ) else {
            continue;
        };
        out.push((group, binding, name));
    }
    out
}

/// Whole-word occurrence count of `word` in `hay`, so `scene_color` never
/// matches inside `scene_color_sampler` and over-counts a binding's reads.
fn whole_word_count(hay: &str, word: &str) -> usize {
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let bytes = hay.as_bytes();
    let mut count = 0;
    let mut cursor = 0;
    while let Some(found) = hay[cursor..].find(word) {
        let start = cursor + found;
        let end = start + word.len();
        let before_ok = start == 0 || !is_ident(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident(bytes[end]);
        if before_ok && after_ok {
            count += 1;
        }
        cursor = end;
    }
    count
}

/// Guards every water `WESL` shader against the *inert binding* defect class: a
/// `@group(G) @binding(B) var NAME` declaration whose `NAME` is never read in
/// the body. That is exactly the fault the surface `SSGI` `@group(7)` uniform
/// exhibited before `water_ssgi_gather` consumed it - a binding that reserves a
/// bind-group slot and a resource on device yet contributes nothing, a silent
/// fake implementation. Pinning it across all water sources stops any binding
/// from regressing into that dead no-op.
#[test]
fn water_wesl_shaders_consume_every_binding() {
    const SOURCES: &[(&str, &str)] = &[
        ("water.wesl", include_str!("../shaders/water.wesl")),
        (
            "water_butterfly.wesl",
            include_str!("../shaders/water_butterfly.wesl"),
        ),
        (
            "water_flip.wesl",
            include_str!("../shaders/water_flip.wesl"),
        ),
        (
            "water_flip_mac.wesl",
            include_str!("../shaders/water_flip_mac.wesl"),
        ),
        (
            "water_flip_mac_g2p.wesl",
            include_str!("../shaders/water_flip_mac_g2p.wesl"),
        ),
        (
            "water_flip_mac_p2g.wesl",
            include_str!("../shaders/water_flip_mac_p2g.wesl"),
        ),
        (
            "water_ocean.wesl",
            include_str!("../shaders/water_ocean.wesl"),
        ),
        ("water_pbf.wesl", include_str!("../shaders/water_pbf.wesl")),
        (
            "water_render_fx.wesl",
            include_str!("../shaders/water_render_fx.wesl"),
        ),
        (
            "water_spectrum_fft.wesl",
            include_str!("../shaders/water_spectrum_fft.wesl"),
        ),
        (
            "water_surface.wesl",
            include_str!("../shaders/water_surface.wesl"),
        ),
        (
            "water_surface_mesh.wesl",
            include_str!("../shaders/water_surface_mesh.wesl"),
        ),
        (
            "water_surface_raster.wesl",
            include_str!("../shaders/water_surface_raster.wesl"),
        ),
    ];
    for (name, src) in SOURCES {
        let code = strip_line_comments(src);
        for (group, binding, ident) in binding_declarations(src) {
            let uses = whole_word_count(&code, &ident);
            assert!(
                uses >= 2,
                "{name}: @group({group}) @binding({binding}) `{ident}` is declared \
                 but never read ({uses} code reference(s)); an inert binding is a \
                 fake implementation - consume it or drop the binding",
            );
        }
    }
}
