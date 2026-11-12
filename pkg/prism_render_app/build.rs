use std::{borrow::Cow, env, fs, path::PathBuf};

struct MaterialResolver(wesl::VirtualResolver<'static>);

fn material_resolver() -> MaterialResolver {
    let mut resolver = wesl::VirtualResolver::new();
    for (module, path) in [
        (
            "prism_render_material::shaders::material",
            "../prism_render_material/src/shaders/material.wesl",
        ),
        (
            "prism_render_material::shaders::material_unpack",
            "../prism_render_material/src/shaders/material_unpack.wesl",
        ),
        (
            "prism_render_material::shaders::material_sample",
            "../prism_render_material/src/shaders/material_sample.wesl",
        ),
        (
            "prism_render_material::shaders::material_virtual_texture",
            "../prism_render_material/src/shaders/material_virtual_texture.wesl",
        ),
        (
            "prism_render_app::shaders::lighting",
            "src/shaders/native/lighting.wesl",
        ),
        (
            "prism_render_app::shaders::tangent",
            "src/shaders/native/tangent.wesl",
        ),
        (
            "prism_render_app::shaders::specular_aa",
            "src/shaders/native/specular_aa.wesl",
        ),
        (
            "prism_render_app::shaders::thin_film",
            "src/shaders/native/thin_film.wesl",
        ),
        (
            "prism_render_app::shaders::glint",
            "src/shaders/native/glint.wesl",
        ),
        (
            "prism_render_app::shaders::lobe_budget",
            "src/shaders/native/lobe_budget.wesl",
        ),
        (
            "prism_render_app::shaders::brdf",
            "src/shaders/native/brdf.wesl",
        ),
        (
            "prism_composite_material",
            "src/shaders/composite_material.wesl",
        ),
        ("prism_surface_encode", "src/shaders/surface_encode.wesl"),
        (
            "prism_surface_identity",
            "src/shaders/surface_identity.wesl",
        ),
        (
            "prism_native_gi_material_evaluator",
            "src/shaders/native_gi_material_evaluator.wesl",
        ),
    ] {
        println!("cargo:rerun-if-changed={path}");
        resolver.add_module(
            module.parse().expect("material shader module path"),
            Cow::Owned(fs::read_to_string(path).expect("shipping material shader source")),
        );
    }
    MaterialResolver(resolver)
}

impl wesl::Resolver for MaterialResolver {
    fn resolve_source(
        &self,
        path: &wesl::syntax::ModulePath,
    ) -> Result<Cow<'_, str>, wesl::ResolveError> {
        self.0.resolve_source(&self.canonical_path(path))
    }

    fn canonical_path(&self, path: &wesl::syntax::ModulePath) -> wesl::syntax::ModulePath {
        let mut canonical = path.clone();
        if let wesl::syntax::PathOrigin::Package(package) = &mut canonical.origin
            && let Some((_, dependency)) = package.rsplit_once('/')
        {
            *package = dependency.to_owned();
        }
        canonical
    }
}

fn main() {
    let resolver = material_resolver();
    let options = wesl::CompileOptions {
        imports: true,
        condcomp: true,
        ..Default::default()
    };
    let mut options = options;
    options
        .features
        .flags
        .insert("MATERIAL_GROUP_2".into(), false.into());
    let linked = wesl::compile_sourcemap(
        &"prism_surface_encode".parse().expect("surface entry path"),
        &resolver,
        &wesl::EscapeMangler,
        &options,
    )
    .unwrap_or_else(|error| {
        panic!(
            "surface encode WESL link: {}",
            error.diagnostic().render_plain()
        )
    })
    .to_string();
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo build output directory"));
    fs::write(out.join("surface_encode.wgsl"), linked).expect("linked surface encode WGSL");
    let linked = wesl::compile_sourcemap(
        &"prism_surface_identity"
            .parse()
            .expect("surface identity entry path"),
        &material_resolver(),
        &wesl::EscapeMangler,
        &options,
    )
    .unwrap_or_else(|error| {
        panic!(
            "surface identity WESL link: {}",
            error.diagnostic().render_plain()
        )
    })
    .to_string();
    fs::write(out.join("surface_identity.wgsl"), linked).expect("linked surface identity WGSL");
    let linked = wesl::compile_sourcemap(
        &"prism_native_gi_material_evaluator"
            .parse()
            .expect("native GI material evaluator entry path"),
        &material_resolver(),
        &wesl::EscapeMangler,
        &options,
    )
    .unwrap_or_else(|error| {
        panic!(
            "native GI material evaluator WESL link: {}",
            error.diagnostic().render_plain()
        )
    })
    .to_string();
    fs::write(out.join("native_gi_material_evaluator.wgsl"), linked)
        .expect("linked native GI material evaluator WGSL");
    let mut bindless_options = options.clone();
    bindless_options
        .features
        .flags
        .insert("MATERIAL_BINDLESS".into(), true.into());
    let linked = wesl::compile_sourcemap(
        &"prism_native_gi_material_evaluator"
            .parse()
            .expect("bindless native GI material evaluator entry path"),
        &material_resolver(),
        &wesl::EscapeMangler,
        &bindless_options,
    )
    .unwrap_or_else(|error| {
        panic!(
            "bindless native GI material evaluator WESL link: {}",
            error.diagnostic().render_plain()
        )
    })
    .to_string();
    fs::write(
        out.join("native_gi_material_evaluator_bindless.wgsl"),
        linked,
    )
    .expect("linked bindless native GI material evaluator WGSL");
    let composite_options = wesl::CompileOptions {
        // This module is a library of callable material helpers, not a shader
        // entry point. Preserve its public declarations for the runtime
        // composite shader that appends it and calls them later.
        strip: false,
        ..options
    };
    let compile_composite = |mut flags: wesl::CompileOptions| {
        flags
            .features
            .flags
            .insert("MATERIAL_GROUP_2".into(), true.into());
        let linked = wesl::compile_sourcemap(
            &"prism_composite_material"
                .parse()
                .expect("composite material entry path"),
            &material_resolver(),
            &wesl::EscapeMangler,
            &flags,
        )
        .unwrap_or_else(|error| {
            panic!(
                "composite material WESL link: {}",
                error.diagnostic().render_plain()
            )
        })
        .to_string();
        assert!(
            linked.contains("fn prism_composite_material_surface("),
            "linked composite material library must retain its runtime entry helper"
        );
        linked
    };
    fs::write(
        out.join("composite_material.wgsl"),
        compile_composite(composite_options.clone()),
    )
    .expect("linked canonical composite material WGSL");
    let mut bindless_composite_options = composite_options;
    bindless_composite_options
        .features
        .flags
        .insert("MATERIAL_BINDLESS".into(), true.into());
    fs::write(
        out.join("composite_material_bindless.wgsl"),
        compile_composite(bindless_composite_options),
    )
    .expect("linked bindless composite material WGSL");
}
