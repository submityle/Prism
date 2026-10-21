//! Real WESL linking and WGSL validation with host-owned GPU publication.
//! No renderer, pipeline cache or asset server is duplicated here. The host
//! constructs its real pipeline in publish_shader's fallible build closure.

use crate::reload::{
    PreparedReload, ReloadCancellation, ReloadError, ReloadLimits, ReloadPipeline, ReloadProvider,
    ReloadSlot, ReloadSources,
};
use prism_render_driver::{ShaderModuleDescriptor, ShaderSource};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

/// Exact schema accepted by the WGSL provider.
pub const WGSL_SCHEMA: &[u8] = b"prism.shader.wgsl/1";
/// Exact schema accepted by the WESL provider.
pub const WESL_SCHEMA: &[u8] = b"prism.shader.wesl/1";

enum Language {
    Wgsl(PathBuf),
    Wesl {
        root: String,
        modules: BTreeMap<String, PathBuf>,
    },
}

/// Shader preparation reuses the actual WESL compiler and Naga validator.
pub struct ShaderReloadProvider {
    language: Language,
}
impl ShaderReloadProvider {
    /// One WGSL file, without filesystem imports.
    pub fn wgsl(path: PathBuf) -> Self {
        Self {
            language: Language::Wgsl(path),
        }
    }
    /// WESL module names mapped to explicitly captured source paths. Only this
    /// virtual resolver is available to imports; there is no implicit disk I/O.
    pub fn wesl(root: String, modules: BTreeMap<String, PathBuf>) -> Self {
        Self {
            language: Language::Wesl { root, modules },
        }
    }
}

/// Validated shader source plus its actual compiler-derived entry points.
pub struct CompiledReloadShader {
    descriptor: ShaderModuleDescriptor,
    entries: Vec<ShaderEntry>,
}
/// An entry point reflected from the validated Naga module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderEntry {
    /// Name used by the host pipeline descriptor.
    pub name: String,
    /// Shader stage reported by Naga.
    pub stage: naga::ShaderStage,
    /// Compute workgroup size, zero for other stages.
    pub workgroup_size: [u32; 3],
}
impl CompiledReloadShader {
    /// Real RHI descriptor. A host still needs device/layout validation.
    pub fn descriptor(&self) -> &ShaderModuleDescriptor {
        &self.descriptor
    }
    /// Actual, validated entry points in source order.
    pub fn entries(&self) -> &[ShaderEntry] {
        &self.entries
    }
}

impl ReloadProvider for ShaderReloadProvider {
    type Artifact = CompiledReloadShader;
    fn prepare(
        &self,
        sources: &ReloadSources,
        schema: &[u8],
        cancellation: &ReloadCancellation,
        limits: &ReloadLimits,
    ) -> Result<CompiledReloadShader, ReloadError> {
        cancellation.check()?;
        let text = match &self.language {
            Language::Wgsl(path) => {
                if schema != WGSL_SCHEMA {
                    return Err(ReloadError::provider("unsupported WGSL schema"));
                }
                sources.text(path)?.to_owned()
            }
            Language::Wesl { root, modules } => {
                if schema != WESL_SCHEMA {
                    return Err(ReloadError::provider("unsupported WESL schema"));
                }
                if modules.len() > limits.sources {
                    return Err(ReloadError::Budget("shader modules"));
                }
                let mut resolver = wesl::VirtualResolver::new();
                let mut identities = HashSet::new();
                for (name, path) in modules {
                    cancellation.check()?;
                    let module = name
                        .parse::<wesl::syntax::ModulePath>()
                        .map_err(ReloadError::provider)?;
                    if matches!(module.origin, wesl::syntax::PathOrigin::Relative(_))
                        || !identities.insert(module.clone())
                    {
                        return Err(ReloadError::provider(
                            "relative or duplicate WESL module identity",
                        ));
                    }
                    resolver.add_module(module, Cow::Owned(sources.text(path)?.to_owned()));
                }
                let mut compiler = wesl::Wesl::new(".").set_custom_resolver(resolver);
                compiler.use_stripping(false);
                let root = root
                    .parse::<wesl::syntax::ModulePath>()
                    .map_err(ReloadError::provider)?;
                compiler
                    .compile(&root)
                    .map_err(ReloadError::provider)?
                    .to_string()
            }
        };
        cancellation.check()?;
        if text.len() > limits.output_bytes {
            return Err(ReloadError::Budget("compiled shader bytes"));
        }
        let module = naga::front::wgsl::parse_str(&text)
            .map_err(|error| ReloadError::provider(error.emit_to_string(&text)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| ReloadError::provider(error.emit_to_string(&text)))?;
        cancellation.check()?;
        let entries = module
            .entry_points
            .iter()
            .map(|entry| ShaderEntry {
                name: entry.name.clone(),
                stage: entry.stage,
                workgroup_size: entry.workgroup_size,
            })
            .collect();
        Ok(CompiledReloadShader {
            descriptor: ShaderModuleDescriptor {
                label: Some("Prism editor reload".into()),
                source: ShaderSource::Wgsl(Cow::Owned(text)),
            },
            entries,
        })
    }
}

/// Constructs a real host GPU resource before swapping the last-good handle.
/// Build must use the backend's fallible validation/error-scope API and return
/// an owned candidate, with cleanup on drop. Errors retain the old pipeline.
/// Returning success for syntax validation alone is not GPU publication.
pub fn publish_shader<L>(
    pipeline: &ReloadPipeline<ShaderReloadProvider>,
    prepared: PreparedReload<CompiledReloadShader>,
    slot: &mut ReloadSlot<L>,
    build: impl FnOnce(&CompiledReloadShader) -> Result<L, ReloadError>,
) -> Result<L, ReloadError> {
    pipeline.publish(prepared, slot, build, |live, _, candidate| {
        std::mem::replace(live, candidate)
    })
}
