//! The mesh pipelines the queue specializes through.
//!
//! Specializing a mesh pipeline needs two things the queue cannot own: the shader
//! and target state (the renderer's, injected through
//! [`MeshPipelineProvider`]) and the device (the module must exist before a
//! descriptor can name it). This module is the render-world resource that holds
//! the provider, plus the spine's default provider.
//!
//! # The spine's default is a stand-in, on purpose
//!
//! [`SpineMeshPipeline`] is a flat-shaded pass-through: it draws geometry where
//! its vertices say, in one colour, with no lighting. It exists so the render
//! spine is runnable and testable on its own — a frame that queues a mesh draws
//! it — and it is **replaced** by installing a real provider
//! ([`MeshPipelines::new`]) once the material system's shading is ported onto
//! this spine. It is not a rendering model, and nothing keys off it.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use prism_ecs::prelude::Resource;
use prism_render_architecture::shader_package::{
    ArtifactDigest, ExecutionFamily, ModuleId, ModuleIdentity, ModuleInternError, ModuleInterner,
    PermutationIndex, PermutationSpace, ShaderPackageId,
};
use prism_render_driver::{
    ColorTargetState, DepthStencilState, FragmentState, MultisampleState, PrimitiveState,
    RenderDevice, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderModuleId, TextureFormat,
    VertexState,
};

use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider};

/// The pipeline provider every queued mesh draw specializes through.
#[derive(Resource)]
pub struct MeshPipelines {
    provider: Box<dyn MeshPipelineProvider + Send + Sync>,
    response_modules: ResponseModuleCache,
    recovery: MeshPipelineRecovery,
}

enum MeshPipelineRecovery {
    Spine {
        color_format: TextureFormat,
    },
    Custom {
        factory: Box<MeshPipelineRecoveryFactory>,
        modules_rebuilt: bool,
    },
    Unavailable,
}

/// Rebuilds a custom mesh provider against a replacement device.
///
/// The factory runs after all response modules have been recreated in the
/// supplied cache. It must return a provider whose descriptors reference only
/// resources owned by `device`; returning an error leaves the render app in
/// `Recovering` and retries the same replacement device on the next frame.
type MeshPipelineRecoveryFactory = dyn Fn(
        &dyn RenderDevice,
        &ResponseModuleCache,
    ) -> Result<Box<dyn MeshPipelineProvider + Send + Sync>, String>
    + Send
    + Sync
    + 'static;

/// Why the mesh pipeline provider could not be rebuilt for a replacement
/// device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MeshPipelineRecoveryError {
    /// A custom provider was installed without a device-rebuild recipe.
    CustomProvider,
    /// The host-provided recovery factory rejected this replacement device.
    Factory(String),
}

impl core::fmt::Display for MeshPipelineRecoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CustomProvider => f.write_str(
                "the installed custom mesh pipeline provider has no device recovery recipe",
            ),
            Self::Factory(error) => write!(f, "custom mesh pipeline recovery failed: {error}"),
        }
    }
}

/// A response module declaration plus its resident RHI shader module.
#[derive(Clone, Debug)]
pub struct ResidentResponseModule {
    /// The four-component pipeline identity's module component.
    pub module_id: ModuleId,
    /// The actual backend shader module used by pipeline descriptors.
    pub shader_module: ShaderModuleId,
    /// The declaration-owned dense permutation space.
    pub permutations: PermutationSpace,
    /// The checked full-lobe fallback, when this module supports one.
    pub full_lobe_fallback: Option<PermutationIndex>,
    /// Whether this module belongs to the closed built-in fused set.
    pub builtin: bool,
    /// The interned package/module identity used to create this resident.
    pub identity: ModuleIdentity,
    descriptor: ShaderModuleDescriptor,
}

/// A module request passed to [`ResponseModuleCache`].
#[derive(Clone, Debug)]
pub struct ResponseModuleSpec {
    /// Fused modules must set this only for a built-in response.
    pub execution_family: ExecutionFamily,
    /// Driver-local package/module identity, including code and ABI digests.
    pub identity: ModuleIdentity,
    /// Checked dense permutation declaration.
    pub permutations: PermutationSpace,
    /// Explicit fallback in the same dense permutation space.
    pub full_lobe_fallback: Option<PermutationIndex>,
    /// Shader source or precompiled module bytes consumed by the RHI.
    pub descriptor: ShaderModuleDescriptor,
    /// Marks a response as one of the built-in fused modules.
    pub builtin: bool,
}

/// Errors from the app-side module/package residency seam.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponseModuleCacheError {
    /// The module declaration was rejected by the shared interner.
    Intern(ModuleInternError),
    /// The same interned identity was offered with different shader bytes.
    DescriptorChanged(ModuleId),
    /// A custom response attempted to use the fused path.
    FusedCustom,
}

/// The render-world cache for response modules and the resident composition
/// kernel. Package names and module declarations are interned once at this
/// boundary; hot pipeline keys carry only `ModuleId` and dense indices.
#[derive(Default)]
pub struct ResponseModuleCache {
    interner: ModuleInterner,
    modules: BTreeMap<ModuleId, ResidentResponseModule>,
    composition: Option<ModuleId>,
}

impl ResponseModuleCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns and makes one response shader module resident.
    pub fn intern<D: RenderDevice + ?Sized>(
        &mut self,
        device: &D,
        spec: ResponseModuleSpec,
    ) -> Result<ModuleId, ResponseModuleCacheError> {
        if spec.execution_family == ExecutionFamily::FusedBuiltin && !spec.builtin {
            return Err(ResponseModuleCacheError::FusedCustom);
        }
        let module_id = self
            .interner
            .intern(
                spec.identity.clone(),
                spec.permutations.clone(),
                spec.full_lobe_fallback,
            )
            .map_err(ResponseModuleCacheError::Intern)?;
        if let Some(existing) = self.modules.get(&module_id) {
            if existing.descriptor != spec.descriptor {
                return Err(ResponseModuleCacheError::DescriptorChanged(module_id));
            }
            return Ok(module_id);
        }
        let shader_module = device.create_shader_module(&spec.descriptor);
        self.modules.insert(
            module_id,
            ResidentResponseModule {
                module_id,
                shader_module,
                permutations: spec.permutations,
                full_lobe_fallback: spec.full_lobe_fallback,
                builtin: spec.builtin,
                identity: spec.identity,
                descriptor: spec.descriptor,
            },
        );
        Ok(module_id)
    }

    /// Installs the renderer's reserved spine module without allocating a
    /// second numeric identity. It is still represented in the same resident
    /// table, so the default path participates in permutation validation.
    pub fn install_reserved<D: RenderDevice + ?Sized>(
        &mut self,
        device: &D,
        module_id: ModuleId,
        spec: ResponseModuleSpec,
    ) -> Result<ModuleId, ResponseModuleCacheError> {
        assert_eq!(
            module_id,
            ModuleId::SPINE,
            "only the spine may use the reserved module id"
        );
        if let Some(existing) = self.modules.get(&module_id) {
            if existing.descriptor != spec.descriptor {
                return Err(ResponseModuleCacheError::DescriptorChanged(module_id));
            }
            return Ok(module_id);
        }
        let shader_module = device.create_shader_module(&spec.descriptor);
        self.modules.insert(
            module_id,
            ResidentResponseModule {
                module_id,
                shader_module,
                permutations: spec.permutations,
                full_lobe_fallback: spec.full_lobe_fallback,
                builtin: spec.builtin,
                identity: spec.identity,
                descriptor: spec.descriptor,
            },
        );
        Ok(module_id)
    }

    /// Ensures the fixed composition kernel exists exactly once.
    pub fn ensure_composition_kernel<D: RenderDevice + ?Sized>(
        &mut self,
        device: &D,
    ) -> Result<ModuleId, ResponseModuleCacheError> {
        if let Some(module) = self.composition {
            return Ok(module);
        }
        let identity = ModuleIdentity {
            package: ShaderPackageId::new("prism.composition.kernel"),
            digest: ArtifactDigest([0x43; 32]),
            entry_point: "compose_fs".into(),
            binding_layout_digest: [0x43; 32],
        };
        let spec = ResponseModuleSpec {
            execution_family: ExecutionFamily::Composition,
            identity,
            permutations: PermutationSpace::try_new([]).expect("empty composition space is valid"),
            full_lobe_fallback: None,
            descriptor: ShaderModuleDescriptor::wgsl(COMPOSITION_WGSL),
            builtin: true,
        };
        let module = self.intern(device, spec)?;
        self.composition = Some(module);
        Ok(module)
    }

    /// Looks up a resident response module.
    #[must_use]
    pub fn get(&self, module_id: ModuleId) -> Option<&ResidentResponseModule> {
        self.modules.get(&module_id)
    }

    /// The resident composition kernel, if one was installed.
    #[must_use]
    pub fn composition_kernel(&self) -> Option<&ResidentResponseModule> {
        self.composition.and_then(|module| self.get(module))
    }

    /// Iterates resident modules in deterministic module-id order.
    pub fn iter(&self) -> impl Iterator<Item = &ResidentResponseModule> {
        self.modules.values()
    }

    /// Number of resident response modules, including the composition kernel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether no response module is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Destroys all shader modules owned by this cache.
    pub fn clear<D: RenderDevice + ?Sized>(&mut self, device: &D) {
        for module in self.modules.values() {
            device.destroy_shader_module(module.shader_module);
        }
        self.modules.clear();
        self.composition = None;
    }
}

const COMPOSITION_WGSL: &str = "\
@vertex fn compose_vs(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {\n\
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n\
    return vec4<f32>(positions[vertex], 0.0, 1.0);\n\
}\n\
@fragment fn compose_fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n\
}\n";

impl MeshPipelines {
    /// Wraps `provider` as the render world's mesh pipeline source.
    pub fn new(provider: impl MeshPipelineProvider + Send + Sync + 'static) -> Self {
        Self {
            provider: Box::new(provider),
            response_modules: ResponseModuleCache::new(),
            recovery: MeshPipelineRecovery::Unavailable,
        }
    }

    /// Wraps a custom provider with an explicit replacement-device factory.
    ///
    /// A provider may own shader modules, bind groups, or other device-scoped
    /// state that cannot be cloned across device loss. The factory is the
    /// host's recovery boundary for that state; it receives the recreated
    /// response-module cache so descriptors can use the new shader ids.
    pub fn with_recovery_factory<F>(
        provider: impl MeshPipelineProvider + Send + Sync + 'static,
        factory: F,
    ) -> Self
    where
        F: Fn(
                &dyn RenderDevice,
                &ResponseModuleCache,
            ) -> Result<Box<dyn MeshPipelineProvider + Send + Sync>, String>
            + Send
            + Sync
            + 'static,
    {
        Self {
            provider: Box::new(provider),
            response_modules: ResponseModuleCache::new(),
            recovery: MeshPipelineRecovery::Custom {
                factory: Box::new(factory),
                modules_rebuilt: false,
            },
        }
    }

    /// The spine's default: a flat pass-through pipeline.
    ///
    /// Built against `device` because a pipeline descriptor names a shader
    /// module, and the module has to exist first.
    #[must_use]
    pub fn spine(device: &dyn RenderDevice, color_format: TextureFormat) -> Self {
        let mut response_modules = ResponseModuleCache::new();
        let spec = ResponseModuleSpec {
            execution_family: ExecutionFamily::FusedBuiltin,
            identity: ModuleIdentity {
                package: ShaderPackageId::new("prism.spine"),
                digest: ArtifactDigest([0x53; 32]),
                entry_point: "fs".into(),
                binding_layout_digest: [0x53; 32],
            },
            permutations: PermutationSpace::try_new([]).expect("empty spine space is valid"),
            full_lobe_fallback: None,
            descriptor: ShaderModuleDescriptor::wgsl(SpineMeshPipeline::WGSL),
            builtin: true,
        };
        let module = response_modules
            .install_reserved(device, ModuleId::SPINE, spec)
            .expect("install the spine response module");
        let shader_module = response_modules
            .get(module)
            .expect("resident spine module")
            .shader_module;
        Self {
            provider: Box::new(SpineMeshPipeline::from_module(shader_module, color_format)),
            response_modules,
            recovery: MeshPipelineRecovery::Spine { color_format },
        }
    }

    /// The wrapped provider.
    #[must_use]
    pub fn provider(&self) -> &(dyn MeshPipelineProvider + Send + Sync) {
        &*self.provider
    }

    /// The response module declarations resident in this render-world provider.
    #[must_use]
    pub fn response_modules(&self) -> &ResponseModuleCache {
        &self.response_modules
    }

    /// Releases resident shader modules after all dependent PSOs are gone.
    pub fn clear_response_modules<D: RenderDevice + ?Sized>(&mut self, device: &D) {
        self.response_modules.clear(device);
    }

    /// Rebuilds the provider and its resident shader modules on `device`.
    ///
    /// The built-in spine carries a complete recipe. A custom provider created
    /// with [`Self::new`] is rejected instead of being silently replaced by the
    /// spine, so recovery remains observable and the app stays paused.
    pub(crate) fn recover(
        &mut self,
        device: &dyn RenderDevice,
    ) -> Result<(), MeshPipelineRecoveryError> {
        match &mut self.recovery {
            MeshPipelineRecovery::Spine { color_format } => {
                Self::rebuild_response_modules(&mut self.response_modules, device);
                self.provider = match self.response_modules.get(ModuleId::SPINE) {
                    Some(module) => Box::new(SpineMeshPipeline::from_module(
                        module.shader_module,
                        *color_format,
                    )),
                    None => Box::new(SpineMeshPipeline::new(device, *color_format)),
                };
                Ok(())
            }
            MeshPipelineRecovery::Custom {
                factory,
                modules_rebuilt,
            } => {
                if !*modules_rebuilt {
                    Self::rebuild_response_modules(&mut self.response_modules, device);
                    *modules_rebuilt = true;
                }
                let provider = factory(device, &self.response_modules)
                    .map_err(MeshPipelineRecoveryError::Factory)?;
                self.provider = provider;
                Ok(())
            }
            MeshPipelineRecovery::Unavailable => Err(MeshPipelineRecoveryError::CustomProvider),
        }
    }

    pub(crate) fn check_recovery(&self) -> Result<(), MeshPipelineRecoveryError> {
        match &self.recovery {
            MeshPipelineRecovery::Spine { .. } | MeshPipelineRecovery::Custom { .. } => Ok(()),
            MeshPipelineRecovery::Unavailable => Err(MeshPipelineRecoveryError::CustomProvider),
        }
    }

    /// Starts a fresh provider rebuild for a newly created device. A failed
    /// factory retry on the same device intentionally keeps its rebuilt module
    /// ids, while a later device loss must recreate them again.
    pub(crate) fn begin_recovery(&mut self) {
        if let MeshPipelineRecovery::Custom {
            modules_rebuilt, ..
        } = &mut self.recovery
        {
            *modules_rebuilt = false;
        }
    }

    fn rebuild_response_modules(cache: &mut ResponseModuleCache, device: &dyn RenderDevice) {
        for module in cache.modules.values_mut() {
            module.shader_module = device.create_shader_module(&module.descriptor);
        }
    }
}

impl MeshPipelineProvider for MeshPipelines {
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        self.provider.descriptor(variant)
    }

    fn transformed_descriptor(
        &self,
        variant: &MeshPipelineKey,
    ) -> Option<RenderPipelineDescriptor> {
        self.provider.transformed_descriptor(variant)
    }
}

/// The spine's stand-in pipeline: position in, one colour out.
///
/// See the module docs — this is what makes the spine runnable before the
/// material system's shading is ported, and installing a different
/// [`MeshPipelineProvider`] replaces it.
pub struct SpineMeshPipeline {
    module: ShaderModuleId,
    color_format: TextureFormat,
}

impl SpineMeshPipeline {
    /// The vertex stage: the mesh's first attribute is its clip-space position.
    ///
    /// The fragment stage is one flat colour. Both entry points live in one
    /// module, so a variant costs no extra shader creation.
    const WGSL: &'static str = "\
@vertex fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(position, 1.0);\n\
}\n\
@vertex fn vs_transform(@location(0) position: vec3<f32>,\n\
    @location(13) row0: vec4<f32>, @location(14) row1: vec4<f32>,\n\
    @location(15) row2: vec4<f32>) -> @builtin(position) vec4<f32> {\n\
    let p = vec4<f32>(position, 1.0);\n\
    return vec4<f32>(dot(row0, p), dot(row1, p), dot(row2, p), 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);\n\
}\n";

    /// Creates the pipeline source against `device`.
    #[must_use]
    pub fn new(device: &dyn RenderDevice, color_format: TextureFormat) -> Self {
        Self {
            module: device.create_shader_module(&ShaderModuleDescriptor::wgsl(Self::WGSL)),
            color_format,
        }
    }

    fn from_module(module: ShaderModuleId, color_format: TextureFormat) -> Self {
        Self {
            module,
            color_format,
        }
    }
}

impl MeshPipelineProvider for SpineMeshPipeline {
    fn transformed_descriptor(
        &self,
        variant: &MeshPipelineKey,
    ) -> Option<RenderPipelineDescriptor> {
        let mut descriptor = self.descriptor(variant);
        descriptor.vertex.entry_point = "vs_transform".into();
        Some(descriptor)
    }

    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("prism-spine-mesh".into()),
            layout: None,
            vertex: VertexState {
                module: self.module,
                entry_point: "vs".into(),
                // Replaced by the specializer with the mesh's own layout, which
                // is the layout the pipeline key hashes.
                buffers: Vec::new(),
            },
            primitive: PrimitiveState::default(),
            // The pass a variant names decides whether there is depth; a variant
            // with no depth format must not declare depth state, or the pipeline
            // would not match its pass.
            depth_stencil: variant.depth_format.map(DepthStencilState::reverse_z),
            multisample: MultisampleState {
                count: variant.sample_count,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: self.module,
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState::opaque(self.color_format))],
            }),
        }
    }
}
