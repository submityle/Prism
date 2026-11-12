//! Render-world ownership of the unified lighting plan.

use bytemuck::{Pod, Zeroable};
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::lighting::contract::{
    BudgetCompiler, ChangeFields, ChangeJournal, GiCandidateAvailability, GiCandidateRejection,
    LightingPlan, PlanCaps, RecipeExecution, RecipeId, RejectionExplanation, SceneStats, Tier,
};
use prism_render_architecture::lighting::{DebugMeta, DebugView, LightScene};
use prism_render_architecture::ray_scene::GpuBvhBuffers;
use prism_render_driver::RenderDevice;
use prism_render_material::{
    BuiltinProgram, DynamicChannels, MaterialDomain, MaterialExecutionBinding,
    MaterialExecutionTables, MaterialRecord, MaterialRenderClass, ProgramRef, RenderDomain,
    ShadingPlan,
};
use prism_render_shading::{
    classify_material_header_with_execution, evaluate_unlit_response,
    surface_sample_from_parameters, ClassificationError,
};
use std::collections::BTreeSet;
use std::sync::Arc;

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
    /// Canonical light-scene cursor paired with the compiled plan.
    pub light_scene_sequence: u64,
    /// Canonical scene lineage paired with the compiled plan.
    pub light_scene_identity: u64,
    /// Canonical light-scene epoch paired with the compiled plan.
    pub light_scene_epoch: u64,
}

/// Render-world debug snapshot consumed by diagnostic views.
///
/// The metadata, view registrations, rejection explanations, and EXPLAIN text
/// are published as one snapshot so consumers cannot observe mixed-frame data.
#[derive(Clone, Debug, Default, PartialEq, Resource)]
pub struct LightingDebugViewState {
    /// Frame for which this diagnostic snapshot was compiled.
    pub compiled_frame: u64,
    /// Plan compilation count paired with [`compiled_frame`].
    pub plan_compile_count: u64,
    /// Selected recipes and their contract-driven display metadata.
    pub selected_recipes: Vec<LightingDebugRecipe>,
    /// Structured reasons for every recipe rejected by the plan.
    pub rejection_explanations: Vec<RejectionExplanation>,
    /// The planner explanation retained for consumers without recompilation.
    pub explain: String,
}

/// One selected recipe exposed to the DebugView consumer chain.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingDebugRecipe {
    /// Recipe identity from the lighting contract.
    pub recipe: RecipeId,
    /// Display semantics declared by the recipe contract.
    pub meta: DebugMeta,
    /// Related views registered by the recipe metadata.
    pub views: Vec<DebugView>,
}

impl LightingDebugViewState {
    /// Creates a debug snapshot immediately consistent with a plan.
    #[must_use]
    pub fn from_plan(state: &LightingPlanState) -> Self {
        let mut snapshot = Self::default();
        snapshot.update_from_plan(state);
        snapshot
    }

    /// Replaces the snapshot from one compiled plan without recompiling it.
    ///
    /// Returns whether the paired frame/compile identity changed.
    pub(crate) fn update_from_plan(&mut self, state: &LightingPlanState) -> bool {
        let changed = self.compiled_frame != state.compiled_frame
            || self.plan_compile_count != state.compile_count;
        self.compiled_frame = state.compiled_frame;
        self.plan_compile_count = state.compile_count;
        self.selected_recipes = state
            .plan
            .debug_metadata()
            .into_iter()
            .map(|(recipe, meta)| LightingDebugRecipe {
                recipe,
                meta,
                views: meta.combine_with.to_vec(),
            })
            .collect();
        self.rejection_explanations = state.plan.rejection_explanations();
        self.explain.clone_from(&state.explain);
        changed
    }
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

/// Extended emitter consumed by the deterministic direct-light recipe.
///
/// `kind` is encoded in `shape_kind[0]`: 0 is a rectangle, 1 is a disk, and
/// 2 is a capsule. The vector fields are shape-specific but always carry
/// normalized directions where a direction is required. Radiance is linear
/// scene radiance and is already converted from the canonical photometric
/// unit by the main-world adapter.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingAreaInput {
    /// Center for a rectangle/disk, or endpoint A for a capsule.
    pub origin: [f32; 4],
    /// Propagation direction for a rectangle/disk, or endpoint B for a capsule.
    pub vector_a: [f32; 4],
    /// Rectangle right axis; unused for disk/capsule.
    pub vector_b: [f32; 4],
    /// Rectangle up axis; unused for disk/capsule.
    pub vector_c: [f32; 4],
    /// Emitted scene-linear radiance.
    pub radiance: [f32; 4],
    /// Rectangle half extents, disk radius, or capsule radius.
    pub shape: [f32; 4],
    /// `[kind, visibility, 0, 0]`.
    pub shape_kind: [u32; 4],
}

impl Default for LightingAreaInput {
    fn default() -> Self {
        Self {
            origin: [0.0; 4],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [1.0, 0.0, 0.0, 0.0],
            vector_c: [0.0, 0.0, 1.0, 0.0],
            radiance: [0.0; 4],
            shape: [0.0; 4],
            shape_kind: [0, 1, 0, 0],
        }
    }
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
    /// World-space receiver position.
    pub position: [f32; 3],
    /// World-space receiver normal; normalized at evaluation.
    pub normal: [f32; 3],
    /// Dimensionless diffuse reflectance, not a pre-divided BRDF.
    pub albedo: [f32; 3],
    /// Incident diffuse irradiance in the same scene-linear convention as lights.
    /// This is not outgoing radiance: the Lambert BRDF is applied exactly once.
    pub ambient: [f32; 3],
}

/// One deterministic indirect-light candidate for the initial GI reservoir.
/// `source_pdf` and `target_pdf` use solid-angle measure. `target_pdf` is the
/// non-normalized luminance of the Lambertian directional integrand at the
/// owning surface, including `albedo / PI`, the incident cosine, and
/// visibility. Misses remain proposals with positive `source_pdf` and zero
/// `target_pdf`, so the RIS denominator counts every attempted sample.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LightingGiCandidateInput {
    pub position: [f32; 3],
    pub source_pdf: f32,
    pub normal: [f32; 3],
    pub target_pdf: f32,
    pub radiance: [f32; 3],
    pub selection_u01: f32,
    pub visibility: f32,
    pub _padding: [f32; 3],
}

impl Default for LightingGiCandidateInput {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, 0.0],
            source_pdf: 1.0,
            normal: [0.0, 1.0, 0.0],
            target_pdf: 0.0,
            radiance: [0.0; 3],
            selection_u01: 0.5,
            visibility: 1.0,
            _padding: [0.0; 3],
        }
    }
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

/// The material slots that actually entered the current visible raster work.
///
/// GI surface mappings are only meaningful when they name this same snapshot;
/// a separately authored list can otherwise point at a material that did not
/// produce any raster identity in the frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VisibleMaterialSnapshot {
    /// Main-world visibility transaction epoch that produced the work list.
    pub frame_epoch: u64,
    /// Stable material-slot indices present in visible raster work.
    pub material_indices: Vec<u32>,
}

impl VisibleMaterialSnapshot {
    fn from_scene(scene: &crate::visibility::PrismRenderScene) -> Self {
        let mut material_indices = BTreeSet::new();
        for (_, view) in scene.view_results() {
            material_indices.extend(view.work.iter().map(|item| item.material.index));
        }
        Self {
            frame_epoch: scene.last_transaction().frame_epoch,
            material_indices: material_indices.into_iter().collect(),
        }
    }
}

/// Why a GI surface-to-material mapping was rejected before extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceMaterialMappingRejection {
    /// A surface batch was published without a complete mapping.
    MissingMapping { surfaces: usize },
    /// The mapping length does not match the surface batch.
    LengthMismatch { surfaces: usize, mappings: usize },
    /// There was no visible raster material snapshot to validate against.
    MissingVisibleSnapshot,
    /// The producer supplied a mapping epoch from another visible frame.
    FrameEpochMismatch { expected: u64, actual: u64 },
    /// The mapping names a material that did not produce visible raster work.
    MaterialNotVisible { material_index: u32 },
    /// The same material slot was assigned to more than one GI surface.
    DuplicateMaterial { material_index: u32 },
}

fn admit_surface_material_mapping(
    surfaces: usize,
    mapping: &[u32],
    mapping_epoch: u64,
    visible: &VisibleMaterialSnapshot,
) -> Result<Vec<u32>, SurfaceMaterialMappingRejection> {
    if surfaces == 0 && mapping.is_empty() {
        return Ok(Vec::new());
    }
    if mapping.is_empty() {
        return Err(SurfaceMaterialMappingRejection::MissingMapping { surfaces });
    }
    if mapping.len() != surfaces {
        return Err(SurfaceMaterialMappingRejection::LengthMismatch {
            surfaces,
            mappings: mapping.len(),
        });
    }
    if visible.frame_epoch == 0 || visible.material_indices.is_empty() {
        return Err(SurfaceMaterialMappingRejection::MissingVisibleSnapshot);
    }
    if mapping_epoch != 0 && mapping_epoch != visible.frame_epoch {
        return Err(SurfaceMaterialMappingRejection::FrameEpochMismatch {
            expected: visible.frame_epoch,
            actual: mapping_epoch,
        });
    }

    let mut seen = BTreeSet::new();
    for &material_index in mapping {
        if visible
            .material_indices
            .binary_search(&material_index)
            .is_err()
        {
            return Err(SurfaceMaterialMappingRejection::MaterialNotVisible { material_index });
        }
        if !seen.insert(material_index) {
            return Err(SurfaceMaterialMappingRejection::DuplicateMaterial { material_index });
        }
    }
    Ok(mapping.to_vec())
}

/// Non-light main-world inputs: budget metadata, invalidation records, surface
/// samples, and GI proposals. Analytic lights are owned only by
/// [`crate::lighting_scene::LightingLightScene`].
#[derive(Clone, Debug, Resource)]
#[allow(missing_docs)]
pub struct LightingSceneInputs {
    pub stats: SceneStats,
    pub journal: ChangeJournal,
    /// Per-surface GI candidates. Each surface owns its own ordered RIS batch;
    /// candidates are never implicitly broadcast across unrelated surfaces.
    pub gi_candidates: Vec<Vec<LightingGiCandidateInput>>,
    pub surfaces: Vec<LightingSurfaceInput>,
    /// Material slots corresponding one-to-one with [`surfaces`]. The Native
    /// composite resolves this explicit relation per pixel; it never treats a
    /// GBuffer material slot as a GI-array index. A material slot may occur at
    /// most once in this list. Missing or duplicate mappings fail closed.
    pub surface_material_indices: Vec<u32>,
    /// Optional visibility transaction epoch for `surface_material_indices`.
    /// Zero means the current visible snapshot supplies the epoch; a non-zero
    /// value is checked explicitly during extraction.
    pub surface_material_snapshot_epoch: u64,
    /// Frame-level controls for the builtin NPR response. These controls are
    /// copied with the lighting snapshot so the Native composite and its CPU
    /// reference never observe different stylized policies.
    pub stylized: NativeStylizedParams,
}

/// Native frame-level controls for the builtin stylized response.
///
/// This is intentionally separate from material data: materials provide
/// relative NPR modifiers, while the scene owns the shared artistic policy.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct NativeStylizedParams {
    /// Cel quantization band count.
    pub bands: u32,
    /// Cosine wrap in `[0, 1]`.
    pub wrap: f32,
    /// Band-edge softness in `[0, 1]`.
    pub ramp_softness: f32,
    /// Stepped-shadow threshold in `[0, 1]`.
    pub shadow_threshold: f32,
    /// Stepped-shadow transition half-width.
    pub shadow_softness: f32,
    /// Stylized specular intensity.
    pub specular_intensity: f32,
    /// Stylized specular threshold.
    pub specular_threshold: f32,
    /// Stylized specular transition half-width.
    pub specular_softness: f32,
    /// Stylized specular tint.
    pub specular_color: [f32; 3],
    /// Stylized rim intensity.
    pub rim_intensity: f32,
    /// Stylized rim tint.
    pub rim_color: [f32; 3],
    /// Stylized rim exponent.
    pub rim_power: f32,
}

impl Default for NativeStylizedParams {
    fn default() -> Self {
        Self {
            bands: 4,
            wrap: 0.0,
            ramp_softness: 0.0,
            shadow_threshold: 0.0,
            shadow_softness: 0.0,
            specular_intensity: 0.0,
            specular_threshold: 0.5,
            specular_softness: 0.05,
            specular_color: [1.0; 3],
            rim_intensity: 0.0,
            rim_color: [1.0; 3],
            rim_power: 4.0,
        }
    }
}

impl NativeStylizedParams {
    /// Returns whether the packed controls are safe for a shader invocation.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.bands != 0
            && self.wrap.is_finite()
            && (0.0..=1.0).contains(&self.wrap)
            && self.ramp_softness.is_finite()
            && (0.0..=1.0).contains(&self.ramp_softness)
            && self.shadow_threshold.is_finite()
            && (0.0..=1.0).contains(&self.shadow_threshold)
            && self.shadow_softness.is_finite()
            && (0.0..=1.0).contains(&self.shadow_softness)
            && self.specular_intensity.is_finite()
            && self.specular_intensity >= 0.0
            && self.specular_threshold.is_finite()
            && (0.0..=1.0).contains(&self.specular_threshold)
            && self.specular_softness.is_finite()
            && (0.0..=1.0).contains(&self.specular_softness)
            && self
                .specular_color
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.rim_intensity.is_finite()
            && self.rim_intensity >= 0.0
            && self
                .rim_color
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.rim_power.is_finite()
            && self.rim_power >= 0.0
    }
}

/// Immutable outgoing radiance authored for one stable BVH primitive.
///
/// `primitive_id` is the stable ID stored in the packed BVH triangle record;
/// it is deliberately not a packed-array index because BVH construction may
/// reorder triangles into leaf-contiguous order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightingTriangleRadiance {
    /// Stable primitive ID from the corresponding packed BVH triangle.
    pub primitive_id: u32,
    /// Linear, non-negative outgoing RGB radiance.
    pub rgb: [f32; 3],
}

/// Binds one stable BVH primitive to the existing authoritative material
/// record and its verified execution identity.
///
/// This is an association only: material parameters remain owned by
/// [`MaterialRecord`] and are never copied into a second material model.
#[derive(Clone, Copy, Debug)]
pub struct LightingTriangleMaterial<'a> {
    /// Stable primitive ID stored in the packed BVH triangle.
    pub primitive_id: u32,
    /// Authoritative material record published by the material registry.
    pub material: &'a MaterialRecord,
    /// Verified plan/program/package/permutation binding for this material.
    pub execution: MaterialExecutionBinding,
    /// Runtime channels paired with the record at this material epoch.
    pub dynamic: DynamicChannels,
}

/// Version and execution proof retained after authoritative material
/// evaluation has produced triangle radiance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LightingMaterialProof {
    /// Stable primitive whose radiance was evaluated.
    pub primitive_id: u32,
    /// Generational material identity used for the evaluation.
    pub material_handle: GenerationalHandle,
    /// Published record revision used for the evaluation.
    pub material_revision: u32,
    /// Registry snapshot epoch covering record, dynamic and execution state.
    pub material_epoch: u64,
    /// Exact verified execution words copied into the GPU material header.
    pub execution_binding: [u32; 5],
}

/// A material cannot enter the triangle-radiance cache unless its response is
/// exactly evaluable from the data carried by the current packed hit ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightingMaterialTraceError {
    /// A material handle uses the reserved invalid index.
    InvalidMaterialHandle {
        /// Primitive associated with the invalid material handle.
        primitive_id: u32,
    },
    /// Only ordinary surface-domain material records are accepted.
    UnsupportedMaterialDomain {
        /// Primitive associated with the unsupported material domain.
        primitive_id: u32,
        /// Authored material domain.
        domain: MaterialDomain,
    },
    /// The current trace slice has no non-surface object-domain evaluator.
    UnsupportedRenderDomain {
        /// Primitive associated with the unsupported render domain.
        primitive_id: u32,
        /// Runtime render domain.
        domain: RenderDomain,
    },
    /// Blended, masked and transmissive classes need hit-time coverage/state.
    UnsupportedRenderClass {
        /// Primitive associated with the unsupported render class.
        primitive_id: u32,
        /// Runtime render class.
        render_class: MaterialRenderClass,
    },
    /// Texture-backed responses need hit UVs and a texture sampling context.
    TexturedMaterial {
        /// Primitive associated with the textured material.
        primitive_id: u32,
        /// Number of bound texture rows.
        textures: usize,
    },
    /// Runtime channels need hit-time coordinates or response evaluation.
    DynamicMaterial {
        /// Primitive associated with the dynamic material.
        primitive_id: u32,
    },
    /// The header did not resolve through the verified execution tables.
    Classification {
        /// Primitive whose header failed verified classification.
        primitive_id: u32,
        /// Verified-classification failure.
        error: ClassificationError,
    },
    /// The bound dense execution plan was not a single-program plan.
    UnsupportedExecutionPlan {
        /// Primitive associated with the unsupported plan shape.
        primitive_id: u32,
    },
    /// Authoring and verified execution identities disagree.
    PlanExecutionMismatch {
        /// Primitive whose authoring and verified identities disagree.
        primitive_id: u32,
        /// Program selected by the authored response plan.
        authored: ProgramRef,
        /// Program selected by the verified execution tables.
        verified: ProgramRef,
    },
    /// The packed hit ABI can currently evaluate only builtin Unlit exactly.
    UnsupportedProgram {
        /// Primitive associated with the unsupported response program.
        primitive_id: u32,
        /// Verified response program.
        program: ProgramRef,
    },
    /// Authoritative evaluation produced a negative or non-finite value.
    InvalidRadiance {
        /// Primitive whose evaluated radiance was invalid.
        primitive_id: u32,
    },
    /// Primitive coverage or the packed BVH itself is invalid.
    Trace(crate::lighting_trace_gpu::TracePreparationError),
}

/// Optional host-owned packed scene used by the software-BVH GI trace pass.
/// Surface IDs are stable identities aligned one-to-one with `surfaces` in
/// [`LightingSceneInputs`]; malformed alignment is rejected by host preparation.
#[derive(Clone, Debug, Resource)]
pub struct LightingTraceSceneInput {
    /// Packed BVH in the architecture crate's frozen GPU ABI.
    pub bvh: Arc<GpuBvhBuffers>,
    /// Stable identities for the current lighting surfaces, in matching order.
    pub surface_ids: Vec<u64>,
    /// Deterministic cosine rays generated for each surface each frame.
    pub rays_per_surface: u32,
    /// Positive minimum trace distance used for self-intersection rejection.
    pub ray_epsilon: f32,
    /// Maximum world-space ray distance.
    pub ray_tmax: f32,
    /// Per-primitive outgoing radiance for trace-hit GI evaluation. Entries
    /// may arrive in any order, but IDs must exactly cover the packed BVH.
    /// Sharing this immutable table and advancing its revision only when
    /// material/texture evaluation changes lets the renderer avoid per-frame
    /// allocation and upload.
    pub outgoing_radiance: Arc<[LightingTriangleRadiance]>,
    /// Monotonic content revision for the outgoing-radiance table.
    pub radiance_revision: u64,
    /// Material identity/version proofs for radiance built through the
    /// authoritative material path. Host-authored radiance has no proofs.
    pub material_proofs: Arc<[LightingMaterialProof]>,
    packed_radiance: Arc<std::sync::Mutex<Option<PackedRadianceCache>>>,
}

#[derive(Debug)]
struct PackedRadianceCache {
    // Retain owners, not raw pointer keys: this prevents allocator ABA and
    // forces Arc::make_mut to copy before changing a cached input snapshot.
    bvh: Arc<GpuBvhBuffers>,
    radiance: Arc<[LightingTriangleRadiance]>,
    revision: u64,
    result: Result<Arc<[[f32; 4]]>, crate::lighting_trace_gpu::TracePreparationError>,
}

impl LightingTraceSceneInput {
    /// Creates an immutable trace snapshot and its revision-scoped upload cache.
    #[must_use]
    pub fn new(
        bvh: Arc<GpuBvhBuffers>,
        surface_ids: Vec<u64>,
        rays_per_surface: u32,
        ray_epsilon: f32,
        ray_tmax: f32,
        outgoing_radiance: Arc<[LightingTriangleRadiance]>,
        radiance_revision: u64,
    ) -> Self {
        Self {
            bvh,
            surface_ids,
            rays_per_surface,
            ray_epsilon,
            ray_tmax,
            outgoing_radiance,
            radiance_revision,
            material_proofs: Arc::from([]),
            packed_radiance: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Builds the exact material subset that the current triangle-only hit ABI
    /// can evaluate without inventing UVs, instance identity, lighting or a
    /// second response implementation.
    pub fn from_untextured_unlit_materials(
        bvh: Arc<GpuBvhBuffers>,
        surface_ids: Vec<u64>,
        rays_per_surface: u32,
        ray_epsilon: f32,
        ray_tmax: f32,
        material_epoch: u64,
        execution_tables: &MaterialExecutionTables,
        materials: &[LightingTriangleMaterial<'_>],
    ) -> Result<Self, LightingMaterialTraceError> {
        let mut radiance = Vec::with_capacity(materials.len());
        let mut proofs = Vec::with_capacity(materials.len());
        for binding in materials {
            let primitive_id = binding.primitive_id;
            let material = binding.material;
            if !material.handle.is_valid() {
                return Err(LightingMaterialTraceError::InvalidMaterialHandle { primitive_id });
            }
            if material.domain != MaterialDomain::Surface {
                return Err(LightingMaterialTraceError::UnsupportedMaterialDomain {
                    primitive_id,
                    domain: material.domain,
                });
            }
            if material.render_domain != RenderDomain::Surface {
                return Err(LightingMaterialTraceError::UnsupportedRenderDomain {
                    primitive_id,
                    domain: material.render_domain,
                });
            }
            if !matches!(
                material.render_class,
                MaterialRenderClass::Opaque | MaterialRenderClass::OpaqueTwoSided
            ) {
                return Err(LightingMaterialTraceError::UnsupportedRenderClass {
                    primitive_id,
                    render_class: material.render_class,
                });
            }
            if !material.textures.is_empty() {
                return Err(LightingMaterialTraceError::TexturedMaterial {
                    primitive_id,
                    textures: material.textures.len(),
                });
            }
            if !binding.dynamic.is_inert() {
                return Err(LightingMaterialTraceError::DynamicMaterial { primitive_id });
            }

            let mut header =
                material.header_with_execution(0, 0, material_epoch, Some(binding.execution));
            let [wetness, snow, damage, dissolve] = binding.dynamic.to_words();
            header.dynamic_wetness = wetness;
            header.dynamic_snow = snow;
            header.dynamic_damage = damage;
            header.dynamic_dissolve = dissolve;
            let verified =
                classify_material_header_with_execution(&header, execution_tables, u32::MAX)
                    .map_err(|error| LightingMaterialTraceError::Classification {
                        primitive_id,
                        error,
                    })?;
            let (left, right, _) = execution_tables
                .plan_endpoints(binding.execution.shading_plan_index)
                .map_err(|_| LightingMaterialTraceError::Classification {
                    primitive_id,
                    error: ClassificationError::InvalidExecutionBinding,
                })?;
            if right.is_some() {
                return Err(LightingMaterialTraceError::UnsupportedExecutionPlan { primitive_id });
            }
            let authored = match material.response_plan {
                ShadingPlan::Single { program } => program,
                ShadingPlan::MaskBlend { .. } => {
                    return Err(LightingMaterialTraceError::UnsupportedExecutionPlan {
                        primitive_id,
                    });
                }
            };
            if authored != left || authored != verified {
                return Err(LightingMaterialTraceError::PlanExecutionMismatch {
                    primitive_id,
                    authored,
                    verified,
                });
            }
            if verified != ProgramRef::Builtin(BuiltinProgram::Unlit) {
                return Err(LightingMaterialTraceError::UnsupportedProgram {
                    primitive_id,
                    program: verified,
                });
            }

            let rgb = evaluate_unlit_response(surface_sample_from_parameters(&material.surface));
            if rgb
                .iter()
                .any(|channel| !channel.is_finite() || *channel < 0.0)
            {
                return Err(LightingMaterialTraceError::InvalidRadiance { primitive_id });
            }
            radiance.push(LightingTriangleRadiance { primitive_id, rgb });
            proofs.push(LightingMaterialProof {
                primitive_id,
                material_handle: material.handle,
                material_revision: material.revision,
                material_epoch,
                execution_binding: binding.execution.to_words(),
            });
        }

        let mut input = Self::new(
            bvh,
            surface_ids,
            rays_per_surface,
            ray_epsilon,
            ray_tmax,
            Arc::from(radiance),
            material_epoch,
        );
        input.material_proofs = Arc::from(proofs);
        input
            .build_packed_outgoing_radiance()
            .map_err(LightingMaterialTraceError::Trace)?;
        Ok(input)
    }

    /// Produces GPU upload order matching packed BVH triangle indices.
    pub(crate) fn packed_outgoing_radiance(
        &self,
    ) -> Result<Arc<[[f32; 4]]>, crate::lighting_trace_gpu::TracePreparationError> {
        let mut cache = self
            .packed_radiance
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.as_ref() {
            if Arc::ptr_eq(&cached.bvh, &self.bvh)
                && Arc::ptr_eq(&cached.radiance, &self.outgoing_radiance)
                && cached.revision == self.radiance_revision
            {
                return cached.result.clone();
            }
        }
        let result = self.build_packed_outgoing_radiance().map(Arc::from);
        *cache = Some(PackedRadianceCache {
            bvh: Arc::clone(&self.bvh),
            radiance: Arc::clone(&self.outgoing_radiance),
            revision: self.radiance_revision,
            result: result.clone(),
        });
        result
    }

    fn build_packed_outgoing_radiance(
        &self,
    ) -> Result<Vec<[f32; 4]>, crate::lighting_trace_gpu::TracePreparationError> {
        crate::lighting_trace_gpu::BorrowedPackedTraceScene::from(self.bvh.as_ref())
            .validate()
            .map_err(crate::lighting_trace_gpu::TracePreparationError::Trace)?;
        let triangles = self.bvh.triangles.chunks_exact(12);
        let triangle_count = self.bvh.triangles.len() / 12;
        if self.outgoing_radiance.len() != triangle_count {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::RadianceCountMismatch {
                    triangles: triangle_count,
                    radiance: self.outgoing_radiance.len(),
                },
            );
        }
        let mut sorted = self.outgoing_radiance.iter().copied().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|entry| entry.primitive_id);
        if let Some(pair) = sorted
            .windows(2)
            .find(|pair| pair[0].primitive_id == pair[1].primitive_id)
        {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::DuplicateRadiancePrimitive(
                    pair[0].primitive_id,
                ),
            );
        }
        let mut packed_ids = triangles
            .clone()
            .map(|triangle| triangle[9])
            .collect::<Vec<_>>();
        packed_ids.sort_unstable();
        if let Some(pair) = packed_ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::DuplicatePackedPrimitive(pair[0]),
            );
        }
        if let Some((radiance_id, packed_id)) = sorted
            .iter()
            .map(|entry| entry.primitive_id)
            .zip(packed_ids.iter().copied())
            .find(|(radiance_id, packed_id)| radiance_id != packed_id)
        {
            return Err(
                crate::lighting_trace_gpu::TracePreparationError::RadiancePrimitiveSetMismatch {
                    radiance_id,
                    packed_id,
                },
            );
        }
        let mut packed = Vec::with_capacity(triangle_count);
        for (packed_index, triangle) in triangles.enumerate() {
            let primitive_id = triangle[9];
            let entry = sorted
                .binary_search_by_key(&primitive_id, |entry| entry.primitive_id)
                .ok()
                .map(|index| sorted[index])
                .ok_or(
                    crate::lighting_trace_gpu::TracePreparationError::MissingRadiancePrimitive {
                        packed_index,
                        primitive_id,
                    },
                )?;
            if entry
                .rgb
                .iter()
                .any(|channel| !channel.is_finite() || *channel < 0.0)
            {
                return Err(
                    crate::lighting_trace_gpu::TracePreparationError::InvalidRadiance(packed_index),
                );
            }
            packed.push([entry.rgb[0], entry.rgb[1], entry.rgb[2], 0.0]);
        }
        Ok(packed)
    }
}

impl LightingSceneInputs {
    /// Creates inputs from the host's configured scene estimate.
    #[must_use]
    pub fn with_stats(stats: SceneStats) -> Self {
        Self {
            stats,
            journal: ChangeJournal::new(),
            gi_candidates: Vec::new(),
            surfaces: Vec::new(),
            surface_material_indices: Vec::new(),
            surface_material_snapshot_epoch: 0,
            stylized: NativeStylizedParams::default(),
        }
    }
}

impl Default for LightingSceneInputs {
    fn default() -> Self {
        Self {
            stats: SceneStats::default(),
            journal: ChangeJournal::new(),
            gi_candidates: Vec::new(),
            surfaces: Vec::new(),
            surface_material_indices: Vec::new(),
            surface_material_snapshot_epoch: 0,
            stylized: NativeStylizedParams::default(),
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
    /// Canonical generation-safe handles aligned with `directionals`.
    pub directional_handles: Vec<prism_render_architecture::lighting::LightSceneHandle>,
    pub punctuals: Vec<LightingPunctualInput>,
    /// Canonical generation-safe handles aligned with `punctuals`.
    pub punctual_handles: Vec<prism_render_architecture::lighting::LightSceneHandle>,
    pub areas: Vec<LightingAreaInput>,
    /// Canonical generation-safe handles aligned with `areas`.
    pub area_handles: Vec<prism_render_architecture::lighting::LightSceneHandle>,
    /// Recoverable per-emitter diagnostics from canonical direct-light adaptation.
    pub light_scene_rejections: Vec<crate::lighting_scene::LightingSceneExtractError>,
    /// Canonical scene sequence represented by this extracted snapshot.
    pub light_scene_sequence: u64,
    /// Canonical scene lineage represented by this extracted snapshot.
    pub light_scene_identity: u64,
    /// Canonical scene epoch represented by this extracted snapshot.
    pub light_scene_epoch: u64,
    /// Snapshot-wide canonical conversion failure, if any.
    pub light_scene_error: Option<crate::lighting_scene::LightingSceneExtractError>,
    /// Per-surface GI candidates, aligned with `surfaces` when present.
    pub gi_candidates: Vec<Vec<LightingGiCandidateInput>>,
    pub surfaces: Vec<LightingSurfaceInput>,
    /// Explicit material-slot mapping for the lighting surfaces. This is
    /// intentionally separate from `LightingSurfaceInput`: a material can be
    /// reused by many world surfaces, so no positional or material-slot
    /// inference is safe here.
    pub surface_material_indices: Vec<u32>,
    /// Visible raster snapshot used to admit `surface_material_indices`.
    pub visible_material_snapshot: VisibleMaterialSnapshot,
    /// Effective visibility epoch paired with the admitted mapping.
    pub surface_material_snapshot_epoch: u64,
    /// Mapping rejection retained for fail-closed plan diagnostics.
    pub surface_material_mapping_rejection: Option<SurfaceMaterialMappingRejection>,
    /// Frame-level NPR controls paired with the light snapshot.
    pub stylized: NativeStylizedParams,
    /// Optional packed scene used to generate and trace this frame's GI rays.
    pub trace_scene: Option<LightingTraceSceneInput>,
}

impl ExtractedLightingInputs {
    /// Validates the candidate producer against this frame and the actual
    /// device. A supplied trace scene is an explicit source request: invalid
    /// trace data never silently selects the host-candidate route.
    pub(crate) fn candidate_availability(
        &self,
        capabilities: &prism_render_driver::DeviceCapabilities,
    ) -> GiCandidateAvailability {
        use crate::lighting_gpu::{
            GI_CANDIDATES_PER_SURFACE, GI_CANDIDATE_BUFFER_BYTES, LIGHTING_PARAMS_BYTES,
            LIGHTING_SURFACE_BUFFER_BYTES, MAX_LIGHTING_SURFACES, OUTPUT_BYTES,
            TRACE_RADIANCE_BYTES,
        };
        use GiCandidateAvailability::{HostProvided, Missing, Rejected, SoftwareBvhGpu};
        use GiCandidateRejection::{DeviceLimits, InvalidInputs, TraceBackendUnavailable};

        if self.surface_material_mapping_rejection.is_some() {
            return Rejected(InvalidInputs);
        }
        if self.surfaces.is_empty() {
            return Missing;
        }
        if self.surfaces.len() > MAX_LIGHTING_SURFACES
            || self.surfaces.iter().any(|surface| {
                let normal_len = dot(surface.normal, surface.normal);
                surface.position.iter().any(|value| !value.is_finite())
                    || surface.normal.iter().any(|value| !value.is_finite())
                    || !normal_len.is_finite()
                    || normal_len <= 1.0e-20
                    || surface
                        .albedo
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Rejected(InvalidInputs);
        }
        let limits = capabilities.limits;
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_bind_groups == 0
            || limits.max_bindings_per_bind_group < 7
        {
            return Rejected(TraceBackendUnavailable);
        }
        if [
            GI_CANDIDATE_BUFFER_BYTES,
            LIGHTING_SURFACE_BUFFER_BYTES,
            OUTPUT_BYTES,
        ]
        .iter()
        .any(|bytes| *bytes > limits.max_storage_buffer_binding_size)
            || LIGHTING_PARAMS_BYTES > limits.max_uniform_buffer_binding_size
            || (self.surfaces.len() as u32).div_ceil(64)
                > limits.max_compute_workgroups_per_dimension
        {
            return Rejected(DeviceLimits);
        }
        if let Some(trace) = self.trace_scene.as_ref() {
            if trace.surface_ids.len() != self.surfaces.len()
                || trace.rays_per_surface == 0
                || trace.rays_per_surface as usize > GI_CANDIDATES_PER_SURFACE
                || !trace.ray_epsilon.is_finite()
                || trace.ray_epsilon <= 0.0
                || !trace.ray_tmax.is_finite()
                || trace.ray_tmax <= trace.ray_epsilon
                || trace.packed_outgoing_radiance().is_err()
            {
                return Rejected(InvalidInputs);
            }
            // Bounded stack scratch avoids a per-frame allocation for ID
            // uniqueness checks; surface validation bounds the copied prefix.
            let mut ids = [0_u64; MAX_LIGHTING_SURFACES];
            let ids = &mut ids[..trace.surface_ids.len()];
            ids.copy_from_slice(&trace.surface_ids);
            ids.sort_unstable();
            if ids.windows(2).any(|pair| pair[0] == pair[1]) {
                return Rejected(InvalidInputs);
            }
            let ray_count = self.surfaces.len() as u64 * u64::from(trace.rays_per_surface);
            let node_bytes = trace.bvh.nodes.len() as u64 * 4;
            let triangle_bytes = trace.bvh.triangles.len() as u64 * 4;
            if [
                node_bytes,
                triangle_bytes,
                ray_count * 48,
                crate::lighting_trace_gpu::HOST_TRACE_OUTPUT_BYTES,
                TRACE_RADIANCE_BYTES,
            ]
            .iter()
            .any(|bytes| *bytes > limits.max_storage_buffer_binding_size)
                || ray_count.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            {
                return Rejected(DeviceLimits);
            }
            return SoftwareBvhGpu;
        }
        if self.gi_candidates.is_empty() {
            return Missing;
        }
        if self.gi_candidates.len() != self.surfaces.len()
            || self.gi_candidates.iter().any(|batch| {
                batch.len() > GI_CANDIDATES_PER_SURFACE
                    || batch.iter().any(|candidate| {
                        candidate
                            .position
                            .iter()
                            .chain(&candidate.normal)
                            .any(|value| !value.is_finite())
                            || candidate
                                .radiance
                                .iter()
                                .any(|value| !value.is_finite() || *value < 0.0)
                            || !candidate.source_pdf.is_finite()
                            || candidate.source_pdf <= 0.0
                            || !candidate.target_pdf.is_finite()
                            || candidate.target_pdf < 0.0
                            || !candidate.selection_u01.is_finite()
                            || !(0.0..=1.0).contains(&candidate.selection_u01)
                            || !candidate.visibility.is_finite()
                            || !(0.0..=1.0).contains(&candidate.visibility)
                    })
            })
        {
            return Rejected(InvalidInputs);
        }
        HostProvided
    }
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
            GiCandidateAvailability::Missing,
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
            light_scene_sequence: 0,
            light_scene_identity: 0,
            light_scene_epoch: 0,
        }
    }
}

/// Copies the host-owned lighting snapshot across the one-way app boundary.
pub(crate) fn extract_lighting_inputs(main: &mut World, render: &mut World) {
    let inputs = main
        .get_resource::<LightingSceneInputs>()
        .cloned()
        .unwrap_or_default();
    let visible_material_snapshot = main
        .get_resource::<crate::visibility::PrismRenderScene>()
        .map(VisibleMaterialSnapshot::from_scene)
        .unwrap_or_default();
    // Pure lighting/trace producers may legitimately have no raster work in
    // this frame. In that case there is no identity mapping to consume; keep
    // the producer's candidate batch independent of raster admission. Once a
    // visible material exists, every mapping is checked against that exact
    // snapshot before it can reach the render world.
    let mapping_admission = if visible_material_snapshot.material_indices.is_empty() {
        Ok(inputs.surface_material_indices.clone())
    } else {
        admit_surface_material_mapping(
            inputs.surfaces.len(),
            &inputs.surface_material_indices,
            inputs.surface_material_snapshot_epoch,
            &visible_material_snapshot,
        )
    };
    let (surface_material_indices, surface_material_mapping_rejection) = match mapping_admission {
        Ok(indices) => (indices, None),
        Err(rejection) => (Vec::new(), Some(rejection)),
    };
    let surface_material_snapshot_epoch = if surface_material_mapping_rejection.is_none()
        && !visible_material_snapshot.material_indices.is_empty()
    {
        visible_material_snapshot.frame_epoch
    } else {
        inputs.surface_material_snapshot_epoch
    };
    // The canonical scene is an explicit main-world input. Extraction must not
    // create or mutate it: doing so would turn a missing producer into an
    // implicit lighting authority. A temporary empty snapshot keeps the
    // render path deterministic until the application publishes its scene.
    let empty_light_scene = LightScene::default();
    let published_light_scene = main.get_resource::<crate::lighting_scene::LightingLightScene>();
    let light_scene =
        published_light_scene.map_or(&empty_light_scene, |published| &published.scene);
    let light_scene_sequence = light_scene.sequence();
    let light_scene_identity = light_scene.identity();
    let light_scene_epoch = light_scene.scene_epoch();
    let light_count = light_scene.len();
    let native_changed = render
        .get_resource::<ExtractedLightingInputs>()
        .is_none_or(|previous| {
            previous.light_scene_identity != light_scene_identity
                || previous.light_scene_sequence != light_scene_sequence
                || previous.light_scene_epoch != light_scene_epoch
        });
    let native = native_changed.then(|| crate::lighting_scene::derive_native_lights(light_scene));
    let mut stats = inputs.stats;
    // SceneStats may carry a host estimate when no canonical lighting scene
    // has been published yet. Once the authoritative scene exists, including
    // an explicitly empty scene, its exact count supersedes that estimate.
    if published_light_scene.is_some() {
        stats.lights = u32::try_from(light_count).unwrap_or(u32::MAX);
    }
    let trace_scene = main.get_resource::<LightingTraceSceneInput>().cloned();
    if let Some(extracted) = render.get_resource_mut::<ExtractedLightingInputs>() {
        extracted.stats = stats;
        extracted.journal_snapshot = inputs.journal.snapshot();
        if native_changed {
            match native.expect("changed canonical scene must have a native snapshot") {
                Ok(native) => {
                    extracted.directionals = native.directionals;
                    extracted.directional_handles = native.directional_handles;
                    extracted.punctuals = native.punctuals;
                    extracted.punctual_handles = native.punctual_handles;
                    extracted.areas = native.areas;
                    extracted.area_handles = native.area_handles;
                    extracted.light_scene_rejections = native.rejections;
                    extracted.light_scene_error = None;
                }
                Err(error) => {
                    extracted.directionals.clear();
                    extracted.directional_handles.clear();
                    extracted.punctuals.clear();
                    extracted.punctual_handles.clear();
                    extracted.areas.clear();
                    extracted.area_handles.clear();
                    extracted.light_scene_rejections.clear();
                    extracted.light_scene_error = Some(error);
                }
            }
            extracted.light_scene_sequence = light_scene_sequence;
            extracted.light_scene_identity = light_scene_identity;
            extracted.light_scene_epoch = light_scene_epoch;
        }
        extracted.gi_candidates = inputs.gi_candidates;
        extracted.surfaces = inputs.surfaces;
        extracted.surface_material_indices = surface_material_indices;
        extracted.visible_material_snapshot = visible_material_snapshot;
        extracted.surface_material_snapshot_epoch = surface_material_snapshot_epoch;
        extracted.surface_material_mapping_rejection = surface_material_mapping_rejection;
        extracted.stylized = inputs.stylized;
        extracted.trace_scene = trace_scene;
    } else {
        let (
            directionals,
            directional_handles,
            punctuals,
            punctual_handles,
            areas,
            area_handles,
            light_scene_rejections,
            light_scene_error,
        ) = match native.expect("first extraction must have a native snapshot") {
            Ok(native) => (
                native.directionals,
                native.directional_handles,
                native.punctuals,
                native.punctual_handles,
                native.areas,
                native.area_handles,
                native.rejections,
                None,
            ),
            Err(error) => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Some(error),
            ),
        };
        render.insert_resource(ExtractedLightingInputs {
            stats,
            journal_snapshot: inputs.journal.snapshot(),
            directionals,
            directional_handles,
            punctuals,
            punctual_handles,
            areas,
            area_handles,
            light_scene_rejections,
            light_scene_sequence,
            light_scene_identity,
            light_scene_epoch,
            light_scene_error,
            gi_candidates: inputs.gi_candidates,
            surfaces: inputs.surfaces,
            surface_material_indices,
            visible_material_snapshot,
            surface_material_snapshot_epoch,
            surface_material_mapping_rejection,
            stylized: inputs.stylized,
            trace_scene,
        });
    }
}

/// Compiles the single plan once for each extracted frame.
pub(crate) fn compile_lighting_plan<I>(
    record: Res<RenderRecord>,
    inputs: Res<ExtractedLightingInputs>,
    runtime: Res<crate::RenderRuntimeResource<I>>,
    mut state: ResMut<LightingPlanState>,
    mut stats: ResMut<RenderFrameStats>,
) where
    I: prism_render_driver::Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + prism_render_driver::FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if state.compiled_frame == record.frame_index {
        return;
    }
    let previous_epoch = state.journal_snapshot.epoch;
    let previous_light_scene_epoch = state.light_scene_epoch;
    let previous_light_scene_identity = state.light_scene_identity;
    state.scene_stats = inputs.stats;
    state.journal_snapshot = inputs.journal_snapshot;
    state.light_scene_sequence = inputs.light_scene_sequence;
    state.light_scene_identity = inputs.light_scene_identity;
    state.light_scene_epoch = inputs.light_scene_epoch;
    let availability = inputs.candidate_availability(runtime.runtime().device().capabilities());
    state.plan = BudgetCompiler::compile(
        state.scene_stats,
        state.plan.budget,
        state.caps,
        state.tier,
        availability,
    );
    state.executions = state.plan.executions();
    state.explain = state.plan.explain();
    state.compiled_frame = record.frame_index;
    state.compile_count = state.compile_count.saturating_add(1);
    stats.lighting_plan_compiles = state.compile_count;
    stats.lighting_recipes_selected = state.plan.recipes.len() as u64;
    stats.lighting_recipes_rejected = state.plan.rejected.len() as u64;
    stats.lighting_executions_planned = state.executions.len() as u64;
    let journal_invalidations = changed_epoch_fields(previous_epoch, state.journal_snapshot.epoch);
    let scene_invalidated_without_journal = (previous_light_scene_identity
        != state.light_scene_identity
        || previous_light_scene_epoch != state.light_scene_epoch)
        && previous_epoch.lighting == state.journal_snapshot.epoch.lighting;
    stats.lighting_cache_invalidations =
        u64::from(journal_invalidations + u32::from(scene_invalidated_without_journal));
}

/// Synchronizes plan diagnostics after compilation and before queue/execution.
pub(crate) fn sync_lighting_debug_view(
    plan: Res<LightingPlanState>,
    mut debug: ResMut<LightingDebugViewState>,
    mut stats: ResMut<RenderFrameStats>,
) {
    if debug.update_from_plan(&plan) {
        stats.lighting_debug_updates = stats.lighting_debug_updates.saturating_add(1);
        stats.lighting_debug_selected_recipes = debug.selected_recipes.len() as u64;
        stats.lighting_debug_rejections = debug.rejection_explanations.len() as u64;
    }
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
/// stable Lambert contract: ambient irradiance, directional lights, point or
/// spot lights, and the deterministic four-sample finite-area estimator.
/// Every incident irradiance term is multiplied by `albedo / PI`; the result
/// is outgoing diffuse radiance, matching the indirect-light contract.
#[must_use]
pub(crate) fn evaluate_deterministic_direct(
    surface: LightingSurfaceInput,
    directionals: &[LightingDirectionalInput],
    punctuals: &[LightingPunctualInput],
    areas: &[LightingAreaInput],
) -> [f32; 3] {
    let normal = normalize_or(surface.normal, [0.0, 1.0, 0.0]);
    let diffuse_brdf = scale(surface.albedo, core::f32::consts::FRAC_1_PI);
    let mut radiance = mul(diffuse_brdf, surface.ambient);
    for light in directionals {
        let direction = normalize_or(light.direction_to_light, [0.0, 1.0, 0.0]);
        let cosine = dot(normal, direction).max(0.0) * light.visibility.clamp(0.0, 1.0);
        radiance = add(
            radiance,
            mul(diffuse_brdf, scale(light.illuminance, cosine)),
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
        radiance = add(radiance, mul(diffuse_brdf, scale(light.intensity, cosine)));
    }
    for light in areas {
        radiance = add(
            radiance,
            mul(diffuse_brdf, evaluate_area_direct(light, surface, normal)),
        );
    }
    radiance
}

fn evaluate_area_direct(
    light: &LightingAreaInput,
    surface: LightingSurfaceInput,
    normal: [f32; 3],
) -> [f32; 3] {
    let measure = area_measure(light);
    if !measure.is_finite() || measure <= 0.0 {
        return [0.0; 3];
    }
    let mut result = [0.0; 3];
    for (sample_index, _) in [0_u32, 1, 2, 3].into_iter().enumerate() {
        let point = area_sample(light, sample_index as u32);
        let to_light = sub(point, surface.position);
        let distance_squared = dot(to_light, to_light);
        if !distance_squared.is_finite() || distance_squared <= 1.0e-8 {
            continue;
        }
        let direction = scale(to_light, distance_squared.sqrt().recip());
        let cosine_surface = dot(normal, direction).max(0.0);
        let cosine_source = if light.shape_kind[0] == 0 || light.shape_kind[0] == 1 {
            dot(
                normalize_or(
                    [light.vector_a[0], light.vector_a[1], light.vector_a[2]],
                    [0.0, -1.0, 0.0],
                ),
                scale(direction, -1.0),
            )
            .max(0.0)
        } else {
            let axis = normalize_or(
                sub(
                    [light.vector_a[0], light.vector_a[1], light.vector_a[2]],
                    [light.origin[0], light.origin[1], light.origin[2]],
                ),
                [0.0, 1.0, 0.0],
            );
            let center = scale(
                add(
                    [light.origin[0], light.origin[1], light.origin[2]],
                    [light.vector_a[0], light.vector_a[1], light.vector_a[2]],
                ),
                0.5,
            );
            let radial = normalize_or(sub(point, center), basis_tangent(axis));
            dot(radial, scale(direction, -1.0)).max(0.0)
        };
        result = add(
            result,
            scale(
                [light.radiance[0], light.radiance[1], light.radiance[2]],
                cosine_surface * cosine_source / distance_squared,
            ),
        );
    }
    scale(
        result,
        measure * 0.25 * (light.shape_kind[1] as f32).clamp(0.0, 1.0),
    )
}

fn area_sample(light: &LightingAreaInput, sample_index: u32) -> [f32; 3] {
    let origin = [light.origin[0], light.origin[1], light.origin[2]];
    let vector_a = [light.vector_a[0], light.vector_a[1], light.vector_a[2]];
    if light.shape_kind[0] == 0 {
        let uv = [
            [-0.57735026, -0.57735026],
            [0.57735026, -0.57735026],
            [-0.57735026, 0.57735026],
            [0.57735026, 0.57735026],
        ][sample_index as usize];
        return add(
            add(
                origin,
                scale(
                    [light.vector_b[0], light.vector_b[1], light.vector_b[2]],
                    light.shape[0] * uv[0],
                ),
            ),
            scale(
                [light.vector_c[0], light.vector_c[1], light.vector_c[2]],
                light.shape[1] * uv[1],
            ),
        );
    }
    if light.shape_kind[0] == 1 {
        let offsets = [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]][sample_index as usize];
        return add(
            add(
                origin,
                scale(basis_tangent(vector_a), light.shape[0] * offsets[0]),
            ),
            scale(basis_bitangent(vector_a), light.shape[0] * offsets[1]),
        );
    }
    let axis = normalize_or(sub(vector_a, origin), [0.0, 1.0, 0.0]);
    let t = [0.125, 0.375, 0.625, 0.875][sample_index as usize];
    let angle: f32 = [0.0, 1.5707963, 3.1415927, 4.712389][sample_index as usize];
    let center = add(origin, scale(sub(vector_a, origin), t));
    let radial = add(
        scale(basis_tangent(axis), angle.cos()),
        scale(basis_bitangent(axis), angle.sin()),
    );
    add(center, scale(radial, light.shape[0]))
}

fn area_measure(light: &LightingAreaInput) -> f32 {
    if light.shape_kind[0] == 0 {
        return 4.0 * light.shape[0] * light.shape[1];
    }
    if light.shape_kind[0] == 1 {
        return core::f32::consts::PI * light.shape[0] * light.shape[0];
    }
    let length = sub(
        [light.vector_a[0], light.vector_a[1], light.vector_a[2]],
        [light.origin[0], light.origin[1], light.origin[2]],
    );
    let length = dot(length, length).sqrt();
    2.0 * core::f32::consts::PI * light.shape[0] * length
        + 4.0 * core::f32::consts::PI * light.shape[0] * light.shape[0]
}

fn basis_tangent(direction: [f32; 3]) -> [f32; 3] {
    let n = normalize_or(direction, [0.0, 1.0, 0.0]);
    let reference = if n[1].abs() > 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize_or(cross(reference, n), [1.0, 0.0, 0.0])
}

fn basis_bitangent(direction: [f32; 3]) -> [f32; 3] {
    let n = normalize_or(direction, [0.0, 1.0, 0.0]);
    normalize_or(cross(n, basis_tangent(n)), [0.0, 0.0, 1.0])
}

/// Evaluates the deterministic initial GI reservoir used by the GPU twin.
/// This is a bounded one-bounce estimator: temporal/spatial reuse and scene
/// traversal remain separate contracts, while the selected sample and its RIS
/// normalization are fully executable and parity-testable here.
#[must_use]
pub(crate) fn evaluate_deterministic_gi(
    surface: LightingSurfaceInput,
    candidates: &[LightingGiCandidateInput],
) -> [f32; 3] {
    const MAX_CANDIDATES: usize = 4;
    let primary_normal = normalize_or(surface.normal, [0.0, 1.0, 0.0]);
    let mut weight_sum = 0.0;
    let proposal_count = candidates.len().min(MAX_CANDIDATES) as u32;
    let mut selected = None;
    for candidate in candidates.iter().take(MAX_CANDIDATES) {
        if !candidate.source_pdf.is_finite()
            || candidate.source_pdf <= 0.0
            || !candidate.target_pdf.is_finite()
            || candidate.target_pdf < 0.0
        {
            continue;
        }
        let weight = candidate.target_pdf / candidate.source_pdf;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        weight_sum += weight;
        if candidate.target_pdf == 0.0 {
            continue;
        }
        let u = candidate.selection_u01.clamp(0.0, 1.0);
        if u * weight_sum < weight {
            selected = Some(*candidate);
        }
    }
    let Some(candidate) = selected else {
        return [0.0; 3];
    };
    if proposal_count == 0 || weight_sum <= 0.0 || candidate.target_pdf <= 0.0 {
        return [0.0; 3];
    }
    let to_sample = sub(candidate.position, surface.position);
    let distance_squared = dot(to_sample, to_sample);
    if !distance_squared.is_finite() || distance_squared <= 1.0e-8 {
        return [0.0; 3];
    }
    let direction = scale(to_sample, distance_squared.sqrt().recip());
    let cosine_primary = dot(primary_normal, direction).max(0.0);
    let visibility = candidate.visibility.clamp(0.0, 1.0);
    let reservoir_weight = weight_sum / (proposal_count as f32 * candidate.target_pdf);
    scale(
        mul(surface.albedo, candidate.radiance),
        cosine_primary / core::f32::consts::PI * visibility * reservoir_weight,
    )
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

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if length_squared.is_finite() && length_squared > 1.0e-12 {
        scale(value, length_squared.sqrt().recip())
    } else {
        fallback
    }
}

#[cfg(test)]
mod lighting_input_defaults_tests {
    use super::*;

    #[test]
    fn absent_scene_has_no_synthetic_lighting_surfaces() {
        assert!(LightingSceneInputs::default().surfaces.is_empty());
        assert!(LightingSceneInputs::with_stats(SceneStats::default())
            .surfaces
            .is_empty());
    }

    #[test]
    fn matching_visible_material_mapping_is_admitted() {
        let visible = VisibleMaterialSnapshot {
            frame_epoch: 17,
            material_indices: vec![2, 7],
        };
        assert_eq!(
            admit_surface_material_mapping(2, &[2, 7], 17, &visible),
            Ok(vec![2, 7])
        );
        // A zero producer epoch is bound to the same visible snapshot during
        // extraction; it is not a second, independently versioned source.
        assert_eq!(
            admit_surface_material_mapping(2, &[2, 7], 0, &visible),
            Ok(vec![2, 7])
        );
    }

    #[test]
    fn stale_or_invisible_material_mapping_is_rejected() {
        let visible = VisibleMaterialSnapshot {
            frame_epoch: 17,
            material_indices: vec![2, 7],
        };
        assert_eq!(
            admit_surface_material_mapping(2, &[2, 7], 16, &visible),
            Err(SurfaceMaterialMappingRejection::FrameEpochMismatch {
                expected: 17,
                actual: 16,
            })
        );
        assert_eq!(
            admit_surface_material_mapping(1, &[9], 17, &visible),
            Err(SurfaceMaterialMappingRejection::MaterialNotVisible {
                material_index: 9,
            })
        );
    }
}

#[cfg(test)]
mod radiance_tests {
    use super::*;
    use prism_render_architecture::{
        abi::{AbiHash, AbiVersion},
        ray_scene::{
            Bvh, GpuBvhBuffers, RayHitAttributeRecord, RayHitIdentity, Triangle, TRIANGLE_WORDS,
        },
        shader_package::{
            ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
            ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
        },
    };
    use prism_render_material::{
        fallback_material_record, ChannelMask, CompiledEntry, ExecutionFamily, GpuMaterialTexture,
    };

    fn execution_fixture(
        program: ProgramRef,
    ) -> (MaterialExecutionTables, MaterialExecutionBinding) {
        let manifest = ShaderPackageManifest {
            id: ShaderPackageId::new("lighting-material-trace-test"),
            source_revision: "rev-1".into(),
            entry_point: "main".into(),
            abi_version: AbiVersion::from_parts(1, 0),
            abi_hash: AbiHash::from_bytes(b"lighting-material-trace-test"),
            variants: ShaderPackageVariantManifest::from_digests(
                &PermutationSpace::new([2]),
                [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
            )
            .unwrap(),
        };
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            (2 << 16) | 17,
            1,
            (3 << 16) | 14,
            0,
            1,
            (5 << 16) | 15,
            5,
            3,
            0x6e69616d,
            0,
            (6 << 16) | 16,
            3,
            17,
            1,
            1,
            1,
            (2 << 16) | 19,
            1,
            (3 << 16) | 33,
            2,
            1,
            (5 << 16) | 54,
            1,
            3,
            0,
            2,
            (2 << 16) | 248,
            4,
            (1 << 16) | 253,
            (1 << 16) | 56,
        ];
        let entry = CompiledEntry::from_spirv(manifest.clone(), binary).unwrap();
        let mut packages = ShaderPackageRegistry::new();
        packages
            .register_verified(entry.artifact().clone())
            .unwrap();
        packages
            .validate(
                &manifest.id,
                &ManifestExpectation::new(manifest.abi_version, manifest.abi_hash),
            )
            .unwrap();

        let mut tables = MaterialExecutionTables::new();
        let package = tables.register_package(&entry, &packages).unwrap();
        let response = tables
            .register_program(
                program,
                &entry,
                package,
                ModuleId(7),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let family = match program {
            ProgramRef::Builtin(BuiltinProgram::Lit) => ExecutionFamily::FusedBuiltin,
            ProgramRef::Builtin(BuiltinProgram::Npr | BuiltinProgram::Unlit) => {
                ExecutionFamily::SplitResponse
            }
            ProgramRef::Registered(_) => ExecutionFamily::SplitResponse,
        };
        let plan = tables
            .register_plan(
                ShadingPlan::Single { program },
                response,
                None,
                family,
                ChannelMask::default(),
            )
            .unwrap();
        let binding = tables
            .bind_raw(plan.get(), response.get(), package.get(), 1)
            .unwrap();
        (tables, binding)
    }

    fn material(
        index: u32,
        generation: u32,
        revision: u32,
        program: BuiltinProgram,
        base_color: [f32; 3],
        emission_color: [f32; 3],
        emission_luminance: f32,
    ) -> MaterialRecord {
        let mut record = fallback_material_record(
            GenerationalHandle::new(index, generation),
            u64::from(revision),
        );
        record.response_plan = ShadingPlan::single_builtin(program);
        record.surface.base_color = [base_color[0], base_color[1], base_color[2], 1.0];
        record.surface.emission_color =
            [emission_color[0], emission_color[1], emission_color[2], 1.0];
        record.surface.emission_luminance = emission_luminance;
        record
    }

    fn one_triangle_bvh() -> Arc<GpuBvhBuffers> {
        Arc::new(GpuBvhBuffers::from_bvh(&Bvh::build(&[Triangle::new(
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            19,
        )])))
    }

    fn packed_bvh_with_hit_attributes(bvh: Bvh) -> Arc<GpuBvhBuffers> {
        let packed = GpuBvhBuffers::from_bvh(&bvh);
        let attributes = packed
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| {
                RayHitAttributeRecord::new(
                    RayHitIdentity::new(1, triangle[9], 1),
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0, 1.0]; 3],
                )
            })
            .collect();
        Arc::new(
            packed
                .with_hit_attributes(attributes)
                .expect("one hit-attribute record is required for every packed triangle"),
        )
    }

    fn build_one<'a>(
        tables: &MaterialExecutionTables,
        binding: MaterialExecutionBinding,
        record: &'a MaterialRecord,
        dynamic: DynamicChannels,
    ) -> Result<LightingTraceSceneInput, LightingMaterialTraceError> {
        LightingTraceSceneInput::from_untextured_unlit_materials(
            one_triangle_bvh(),
            vec![1],
            1,
            0.001,
            10.0,
            41,
            tables,
            &[LightingTriangleMaterial {
                primitive_id: 19,
                material: record,
                execution: binding,
                dynamic,
            }],
        )
    }

    #[test]
    fn authoritative_unlit_materials_feed_trace_and_candidate_radiance() {
        let (tables, binding) = execution_fixture(ProgramRef::Builtin(BuiltinProgram::Unlit));
        let material_11 = material(
            7,
            3,
            9,
            BuiltinProgram::Unlit,
            [0.2, 0.4, 0.6],
            [0.1, 0.2, 0.3],
            2.0,
        );
        let material_73 = material(
            8,
            4,
            12,
            BuiltinProgram::Unlit,
            [0.7, 0.3, 0.1],
            [0.4, 0.2, 0.1],
            0.5,
        );
        let packed = GpuBvhBuffers::from_bvh(&Bvh::build(&[
            Triangle::new(
                [-2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.5, 1.0, -1.0],
                73,
            ),
            Triangle::new([1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [1.5, 1.0, -1.0], 11),
        ]));
        let attributes = packed
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| {
                let material = if triangle[9] == 11 {
                    &material_11
                } else {
                    &material_73
                };
                RayHitAttributeRecord::new(
                    RayHitIdentity::new(1, triangle[9], material.handle.generation)
                        .with_material_proof(
                            material.handle.index,
                            material.revision,
                            41,
                            binding.to_words(),
                        ),
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0, 1.0]; 3],
                )
            })
            .collect();
        let bvh = Arc::new(
            packed
                .with_hit_attributes(attributes)
                .expect("one proof-bearing hit-attribute record is required per triangle"),
        );
        let input = LightingTraceSceneInput::from_untextured_unlit_materials(
            Arc::clone(&bvh),
            vec![101, 102],
            1,
            0.001,
            10.0,
            41,
            &tables,
            &[
                LightingTriangleMaterial {
                    primitive_id: 11,
                    material: &material_11,
                    execution: binding,
                    dynamic: DynamicChannels::NONE,
                },
                LightingTriangleMaterial {
                    primitive_id: 73,
                    material: &material_73,
                    execution: binding,
                    dynamic: DynamicChannels::NONE,
                },
            ],
        )
        .unwrap();

        let expected_11 =
            evaluate_unlit_response(surface_sample_from_parameters(&material_11.surface));
        let expected_73 =
            evaluate_unlit_response(surface_sample_from_parameters(&material_73.surface));
        assert_eq!(input.radiance_revision, 41);
        assert_eq!(input.outgoing_radiance[0].rgb, expected_11);
        assert_eq!(input.outgoing_radiance[1].rgb, expected_73);
        assert_eq!(
            input.material_proofs,
            Arc::from([
                LightingMaterialProof {
                    primitive_id: 11,
                    material_handle: material_11.handle,
                    material_revision: 9,
                    material_epoch: 41,
                    execution_binding: binding.to_words(),
                },
                LightingMaterialProof {
                    primitive_id: 73,
                    material_handle: material_73.handle,
                    material_revision: 12,
                    material_epoch: 41,
                    execution_binding: binding.to_words(),
                },
            ])
        );

        let packed_ids = bvh
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| triangle[9])
            .collect::<Vec<_>>();
        assert_ne!(packed_ids, [11, 73], "fixture must exercise BVH reordering");
        let packed_radiance = input.packed_outgoing_radiance().unwrap();
        for (primitive_id, radiance) in packed_ids.iter().zip(packed_radiance.iter()) {
            let expected = if *primitive_id == 11 {
                expected_11
            } else {
                expected_73
            };
            assert_eq!(&radiance[..3], &expected);
        }

        let scene = crate::lighting_trace_gpu::PackedTraceScene::from(bvh.as_ref());
        let rays = [
            crate::lighting_trace_gpu::TraceRay {
                origin: [-1.5, 0.0, 0.0],
                t_min: 0.001,
                direction: [0.0, 0.0, -1.0],
                t_max: 10.0,
                direction_pdf: 0.25,
                selection_u01: 0.25,
            },
            crate::lighting_trace_gpu::TraceRay {
                origin: [1.5, 0.0, 0.0],
                t_min: 0.001,
                direction: [0.0, 0.0, -1.0],
                t_max: 10.0,
                direction_pdf: 0.25,
                selection_u01: 0.75,
            },
        ];
        let hits = crate::lighting_trace_gpu::trace_reference(&scene, &rays).unwrap();
        assert_eq!(hits[0].primitive, 73);
        assert_eq!(hits[1].primitive, 11);
        let surfaces = [
            LightingSurfaceInput {
                position: rays[0].origin,
                normal: [0.0, 0.0, -1.0],
                albedo: [1.0; 3],
                ambient: [0.0; 3],
            },
            LightingSurfaceInput {
                position: rays[1].origin,
                normal: [0.0, 0.0, -1.0],
                albedo: [1.0; 3],
                ambient: [0.0; 3],
            },
        ];
        let packed_rgb = packed_radiance
            .iter()
            .map(|entry| [entry[0], entry[1], entry[2]])
            .collect::<Vec<_>>();
        let candidates = crate::lighting_trace_gpu::build_gi_candidates_reference(
            &scene,
            &rays,
            &hits,
            &surfaces,
            &input.surface_ids,
            7,
            &packed_rgb,
        )
        .unwrap();
        assert_eq!(candidates[0][0].radiance, expected_73);
        assert_eq!(candidates[1][0].radiance, expected_11);
        assert!(candidates.iter().all(|batch| batch[0].target_pdf > 0.0));

        let newer = LightingTraceSceneInput::from_untextured_unlit_materials(
            bvh,
            vec![101, 102],
            1,
            0.001,
            10.0,
            42,
            &tables,
            &[
                LightingTriangleMaterial {
                    primitive_id: 11,
                    material: &material_11,
                    execution: binding,
                    dynamic: DynamicChannels::NONE,
                },
                LightingTriangleMaterial {
                    primitive_id: 73,
                    material: &material_73,
                    execution: binding,
                    dynamic: DynamicChannels::NONE,
                },
            ],
        )
        .unwrap();
        assert_eq!(newer.radiance_revision, 42);
        assert!(newer
            .material_proofs
            .iter()
            .all(|proof| proof.material_epoch == 42));
        assert!(!Arc::ptr_eq(
            &packed_radiance,
            &newer.packed_outgoing_radiance().unwrap()
        ));
    }

    #[test]
    fn authoritative_material_path_rejects_inputs_the_hit_abi_cannot_evaluate() {
        let unlit = ProgramRef::Builtin(BuiltinProgram::Unlit);
        let (unlit_tables, unlit_binding) = execution_fixture(unlit);
        let base = material(
            7,
            3,
            9,
            BuiltinProgram::Unlit,
            [0.2, 0.4, 0.6],
            [0.0; 3],
            0.0,
        );

        let mut textured = base.clone();
        textured
            .textures
            .push(GpuMaterialTexture::non_virtual(4, 2, 1, 0));
        assert_eq!(
            build_one(
                &unlit_tables,
                unlit_binding,
                &textured,
                DynamicChannels::NONE,
            )
            .unwrap_err(),
            LightingMaterialTraceError::TexturedMaterial {
                primitive_id: 19,
                textures: 1,
            }
        );
        assert_eq!(
            build_one(
                &unlit_tables,
                unlit_binding,
                &base,
                DynamicChannels {
                    wetness: 0.5,
                    ..DynamicChannels::NONE
                },
            )
            .unwrap_err(),
            LightingMaterialTraceError::DynamicMaterial { primitive_id: 19 }
        );

        let lit = ProgramRef::Builtin(BuiltinProgram::Lit);
        let (lit_tables, lit_binding) = execution_fixture(lit);
        let lit_material = material(8, 4, 10, BuiltinProgram::Lit, [0.3; 3], [0.0; 3], 0.0);
        assert_eq!(
            build_one(
                &lit_tables,
                lit_binding,
                &lit_material,
                DynamicChannels::NONE,
            )
            .unwrap_err(),
            LightingMaterialTraceError::UnsupportedProgram {
                primitive_id: 19,
                program: lit,
            }
        );
        assert_eq!(
            build_one(&lit_tables, lit_binding, &base, DynamicChannels::NONE,).unwrap_err(),
            LightingMaterialTraceError::PlanExecutionMismatch {
                primitive_id: 19,
                authored: unlit,
                verified: lit,
            }
        );

        let mut invalid = base.clone();
        invalid.surface.base_color[0] = -1.0;
        assert_eq!(
            build_one(
                &unlit_tables,
                unlit_binding,
                &invalid,
                DynamicChannels::NONE,
            )
            .unwrap_err(),
            LightingMaterialTraceError::InvalidRadiance { primitive_id: 19 }
        );

        let mut blended = base.clone();
        blended.response_plan = ShadingPlan::MaskBlend {
            left: unlit,
            right: ProgramRef::Builtin(BuiltinProgram::Npr),
            mask: prism_render_material::ResponseMaskSource::RegionMask,
            domain: prism_render_material::BlendDomain::FullRadiance,
        };
        assert_eq!(
            build_one(
                &unlit_tables,
                unlit_binding,
                &blended,
                DynamicChannels::NONE,
            )
            .unwrap_err(),
            LightingMaterialTraceError::UnsupportedExecutionPlan { primitive_id: 19 }
        );
    }

    #[test]
    fn radiance_table_is_reordered_by_stable_primitive_id() {
        let bvh = Bvh::build(&[
            Triangle::new(
                [-2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.5, 1.0, -1.0],
                73,
            ),
            Triangle::new([1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [1.5, 1.0, -1.0], 11),
        ]);
        let bvh = Arc::new(GpuBvhBuffers::from_bvh(&bvh));
        let input = LightingTraceSceneInput::new(
            Arc::clone(&bvh),
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 11,
                    rgb: [11.0, 1.0, 0.0],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [73.0, 1.0, 0.0],
                },
            ]),
            4,
        );
        let packed = input.packed_outgoing_radiance().unwrap();
        assert!(Arc::ptr_eq(
            &packed,
            &input.clone().packed_outgoing_radiance().unwrap()
        ));
        let mut changed = input.clone();
        Arc::make_mut(&mut changed.outgoing_radiance)[0].rgb[0] = 99.0;
        let changed_packed = changed.packed_outgoing_radiance().unwrap();
        assert!(!Arc::ptr_eq(&packed, &changed_packed));
        assert!(changed_packed.iter().any(|rgb| rgb[0] == 99.0));
        assert_eq!(&*input.packed_outgoing_radiance().unwrap(), &*packed);
        changed.radiance_revision += 1;
        assert!(!Arc::ptr_eq(
            &changed_packed,
            &changed.packed_outgoing_radiance().unwrap()
        ));
        let primitive_ids = bvh
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| triangle[9])
            .collect::<Vec<_>>();
        assert_ne!(
            primitive_ids,
            [11, 73],
            "fixture must exercise BVH reordering"
        );
        assert_eq!(
            packed.iter().map(|entry| entry[0]).collect::<Vec<_>>(),
            primitive_ids
                .iter()
                .map(|primitive_id| *primitive_id as f32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn radiance_table_rejects_duplicate_and_mismatched_primitive_ids() {
        let bvh = Arc::new(GpuBvhBuffers::from_bvh(&Bvh::build(&[
            Triangle::new(
                [-2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.5, 1.0, -1.0],
                73,
            ),
            Triangle::new([1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [1.5, 1.0, -1.0], 12),
        ])));
        let duplicate = LightingTraceSceneInput::new(
            Arc::clone(&bvh),
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [1.0; 3],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [2.0; 3],
                },
            ]),
            0,
        );
        assert_eq!(
            duplicate.packed_outgoing_radiance(),
            Err(crate::lighting_trace_gpu::TracePreparationError::DuplicateRadiancePrimitive(73))
        );
        let mismatch = LightingTraceSceneInput::new(
            bvh,
            vec![1],
            1,
            0.001,
            10.0,
            Arc::from([
                LightingTriangleRadiance {
                    primitive_id: 11,
                    rgb: [1.0; 3],
                },
                LightingTriangleRadiance {
                    primitive_id: 73,
                    rgb: [1.0; 3],
                },
            ]),
            0,
        );
        assert_eq!(
            mismatch.packed_outgoing_radiance(),
            Err(
                crate::lighting_trace_gpu::TracePreparationError::RadiancePrimitiveSetMismatch {
                    radiance_id: 11,
                    packed_id: 12,
                }
            )
        );
    }
}
