//! Verified response identities and deterministic, variable-length work buckets.
use alloc::{collections::BTreeMap, vec::Vec};
use prism_render_material::{
    decode_illumination_blend_weight_strict, BuiltinProgram, ChannelMask, ExecutionFamily,
    GpuMaterialHeader, IlluminationBlendMetadataError, MaterialExecutionBinding,
    MaterialExecutionTables, MaterialFeatureFlags, MaterialRenderClass, ProgramId, ProgramRef,
    RenderDomain, ResponseProgramIndex,
};

const RESPONSE_META_MASK: u32 = u16::MAX as u32;
const RESPONSE_META_CHANNEL_SHIFT: u32 = 16;
const RESPONSE_META_NONE: u32 = u32::MAX;
const INVALID_RESPONSE_BUCKET: u32 = u32::MAX;

#[inline]
fn pack_response_meta(mask_word: u32, channels: ChannelMask) -> u32 {
    (mask_word & RESPONSE_META_MASK) | (channels.0 << RESPONSE_META_CHANNEL_SHIFT)
}

fn validate_continuous_mask(mask_word: u32) -> Result<u32, ClassificationError> {
    match decode_illumination_blend_weight_strict(mask_word) {
        Ok(Some(_)) => Ok(mask_word),
        Ok(None) => Err(ClassificationError::InvalidContinuousMask),
        Err(IlluminationBlendMetadataError::ReservedBits(_)) => {
            Err(ClassificationError::InvalidContinuousMask)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RenderPhase {
    Opaque,
    Transparent,
    Stylized,
    Custom,
}

const PHASES: [RenderPhase; 4] = [
    RenderPhase::Opaque,
    RenderPhase::Transparent,
    RenderPhase::Stylized,
    RenderPhase::Custom,
];
const BUILTINS: [ProgramRef; 3] = [
    ProgramRef::Builtin(BuiltinProgram::Lit),
    ProgramRef::Builtin(BuiltinProgram::Npr),
    ProgramRef::Builtin(BuiltinProgram::Unlit),
];

/// Complete dispatch key, independent of the closure graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResponseBucketKey {
    pub execution_family: ExecutionFamily,
    pub program_id: ProgramId,
    pub render_phase: RenderPhase,
}

impl ResponseBucketKey {
    #[must_use]
    pub const fn new(
        execution_family: ExecutionFamily,
        program_id: ProgramId,
        render_phase: RenderPhase,
    ) -> Self {
        Self {
            execution_family,
            program_id,
            render_phase,
        }
    }

    /// Builtin ids are Lit=0, Npr=1, Unlit=2. Registered id 0 is legal;
    /// construction from execution tables rejects aliases within one family.
    #[must_use]
    pub const fn from_program(
        family: ExecutionFamily,
        program: ProgramRef,
        render_phase: RenderPhase,
    ) -> Self {
        let id = match program {
            ProgramRef::Builtin(BuiltinProgram::Lit) => ProgramId(0),
            ProgramRef::Builtin(BuiltinProgram::Npr) => ProgramId(1),
            ProgramRef::Builtin(BuiltinProgram::Unlit) => ProgramId(2),
            ProgramRef::Registered(id) => id,
        };
        Self::new(family, id, render_phase)
    }
}

/// Dense bucket id and upload key. The fourth upload word is always zero.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResponseBucket {
    pub bucket_id: u32,
    pub execution_family: ExecutionFamily,
    pub program_id: ProgramId,
    pub render_phase: RenderPhase,
    pub reserved: u32,
}

impl ResponseBucket {
    #[must_use]
    pub const fn key(self) -> ResponseBucketKey {
        ResponseBucketKey::new(self.execution_family, self.program_id, self.render_phase)
    }
}

/// Immutable sorted dispatch identities shared by replay, upload and dispatch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResponseBucketTable {
    buckets: Vec<ResponseBucket>,
    by_key: BTreeMap<ResponseBucketKey, u32>,
}

impl ResponseBucketTable {
    /// Keys must describe identities owned by the caller. Use
    /// from_execution_tables to validate registered program namespaces.
    pub fn from_keys(keys: impl IntoIterator<Item = ResponseBucketKey>) -> Self {
        let mut keys: Vec<_> = keys.into_iter().collect();
        keys.sort_unstable();
        keys.dedup();
        let mut table = Self::default();
        for key in keys {
            let bucket_id = table.buckets.len() as u32;
            table.buckets.push(ResponseBucket {
                bucket_id,
                execution_family: key.execution_family,
                program_id: key.program_id,
                render_phase: key.render_phase,
                reserved: 0,
            });
            table.by_key.insert(key, bucket_id);
        }
        table
    }

    #[must_use]
    pub fn builtin() -> Self {
        Self::from_keys(BUILTINS.into_iter().flat_map(|program| {
            builtin_families(program)
                .into_iter()
                .flat_map(move |family| {
                    PHASES
                        .into_iter()
                        .map(move |phase| ResponseBucketKey::from_program(family, program, phase))
                })
        }))
    }

    /// Includes fallback programs, verified plan endpoints, and both binary
    /// Lit/Npr endpoints under each selectable builtin's execution family.
    /// Raw-id aliases between distinct ProgramRefs cannot share this ABI.
    pub fn from_execution_tables(
        execution: &MaterialExecutionTables,
    ) -> Result<Self, ResponseBucketValidationError> {
        let mut identities = BTreeMap::new();
        let mut keys = Vec::new();
        let mut insert = |family, program| -> Result<(), ResponseBucketValidationError> {
            let raw =
                ResponseBucketKey::from_program(family, program, RenderPhase::Opaque).program_id;
            if let Some(&existing) = identities.get(&(family, raw)) {
                if existing != program {
                    return Err(ResponseBucketValidationError::AmbiguousProgramIdentity {
                        execution_family: family,
                        program_id: raw,
                        first: existing,
                        second: program,
                    });
                }
            }
            identities.insert((family, raw), program);
            keys.extend(
                PHASES.map(|phase| ResponseBucketKey::from_program(family, program, phase)),
            );
            Ok(())
        };
        for [left, right, family, _channels, _mask_kind, _mask_index, _domain, _reserved] in
            execution.gpu_plan_rows()
        {
            let family = match family {
                0 => ExecutionFamily::FusedBuiltin,
                1 => ExecutionFamily::SplitResponse,
                2 => ExecutionFamily::Composition,
                _ => return Err(ResponseBucketValidationError::InvalidExecutionEntry),
            };
            for index in [left, right] {
                if index == u32::MAX {
                    continue;
                }
                let index = ResponseProgramIndex::from_raw(index, execution.programs())
                    .map_err(|_| ResponseBucketValidationError::InvalidExecutionEntry)?;
                let program = execution
                    .program(index)
                    .map_err(|_| ResponseBucketValidationError::InvalidExecutionEntry)?
                    .program();
                for selected in selectable_programs(program) {
                    insert(execution_family_for_program(family, selected), selected)?;
                }
            }
        }
        Ok(Self::from_keys(keys))
    }

    #[must_use]
    pub fn buckets(&self) -> &[ResponseBucket] {
        &self.buckets
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
    #[must_use]
    pub fn bucket_id(&self, key: ResponseBucketKey) -> Option<u32> {
        self.by_key.get(&key).copied()
    }

    /// The backend registry owns the compiled entry catalogue, not this crate.
    pub fn validate_compiled_entries(
        &self,
        mut is_compiled: impl FnMut(ResponseBucketKey) -> bool,
    ) -> Result<(), ResponseBucketValidationError> {
        for bucket in &self.buckets {
            if !is_compiled(bucket.key()) {
                return Err(ResponseBucketValidationError::MissingCompiledEntry {
                    bucket: *bucket,
                });
            }
        }
        Ok(())
    }
}

fn builtin_families(program: ProgramRef) -> Vec<ExecutionFamily> {
    match program {
        ProgramRef::Builtin(builtin) => {
            let manifest = prism_render_material::ResponseManifest::from_builtin(builtin)
                .expect("generated builtin response manifest must exist");
            let mut families = Vec::with_capacity(2);
            if manifest.execution.supports_fused {
                families.push(ExecutionFamily::FusedBuiltin);
            }
            if manifest.execution.supports_split {
                families.push(ExecutionFamily::SplitResponse);
            }
            families
        }
        ProgramRef::Registered(_) => vec![ExecutionFamily::SplitResponse],
    }
}

fn selectable_programs(program: ProgramRef) -> Vec<ProgramRef> {
    match program {
        ProgramRef::Builtin(BuiltinProgram::Lit | BuiltinProgram::Npr) => {
            vec![BUILTINS[0], BUILTINS[1]]
        }
        _ => vec![program],
    }
}

fn execution_family_for_program(family: ExecutionFamily, program: ProgramRef) -> ExecutionFamily {
    if matches!(family, ExecutionFamily::Composition) {
        return family;
    }
    if matches!(family, ExecutionFamily::SplitResponse) {
        return family;
    }
    match program {
        ProgramRef::Builtin(builtin) => {
            let manifest = prism_render_material::ResponseManifest::from_builtin(builtin)
                .expect("generated builtin response manifest must exist");
            if manifest.execution.supports_fused {
                ExecutionFamily::FusedBuiltin
            } else {
                ExecutionFamily::SplitResponse
            }
        }
        ProgramRef::Registered(_) => ExecutionFamily::SplitResponse,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseBucketValidationError {
    MissingCompiledEntry {
        bucket: ResponseBucket,
    },
    InvalidExecutionEntry,
    AmbiguousProgramIdentity {
        execution_family: ExecutionFamily,
        program_id: ProgramId,
        first: ProgramRef,
        second: ProgramRef,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationError {
    Inactive,
    NonSurfaceClass,
    /// Partial, unknown, stale, or ownership-invalid execution tuple.
    InvalidExecutionBinding,
    ExecutionContextRequired,
    InvalidProgramId,
    InvalidContinuousMask,
    InvalidContinuousPlan,
}

/// Four ABI response kinds, not closure identities.
#[must_use]
pub const fn response_kind_for_program(program: ProgramRef) -> u32 {
    match program {
        ProgramRef::Builtin(BuiltinProgram::Lit) => 0,
        ProgramRef::Builtin(BuiltinProgram::Npr) => 1,
        ProgramRef::Builtin(BuiltinProgram::Unlit) => 2,
        ProgramRef::Registered(_) => 3,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadingWorkItem {
    pub pixel_index: u32,
    pub material_index: u32,
    /// Dense id in the current dynamic bucket table.
    pub shading_class: u32,
    /// Lit=0, Npr=1, Unlit=2, Registered=3 for the left arm.
    pub response_kind: u32,
    pub flags: u32,
    /// Dense id for the right response arm, or `u32::MAX` for single-arm work.
    pub right_shading_class: u32,
    /// Response kind for the right response arm.
    pub right_response_kind: u32,
    /// Low 16 bits are the continuous mask UNORM; high bits are the channel
    /// mask. `u32::MAX` means that this work item is not a blend.
    pub response_meta: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingWorkPlanError {
    Classification {
        pixel_index: u32,
        error: ClassificationError,
    },
    MissingBucket {
        pixel_index: u32,
        key: ResponseBucketKey,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadingWorkPlan {
    pub offsets: Vec<u32>,
    pub counts: Vec<u32>,
    pub buckets: ResponseBucketTable,
    pub work: Vec<ShadingWorkItem>,
}

impl ShadingWorkPlan {
    pub fn build(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
    ) -> Result<Self, ShadingWorkPlanError> {
        Self::build_with_region(
            pixels.into_iter().map(|(i, p)| (i, p, u32::MAX)),
            materials,
            capacity,
        )
    }

    pub fn build_with_region(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel, u32)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
    ) -> Result<Self, ShadingWorkPlanError> {
        Self::build_with_region_and_buckets(
            pixels,
            materials,
            capacity,
            &ResponseBucketTable::builtin(),
        )
    }

    pub fn build_with_region_and_buckets(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel, u32)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
        buckets: &ResponseBucketTable,
    ) -> Result<Self, ShadingWorkPlanError> {
        Self::build_with_region_and_buckets_and_execution(
            pixels, materials, capacity, buckets, None,
        )
    }

    /// Background and stale visibility are filtered. A live material's invalid
    /// execution binding or missing dispatch entry is a hard error.
    pub fn build_with_region_and_buckets_and_execution(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel, u32)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
        buckets: &ResponseBucketTable,
        execution: Option<&MaterialExecutionTables>,
    ) -> Result<Self, ShadingWorkPlanError> {
        let mut classified: Vec<Vec<ShadingWorkItem>> =
            (0..buckets.len()).map(|_| Vec::new()).collect();
        let mut accepted = 0usize;
        for (pixel_index, pixel, region) in pixels {
            if !pixel.is_valid() || accepted >= capacity as usize {
                continue;
            }
            let Some(header) = materials.get(pixel.material_index as usize) else {
                continue;
            };
            if header.generation != pixel.material_generation {
                continue;
            }
            let item = if header.feature_flags & MaterialFeatureFlags::ILLUMINATION_BLEND.0 != 0 {
                let (family, left, right, channels) =
                    continuous_response_arms(header, execution, pixel.coverage_and_flags).map_err(
                        |error| ShadingWorkPlanError::Classification { pixel_index, error },
                    )?;
                let left_key = ResponseBucketKey::from_program(
                    family,
                    left,
                    render_phase_for_header(header, left),
                );
                let right_key = ResponseBucketKey::from_program(
                    family,
                    right,
                    render_phase_for_header(header, right),
                );
                let left_bucket =
                    buckets
                        .bucket_id(left_key)
                        .ok_or(ShadingWorkPlanError::MissingBucket {
                            pixel_index,
                            key: left_key,
                        })?;
                let right_bucket =
                    buckets
                        .bucket_id(right_key)
                        .ok_or(ShadingWorkPlanError::MissingBucket {
                            pixel_index,
                            key: right_key,
                        })?;
                let response_meta = pack_response_meta(
                    validate_continuous_mask(pixel.coverage_and_flags).map_err(|error| {
                        ShadingWorkPlanError::Classification { pixel_index, error }
                    })?,
                    channels,
                );
                (
                    left_bucket,
                    ShadingWorkItem {
                        pixel_index,
                        material_index: pixel.material_index,
                        shading_class: left_bucket,
                        response_kind: response_kind_for_program(left),
                        flags: header.feature_flags,
                        right_shading_class: right_bucket,
                        right_response_kind: response_kind_for_program(right),
                        response_meta,
                    },
                )
            } else {
                let (family, program) = selected_execution_identity(header, execution, region)
                    .map_err(|error| ShadingWorkPlanError::Classification { pixel_index, error })?;
                let key = ResponseBucketKey::from_program(
                    family,
                    program,
                    render_phase_for_header(header, program),
                );
                let bucket_id = buckets
                    .bucket_id(key)
                    .ok_or(ShadingWorkPlanError::MissingBucket { pixel_index, key })?;
                (
                    bucket_id,
                    ShadingWorkItem {
                        pixel_index,
                        material_index: pixel.material_index,
                        shading_class: bucket_id,
                        response_kind: response_kind_for_program(program),
                        flags: header.feature_flags,
                        right_shading_class: INVALID_RESPONSE_BUCKET,
                        right_response_kind: 0,
                        response_meta: RESPONSE_META_NONE,
                    },
                )
            };
            classified[item.0 as usize].push(item.1);
            accepted += 1;
        }
        let mut plan = Self {
            offsets: vec![0; buckets.len()],
            counts: vec![0; buckets.len()],
            buckets: buckets.clone(),
            work: Vec::new(),
        };
        for (bucket, items) in classified.into_iter().enumerate() {
            plan.offsets[bucket] = plan.work.len() as u32;
            plan.counts[bucket] = items.len() as u32;
            plan.work.extend(items);
        }
        Ok(plan)
    }
}

fn validate_surface(header: &GpuMaterialHeader) -> Result<(), ClassificationError> {
    if header.is_active == 0 {
        return Err(ClassificationError::Inactive);
    }
    // This classifier feeds opaque visibility resolve, not forward blend bins.
    // Transmissive/Transparent/Additive are known classes but are ineligible.
    if !matches!(
        MaterialRenderClass::from_u32(header.render_class),
        Some(
            MaterialRenderClass::Opaque
                | MaterialRenderClass::OpaqueTwoSided
                | MaterialRenderClass::Masked
                | MaterialRenderClass::MaskedTwoSided
        )
    ) || !matches!(header.render_domain, x if x == RenderDomain::Surface as u32 || x == RenderDomain::Hair as u32)
    {
        return Err(ClassificationError::NonSurfaceClass);
    }
    Ok(())
}

fn response_execution_identity(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
) -> Result<(ExecutionFamily, ProgramRef), ClassificationError> {
    let binding = match response_execution_binding(header, execution)? {
        Some(binding) => binding,
        None => return Ok((ExecutionFamily::FusedBuiltin, BUILTINS[0])),
    };
    let execution = execution.ok_or(ClassificationError::ExecutionContextRequired)?;
    let family = execution
        .execution_family_for_binding(binding)
        .map_err(|_| ClassificationError::InvalidExecutionBinding)?;
    let program = execution
        .program(binding.response_program_index)
        .map_err(|_| ClassificationError::InvalidExecutionBinding)?
        .program();
    Ok((family, program))
}

fn response_execution_binding(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
) -> Result<Option<MaterialExecutionBinding>, ClassificationError> {
    let words = header.execution_binding_words();
    let sentinel = GpuMaterialHeader::EXECUTION_BINDING_SENTINEL;
    if words.iter().all(|word| *word == sentinel) {
        return Ok(None);
    }
    if words.contains(&sentinel) {
        return Err(ClassificationError::InvalidExecutionBinding);
    }
    let execution = execution.ok_or(ClassificationError::ExecutionContextRequired)?;
    execution
        .bind_raw(
            words[0],
            words[1],
            words[2],
            u64::from(words[3]) | (u64::from(words[4]) << 32),
        )
        .map(Some)
        .map_err(|_| ClassificationError::InvalidExecutionBinding)
}

fn continuous_response_arms(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
    mask_word: u32,
) -> Result<(ExecutionFamily, ProgramRef, ProgramRef, ChannelMask), ClassificationError> {
    validate_surface(header)?;
    let execution = execution.ok_or(ClassificationError::ExecutionContextRequired)?;
    let binding = response_execution_binding(header, Some(execution))?
        .ok_or(ClassificationError::InvalidContinuousPlan)?;
    let (left, right, family) = execution
        .plan_endpoints(binding.shading_plan_index)
        .map_err(|_| ClassificationError::InvalidContinuousPlan)?;
    let right = right.ok_or(ClassificationError::InvalidContinuousPlan)?;
    let channels = execution
        .plan_blended_channels(binding.shading_plan_index)
        .map_err(|_| ClassificationError::InvalidContinuousPlan)?;
    if channels.0 > u16::MAX as u32 {
        return Err(ClassificationError::InvalidContinuousPlan);
    }
    let mask_word = validate_continuous_mask(mask_word)?;
    if pack_response_meta(mask_word, channels) == RESPONSE_META_NONE {
        return Err(ClassificationError::InvalidContinuousPlan);
    }
    Ok((family, left, right, channels))
}

fn select_program(header: &GpuMaterialHeader, program: ProgramRef, region: u32) -> ProgramRef {
    if header.feature_flags & MaterialFeatureFlags::REGION_ILLUMINATION.0 == 0 {
        return program;
    }
    if matches!(
        program,
        ProgramRef::Builtin(BuiltinProgram::Lit | BuiltinProgram::Npr)
    ) {
        match region {
            0 => BUILTINS[0],
            1 => BUILTINS[1],
            _ => program,
        }
    } else {
        program
    }
}

fn selected_execution_identity(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
    region: u32,
) -> Result<(ExecutionFamily, ProgramRef), ClassificationError> {
    validate_surface(header)?;
    let (family, program) = response_execution_identity(header, execution)?;
    let selected = select_program(header, program, region);
    Ok((execution_family_for_program(family, selected), selected))
}

fn render_phase_for_header(header: &GpuMaterialHeader, program: ProgramRef) -> RenderPhase {
    if matches!(
        MaterialRenderClass::from_u32(header.render_class),
        Some(
            MaterialRenderClass::Transmissive
                | MaterialRenderClass::Transparent
                | MaterialRenderClass::Additive
        )
    ) {
        return RenderPhase::Transparent;
    }
    match program {
        ProgramRef::Builtin(BuiltinProgram::Npr) => RenderPhase::Stylized,
        ProgramRef::Registered(_) => RenderPhase::Custom,
        _ => RenderPhase::Opaque,
    }
}

/// Binary selector 0/1 changes the actual builtin Lit/Npr entry and its phase.
/// Unlit/Registered and other selectors preserve the verified program.
pub fn response_bucket_key_for_header(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
    region: u32,
) -> Result<ResponseBucketKey, ClassificationError> {
    let (family, program) = selected_execution_identity(header, execution, region)?;
    Ok(ResponseBucketKey::from_program(
        family,
        program,
        render_phase_for_header(header, program),
    ))
}

pub fn classify_material_header(
    header: &GpuMaterialHeader,
) -> Result<ProgramRef, ClassificationError> {
    classify_material_header_with_context(header, None, u32::MAX)
}
pub fn classify_material_header_with_region(
    header: &GpuMaterialHeader,
    region: u32,
) -> Result<ProgramRef, ClassificationError> {
    classify_material_header_with_context(header, None, region)
}
pub fn classify_material_header_with_execution(
    header: &GpuMaterialHeader,
    execution: &MaterialExecutionTables,
    region: u32,
) -> Result<ProgramRef, ClassificationError> {
    classify_material_header_with_context(header, Some(execution), region)
}
pub fn classify_material_header_with_context(
    header: &GpuMaterialHeader,
    execution: Option<&MaterialExecutionTables>,
    region: u32,
) -> Result<ProgramRef, ClassificationError> {
    selected_execution_identity(header, execution, region).map(|(_, program)| program)
}

/// Direct-program entry for callers that have already verified execution.
/// Without tables it cannot verify package ownership; malformed partial
/// tuples and reserved program ids are still rejected.
pub fn classify_material_header_with_program(
    header: &GpuMaterialHeader,
    program: ProgramRef,
    region: u32,
) -> Result<ProgramRef, ClassificationError> {
    validate_surface(header)?;
    let words = header.execution_binding_words();
    let sentinel = GpuMaterialHeader::EXECUTION_BINDING_SENTINEL;
    if words.contains(&sentinel) && !words.iter().all(|word| *word == sentinel) {
        return Err(ClassificationError::InvalidExecutionBinding);
    }
    if program == ProgramRef::Registered(ProgramId(u32::MAX)) {
        return Err(ClassificationError::InvalidProgramId);
    }
    Ok(select_program(header, program, region))
}

#[cfg(test)]
pub(crate) mod tests;
