//! Explicit compiled material execution tables.
//!
//! The table is the only place where authoring identities become hot-path
//! indices.  A package can enter the table only through a validated
//! [`CompiledEntry`], and every binding is checked against the owning plan,
//! program, package and permutation space before it is returned.

use alloc::{string::String, vec::Vec};

use prism_render_architecture::shader_package::{
    ArtifactDigest, ExecutionFamily, ModuleId, PackageState, PermutationError, PermutationIndex,
    PermutationSpace, ShaderPackageManifest, ShaderPackageRegistry,
};

use crate::{
    BuiltinProgram, ChannelMask, CompiledEntry, ProgramId, ProgramRef, ResponseMaskSource,
    ShadingPlan,
};

/// GPU response-program kind written by the execution-table upload.
pub const RESPONSE_PROGRAM_KIND_LIT: u32 = 0;
pub const RESPONSE_PROGRAM_KIND_NPR: u32 = 1;
pub const RESPONSE_PROGRAM_KIND_UNLIT: u32 = 2;
pub const RESPONSE_PROGRAM_KIND_CUSTOM: u32 = 3;

/// Fixed ABI width of one execution-plan row. Keeping the row a multiple of
/// 16 bytes makes storage-buffer indexing portable and leaves explicit slots
/// for mask resolution rather than encoding semantics in a hash.
pub const EXECUTION_PLAN_ROW_WORDS: usize = 8;
pub const EXECUTION_MASK_NONE: u32 = 0;
pub const EXECUTION_MASK_TEXTURE: u32 = 1;
pub const EXECUTION_MASK_VERTEX_ATTRIBUTE: u32 = 2;
pub const EXECUTION_MASK_UV_CHANNEL: u32 = 3;
pub const EXECUTION_MASK_REGION: u32 = 4;
pub const EXECUTION_MASK_DYNAMIC_CHANNEL: u32 = 5;

/// The kind of dense table index that failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTable {
    Plan,
    Program,
    Package,
}

/// A checked dense index into one execution table.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PlanIndex(u32);

/// A checked dense index into the compiled response-program table.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ResponseProgramIndex(u32);

/// A checked dense index into the verified shader-package table.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PackageIndex(u32);

pub type ShadingPlanIndex = PlanIndex;
pub type ProgramIndex = ResponseProgramIndex;

macro_rules! dense_index_impl {
    ($name:ident, $kind:expr) => {
        impl $name {
            /// The raw value which is never a valid table index.
            pub const INVALID: u32 = u32::MAX;

            /// Builds an index only when `raw` addresses an existing entry.
            pub fn from_raw(raw: u32, len: usize) -> Result<Self, MaterialExecutionError> {
                if raw == Self::INVALID {
                    return Err(MaterialExecutionError::ReservedIndex { kind: $kind });
                }
                if (raw as usize) >= len {
                    return Err(MaterialExecutionError::IndexOutOfBounds {
                        kind: $kind,
                        index: raw,
                        len,
                    });
                }
                Ok(Self(raw))
            }

            /// Returns the dense table coordinate.
            #[must_use]
            pub const fn get(self) -> u32 {
                self.0
            }
        }
    };
}

dense_index_impl!(PlanIndex, ExecutionTable::Plan);
dense_index_impl!(ResponseProgramIndex, ExecutionTable::Program);
dense_index_impl!(PackageIndex, ExecutionTable::Package);

/// Errors raised while building or resolving execution bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialExecutionError {
    ReservedIndex {
        kind: ExecutionTable,
    },
    IndexOutOfBounds {
        kind: ExecutionTable,
        index: u32,
        len: usize,
    },
    InvalidProgramId,
    PackageUnavailable,
    PackageNotReady,
    PackageNotVerified,
    PackageChanged,
    PackageCodeChanged,
    DuplicateProgram(ProgramRef),
    DuplicatePlan,
    UnknownProgram(ResponseProgramIndex),
    PackageMismatch {
        program: ResponseProgramIndex,
        expected: PackageIndex,
        found: PackageIndex,
    },
    PlanProgramMismatch {
        plan: PlanIndex,
        program: ResponseProgramIndex,
    },
    InvalidPlanShape,
    InvalidExecutionFamily,
    Permutation(PermutationError),
    PermutationCountMismatch {
        declared: u32,
        actual: u64,
    },
    UnresolvedMaskSource(String),
}

/// One verified package table entry.
#[derive(Clone, Debug)]
pub struct MaterialExecutionPackage {
    manifest: ShaderPackageManifest,
    digest: ArtifactDigest,
}

impl MaterialExecutionPackage {
    #[must_use]
    pub fn manifest(&self) -> &ShaderPackageManifest {
        &self.manifest
    }

    #[must_use]
    pub const fn digest(&self) -> ArtifactDigest {
        self.digest
    }
}

/// One compiled response-program table entry.
#[derive(Clone, Debug)]
pub struct MaterialExecutionProgram {
    program: ProgramRef,
    package: PackageIndex,
    module_id: ModuleId,
    permutations: PermutationSpace,
    digest: ArtifactDigest,
    /// The validated executable and its response-function source payload.
    /// Keeping this cold-path copy beside the dense identity lets the render
    /// linker build the finite WESL dispatch table without re-opening an
    /// authoring registry or inventing a fallback implementation.
    compiled: CompiledEntry,
}

impl MaterialExecutionProgram {
    #[must_use]
    pub const fn program(&self) -> ProgramRef {
        self.program
    }

    #[must_use]
    pub const fn package(&self) -> PackageIndex {
        self.package
    }

    #[must_use]
    pub const fn module_id(&self) -> ModuleId {
        self.module_id
    }

    #[must_use]
    pub const fn permutations(&self) -> &PermutationSpace {
        &self.permutations
    }

    #[must_use]
    pub const fn digest(&self) -> ArtifactDigest {
        self.digest
    }

    #[must_use]
    pub fn compiled_entry(&self) -> &CompiledEntry {
        &self.compiled
    }
}

#[derive(Clone, Debug)]
struct MaterialExecutionPlan {
    source: ShadingPlan,
    execution_family: ExecutionFamily,
    left: ResponseProgramIndex,
    right: Option<ResponseProgramIndex>,
    blended_channels: ChannelMask,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
}

/// Resolves authoring mask names to stable runtime indices. Intrinsic masks
/// need no resolver; texture/channel names must be admitted by the owning
/// asset/material registry before entering the GPU table.
pub trait MaskSourceResolver {
    fn texture_index(&self, semantic: &str) -> Option<u32>;
}

fn resolve_mask_source(
    source: &ShadingPlan,
    resolver: Option<&dyn MaskSourceResolver>,
) -> Result<(u32, u32), MaterialExecutionError> {
    let ShadingPlan::MaskBlend { mask, .. } = source else {
        return Ok((EXECUTION_MASK_NONE, 0));
    };
    match mask {
        ResponseMaskSource::VertexAttr => Ok((EXECUTION_MASK_VERTEX_ATTRIBUTE, 0)),
        ResponseMaskSource::UvChannel => Ok((EXECUTION_MASK_UV_CHANNEL, 0)),
        ResponseMaskSource::RegionMask => Ok((EXECUTION_MASK_REGION, 0)),
        ResponseMaskSource::Channel(name) => crate::SURFACE_CHANNELS
            .iter()
            .find(|channel| channel.id == name)
            .map(|channel| (EXECUTION_MASK_DYNAMIC_CHANNEL, channel.header_word))
            .ok_or_else(|| MaterialExecutionError::UnresolvedMaskSource(name.clone())),
        ResponseMaskSource::Texture(name) => resolver
            .and_then(|resolver| resolver.texture_index(name))
            .map(|index| (EXECUTION_MASK_TEXTURE, index))
            .ok_or_else(|| MaterialExecutionError::UnresolvedMaskSource(name.clone())),
    }
}

fn blend_domain_word(source: &ShadingPlan) -> u32 {
    match source {
        ShadingPlan::Single { .. } => 0,
        ShadingPlan::MaskBlend { domain, .. } => match domain {
            crate::BlendDomain::FullRadiance => 0,
            crate::BlendDomain::DirectOnly => 1,
            crate::BlendDomain::CustomChannelMask(_) => 2,
        },
    }
}

/// A fully checked hot-path execution binding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialExecutionBinding {
    pub shading_plan_index: PlanIndex,
    pub response_program_index: ResponseProgramIndex,
    pub package_index: PackageIndex,
    pub permutation_index: PermutationIndex,
}

impl MaterialExecutionBinding {
    /// Returns the five words used by a GPU header extension: three dense
    /// table indices followed by the checked permutation's low/high words.
    #[must_use]
    pub const fn to_words(self) -> [u32; 5] {
        [
            self.shading_plan_index.get(),
            self.response_program_index.get(),
            self.package_index.get(),
            self.permutation_index.get() as u32,
            (self.permutation_index.get() >> 32) as u32,
        ]
    }
}

/// Compiled plan/program/package tables used to resolve material headers.
#[derive(Clone, Debug, Default)]
pub struct MaterialExecutionTables {
    plans: Vec<MaterialExecutionPlan>,
    programs: Vec<MaterialExecutionProgram>,
    packages: Vec<MaterialExecutionPackage>,
}

impl MaterialExecutionTables {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a package only after checking that the registry contains the same
    /// ready state and the same verified executable digest.
    pub fn register_package(
        &mut self,
        compiled: &CompiledEntry,
        packages: &ShaderPackageRegistry,
    ) -> Result<PackageIndex, MaterialExecutionError> {
        let manifest = compiled.package_manifest();
        let registered = packages
            .get(&manifest.id)
            .ok_or(MaterialExecutionError::PackageUnavailable)?;
        if registered.manifest.id != manifest.id
            || registered.manifest.source_revision != manifest.source_revision
            || registered.manifest.entry_point != manifest.entry_point
            || registered.manifest.abi_version != manifest.abi_version
            || registered.manifest.abi_hash != manifest.abi_hash
            || registered.manifest.permutation_count != manifest.permutation_count
        {
            return Err(MaterialExecutionError::PackageChanged);
        }
        if registered.state != PackageState::Ready {
            return Err(MaterialExecutionError::PackageNotReady);
        }
        let artifact = packages
            .verified_artifact(&manifest.id)
            .ok_or(MaterialExecutionError::PackageNotVerified)?;
        if artifact.digest() != compiled.digest() {
            return Err(MaterialExecutionError::PackageCodeChanged);
        }

        if let Some((index, existing)) = self
            .packages
            .iter()
            .enumerate()
            .find(|(_, existing)| existing.manifest.id == manifest.id)
        {
            if existing.digest != compiled.digest() || !same_manifest(&existing.manifest, manifest)
            {
                return Err(MaterialExecutionError::PackageChanged);
            }
            let raw =
                u32::try_from(index).map_err(|_| MaterialExecutionError::IndexOutOfBounds {
                    kind: ExecutionTable::Package,
                    index: u32::MAX,
                    len: self.packages.len(),
                })?;
            return PackageIndex::from_raw(raw, self.packages.len());
        }

        let index = checked_new_package_index(self.packages.len())?;
        self.packages.push(MaterialExecutionPackage {
            manifest: manifest.clone(),
            digest: compiled.digest(),
        });
        Ok(index)
    }

    /// Adds a compiled response entry. The permutation space must exactly
    /// describe the package's declared compiled variant count.
    pub fn register_program(
        &mut self,
        program: ProgramRef,
        compiled: &CompiledEntry,
        package: PackageIndex,
        module_id: ModuleId,
        permutations: PermutationSpace,
    ) -> Result<ResponseProgramIndex, MaterialExecutionError> {
        if matches!(program, ProgramRef::Registered(ProgramId(id)) if id == u32::MAX) {
            return Err(MaterialExecutionError::InvalidProgramId);
        }
        let package_entry = self.package(package)?;
        if !same_manifest(&package_entry.manifest, compiled.package_manifest())
            || package_entry.digest != compiled.digest()
        {
            return Err(MaterialExecutionError::PackageCodeChanged);
        }
        let actual = permutations
            .checked_total()
            .map_err(MaterialExecutionError::Permutation)?;
        if actual != u64::from(compiled.package_manifest().permutation_count) {
            return Err(MaterialExecutionError::PermutationCountMismatch {
                declared: compiled.package_manifest().permutation_count,
                actual,
            });
        }
        if self.programs.iter().any(|entry| entry.program == program) {
            return Err(MaterialExecutionError::DuplicateProgram(program));
        }
        let index = checked_new_program_index(self.programs.len())?;
        self.programs.push(MaterialExecutionProgram {
            program,
            package,
            module_id,
            permutations,
            digest: compiled.digest(),
            compiled: compiled.clone(),
        });
        Ok(index)
    }

    /// Adds a plan after checking that its dense program references match the
    /// authoring plan shape. `execution_family` is explicit and therefore is
    /// not inferred from a hash or from a stale authoring field.
    pub fn register_plan(
        &mut self,
        source: ShadingPlan,
        left: ResponseProgramIndex,
        right: Option<ResponseProgramIndex>,
        execution_family: ExecutionFamily,
        blended_channels: ChannelMask,
    ) -> Result<PlanIndex, MaterialExecutionError> {
        self.register_plan_with_mask_resolver(
            source,
            left,
            right,
            execution_family,
            blended_channels,
            None,
        )
    }

    pub fn register_plan_with_mask_resolver(
        &mut self,
        source: ShadingPlan,
        left: ResponseProgramIndex,
        right: Option<ResponseProgramIndex>,
        execution_family: ExecutionFamily,
        blended_channels: ChannelMask,
        resolver: Option<&dyn MaskSourceResolver>,
    ) -> Result<PlanIndex, MaterialExecutionError> {
        let (mask_kind, mask_index) = resolve_mask_source(&source, resolver)?;
        let left_entry = self.program(left)?;
        if let Some(right) = right {
            self.program(right)?;
        }
        match &source {
            ShadingPlan::Single { .. } if right.is_some() => {
                return Err(MaterialExecutionError::InvalidPlanShape)
            }
            ShadingPlan::MaskBlend { .. } if right.is_none() => {
                return Err(MaterialExecutionError::InvalidPlanShape)
            }
            ShadingPlan::Single { program } => {
                if left_entry.program != *program {
                    return Err(MaterialExecutionError::PlanProgramMismatch {
                        plan: PlanIndex(u32::MAX),
                        program: left,
                    });
                }
                if execution_family == ExecutionFamily::Composition
                    || (execution_family == ExecutionFamily::FusedBuiltin
                        && !matches!(program, ProgramRef::Builtin(_)))
                {
                    return Err(MaterialExecutionError::InvalidExecutionFamily);
                }
            }
            ShadingPlan::MaskBlend {
                left: source_left,
                right: source_right,
                ..
            } => {
                let right = match right {
                    Some(right) => right,
                    None => return Err(MaterialExecutionError::InvalidPlanShape),
                };
                if left_entry.program != *source_left
                    || self.program(right)?.program != *source_right
                {
                    return Err(MaterialExecutionError::PlanProgramMismatch {
                        plan: PlanIndex(u32::MAX),
                        program: left,
                    });
                }
                let builtin_pair = matches!(source_left, ProgramRef::Builtin(_))
                    && matches!(source_right, ProgramRef::Builtin(_));
                if execution_family == ExecutionFamily::SplitResponse
                    || (execution_family == ExecutionFamily::FusedBuiltin && !builtin_pair)
                {
                    return Err(MaterialExecutionError::InvalidExecutionFamily);
                }
            }
        }
        if self.plans.iter().any(|entry| {
            entry.source == source
                && entry.left == left
                && entry.right == right
                && entry.execution_family == execution_family
                && entry.blended_channels == blended_channels
                && entry.mask_kind == mask_kind
                && entry.mask_index == mask_index
                && entry.blend_domain == blend_domain_word(&source)
        }) {
            return Err(MaterialExecutionError::DuplicatePlan);
        }
        let index = checked_new_plan_index(self.plans.len())?;
        let blend_domain = blend_domain_word(&source);
        self.plans.push(MaterialExecutionPlan {
            source,
            execution_family,
            left,
            right,
            blended_channels,
            mask_kind,
            mask_index,
            blend_domain,
        });
        Ok(index)
    }

    /// Registers a plan whose response contracts have already passed the
    /// authoritative [`ResponseRegistry`] validation. The compiled entries in
    /// `validated` must be the same entries used to create the program rows.
    pub fn register_validated_plan(
        &mut self,
        source: ShadingPlan,
        validated: &crate::ValidatedPlan<'_>,
        left: ResponseProgramIndex,
        right: Option<ResponseProgramIndex>,
    ) -> Result<PlanIndex, MaterialExecutionError> {
        let left_entry = self.program(left)?;
        if left_entry.digest != validated.left.compiled_entry().digest() {
            return Err(MaterialExecutionError::PackageCodeChanged);
        }
        if let Some(right_index) = right {
            let right_entry = self.program(right_index)?;
            let validated_right = validated
                .right
                .ok_or(MaterialExecutionError::InvalidPlanShape)?;
            if right_entry.digest != validated_right.compiled_entry().digest() {
                return Err(MaterialExecutionError::PackageCodeChanged);
            }
        } else if validated.right.is_some() {
            return Err(MaterialExecutionError::InvalidPlanShape);
        }
        self.register_plan(
            source,
            left,
            right,
            validated.execution_family,
            validated.blended_channels,
        )
    }

    /// Resolves raw serialized indices only after checking every dense table
    /// coordinate and the package/permutation ownership relation.
    pub fn bind_raw(
        &self,
        plan: u32,
        program: u32,
        package: u32,
        permutation: u64,
    ) -> Result<MaterialExecutionBinding, MaterialExecutionError> {
        let plan = PlanIndex::from_raw(plan, self.plans.len())?;
        let program = ResponseProgramIndex::from_raw(program, self.programs.len())?;
        let package = PackageIndex::from_raw(package, self.packages.len())?;
        let permutation = self
            .program(program)?
            .permutations
            .check_index(permutation)
            .map_err(MaterialExecutionError::Permutation)?;
        self.bind(plan, program, package, permutation)
    }

    /// Resolves a typed binding and verifies that the selected program belongs
    /// to the selected plan and package.
    pub fn bind(
        &self,
        plan: PlanIndex,
        program: ResponseProgramIndex,
        package: PackageIndex,
        permutation: PermutationIndex,
    ) -> Result<MaterialExecutionBinding, MaterialExecutionError> {
        let plan_entry = self.plan(plan)?;
        let program_entry = self.program(program)?;
        self.package(package)?;
        if plan_entry.left != program && plan_entry.right != Some(program) {
            return Err(MaterialExecutionError::PlanProgramMismatch { plan, program });
        }
        if program_entry.package != package {
            return Err(MaterialExecutionError::PackageMismatch {
                program,
                expected: program_entry.package,
                found: package,
            });
        }
        let permutation = program_entry
            .permutations
            .check_index(permutation.get())
            .map_err(MaterialExecutionError::Permutation)?;
        Ok(MaterialExecutionBinding {
            shading_plan_index: plan,
            response_program_index: program,
            package_index: package,
            permutation_index: permutation,
        })
    }

    #[must_use]
    pub fn plans(&self) -> usize {
        self.plans.len()
    }

    #[must_use]
    pub fn programs(&self) -> usize {
        self.programs.len()
    }

    #[must_use]
    pub fn packages(&self) -> usize {
        self.packages.len()
    }

    /// Returns the compact GPU plan rows. The row contains the program indices
    /// accepted by the plan plus its verified execution family; the shader uses
    /// all three words to reject a forged plan/program tuple and to derive the
    /// response bucket key without a legacy-class projection.
    #[must_use]
    pub fn gpu_plan_rows(&self) -> Vec<[u32; EXECUTION_PLAN_ROW_WORDS]> {
        self.plans
            .iter()
            .map(|plan| {
                [
                    plan.left.get(),
                    plan.right.map_or(u32::MAX, |index| index.get()),
                    plan.execution_family as u32,
                    plan.blended_channels.0,
                    plan.mask_kind,
                    plan.mask_index,
                    plan.blend_domain,
                    0,
                ]
            })
            .collect()
    }

    /// Returns compact GPU program rows: response kind, owning package and the
    /// full 64-bit permutation-space count split into two words, followed by
    /// the stable `ProgramId` used by response bucket identity.
    #[must_use]
    pub fn gpu_program_rows(&self) -> Vec<[u32; 5]> {
        self.programs
            .iter()
            .map(|program| {
                let kind = match program.program {
                    ProgramRef::Builtin(BuiltinProgram::Lit) => RESPONSE_PROGRAM_KIND_LIT,
                    ProgramRef::Builtin(BuiltinProgram::Npr) => RESPONSE_PROGRAM_KIND_NPR,
                    ProgramRef::Builtin(BuiltinProgram::Unlit) => RESPONSE_PROGRAM_KIND_UNLIT,
                    ProgramRef::Registered(_) => RESPONSE_PROGRAM_KIND_CUSTOM,
                };
                let count = program.permutations.checked_total().unwrap_or(0);
                let program_id = match program.program {
                    ProgramRef::Builtin(BuiltinProgram::Lit) => 0,
                    ProgramRef::Builtin(BuiltinProgram::Npr) => 1,
                    ProgramRef::Builtin(BuiltinProgram::Unlit) => 2,
                    ProgramRef::Registered(id) => id.0,
                };
                [
                    kind,
                    program.package.get(),
                    count as u32,
                    (count >> 32) as u32,
                    program_id,
                ]
            })
            .collect()
    }

    /// Returns the verified execution family for a checked binding.
    #[must_use]
    pub fn execution_family_for_binding(
        &self,
        binding: MaterialExecutionBinding,
    ) -> Result<ExecutionFamily, MaterialExecutionError> {
        Ok(self.plan(binding.shading_plan_index)?.execution_family)
    }

    /// Number of verified package rows exposed to the GPU bounds check.
    #[must_use]
    pub const fn gpu_package_count(&self) -> u32 {
        self.packages.len() as u32
    }

    #[must_use]
    pub fn package(
        &self,
        index: PackageIndex,
    ) -> Result<&MaterialExecutionPackage, MaterialExecutionError> {
        self.packages
            .get(index.get() as usize)
            .ok_or(MaterialExecutionError::IndexOutOfBounds {
                kind: ExecutionTable::Package,
                index: index.get(),
                len: self.packages.len(),
            })
    }

    #[must_use]
    pub fn program(
        &self,
        index: ResponseProgramIndex,
    ) -> Result<&MaterialExecutionProgram, MaterialExecutionError> {
        self.programs
            .get(index.get() as usize)
            .ok_or(MaterialExecutionError::UnknownProgram(index))
    }

    fn plan(&self, index: PlanIndex) -> Result<&MaterialExecutionPlan, MaterialExecutionError> {
        self.plans
            .get(index.get() as usize)
            .ok_or(MaterialExecutionError::IndexOutOfBounds {
                kind: ExecutionTable::Plan,
                index: index.get(),
                len: self.plans.len(),
            })
    }
}

fn checked_raw(kind: ExecutionTable, len: usize) -> Result<u32, MaterialExecutionError> {
    let raw = u32::try_from(len).map_err(|_| MaterialExecutionError::IndexOutOfBounds {
        kind,
        index: u32::MAX,
        len,
    })?;
    if raw == u32::MAX {
        return Err(MaterialExecutionError::ReservedIndex { kind });
    }
    Ok(raw)
}

fn checked_new_plan_index(len: usize) -> Result<PlanIndex, MaterialExecutionError> {
    Ok(PlanIndex(checked_raw(ExecutionTable::Plan, len)?))
}

fn checked_new_program_index(len: usize) -> Result<ResponseProgramIndex, MaterialExecutionError> {
    Ok(ResponseProgramIndex(checked_raw(
        ExecutionTable::Program,
        len,
    )?))
}

fn checked_new_package_index(len: usize) -> Result<PackageIndex, MaterialExecutionError> {
    Ok(PackageIndex(checked_raw(ExecutionTable::Package, len)?))
}

fn same_manifest(left: &ShaderPackageManifest, right: &ShaderPackageManifest) -> bool {
    left.id == right.id
        && left.source_revision == right.source_revision
        && left.entry_point == right.entry_point
        && left.abi_version == right.abi_version
        && left.abi_hash == right.abi_hash
        && left.permutation_count == right.permutation_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::{AbiHash, AbiVersion};
    use prism_render_architecture::shader_package::{
        ManifestExpectation, ShaderPackageId, ShaderPackageManifest,
    };

    fn package_manifest() -> ShaderPackageManifest {
        ShaderPackageManifest {
            id: ShaderPackageId::new("execution-test"),
            source_revision: "rev-1".into(),
            entry_point: "main".into(),
            abi_version: AbiVersion::from_parts(1, 0),
            abi_hash: AbiHash::from_bytes(b"execution-test"),
            permutation_count: 2,
        }
    }

    fn binary() -> Vec<u32> {
        vec![
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
        ]
    }

    fn compiled() -> CompiledEntry {
        CompiledEntry::from_spirv(package_manifest(), binary()).unwrap()
    }

    fn ready_packages(entry: &CompiledEntry) -> ShaderPackageRegistry {
        let mut packages = ShaderPackageRegistry::new();
        packages
            .register_verified(entry.artifact().clone())
            .unwrap();
        let expectation =
            ManifestExpectation::new(package_manifest().abi_version, package_manifest().abi_hash);
        packages
            .validate(&package_manifest().id, &expectation)
            .unwrap();
        packages
    }

    #[test]
    fn only_verified_compiled_packages_enter_the_table() {
        let entry = compiled();
        let packages = ready_packages(&entry);
        let mut tables = MaterialExecutionTables::new();
        let package = tables.register_package(&entry, &packages).unwrap();
        assert_eq!(package.get(), 0);
        assert_eq!(tables.package(package).unwrap().digest(), entry.digest());
    }

    #[test]
    fn raw_sentinels_and_out_of_range_indices_are_rejected() {
        assert!(matches!(
            PlanIndex::from_raw(u32::MAX, 1),
            Err(MaterialExecutionError::ReservedIndex { .. })
        ));
        assert!(matches!(
            ResponseProgramIndex::from_raw(1, 1),
            Err(MaterialExecutionError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            PackageIndex::from_raw(0, 0),
            Err(MaterialExecutionError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn binding_checks_plan_program_package_and_permutation_ownership() {
        let entry = compiled();
        let packages = ready_packages(&entry);
        let mut tables = MaterialExecutionTables::new();
        let package = tables.register_package(&entry, &packages).unwrap();
        let program = tables
            .register_program(
                ProgramRef::Builtin(BuiltinProgram::Lit),
                &entry,
                package,
                ModuleId(7),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let plan = tables
            .register_plan(
                ShadingPlan::single_builtin(BuiltinProgram::Lit),
                program,
                None,
                ExecutionFamily::FusedBuiltin,
                ChannelMask::default(),
            )
            .unwrap();
        let binding = tables
            .bind_raw(plan.get(), program.get(), package.get(), 1)
            .unwrap();
        assert_eq!(binding.to_words(), [0, 0, 0, 1, 0]);
        assert!(matches!(
            tables.bind_raw(plan.get(), program.get(), package.get(), 2),
            Err(MaterialExecutionError::Permutation(_))
        ));
        assert!(matches!(
            tables.bind_raw(plan.get(), program.get(), u32::MAX, 0),
            Err(MaterialExecutionError::ReservedIndex { .. })
        ));
    }

    #[test]
    fn gpu_descriptor_rows_encode_the_verified_dense_tables() {
        let entry = compiled();
        let packages = ready_packages(&entry);
        let mut tables = MaterialExecutionTables::new();
        let package = tables.register_package(&entry, &packages).unwrap();
        let program = tables
            .register_program(
                ProgramRef::Builtin(BuiltinProgram::Lit),
                &entry,
                package,
                ModuleId(7),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let plan = tables
            .register_plan(
                ShadingPlan::single_builtin(BuiltinProgram::Lit),
                program,
                None,
                ExecutionFamily::FusedBuiltin,
                ChannelMask::default(),
            )
            .unwrap();

        assert_eq!(
            tables.gpu_plan_rows(),
            vec![[
                program.get(),
                u32::MAX,
                ExecutionFamily::FusedBuiltin as u32,
                0,
                0,
                0,
                0,
                0,
            ]]
        );
        assert_eq!(
            tables.gpu_program_rows(),
            vec![[RESPONSE_PROGRAM_KIND_LIT, package.get(), 2, 0, 0]]
        );
        assert_eq!(tables.gpu_package_count(), 1);
        assert_eq!(
            tables
                .bind_raw(plan.get(), program.get(), package.get(), 1)
                .unwrap()
                .to_words(),
            [0, 0, 0, 1, 0]
        );
    }

    #[test]
    fn mask_blend_preserves_validated_fused_and_composition_families() {
        use crate::{BlendDomain, ResponseChannel, ResponseMaskSource};

        let entry = compiled();
        let packages = ready_packages(&entry);
        let mut tables = MaterialExecutionTables::new();
        let package = tables.register_package(&entry, &packages).unwrap();
        let references = [
            ProgramRef::Builtin(BuiltinProgram::Lit),
            ProgramRef::Builtin(BuiltinProgram::Npr),
            ProgramRef::Registered(ProgramId(0)),
        ];
        let programs: Vec<_> = references
            .into_iter()
            .enumerate()
            .map(|(index, program)| {
                tables
                    .register_program(
                        program,
                        &entry,
                        package,
                        ModuleId(index as u64),
                        PermutationSpace::try_new([2]).unwrap(),
                    )
                    .unwrap()
            })
            .collect();
        let channels = ChannelMask::one(ResponseChannel::Direct);
        for right in [1, 2] {
            let source = ShadingPlan::MaskBlend {
                left: references[0],
                right: references[right],
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::DirectOnly,
            };
            assert_eq!(
                tables.register_plan(
                    source.clone(),
                    programs[0],
                    Some(programs[right]),
                    ExecutionFamily::SplitResponse,
                    channels,
                ),
                Err(MaterialExecutionError::InvalidExecutionFamily),
            );
            let fused = tables.register_plan(
                source.clone(),
                programs[0],
                Some(programs[right]),
                ExecutionFamily::FusedBuiltin,
                channels,
            );
            if right == 1 {
                let plan = fused.unwrap();
                assert_eq!(
                    tables.gpu_plan_rows()[plan.get() as usize][2],
                    ExecutionFamily::FusedBuiltin as u32
                );
            } else {
                assert_eq!(fused, Err(MaterialExecutionError::InvalidExecutionFamily));
            }
            tables
                .register_plan(
                    source,
                    programs[0],
                    Some(programs[right]),
                    ExecutionFamily::Composition,
                    channels,
                )
                .unwrap();
        }
        assert_eq!(
            tables.register_plan(
                ShadingPlan::Single {
                    program: references[2]
                },
                programs[2],
                None,
                ExecutionFamily::FusedBuiltin,
                ChannelMask::default(),
            ),
            Err(MaterialExecutionError::InvalidExecutionFamily),
        );
    }
}
