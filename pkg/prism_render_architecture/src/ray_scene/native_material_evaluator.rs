//! ABI contracts for a Native material evaluation pass consumed by GI.
//!
//! A secondary hit is not allowed to treat a triangle-indexed radiance row as
//! a material evaluation. The native evaluator must consume one coherent
//! material snapshot (headers, packed parameters, texture rows and execution
//! tables), and the matching VT publication, then publish a per-hit response
//! stamped with the hit identity. The current portable evaluator deliberately
//! has no direct-image heap: a non-VT authored row is rejected rather than
//! sampled through an unproven resource. The render-app trace pass binds this
//! contract through two groups: per-dispatch trace/output state at group 0 and
//! the immutable material/VT snapshot at group 1.

use bytemuck::{Pod, Zeroable};

use super::hit_attributes::RayHitIdentity;

/// Version of the Native GI material-evaluator binding contract.
pub const NATIVE_GI_MATERIAL_EVALUATOR_ABI_VERSION: u32 = 1;
/// Number of words in [`NativeGiMaterialEvaluatorConfig`].
pub const NATIVE_GI_MATERIAL_EVALUATOR_CONFIG_WORDS: usize = 16;
/// Number of words in one per-hit evaluator result.
pub const NATIVE_GI_MATERIAL_EVALUATION_WORDS: usize = 20;
/// Byte stride of one per-hit evaluator result.
pub const NATIVE_GI_MATERIAL_EVALUATION_BYTES: u64 =
    (NATIVE_GI_MATERIAL_EVALUATION_WORDS * core::mem::size_of::<u32>()) as u64;

/// Binding group used by the candidate pass for Native material resources.
pub const NATIVE_GI_MATERIAL_EVALUATOR_GROUP: u32 = 1;
/// Header snapshot storage binding in [`NATIVE_GI_MATERIAL_EVALUATOR_GROUP`].
pub const NATIVE_GI_MATERIAL_HEADERS_BINDING: u32 = 0;
/// Packed material-parameter word heap binding in the material group.
pub const NATIVE_GI_MATERIAL_PARAMETERS_BINDING: u32 = 1;
/// Material texture-row snapshot binding in the material group.
pub const NATIVE_GI_MATERIAL_TEXTURE_ROWS_BINDING: u32 = 2;
/// Verified execution-plan table binding in the material group.
pub const NATIVE_GI_EXECUTION_PLANS_BINDING: u32 = 3;
/// Verified response-program table binding in the material group.
pub const NATIVE_GI_EXECUTION_PROGRAMS_BINDING: u32 = 4;
/// VT page-table storage binding.
pub const NATIVE_GI_VT_PAGE_TABLE_BINDING: u32 = 5;
/// VT configuration storage binding.
pub const NATIVE_GI_VT_CONFIG_BINDING: u32 = 6;
/// VT physical atlas binding.
pub const NATIVE_GI_VT_ATLAS_BINDING: u32 = 7;
/// VT atlas sampler binding.
pub const NATIVE_GI_VT_SAMPLER_BINDING: u32 = 8;
/// Number of material-group bindings. Bindings 3 and 4 are intentionally not
/// direct-image arrays: the portable slice consumes only source-backed/VT rows.
pub const NATIVE_GI_MATERIAL_EVALUATOR_BINDING_COUNT: u32 = 9;
/// Per-dispatch evaluator configuration group.
pub const NATIVE_GI_EVALUATOR_STATE_GROUP: u32 = 0;
/// Per-dispatch trace-hit input binding.
pub const NATIVE_GI_TRACE_HITS_BINDING: u32 = 0;
/// Per-dispatch hit-attribute input binding.
pub const NATIVE_GI_HIT_ATTRIBUTES_BINDING: u32 = 1;
/// Per-dispatch evaluator configuration binding.
pub const NATIVE_GI_EVALUATOR_CONFIG_BINDING: u32 = 3;
/// Per-hit material-evaluator output binding.
pub const NATIVE_GI_EVALUATIONS_BINDING: u32 = 4;
/// Per-dispatch packed trace-ray input binding.
pub const NATIVE_GI_TRACE_RAYS_BINDING: u32 = 2;
/// Direct-image heap names remain exported for the architecture module's
/// public surface, but this ABI has no such bindings. A non-VT authored row is
/// rejected before an image index can be consumed.
pub const NATIVE_GI_TEXTURE_HEAP_BINDING: u32 = u32::MAX;
/// See [`NATIVE_GI_TEXTURE_HEAP_BINDING`].
pub const NATIVE_GI_SAMPLER_HEAP_BINDING: u32 = u32::MAX;
/// See [`NATIVE_GI_TEXTURE_HEAP_BINDING`]. Row generations are carried inline.
pub const NATIVE_GI_TEXTURE_GENERATIONS_BINDING: u32 = u32::MAX;

/// Configuration stamped beside the resources of one coherent material frame.
///
/// All counts are logical element/word counts, not allocation capacities. The
/// consumer uses them to reject stale or truncated bindings before indexing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct NativeGiMaterialEvaluatorConfig {
    /// Must equal [`NATIVE_GI_MATERIAL_EVALUATOR_ABI_VERSION`].
    pub abi_version: u32,
    /// Material snapshot epoch, low word.
    pub material_epoch_low: u32,
    /// Material snapshot epoch, high word.
    pub material_epoch_high: u32,
    /// Verified execution-table publication version.
    pub execution_table_version: u32,
    /// VT publication generation, low word.
    pub vt_generation_low: u32,
    /// VT publication generation, high word.
    pub vt_generation_high: u32,
    /// Number of material-header rows.
    pub material_header_count: u32,
    /// Number of addressable words in the packed parameter heap.
    pub parameter_word_count: u32,
    /// Number of material texture rows.
    pub texture_row_count: u32,
    /// Number of verified execution-plan rows.
    pub execution_plan_count: u32,
    /// Number of verified response-program rows.
    pub execution_program_count: u32,
    /// Number of bindless texture/sampler slots in both arrays.
    pub texture_slot_count: u32,
    /// Number of words in the VT page table.
    pub vt_page_table_word_count: u32,
    /// Reserved feature flags. Must be zero until a flag is specified.
    pub flags: u32,
    /// Keeps the uniform at the declared 16-word ABI size.
    pub _padding: [u32; 2],
}

impl NativeGiMaterialEvaluatorConfig {
    /// A zeroed config is intentionally incomplete and cannot authorize a pass.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            abi_version: 0,
            material_epoch_low: 0,
            material_epoch_high: 0,
            execution_table_version: 0,
            vt_generation_low: 0,
            vt_generation_high: 0,
            material_header_count: 0,
            parameter_word_count: 0,
            texture_row_count: 0,
            execution_plan_count: 0,
            execution_program_count: 0,
            texture_slot_count: 0,
            vt_page_table_word_count: 0,
            flags: 0,
            _padding: [0; 2],
        }
    }

    /// Returns the snapshot/texture/VT invariants required by the seam.
    pub fn validate(self) -> Result<(), NativeGiMaterialEvaluatorConfigError> {
        if self.abi_version != NATIVE_GI_MATERIAL_EVALUATOR_ABI_VERSION {
            return Err(NativeGiMaterialEvaluatorConfigError::UnsupportedAbiVersion(
                self.abi_version,
            ));
        }
        if self.execution_table_version == 0
            || self.material_header_count == 0
            || self.texture_row_count == 0
        {
            return Err(NativeGiMaterialEvaluatorConfigError::IncompleteMaterialSnapshot);
        }
        if self.parameter_word_count == 0
            || self.execution_plan_count == 0
            || self.execution_program_count == 0
        {
            return Err(NativeGiMaterialEvaluatorConfigError::IncompleteMaterialSnapshot);
        }
        // This slice has no direct-image heap. A non-zero slot count would
        // authorize a resource that is not represented by this bind group.
        if self.texture_slot_count != 0 {
            return Err(NativeGiMaterialEvaluatorConfigError::IncompleteTextureOrVtState);
        }
        let has_vt_resources = self.vt_page_table_word_count != 0;
        let has_vt_generation = self.vt_generation_low != 0 || self.vt_generation_high != 0;
        if has_vt_resources != has_vt_generation {
            return Err(NativeGiMaterialEvaluatorConfigError::IncompleteTextureOrVtState);
        }
        if self.flags != 0 || self._padding != [0; 2] {
            return Err(NativeGiMaterialEvaluatorConfigError::ReservedWordsNonZero);
        }
        Ok(())
    }
}

/// Why a Native evaluator config cannot authorize a candidate pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeGiMaterialEvaluatorConfigError {
    /// The producer and consumer do not agree on the seam version.
    UnsupportedAbiVersion(u32),
    /// A material snapshot or its verified execution tables are absent.
    IncompleteMaterialSnapshot,
    /// Texture rows, texture slots, or the VT page table are absent.
    IncompleteTextureOrVtState,
    /// Reserved feature/padding words were populated before being specified.
    ReservedWordsNonZero,
}

/// One evaluator result for one traced hit.
///
/// The first four words are the evaluated response (`valid` is word three).
/// The remaining twelve words repeat the hit identity so candidate generation
/// can reject a response produced from a different material publication.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct NativeGiMaterialEvaluationRecord {
    /// Evaluated outgoing response for the traced secondary direction.
    pub response_rgb: [f32; 3],
    /// `1` only when the response was produced by the Native evaluator.
    pub valid: u32,
    /// Stable instance identity.
    pub instance_id: u32,
    /// Stable geometry identity.
    pub geometry_id: u32,
    /// Dense material slot.
    pub material_index: u32,
    /// Material generation.
    pub material_generation: u32,
    /// Material record revision.
    pub material_revision: u32,
    /// Material snapshot epoch, low word.
    pub material_epoch_low: u32,
    /// Material snapshot epoch, high word.
    pub material_epoch_high: u32,
    /// Verified execution plan index.
    pub execution_plan_index: u32,
    /// Verified response-program index.
    pub execution_program_index: u32,
    /// Verified package index.
    pub execution_package_index: u32,
    /// Verified permutation index, low word.
    pub execution_permutation_low: u32,
    /// Verified permutation index, high word.
    pub execution_permutation_high: u32,
    /// Response-program kind proven by the execution row.
    pub response_kind: u32,
    /// Execution-family proof for the evaluated arm.
    pub execution_family: u32,
    /// Mask/blend proof: the accepted plan's blended channel mask.
    pub blended_channels: u32,
    /// Response source proof: builtin or registered stable program id.
    pub response_program_id: u32,
}

impl NativeGiMaterialEvaluationRecord {
    /// Number of 32-bit words in the storage-buffer record.
    pub const WORDS: usize = NATIVE_GI_MATERIAL_EVALUATION_WORDS;

    /// Returns the stamped material snapshot identity.
    #[must_use]
    pub const fn snapshot(self) -> super::hit_attributes::RayHitMaterialSnapshot {
        super::hit_attributes::RayHitMaterialSnapshot {
            material_index: self.material_index,
            material_generation: self.material_generation,
            material_revision: self.material_revision,
            material_epoch: (self.material_epoch_low as u64)
                | ((self.material_epoch_high as u64) << 32),
            execution_binding: [
                self.execution_plan_index,
                self.execution_program_index,
                self.execution_package_index,
                self.execution_permutation_low,
                self.execution_permutation_high,
            ],
        }
    }

    /// Validates the output without accepting a triangle-radiance fallback.
    pub fn validate(&self) -> Result<(), NativeGiMaterialEvaluationError> {
        if self.valid != 1 {
            return Err(NativeGiMaterialEvaluationError::NotEvaluated);
        }
        if !self
            .response_rgb
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
        {
            return Err(NativeGiMaterialEvaluationError::InvalidResponse);
        }
        if self.instance_id == u32::MAX
            || self.geometry_id == u32::MAX
            || !self.snapshot().is_complete()
            || self.response_kind > 3
            || self.execution_family > 2
            || self.response_program_id == u32::MAX
        {
            return Err(NativeGiMaterialEvaluationError::MissingSnapshotProof);
        }
        Ok(())
    }

    /// Accepts this result only for the exact hit and immutable evaluator
    /// snapshot that produced it.
    pub fn validate_for_hit(
        &self,
        hit: RayHitIdentity,
        config: NativeGiMaterialEvaluatorConfig,
    ) -> Result<(), NativeGiMaterialEvaluationError> {
        self.validate()?;
        config
            .validate()
            .map_err(|_| NativeGiMaterialEvaluationError::InvalidEvaluatorSnapshot)?;
        let expected_epoch =
            (config.material_epoch_low as u64) | ((config.material_epoch_high as u64) << 32);
        if self.instance_id != hit.instance_id
            || self.geometry_id != hit.geometry_id
            || self.snapshot() != hit.material_snapshot()
            || self.snapshot().material_epoch != expected_epoch
            || self.material_index >= config.material_header_count
            || self.execution_plan_index >= config.execution_plan_count
            || self.execution_program_index >= config.execution_program_count
        {
            return Err(NativeGiMaterialEvaluationError::SnapshotMismatch);
        }
        Ok(())
    }
}

/// Why an evaluator result cannot be consumed by GI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeGiMaterialEvaluationError {
    /// The evaluator did not publish a valid result for the hit.
    NotEvaluated,
    /// The response contains a negative or non-finite channel.
    InvalidResponse,
    /// The response is not stamped with a complete material proof.
    MissingSnapshotProof,
    /// The evaluator resources do not describe a published material snapshot.
    InvalidEvaluatorSnapshot,
    /// The response was produced for another hit or material publication.
    SnapshotMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> NativeGiMaterialEvaluatorConfig {
        NativeGiMaterialEvaluatorConfig {
            abi_version: NATIVE_GI_MATERIAL_EVALUATOR_ABI_VERSION,
            material_epoch_low: 9,
            execution_table_version: 2,
            vt_generation_low: 4,
            material_header_count: 8,
            parameter_word_count: 32,
            texture_row_count: 8,
            execution_plan_count: 4,
            execution_program_count: 4,
            texture_slot_count: 0,
            vt_page_table_word_count: 16,
            ..NativeGiMaterialEvaluatorConfig::empty()
        }
    }

    fn valid_hit() -> RayHitIdentity {
        RayHitIdentity {
            instance_id: 3,
            geometry_id: 5,
            material_index: 2,
            material_generation: 7,
            material_revision: 11,
            material_epoch: 9,
            execution_binding: [1, 2, 3, 4, 0],
        }
    }

    fn valid_evaluation() -> NativeGiMaterialEvaluationRecord {
        NativeGiMaterialEvaluationRecord {
            response_rgb: [0.4, 0.7, 1.2],
            valid: 1,
            instance_id: 3,
            geometry_id: 5,
            material_index: 2,
            material_generation: 7,
            material_revision: 11,
            material_epoch_low: 9,
            material_epoch_high: 0,
            execution_plan_index: 1,
            execution_program_index: 2,
            execution_package_index: 3,
            execution_permutation_low: 4,
            execution_permutation_high: 0,
            response_kind: 2,
            execution_family: 0,
            blended_channels: 0,
            response_program_id: 2,
        }
    }

    #[test]
    fn evaluation_is_bound_to_the_exact_hit_and_snapshot() {
        let evaluation = valid_evaluation();
        assert_eq!(
            evaluation.validate_for_hit(valid_hit(), valid_config()),
            Ok(())
        );

        let mut stale_hit = valid_hit();
        stale_hit.material_generation += 1;
        assert_eq!(
            evaluation.validate_for_hit(stale_hit, valid_config()),
            Err(NativeGiMaterialEvaluationError::SnapshotMismatch)
        );

        let mut stale_config = valid_config();
        stale_config.material_epoch_low += 1;
        assert_eq!(
            evaluation.validate_for_hit(valid_hit(), stale_config),
            Err(NativeGiMaterialEvaluationError::SnapshotMismatch)
        );

        let mut out_of_range = valid_evaluation();
        out_of_range.execution_plan_index = 4;
        assert_eq!(
            out_of_range.validate_for_hit(valid_hit(), valid_config()),
            Err(NativeGiMaterialEvaluationError::SnapshotMismatch)
        );

        let mut invalid_material_index = valid_evaluation();
        invalid_material_index.material_index = valid_config().material_header_count;
        let mut matching_hit = valid_hit();
        matching_hit.material_index = invalid_material_index.material_index;
        assert_eq!(
            invalid_material_index.validate_for_hit(matching_hit, valid_config()),
            Err(NativeGiMaterialEvaluationError::SnapshotMismatch)
        );
    }

    #[test]
    fn evaluator_config_rejects_incomplete_snapshot_and_vt_state() {
        assert_eq!(
            NativeGiMaterialEvaluatorConfig::empty().validate(),
            Err(NativeGiMaterialEvaluatorConfigError::UnsupportedAbiVersion(
                0
            ))
        );
        let mut config = NativeGiMaterialEvaluatorConfig {
            abi_version: NATIVE_GI_MATERIAL_EVALUATOR_ABI_VERSION,
            execution_table_version: 1,
            material_header_count: 1,
            parameter_word_count: 1,
            texture_row_count: 1,
            execution_plan_count: 1,
            execution_program_count: 1,
            texture_slot_count: 0,
            vt_page_table_word_count: 0,
            ..NativeGiMaterialEvaluatorConfig::empty()
        };
        assert_eq!(config.validate(), Ok(()));
        config.vt_page_table_word_count = 1;
        config.vt_generation_low = 1;
        assert_eq!(config.validate(), Ok(()));
        config.vt_page_table_word_count = 0;
        assert_eq!(
            config.validate(),
            Err(NativeGiMaterialEvaluatorConfigError::IncompleteTextureOrVtState)
        );

        config.vt_generation_low = 0;
        assert_eq!(
            config.validate(),
            Ok(()),
            "untextured evaluator snapshots use the disabled VT contract"
        );
        config.texture_slot_count = 1;
        assert_eq!(
            config.validate(),
            Err(NativeGiMaterialEvaluatorConfigError::IncompleteTextureOrVtState)
        );
    }

    #[test]
    fn evaluation_result_requires_native_proof_and_finite_response() {
        let mut result = NativeGiMaterialEvaluationRecord {
            response_rgb: [1.0, 2.0, 3.0],
            valid: 1,
            instance_id: 4,
            geometry_id: 5,
            material_index: 6,
            material_generation: 7,
            material_revision: 8,
            material_epoch_low: 9,
            material_epoch_high: 0,
            execution_plan_index: 10,
            execution_program_index: 11,
            execution_package_index: 12,
            execution_permutation_low: 13,
            execution_permutation_high: 14,
            response_kind: 2,
            execution_family: 0,
            blended_channels: 0,
            response_program_id: 2,
        };
        assert_eq!(result.validate(), Ok(()));
        result.valid = 0;
        assert_eq!(
            result.validate(),
            Err(NativeGiMaterialEvaluationError::NotEvaluated)
        );
    }
}
