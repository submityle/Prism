//! Standalone fail-closed validation for one GPU material execution tuple.
//!
//! This module is intentionally not registered from material::mod: callers
//! may include it while integrating the Native composite without changing the
//! existing material API. Inputs are the exact CPU mirrors uploaded for the
//! frame plus the parameter heap's logical word length. This validates table
//! identity and ranges only; it does not decode surface parameters.

use prism_render_architecture::shader_package::ExecutionFamily;
use prism_render_material::{
    GpuMaterialHeader, LobeMask, EXECUTION_PLAN_ROW_WORDS, SURFACE_CORE_WORDS,
    SURFACE_LOBE_WORDS_BY_SLOT,
};

/// Per-pixel identity supplied by the surface/visibility producer. A GPU
/// material header has no slot field, so slot is the header row address and
/// the other fields bind that row to the producer's generational identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceMaterialIdentity {
    pub slot: u32,
    pub generation: u32,
    pub revision: u32,
}

/// A tuple whose header, parameter range, and execution-table relationships
/// have all been checked. This is not a decoded material surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedMaterialTuple {
    pub identity: SurfaceMaterialIdentity,
    pub parameter_offset_words: u32,
    pub parameter_words: u32,
    pub plan_index: u32,
    pub program_index: u32,
    pub package_index: u32,
    pub permutation_index: u64,
    pub execution_family: ExecutionFamily,
}

/// Why an untrusted GPU material tuple was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialTupleValidationError {
    HeaderSlotOutOfBounds {
        slot: u32,
        len: usize,
    },
    HeaderInactive {
        slot: u32,
    },
    GenerationMismatch {
        slot: u32,
        expected: u32,
        found: u32,
    },
    RevisionMismatch {
        slot: u32,
        expected: u32,
        found: u32,
    },
    UnsupportedLobeBits {
        bits: u32,
    },
    ParameterSizeNotWordAligned {
        bytes: u32,
    },
    ParameterSizeMismatch {
        expected_words: u32,
        found_words: u32,
    },
    ParameterRangeOverflow,
    ParameterRangeOutOfBounds {
        end_words: u32,
        heap_words: u32,
    },
    ReservedExecutionIndex {
        table: &'static str,
    },
    ExecutionIndexOutOfBounds {
        table: &'static str,
        index: u32,
        len: usize,
    },
    InvalidExecutionFamily {
        value: u32,
    },
    ProgramNotOwnedByPlan {
        plan: u32,
        program: u32,
    },
    PackageNotOwnedByProgram {
        program: u32,
        expected: u32,
        found: u32,
    },
    PermutationOutOfRange {
        index: u64,
        limit: u64,
    },
}

/// Validate one header and its complete execution binding against the exact
/// table mirrors and parameter-heap length intended for consumption.
///
/// Both endpoints of a MaskBlend plan are recognized as plan owners. MaskBlend
/// is therefore not rejected here; support for blending its response at the
/// consuming shader remains a separate integration concern.
pub fn validate_material_tuple(
    identity: SurfaceMaterialIdentity,
    headers: &[GpuMaterialHeader],
    parameter_heap_words: u32,
    plan_rows: &[[u32; EXECUTION_PLAN_ROW_WORDS]],
    program_rows: &[[u32; 5]],
    package_count: u32,
) -> Result<ValidatedMaterialTuple, MaterialTupleValidationError> {
    let header = headers.get(identity.slot as usize).ok_or(
        MaterialTupleValidationError::HeaderSlotOutOfBounds {
            slot: identity.slot,
            len: headers.len(),
        },
    )?;
    if header.is_active != 1 {
        return Err(MaterialTupleValidationError::HeaderInactive {
            slot: identity.slot,
        });
    }
    if header.generation != identity.generation {
        return Err(MaterialTupleValidationError::GenerationMismatch {
            slot: identity.slot,
            expected: identity.generation,
            found: header.generation,
        });
    }
    if header.revision != identity.revision {
        return Err(MaterialTupleValidationError::RevisionMismatch {
            slot: identity.slot,
            expected: identity.revision,
            found: header.revision,
        });
    }

    let unsupported = header.lobe_mask & !LobeMask::all().0;
    if unsupported != 0 {
        return Err(MaterialTupleValidationError::UnsupportedLobeBits { bits: unsupported });
    }
    if header.parameter_size % 4 != 0 {
        return Err(MaterialTupleValidationError::ParameterSizeNotWordAligned {
            bytes: header.parameter_size,
        });
    }
    let found_words = header.parameter_size / 4;
    let expected_words = canonical_packed_words(header.lobe_mask) as u32;
    if found_words != expected_words {
        return Err(MaterialTupleValidationError::ParameterSizeMismatch {
            expected_words,
            found_words,
        });
    }
    let end_words = header
        .parameter_offset
        .checked_add(found_words)
        .ok_or(MaterialTupleValidationError::ParameterRangeOverflow)?;
    if end_words > parameter_heap_words {
        return Err(MaterialTupleValidationError::ParameterRangeOutOfBounds {
            end_words,
            heap_words: parameter_heap_words,
        });
    }

    let plan = checked_row(plan_rows, header.execution_plan_index, "plan")?;
    let program = checked_row(program_rows, header.execution_program_index, "program")?;
    if header.execution_package_index == u32::MAX {
        return Err(MaterialTupleValidationError::ReservedExecutionIndex { table: "package" });
    }
    if header.execution_package_index >= package_count {
        return Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
            table: "package",
            index: header.execution_package_index,
            len: package_count as usize,
        });
    }
    let family = match plan[2] {
        value if value == ExecutionFamily::FusedBuiltin as u32 => ExecutionFamily::FusedBuiltin,
        value if value == ExecutionFamily::SplitResponse as u32 => ExecutionFamily::SplitResponse,
        value if value == ExecutionFamily::Composition as u32 => ExecutionFamily::Composition,
        value => return Err(MaterialTupleValidationError::InvalidExecutionFamily { value }),
    };
    // Plan row words 0 and 1 are the verified left/right response-program
    // endpoints; u32::MAX means a single-endpoint plan. Either is accepted.
    if plan[0] == u32::MAX || plan[0] as usize >= program_rows.len() {
        return Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
            table: "plan endpoint program",
            index: plan[0],
            len: program_rows.len(),
        });
    }
    if plan[1] != u32::MAX && plan[1] as usize >= program_rows.len() {
        return Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
            table: "plan endpoint program",
            index: plan[1],
            len: program_rows.len(),
        });
    }
    let program_index = header.execution_program_index;
    if plan[0] != program_index && plan[1] != program_index {
        return Err(MaterialTupleValidationError::ProgramNotOwnedByPlan {
            plan: header.execution_plan_index,
            program: program_index,
        });
    }
    let owning_package = program[1];
    if owning_package != header.execution_package_index {
        return Err(MaterialTupleValidationError::PackageNotOwnedByProgram {
            program: program_index,
            expected: owning_package,
            found: header.execution_package_index,
        });
    }
    let permutation_index = u64::from(header.execution_permutation_low)
        | (u64::from(header.execution_permutation_high) << 32);
    let permutation_limit = u64::from(program[2]) | (u64::from(program[3]) << 32);
    if permutation_index >= permutation_limit {
        return Err(MaterialTupleValidationError::PermutationOutOfRange {
            index: permutation_index,
            limit: permutation_limit,
        });
    }

    Ok(ValidatedMaterialTuple {
        identity,
        parameter_offset_words: header.parameter_offset,
        parameter_words: found_words,
        plan_index: header.execution_plan_index,
        program_index,
        package_index: header.execution_package_index,
        permutation_index,
        execution_family: family,
    })
}

fn checked_row<'a, const N: usize>(
    rows: &'a [[u32; N]],
    index: u32,
    table: &'static str,
) -> Result<&'a [u32; N], MaterialTupleValidationError> {
    if index == u32::MAX {
        return Err(MaterialTupleValidationError::ReservedExecutionIndex { table });
    }
    rows.get(index as usize)
        .ok_or(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
            table,
            index,
            len: rows.len(),
        })
}

/// Return the canonical packed word count from the generated ABI layout facts.
/// This computes only the size; it deliberately does not decode parameter data.
fn canonical_packed_words(mask: u32) -> usize {
    SURFACE_CORE_WORDS
        + SURFACE_LOBE_WORDS_BY_SLOT
            .iter()
            .enumerate()
            .filter(|(slot, words)| **words != 0 && mask & (1u32 << *slot) != 0)
            .map(|(_, words)| *words)
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (
        Vec<GpuMaterialHeader>,
        Vec<[u32; EXECUTION_PLAN_ROW_WORDS]>,
        Vec<[u32; 5]>,
    ) {
        let mut header = GpuMaterialHeader::default();
        header.generation = 7;
        header.revision = 11;
        header.is_active = 1;
        header.parameter_offset = 4;
        header.parameter_size = (SURFACE_CORE_WORDS as u32) * 4;
        header.execution_plan_index = 0;
        header.execution_program_index = 0;
        header.execution_package_index = 0;
        header.execution_permutation_low = 2;
        header.execution_permutation_high = 0;
        (
            vec![header],
            vec![[
                0,
                u32::MAX,
                ExecutionFamily::FusedBuiltin as u32,
                0,
                0,
                0,
                0,
                0,
            ]],
            vec![[0, 0, 4, 0, 0]],
        )
    }

    fn identity() -> SurfaceMaterialIdentity {
        SurfaceMaterialIdentity {
            slot: 0,
            generation: 7,
            revision: 11,
        }
    }

    #[test]
    fn accepts_valid_single_endpoint_tuple() {
        let (headers, plans, programs) = fixture();
        let tuple =
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1).unwrap();
        assert_eq!(tuple.permutation_index, 2);
        assert_eq!(tuple.parameter_words, SURFACE_CORE_WORDS as u32);
    }

    #[test]
    fn rejects_inactive_or_stale_surface_identity() {
        let (mut headers, plans, programs) = fixture();
        headers[0].is_active = 0;
        assert_eq!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::HeaderInactive { slot: 0 })
        );
        headers[0].is_active = 1;
        let stale = SurfaceMaterialIdentity {
            generation: 8,
            ..identity()
        };
        assert!(matches!(
            validate_material_tuple(stale, &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::GenerationMismatch { .. })
        ));
        let stale_revision = SurfaceMaterialIdentity {
            revision: 12,
            ..identity()
        };
        assert!(matches!(
            validate_material_tuple(stale_revision, &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::RevisionMismatch { .. })
        ));
    }

    #[test]
    fn validates_canonical_size_and_heap_range() {
        let (mut headers, plans, programs) = fixture();
        headers[0].parameter_size -= 4;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ParameterSizeMismatch { .. })
        ));
        headers[0].parameter_size = (SURFACE_CORE_WORDS as u32) * 4;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 23, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ParameterRangeOutOfBounds { .. })
        ));
        headers[0].parameter_size = (SURFACE_CORE_WORDS as u32) * 4 + 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ParameterSizeNotWordAligned { .. })
        ));
        headers[0].parameter_size = (SURFACE_CORE_WORDS as u32) * 4;
        headers[0].parameter_offset = u32::MAX;
        assert_eq!(
            validate_material_tuple(identity(), &headers, u32::MAX, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ParameterRangeOverflow)
        );
    }

    #[test]
    fn rejects_surface_and_execution_table_bounds() {
        let (mut headers, plans, programs) = fixture();
        let missing_slot = SurfaceMaterialIdentity {
            slot: 1,
            ..identity()
        };
        assert!(matches!(
            validate_material_tuple(missing_slot, &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::HeaderSlotOutOfBounds { .. })
        ));
        headers[0].execution_plan_index = 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds { table: "plan", .. })
        ));
        headers[0].execution_plan_index = 0;
        headers[0].execution_package_index = 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
                table: "package",
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_owned_program_and_invalid_plan_endpoint() {
        let (mut headers, mut plans, mut programs) = fixture();
        programs.push([0, 0, 4, 0, 1]);
        headers[0].execution_program_index = 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ProgramNotOwnedByPlan { .. })
        ));
        headers[0].execution_program_index = 0;
        plans[0][1] = 2;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
                table: "plan endpoint program",
                ..
            })
        ));
    }

    #[test]
    fn checks_plan_program_package_and_permutation_ownership() {
        let (mut headers, plans, programs) = fixture();
        headers[0].execution_program_index = 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 2),
            Err(MaterialTupleValidationError::ExecutionIndexOutOfBounds {
                table: "program",
                ..
            })
        ));
        headers[0].execution_program_index = 0;
        headers[0].execution_package_index = 1;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 2),
            Err(MaterialTupleValidationError::PackageNotOwnedByProgram { .. })
        ));
        headers[0].execution_package_index = 0;
        headers[0].execution_permutation_low = 4;
        assert!(matches!(
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1),
            Err(MaterialTupleValidationError::PermutationOutOfRange { .. })
        ));
    }

    #[test]
    fn permits_either_verified_plan_endpoint_without_maskblend_rejection() {
        let (mut headers, mut plans, mut programs) = fixture();
        programs.push([0, 0, 4, 0, 1]);
        plans[0][1] = 1;
        headers[0].execution_program_index = 1;
        let tuple =
            validate_material_tuple(identity(), &headers, 24, &plans, &programs, 1).unwrap();
        assert_eq!(tuple.program_index, 1);
    }
}
