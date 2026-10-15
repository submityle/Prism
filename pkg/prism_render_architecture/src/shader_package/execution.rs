//! Shared execution identity. This enum is the declaration, not an encoded
//! material style axis; consumers must match variants rather than bit-pack it.

use super::{ModuleId, PermutationIndex};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionFamily {
    FusedBuiltin,
    SplitResponse,
    Composition,
}

/// A per-pass identity. State is semantic, typed state, never just its hash.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PipelineIdentity<State> {
    pub execution_family: ExecutionFamily,
    pub response_module_id: ModuleId,
    pub permutation_index: PermutationIndex,
    pub typed_pipeline_state: State,
}

impl<State> PipelineIdentity<State> {
    pub const fn new(
        execution_family: ExecutionFamily,
        response_module_id: ModuleId,
        permutation_index: PermutationIndex,
        typed_pipeline_state: State,
    ) -> Self {
        Self {
            execution_family,
            response_module_id,
            permutation_index,
            typed_pipeline_state,
        }
    }

    pub fn from_descriptor<'a, Descriptor: 'a>(
        execution_family: ExecutionFamily,
        response_module_id: ModuleId,
        permutation_index: PermutationIndex,
        descriptor: &'a Descriptor,
    ) -> Self
    where
        State: From<&'a Descriptor>,
    {
        Self::new(
            execution_family,
            response_module_id,
            permutation_index,
            State::from(descriptor),
        )
    }
}
