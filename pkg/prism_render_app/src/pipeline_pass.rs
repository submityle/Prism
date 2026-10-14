//! Driver-local pass sets shared by direct work and draw-bin resolution.

use crate::{DrawFunctionId, MeshPipelineKey, RenderPhase};
use prism_render_architecture::shader_package::{ExecutionFamily, ModuleId, PermutationIndex};
use prism_render_driver::{PrimitiveTopology, TextureFormat};

/// Sorting category, deliberately unrelated to a u32 pass-set table index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassDescriptorClass(u8);

impl PassDescriptorClass {
    /// 255 is reserved as an exhaustion sentinel, never silently wrapped.
    pub fn new(value: u8) -> Result<Self, PassSetError> {
        if value == u8::MAX {
            Err(PassSetError::SortClassExhausted)
        } else {
            Ok(Self(value))
        }
    }
    /// Returns the compact class used only for coarse sorting.
    pub const fn get(self) -> u8 {
        self.0
    }
    /// Builds a class from a table-owned ordinal without allowing the reserved
    /// sentinel to wrap into the sort key.
    pub fn from_ordinal(ordinal: usize) -> Result<Self, PassSetError> {
        let value = u8::try_from(ordinal).map_err(|_| PassSetError::SortClassExhausted)?;
        Self::new(value)
    }
    /// Returns the next class, or an exhaustion error at the reserved value.
    pub fn next(self) -> Result<Self, PassSetError> {
        Self::from_ordinal(usize::from(self.0) + 1)
    }
    pub const SURFACE: Self = Self(0);
}

/// The role of one item in a material's pass set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PassRole {
    /// A fused response or an ordinary single-pass surface draw.
    Surface,
    /// The first response arm in a split execution.
    ResponseLeft,
    /// The second response arm in a split execution.
    ResponseRight,
    /// The resident composition kernel that combines split response output.
    Composition,
}

/// A single pass/module query. Providers own the descriptor's actual GPU
/// state; the specializer derives and validates full PipelineState from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassDescriptor {
    /// Whether this item is a surface, response arm, or composition pass.
    pub role: PassRole,
    pub execution_family: ExecutionFamily,
    pub response_module_id: ModuleId,
    pub permutation_index: PermutationIndex,
    pub fallback_permutation: Option<PermutationIndex>,
    pub render_phase: RenderPhase,
    /// A composition entry must explicitly bind its response resources and
    /// target. It cannot accidentally become another ordinary surface draw.
    pub draw: Option<DrawFunctionId>,
    pub sort_class: PassDescriptorClass,
}

impl PassDescriptor {
    pub fn mesh_variant(
        self,
        topology: PrimitiveTopology,
        color_format: Option<TextureFormat>,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> MeshPipelineKey {
        MeshPipelineKey {
            execution_family: self.execution_family,
            response_module_id: self.response_module_id,
            permutation_index: self.permutation_index,
            fallback_permutation: self.fallback_permutation,
            color_format,
            depth_format,
            sample_count,
            topology,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PassSetError {
    Empty,
    InvalidShape,
    CompositionNeedsDraw,
    SortClassExhausted,
    TableExhausted,
    MissingMaterial,
}

#[derive(Default)]
pub(crate) struct PassDescriptorTable {
    sets: Vec<Vec<PassDescriptor>>,
}

impl PassDescriptorTable {
    pub fn intern(&mut self, passes: Vec<PassDescriptor>) -> Result<u32, PassSetError> {
        if passes.is_empty() {
            return Err(PassSetError::Empty);
        }
        match passes.as_slice() {
            [single]
                if single.execution_family != ExecutionFamily::Composition
                    && single.role == PassRole::Surface =>
            {
                ()
            }
            [left, right, composition]
                if left.execution_family == ExecutionFamily::SplitResponse
                    && right.execution_family == ExecutionFamily::SplitResponse
                    && composition.execution_family == ExecutionFamily::Composition
                    && left.role == PassRole::ResponseLeft
                    && right.role == PassRole::ResponseRight
                    && composition.role == PassRole::Composition =>
            {
                if composition.draw.is_none() {
                    return Err(PassSetError::CompositionNeedsDraw);
                }
                if left.draw.is_none() || right.draw.is_none() {
                    return Err(PassSetError::InvalidShape);
                }
            }
            _ => return Err(PassSetError::InvalidShape),
        }
        if let Some(index) = self.sets.iter().position(|set| *set == passes) {
            return u32::try_from(index).map_err(|_| PassSetError::TableExhausted);
        }
        let index = u32::try_from(self.sets.len()).map_err(|_| PassSetError::TableExhausted)?;
        self.sets.push(passes);
        Ok(index)
    }
    pub fn get(&self, index: u32) -> Option<&[PassDescriptor]> {
        self.sets.get(index as usize).map(Vec::as_slice)
    }

    pub fn has_composition(&self) -> bool {
        self.sets
            .iter()
            .flatten()
            .any(|pass| pass.role == PassRole::Composition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(
        role: PassRole,
        family: ExecutionFamily,
        draw: Option<DrawFunctionId>,
    ) -> PassDescriptor {
        PassDescriptor {
            role,
            execution_family: family,
            response_module_id: ModuleId(1),
            permutation_index: PermutationIndex::ZERO,
            fallback_permutation: None,
            render_phase: RenderPhase::Opaque,
            draw,
            sort_class: PassDescriptorClass::SURFACE,
        }
    }

    #[test]
    fn split_pass_sets_keep_both_arms_and_resident_composition() {
        let mut table = PassDescriptorTable::default();
        let index = table
            .intern(vec![
                pass(
                    PassRole::ResponseLeft,
                    ExecutionFamily::SplitResponse,
                    Some(DrawFunctionId(0)),
                ),
                pass(
                    PassRole::ResponseRight,
                    ExecutionFamily::SplitResponse,
                    Some(DrawFunctionId(0)),
                ),
                pass(
                    PassRole::Composition,
                    ExecutionFamily::Composition,
                    Some(DrawFunctionId(1)),
                ),
            ])
            .unwrap();
        assert_eq!(table.get(index).unwrap().len(), 3);
        assert!(table.has_composition());
    }

    #[test]
    fn sort_class_reserves_255_and_table_index_is_independent() {
        assert_eq!(
            PassDescriptorClass::new(u8::MAX),
            Err(PassSetError::SortClassExhausted)
        );
        assert_eq!(
            PassDescriptorClass::from_ordinal(255),
            Err(PassSetError::SortClassExhausted)
        );
        assert!(PassDescriptorClass::from_ordinal(254).is_ok());
    }
}
