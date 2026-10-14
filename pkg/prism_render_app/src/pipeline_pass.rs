//! Driver-local pass sets shared by direct work and draw-bin resolution.

use prism_render_architecture::shader_package::{ExecutionFamily, ModuleId, PermutationIndex};
use crate::{DrawFunctionId, MeshPipelineKey, RenderPhase};
use prism_render_driver::{PrimitiveTopology, TextureFormat};

/// Sorting category, deliberately unrelated to a u32 pass-set table index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassDescriptorClass(u8);

impl PassDescriptorClass {
    /// 255 is reserved as an exhaustion sentinel, never silently wrapped.
    pub fn new(value: u8) -> Result<Self, PassSetError> {
        if value == u8::MAX { Err(PassSetError::SortClassExhausted) } else { Ok(Self(value)) }
    }
    pub const fn get(self) -> u8 { self.0 }
    pub const SURFACE: Self = Self(0);
}

/// A single pass/module query. Providers own the descriptor's actual GPU
/// state; the specializer derives and validates full PipelineState from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassDescriptor {
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
    pub fn mesh_variant(self, topology: PrimitiveTopology, color_format: Option<TextureFormat>, depth_format: Option<TextureFormat>, sample_count: u32) -> MeshPipelineKey {
        MeshPipelineKey { execution_family: self.execution_family, response_module_id: self.response_module_id, permutation_index: self.permutation_index, fallback_permutation: self.fallback_permutation, color_format, depth_format, sample_count, topology }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PassSetError { Empty, InvalidShape, CompositionNeedsDraw, SortClassExhausted, TableExhausted, MissingMaterial }

#[derive(Default)]
pub(crate) struct PassDescriptorTable { sets: Vec<Vec<PassDescriptor>> }

impl PassDescriptorTable {
    pub fn intern(&mut self, passes: Vec<PassDescriptor>) -> Result<u32, PassSetError> {
        if passes.is_empty() { return Err(PassSetError::Empty); }
        match passes.as_slice() {
            [single] if single.execution_family != ExecutionFamily::Composition => (),
            [left, right, composition] if left.execution_family == ExecutionFamily::SplitResponse && right.execution_family == ExecutionFamily::SplitResponse && composition.execution_family == ExecutionFamily::Composition => {
                if composition.draw.is_none() { return Err(PassSetError::CompositionNeedsDraw); }
                if left.draw.is_none() || right.draw.is_none() { return Err(PassSetError::InvalidShape); }
            }
            _ => return Err(PassSetError::InvalidShape),
        }
        if let Some(index) = self.sets.iter().position(|set| *set == passes) { return u32::try_from(index).map_err(|_| PassSetError::TableExhausted); }
        let index = u32::try_from(self.sets.len()).map_err(|_| PassSetError::TableExhausted)?;
        self.sets.push(passes);
        Ok(index)
    }
    pub fn get(&self, index: u32) -> Option<&[PassDescriptor]> { self.sets.get(index as usize).map(Vec::as_slice) }
}
