//! Material IR, runtime records, and closure contracts.

use crate::{abi::GenerationalHandle, shader_package::ShaderPackageId};

pub type MaterialHandle = GenerationalHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialDomain {
    Surface,
    Decal,
    Volume,
    PostProcess,
    LightFunction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialExecutionPath {
    FixedPbr,
    FixedNpr,
    ClosureTable,
    DiagnosticFallback,
}

#[derive(Clone, Debug)]
pub struct MaterialRecord {
    pub domain: MaterialDomain,
    pub execution: MaterialExecutionPath,
    pub parameter_offset: u32,
    pub shader: Option<ShaderPackageId>,
}
