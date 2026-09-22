//! Build-time HLSL to SPIR-V package metadata.

use crate::abi::{AbiHash, AbiVersion};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderPackageId(pub String);

#[derive(Clone, Debug)]
pub struct ShaderPackageManifest {
    pub id: ShaderPackageId,
    pub source_revision: String,
    pub entry_point: String,
    pub abi_version: AbiVersion,
    pub abi_hash: AbiHash,
    pub permutation_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackageState {
    Missing,
    Validating,
    Ready,
    Rejected,
}
