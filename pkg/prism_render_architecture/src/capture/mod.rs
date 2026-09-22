//! Deterministic render capture and replay metadata.

use crate::{abi::AbiHash, gpu_scene::GpuSceneSnapshot};

#[derive(Clone, Debug)]
pub struct CaptureHeader {
    pub architecture_version: u32,
    pub abi_hash: AbiHash,
    pub scene: GpuSceneSnapshot,
    pub random_seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureMode {
    Quality,
    Performance,
    Crash,
}
