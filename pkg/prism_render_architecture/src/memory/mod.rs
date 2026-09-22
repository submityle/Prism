//! GPU heap and frame-budget contracts.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HeapClass {
    DeviceLocal,
    Upload,
    Readback,
    Transient,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HeapStats {
    pub committed_bytes: u64,
    pub used_bytes: u64,
    pub largest_free_block: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameWorkBudget {
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub relocation_bytes: u64,
    pub acceleration_structure_builds: u32,
}
