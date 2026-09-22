//! Shared virtual-resource scheduling contracts.

use std::hash::Hash;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RequestPriority(pub u32);

#[derive(Clone, Copy, Debug, Default)]
pub struct ResidencyBudget {
    pub soft_bytes: u64,
    pub hard_bytes: u64,
    pub upload_bytes_per_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyState {
    Missing,
    Requested,
    Uploading,
    Resident,
    Retiring,
}

pub trait VirtualResourceClient {
    type Key: Copy + Eq + Hash;

    fn priority(&self, key: Self::Key) -> RequestPriority;
    fn parent(&self, key: Self::Key) -> Option<Self::Key>;
    fn invalidate(&mut self, key: Self::Key, epoch: u64);
}
