//! GPU-generated draw and dispatch work.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkQueueId(pub u32);

#[derive(Clone, Copy, Debug, Default)]
pub struct WorkQueueCapacity {
    pub items: u32,
    pub overflow_is_fatal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorkQueueStats {
    pub produced: u32,
    pub consumed: u32,
    pub dropped: u32,
}
