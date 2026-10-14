//! Resource lifetime intervals over the shared scheduled order.
use super::kernel;
use alloc::vec::Vec;

/// First and last scheduled positions, split into texture and buffer namespaces.
pub(crate) struct Lifetimes {
    pub textures: Vec<Option<(usize, usize)>>,
    pub buffers: Vec<Option<(usize, usize)>>,
}

pub(crate) fn analyze(
    accesses: &[Vec<kernel::Access>],
    order: &[usize],
    texture_count: usize,
    buffer_count: usize,
) -> Lifetimes {
    let mut lives = kernel::lifetimes(accesses, order, texture_count + buffer_count);
    let buffers = lives.split_off(texture_count);
    Lifetimes {
        textures: lives,
        buffers,
    }
}
