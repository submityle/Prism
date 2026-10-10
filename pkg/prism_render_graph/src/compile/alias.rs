//! Transient memory aliasing by greedy interval assignment.
//!
//! Transients dominate a frame's memory, yet most are alive only briefly. If
//! two transients never coexist, they can occupy the *same* bytes. This module
//! turns that observation into a concrete plan: it sizes every transient,
//! sweeps the schedule allocating at first use and freeing after last use
//! through the driver's [`TlsfAllocator`], and records the byte offset each
//! transient received. Backends with placed resources bind one heap and honor
//! the offsets; backends without them ignore the offsets and allocate per
//! resource. Either way the plan reports `heap_size`, the peak simultaneous
//! footprint, which is the real memory the frame needs.
//!
//! [`TlsfAllocator`]: prism_render_driver::TlsfAllocator

use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{Allocation, Extent3d, TlsfAllocator};

use super::lifetime::Lifetimes;
use crate::plan::{AliasPlan, AliasSlot};
use crate::resource::{BufferResource, TextureResource};

/// Minimum alignment for a placed resource slot, in bytes.
const SLOT_ALIGN: u64 = 256;

/// Builds the [`AliasPlan`] for one frame's transients.
pub(crate) fn plan_alias(
    textures: &[TextureResource],
    buffers: &[BufferResource],
    lifetimes: &Lifetimes,
    pass_count: usize,
    swapchain: Extent3d,
) -> AliasPlan {
    let mut tex_size = vec![0u64; textures.len()];
    let mut buf_size = vec![0u64; buffers.len()];
    let mut capacity = 0u64;

    for (i, tr) in textures.iter().enumerate() {
        if tr.is_transient() && lifetimes.textures[i].is_some() {
            // Store the alignment-padded size: feeding `allocate` a size that is
            // already a multiple of `SLOT_ALIGN` keeps every placed offset
            // aligned, so no allocation ever needs front padding.
            let size = align_up(texture_bytes(tr, swapchain), SLOT_ALIGN);
            tex_size[i] = size;
            capacity = capacity.saturating_add(reservation(size));
        }
    }
    for (i, br) in buffers.iter().enumerate() {
        if br.is_transient() && lifetimes.buffers[i].is_some() {
            let size = align_up(br.desc.size.max(1), SLOT_ALIGN);
            buf_size[i] = size;
            capacity = capacity.saturating_add(reservation(size));
        }
    }

    // Per-position start/end lists for a single forward sweep.
    let mut tex_starts: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
    let mut tex_ends: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
    let mut buf_starts: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
    let mut buf_ends: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
    for (i, tr) in textures.iter().enumerate() {
        if tr.is_transient()
            && let Some((first, last)) = lifetimes.textures[i]
        {
            tex_starts[first].push(i);
            tex_ends[last].push(i);
        }
    }
    for (i, br) in buffers.iter().enumerate() {
        if br.is_transient()
            && let Some((first, last)) = lifetimes.buffers[i]
        {
            buf_starts[first].push(i);
            buf_ends[last].push(i);
        }
    }

    let mut allocator = TlsfAllocator::new(capacity.max(SLOT_ALIGN));
    let mut texture_slots: Vec<Option<AliasSlot>> = vec![None; textures.len()];
    let mut buffer_slots: Vec<Option<AliasSlot>> = vec![None; buffers.len()];
    let mut tex_alloc: Vec<Option<Allocation>> = Vec::new();
    tex_alloc.resize_with(textures.len(), || None);
    let mut buf_alloc: Vec<Option<Allocation>> = Vec::new();
    buf_alloc.resize_with(buffers.len(), || None);
    let mut peak = 0u64;

    for pos in 0..pass_count {
        for &i in &tex_starts[pos] {
            let allocation = allocator
                .allocate(tex_size[i], SLOT_ALIGN)
                .expect("transient heap sized to the worst case cannot overflow");
            texture_slots[i] = Some(AliasSlot {
                offset: allocation.offset(),
                size: allocation.size(),
            });
            tex_alloc[i] = Some(allocation);
        }
        for &i in &buf_starts[pos] {
            let allocation = allocator
                .allocate(buf_size[i], SLOT_ALIGN)
                .expect("transient heap sized to the worst case cannot overflow");
            buffer_slots[i] = Some(AliasSlot {
                offset: allocation.offset(),
                size: allocation.size(),
            });
            buf_alloc[i] = Some(allocation);
        }

        peak = peak.max(allocator.allocated());

        for &i in &tex_ends[pos] {
            if let Some(allocation) = tex_alloc[i].take() {
                allocator.free(allocation);
            }
        }
        for &i in &buf_ends[pos] {
            if let Some(allocation) = buf_alloc[i].take() {
                allocator.free(allocation);
            }
        }
    }

    AliasPlan {
        texture_slots,
        buffer_slots,
        heap_size: peak,
    }
}

/// Conservative byte footprint of a texture, including a mip-chain allowance
/// (the geometric series sum `~4/3`) and MSAA sample multiplier.
fn texture_bytes(tr: &TextureResource, swapchain: Extent3d) -> u64 {
    let extent = tr.desc.size.resolve(swapchain);
    let bytes_per_texel = u64::from(tr.desc.format.bytes_per_texel());
    let samples = u64::from(tr.desc.sample_count.max(1));
    let base = u64::from(extent.width)
        .saturating_mul(u64::from(extent.height))
        .saturating_mul(u64::from(extent.depth_or_array_layers))
        .saturating_mul(bytes_per_texel)
        .saturating_mul(samples);
    let total = if tr.desc.mip_level_count > 1 {
        base.saturating_mul(4) / 3
    } else {
        base
    };
    total.max(1)
}

/// Rounds `value` up to the next multiple of `align` (a power of two).
const fn align_up(value: u64, align: u64) -> u64 {
    (value + (align - 1)) & !(align - 1)
}

/// Bytes to reserve in the planning heap for one transient of `size` bytes.
///
/// [`TlsfAllocator::allocate`] widens a request to `size + align - 1` and then
/// rounds that up to a size-class boundary (see [`tlsf_search_ceil`]) before
/// searching. Reserving the rounded figure for *every* transient guarantees the
/// contiguous tail of the heap is always at least as large as the next request,
/// so the sweep's allocation can never fail regardless of the free/alloc order.
/// Over-reserving is free: the reported `heap_size` is the live-set peak from
/// [`TlsfAllocator::allocated`], not the heap capacity.
///
/// [`TlsfAllocator::allocate`]: prism_render_driver::TlsfAllocator::allocate
/// [`TlsfAllocator::allocated`]: prism_render_driver::TlsfAllocator::allocated
const fn reservation(size: u64) -> u64 {
    tlsf_search_ceil(size.saturating_add(SLOT_ALIGN - 1))
}

/// Upper bound on the free-block size TLSF's search demands for a request of
/// `size` bytes, mirroring the internal round-up in `mapping_search` (its
/// second level uses `log2 = 4`). Any free block of at least this size maps to
/// a class the search will accept, so a reservation of this size is sufficient.
const fn tlsf_search_ceil(size: u64) -> u64 {
    const SL_LOG2: u32 = 4;
    const SMALL_BLOCK: u64 = 1 << SL_LOG2;
    if size < SMALL_BLOCK {
        return size;
    }
    let fl = 63 - size.leading_zeros();
    let round = (1u64 << (fl - SL_LOG2)) - 1;
    size.saturating_add(round)
}
