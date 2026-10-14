//! Backend footprint adapter for the shared lifetime/alias allocator.
use super::{kernel, lifetime::Lifetimes};
use crate::plan::{AliasPlan, CompileError};
use crate::resource::{BufferResource, TextureResource};
use alloc::vec;
use prism_render_driver::Extent3d;

const SLOT_ALIGN: u64 = 256;

pub(crate) fn plan_alias(
    textures: &[TextureResource],
    buffers: &[BufferResource],
    lifetimes: &Lifetimes,
    swapchain: Extent3d,
) -> Result<AliasPlan, CompileError> {
    let mut requests = vec![None; textures.len() + buffers.len()];
    for (i, resource) in textures.iter().enumerate() {
        if resource.is_transient()
            && let Some((first, last)) = lifetimes.textures[i]
        {
            let size = kernel::align_up(texture_bytes(resource, swapchain)?, SLOT_ALIGN)?;
            requests[i] = Some(kernel::AllocationRequest {
                size,
                alignment: SLOT_ALIGN,
                first,
                last,
            });
        }
    }
    for (i, resource) in buffers.iter().enumerate() {
        if resource.is_transient()
            && let Some((first, last)) = lifetimes.buffers[i]
        {
            let size = kernel::align_up(resource.desc.size.max(1), SLOT_ALIGN)?;
            requests[textures.len() + i] = Some(kernel::AllocationRequest {
                size,
                alignment: SLOT_ALIGN,
                first,
                last,
            });
        }
    }
    let (mut slots, heap_size) = kernel::allocate(&requests)?;
    let buffer_slots = slots.split_off(textures.len());
    Ok(AliasPlan {
        texture_slots: slots,
        buffer_slots,
        heap_size,
    })
}

fn texture_bytes(resource: &TextureResource, swapchain: Extent3d) -> Result<u64, CompileError> {
    let extent = resource.desc.size.resolve(swapchain);
    let base = [
        u64::from(extent.width),
        u64::from(extent.height),
        u64::from(extent.depth_or_array_layers),
        u64::from(resource.desc.format.bytes_per_texel()),
        u64::from(resource.desc.sample_count.max(1)),
    ]
    .into_iter()
    .try_fold(1u64, |size, factor| {
        size.checked_mul(factor).ok_or(CompileError::MemoryOverflow)
    })?;
    let total = if resource.desc.mip_level_count > 1 {
        base.checked_mul(4).ok_or(CompileError::MemoryOverflow)? / 3
    } else {
        base
    };
    Ok(total.max(1))
}
