//! The bindless texture arrays a material binding needs.
//!
//! A bindless material shader samples `material_texture_heap[slot]`, and a
//! binding-array target is **dense**: every slot below the bound length must name
//! a real view and sampler, whether an image lives there or not. So this layer
//! materializes one entry per slot — the resident image's prepared view and
//! sampler where the heap has one, the reserved fallback texture elsewhere — from
//! the slot heap and the prepared images.
//!
//! The arrays are rebuilt when the heap's occupancy changes (a slot acquired,
//! freed, or reused): a slot's *generation* protects stale rows, but the array
//! itself has to point at the right texture for the rows that are still valid.

use alloc::vec::Vec;

use prism_image::Image;
use prism_render_driver::{Extent3d, RenderDevice, RenderQueue, SamplerId, TextureFormat, TextureViewId};

use super::texture_heap::TextureSlotHeap;
use crate::image::{GpuImage, prepare_image};
use crate::render_asset::RenderAssets;

/// The three reserved fallback textures every bindless array carries.
///
/// They are renderer-owned, not assets: a missing map has to resolve to a
/// *correct* texture (a flat tangent-space normal for a normal slot, white for
/// base colour), and that cannot depend on the simulation having published one.
pub struct FallbackTextures {
    /// Opaque white (`1, 1, 1, 1`).
    pub white: GpuImage,
    /// The flat tangent-space normal (`0.5, 0.5, 1.0`).
    pub flat_normal: GpuImage,
    /// Opaque black (`0, 0, 0, 1`).
    pub black: GpuImage,
}

impl FallbackTextures {
    /// Creates the three 1×1 fallback textures against `device`.
    ///
    /// They go through the same preparation path as an uploaded image, so a
    /// fallback is an ordinary `GpuImage` (view, sampler, extent) and nothing
    /// downstream needs a special case for it.
    #[must_use]
    pub fn new(device: &dyn RenderDevice, queue: &dyn RenderQueue) -> Self {
        let one_by_one = |texel: [u8; 4]| {
            Image::new(
                Extent3d::new_2d(1, 1),
                TextureFormat::Rgba8Unorm,
                texel.to_vec(),
            )
            .expect("a 1x1 RGBA8 image is valid")
        };
        Self {
            white: prepare_image(device, queue, &one_by_one([255, 255, 255, 255]), None),
            flat_normal: prepare_image(device, queue, &one_by_one([128, 128, 255, 255]), None),
            black: prepare_image(device, queue, &one_by_one([0, 0, 0, 255]), None),
        }
    }

    /// The fallback for a reserved slot, by slot index.
    ///
    /// Slot 1 is the flat normal; 0 and 2 are white and black. Any other index is
    /// not reserved and this returns `None`, which the caller treats as a bug
    /// rather than guessing a fallback.
    #[must_use]
    pub const fn for_slot(&self, slot: u32) -> Option<&GpuImage> {
        match slot {
            super::texture_heap::WHITE_SLOT => Some(&self.white),
            super::texture_heap::FLAT_NORMAL_SLOT => Some(&self.flat_normal),
            super::texture_heap::BLACK_SLOT => Some(&self.black),
            _ => None,
        }
    }

    /// Releases the three textures.
    pub fn release(&self, device: &dyn RenderDevice) {
        for texture in [&self.white, &self.flat_normal, &self.black] {
            <GpuImage as crate::render_asset::RenderAsset>::unload(texture, device);
        }
    }
}

/// One dense bindless array pair: a view and a sampler per slot.
pub struct MaterialTextureSlots {
    views: Vec<TextureViewId>,
    samplers: Vec<SamplerId>,
}

impl MaterialTextureSlots {
    /// Materializes the arrays for `heap`'s occupancy.
    ///
    /// Entry `slot` is the resident image's prepared view and sampler, or the
    /// reserved fallback when nothing (or nothing valid) lives there. The arrays
    /// run to `heap.capacity()`, because a binding array's length is fixed at
    /// layout time and every addressable slot has to resolve.
    #[must_use]
    pub fn build(
        device: &dyn RenderDevice,
        images: &RenderAssets<GpuImage>,
        fallbacks: &FallbackTextures,
        heap: &TextureSlotHeap,
    ) -> Self {
        let _ = device;
        let capacity = heap.capacity();
        let mut views = Vec::with_capacity(capacity as usize);
        let mut samplers = Vec::with_capacity(capacity as usize);
        for slot in 0..capacity {
            let view = heap
                .image_at(slot)
                .and_then(|image| images.get(image))
                .or_else(|| fallbacks.for_slot(slot));
            match view {
                Some(image) => {
                    views.push(image.view);
                    samplers.push(image.sampler);
                }
                // A slot with neither a resident image nor a reserved fallback
                // cannot happen: the heap only hands out dynamic slots, and every
                // reserved slot has a fallback. Falling back to white keeps the
                // array dense rather than producing a short one.
                None => {
                    views.push(fallbacks.white.view);
                    samplers.push(fallbacks.white.sampler);
                }
            }
        }
        Self { views, samplers }
    }

    /// The view per slot, in slot order.
    #[must_use]
    pub fn views(&self) -> &[TextureViewId] {
        &self.views
    }

    /// The sampler per slot, in slot order.
    #[must_use]
    pub fn samplers(&self) -> &[SamplerId] {
        &self.samplers
    }

    /// The number of slots the arrays cover.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether the arrays are empty (a zero-capacity heap, which cannot exist).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

