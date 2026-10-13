//! The render world's texture cache.
//!
//! Bevy's render world carries a `TextureCache` mapping an asset to its GPU
//! image so a texture is uploaded once and shared. Prism's equivalent is a thin
//! renderer-side cache over the RHI's [`ContentCache`]: the key is the texture's
//! identity (an asset/content id the caller owns), the value is the RHI texture
//! plus its whole-image view. Identical keys return the same ids, and a texture
//! that is no longer referenced can be evicted and destroyed without disturbing
//! one that is still live.

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    CacheStats, ContentCache, Extent3d, ImageDataLayout, RenderDevice, RenderQueue,
    TextureDescriptor, TextureId, TextureViewDescriptor, TextureViewId, TextureWrite,
};

/// The identity of a cached texture.
///
/// A caller-owned id (an asset handle, a content hash, or a streamed page key);
/// the cache compares it exactly and never inspects the bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TextureKey(pub u64);

/// A cached GPU texture and its whole-image default view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuTexture {
    /// The RHI texture.
    pub texture: TextureId,
    /// A whole-image view over [`texture`](Self::texture).
    pub view: TextureViewId,
}

/// A content-addressed, reference-counted cache of GPU textures.
#[derive(Resource, Default)]
pub struct TextureCache {
    cache: ContentCache<TextureKey, GpuTexture>,
}

impl TextureCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: ContentCache::new(),
        }
    }

    /// The number of distinct cached textures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache holds no textures.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The hit/miss counters and live entry count.
    #[must_use]
    pub fn stats(&mut self) -> CacheStats {
        self.cache.stats()
    }

    /// Returns the cached texture for `key`, creating and uploading it on first
    /// use.
    ///
    /// `data` may be empty, in which case the texture is created but not
    /// written (a render target or a texture filled later). Non-empty `data` is
    /// uploaded into mip `0`, layer `0`, sized to the descriptor's width and
    /// height. A later call with the same key returns the cached ids and ignores
    /// the new descriptor, data, and layout: the key is the texture's identity.
    pub fn get_or_create<D: RenderDevice, Q: RenderQueue>(
        &mut self,
        device: &D,
        queue: &Q,
        key: TextureKey,
        descriptor: &TextureDescriptor,
        data: &[u8],
        layout: ImageDataLayout,
    ) -> GpuTexture {
        let size = descriptor.size;
        self.cache.get_or_create(key, || {
            let texture = device.create_texture(descriptor);
            let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
            if !data.is_empty() {
                queue.write_texture(
                    TextureWrite {
                        texture,
                        mip_level: 0,
                        origin: [0, 0, 0],
                        size: Extent3d::new_2d(size.width, size.height),
                    },
                    data,
                    layout,
                );
            }
            GpuTexture { texture, view }
        })
    }

    /// Drops one reference to `key`, destroying the texture and its view when it
    /// reaches zero. Returns whether the texture was evicted.
    pub fn release<D: RenderDevice>(&mut self, device: &D, key: &TextureKey) -> bool {
        match self.cache.release(key) {
            Some(gpu) => {
                device.destroy_texture_view(gpu.view);
                device.destroy_texture(gpu.texture);
                true
            }
            None => false,
        }
    }

    /// Destroys every cached texture regardless of reference count, for device
    /// loss or teardown.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        self.cache.drain_all(|gpu| {
            device.destroy_texture_view(gpu.view);
            device.destroy_texture(gpu.texture);
        });
    }
}
