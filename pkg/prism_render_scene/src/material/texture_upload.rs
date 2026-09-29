//! Render-world bindless texture arrays for the unified Material ABI.
//!
//! [`BindlessTextureHeap`](super::texture_heap::BindlessTextureHeap) assigns
//! every material texture a stable slot; this module materialises the matching
//! GPU state. It owns:
//!
//! * three **reserved 1x1 fallback textures** — opaque white, the flat
//!   tangent-space normal `(0.5, 0.5, 1.0)`, and opaque black — so every
//!   semantic samples a correct identity value even before any real image
//!   uploads, and
//! * two **dense, fixed-length arrays** (`views`, `samplers`) sized to the
//!   heap's device-clamped capacity. Slot `i` of `views`/`samplers` is bound at
//!   index `i` of the `binding_array<texture_2d<f32>>` /
//!   `binding_array<sampler>` that `material_sample.wesl` samples.
//!
//! The arrays are **densely populated**: any slot without a resident,
//! GPU-uploaded image is backed by the white (or, for reserved slot 1, the
//! flat-normal) fallback rather than left unbound. This keeps the shader free of
//! per-texel bounds branches and avoids depending on
//! `PARTIALLY_BOUND_BINDING_ARRAY`, at the cost of binding a handful of shared
//! 1x1 views for the unused tail.
//!
//! [`compose`](MaterialTextureArrays::compose) rebuilds the arrays each frame
//! from the current heap residency and the [`RenderAssets<GpuImage>`] cache,
//! bumping [`version`](MaterialTextureArrays::version) only when the resolved
//! view set actually changes so the bind group is recreated lazily.
//!
//! Bindless is gated on device support. When the adapter lacks binding arrays
//! (or is a known-buggy Adreno) the resource degrades to a non-bindless stub
//! (`bindless == false`, empty arrays) and the material bind group falls back to
//! the three storage-buffer bindings alone.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    get_adreno_model,
    render_asset::RenderAssets,
    render_resource::{
        AddressMode, Extent3d, FilterMode, MipmapFilterMode, Sampler, SamplerDescriptor, Texture,
        TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureView, TextureViewDescriptor, TextureViewId, WgpuFeatures, WgpuSampler,
        WgpuTextureView,
    },
    renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
    texture::GpuImage,
};

use super::runtime::RenderMaterialRegistry;
use super::texture_heap::{BLACK_SLOT, FIRST_DYNAMIC_SLOT, FLAT_NORMAL_SLOT};

/// Hard ceiling on the bindless slot count, independent of device limits. Keeps
/// the arrays (and the layout's `count`) bounded on GPUs that advertise very
/// large binding-array limits. The heap capacity is clamped to
/// `min(this, device limits)` at render startup.
pub const MAX_BINDLESS_TEXTURES: u32 = 4096;

/// Number of reserved fallback textures (white / flat-normal / black).
const RESERVED_TEXTURE_COUNT: usize = 3;

/// Determines whether the device supports the bindless material path and, if so,
/// how many slots the heap and binding arrays may address.
///
/// Mirrors `bevy_pbr`'s `binding_arrays_are_usable`: requires both
/// `TEXTURE_BINDING_ARRAY` and non-uniform indexing, honours the per-stage
/// binding-array limits for both textures and samplers, and refuses the
/// binding-array-buggy Adreno <= 610 family. Returns the clamped capacity, or
/// `None` when bindless is unavailable or the device cannot fit even one dynamic
/// slot past the reserved block.
fn bindless_capacity(device: &RenderDevice, adapter_info: &RenderAdapterInfo) -> Option<u32> {
    if get_adreno_model(adapter_info).is_some_and(|model| model <= 610) {
        return None;
    }
    if !device.features().contains(
        WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    ) {
        return None;
    }
    let limits = device.limits();
    let capacity = MAX_BINDLESS_TEXTURES
        .min(limits.max_binding_array_elements_per_shader_stage)
        .min(limits.max_binding_array_sampler_elements_per_shader_stage);
    (capacity > FIRST_DYNAMIC_SLOT).then_some(capacity)
}

/// Builds one opaque 1x1 `Rgba8Unorm` texture from a single RGBA texel.
fn create_solid_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &'static str,
    texel: [u8; 4],
) -> Texture {
    device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &texel,
    )
}

/// Dense, device-clamped bindless texture and sampler arrays for the material
/// bind group. See the module docs for the population contract.
#[derive(Resource)]
pub struct MaterialTextureArrays {
    bindless: bool,
    capacity: u32,
    /// Retains ownership of the reserved fallback textures for the lifetime of
    /// the resource; the views below hold GPU references but keeping the
    /// textures explicit documents the ownership and guards against any backend
    /// that does not keep the parent texture alive through a view alone.
    #[expect(
        dead_code,
        reason = "retains GPU texture ownership behind the reserved fallback views"
    )]
    reserved_textures: [Texture; RESERVED_TEXTURE_COUNT],
    /// Views onto the reserved textures: `[white, flat-normal, black]`.
    reserved_views: [TextureView; RESERVED_TEXTURE_COUNT],
    /// Shared filtering sampler bound at every slot without an authored sampler.
    default_sampler: Sampler,
    /// Dense per-slot texture views, length == `capacity` when bindless.
    views: Vec<TextureView>,
    /// Dense per-slot samplers, length == `capacity` when bindless.
    samplers: Vec<Sampler>,
    /// Identity of every bound view last compose, used to bump `version` only on
    /// a real change.
    fingerprint: Vec<TextureViewId>,
    /// Monotonic counter incremented whenever the resolved view set changes, so
    /// the bind group is rebuilt lazily.
    version: u32,
}

impl FromWorld for MaterialTextureArrays {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>().clone();
        let queue = world.resource::<RenderQueue>().clone();
        let capacity = {
            let adapter_info = world.resource::<RenderAdapterInfo>();
            bindless_capacity(&device, adapter_info)
        };

        let reserved_textures = [
            create_solid_texture(
                &device,
                &queue,
                "prism_bindless_white",
                [255, 255, 255, 255],
            ),
            create_solid_texture(
                &device,
                &queue,
                "prism_bindless_flat_normal",
                [128, 128, 255, 255],
            ),
            create_solid_texture(&device, &queue, "prism_bindless_black", [0, 0, 0, 255]),
        ];
        let reserved_views = [
            reserved_textures[0].create_view(&TextureViewDescriptor::default()),
            reserved_textures[1].create_view(&TextureViewDescriptor::default()),
            reserved_textures[2].create_view(&TextureViewDescriptor::default()),
        ];
        let default_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("prism_bindless_default_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (bindless, capacity) = match capacity {
            Some(capacity) => (true, capacity),
            None => (false, 0),
        };
        let slots = capacity as usize;
        let mut arrays = Self {
            bindless,
            capacity,
            reserved_textures,
            reserved_views,
            default_sampler,
            views: Vec::with_capacity(slots),
            samplers: Vec::with_capacity(slots),
            fingerprint: Vec::with_capacity(slots),
            version: 0,
        };
        if bindless {
            for slot in 0..capacity {
                arrays
                    .views
                    .push(arrays.reserved_view_for_slot(slot).clone());
                arrays.samplers.push(arrays.default_sampler.clone());
            }
            arrays.fingerprint = arrays.views.iter().map(TextureView::id).collect();
        }
        arrays
    }
}

impl MaterialTextureArrays {
    /// Whether the bindless material path is active on this device.
    pub fn bindless(&self) -> bool {
        self.bindless
    }

    /// Total addressable bindless slots (0 when the device lacks bindless).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Monotonic version of the current view set; changes only when a bound view
    /// actually changes so the bind group is rebuilt lazily.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The fallback view a slot resolves to before (or without) a real upload.
    /// Reserved slot 1 is the flat tangent-space normal; every other slot,
    /// including the dynamic tail, falls back to opaque white.
    fn reserved_view_for_slot(&self, slot: u32) -> &TextureView {
        let index = match slot {
            FLAT_NORMAL_SLOT => 1,
            BLACK_SLOT => 2,
            // WHITE_SLOT and every other slot (including the dynamic tail) fall
            // back to opaque white in reserved view 0.
            _ => 0,
        };
        &self.reserved_views[index]
    }

    /// Rebuilds the dense view/sampler arrays from the heap's current residency
    /// and the GPU image cache. Slots whose image has not yet produced a
    /// [`GpuImage`] keep their fallback so the shader always samples something
    /// valid. Bumps [`version`](Self::version) only when the resolved view set
    /// changes. A no-op on non-bindless devices.
    pub(crate) fn compose(
        &mut self,
        runtime: &RenderMaterialRegistry,
        images: &RenderAssets<GpuImage>,
    ) {
        if !self.bindless {
            return;
        }
        // Reset every slot to its fallback so retired/absent images never leave
        // a stale view bound.
        for slot in 0..self.capacity {
            let fallback = self.reserved_view_for_slot(slot).clone();
            self.views[slot as usize] = fallback;
            self.samplers[slot as usize] = self.default_sampler.clone();
        }
        // Overlay the resident, uploaded images at their assigned slots.
        for (image, slot) in runtime.texture_slots() {
            if slot >= self.capacity {
                continue;
            }
            if let Some(gpu_image) = images.get(image) {
                self.views[slot as usize] = gpu_image.texture_view.clone();
                self.samplers[slot as usize] = gpu_image.sampler.clone();
            }
            // Otherwise the image is resident in the heap but its GPU upload has
            // not landed yet; leave the white fallback until the RenderAsset
            // appears on a later frame.
        }
        let fingerprint: Vec<TextureViewId> = self.views.iter().map(TextureView::id).collect();
        if fingerprint != self.fingerprint {
            self.fingerprint = fingerprint;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Borrows the raw texture views as the `&[&wgpu::TextureView]` slice the
    /// bind group's `TextureViewArray` binding expects.
    pub fn view_array(&self) -> Vec<&WgpuTextureView> {
        self.views.iter().map(|view| &**view).collect()
    }

    /// Borrows the raw samplers as the `&[&wgpu::Sampler]` slice the bind
    /// group's `SamplerArray` binding expects.
    pub fn sampler_array(&self) -> Vec<&WgpuSampler> {
        self.samplers.iter().map(|sampler| &**sampler).collect()
    }
}

/// Rebuilds [`MaterialTextureArrays`] from the current heap residency and GPU
/// image cache each frame. Runs in `PrepareResources`; a no-op when bindless is
/// unavailable.
pub(crate) fn prepare_material_texture_arrays(
    mut arrays: ResMut<MaterialTextureArrays>,
    runtime: Res<RenderMaterialRegistry>,
    images: Res<RenderAssets<GpuImage>>,
) {
    arrays.compose(&runtime, &images);
}
