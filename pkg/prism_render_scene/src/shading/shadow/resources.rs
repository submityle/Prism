//! Device-side shadow resources: the depth atlas array texture and the storage
//! buffers that mirror the per-frame [`ExtractedShadows`] onto the GPU.
//!
//! Four parallel storage buffers back the resolve pass's shadow bind group: a
//! directional-shadow array, a point-shadow array, a spot-shadow array, and a
//! single-element globals record carrying the live slot counts and the shared
//! atlas resolution.  They
//! are packed from [`ExtractedShadows`] every frame; empty arrays are padded
//! with one disabled element so the storage bindings are never zero-sized and
//! the shader's `enabled == 0` early-out keeps unlit scenes correct.
//!
//! The atlas itself is a single `texture_2d_array<f32>` whose `.r` channel
//! stores either the wgpu NDC depth (directional cascades / spot frusta) or the
//! light-range-normalized linear distance (point-light cube faces), exactly as
//! `shaders/shadow.wesl` and the CPU golden reference
//! [`prism_render_shading::shadow`] expect.  The depth-raster pass renders into
//! each layer as a `RENDER_ATTACHMENT`; the resolve pass samples the whole
//! array through a linear `sampler` and performs the depth comparison manually
//! in-shader to stay a byte-for-byte twin of the CPU reference.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{
        AddressMode, Buffer, BufferUsages, Extent3d, FilterMode, RawBufferVec, Sampler,
        SamplerDescriptor, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::{RenderDevice, RenderQueue},
};

use super::abi::{
    GpuDirectionalShadow, GpuPointShadow, GpuShadowGlobals, GpuSpotShadow,
    MAX_SHADOW_DIRECTIONALS, MAX_SHADOW_POINTS, MAX_SHADOW_SPOTS,
};
use super::pipeline::SHADOW_DEPTH_FORMAT;

/// Texel format of every atlas layer.  A single-channel 32-bit float holds the
/// full-precision NDC depth or normalized linear distance the shadow test
/// compares against; `R32Float` is renderable as a color attachment on every
/// target tier Prism supports, avoiding a hardware depth format whose sampling
/// path differs across backends.
pub(crate) const SHADOW_ATLAS_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Default number of atlas array layers.  A directional light consumes up to
/// four (one per cascade) and a point light six (its cube faces), so sixteen
/// layers comfortably holds a sun plus a couple of shadow-casting point lights.
pub(crate) const DEFAULT_SHADOW_ATLAS_LAYERS: u32 = 16;

/// Default square edge resolution (texels) of each atlas layer.
pub(crate) const DEFAULT_SHADOW_ATLAS_RESOLUTION: u32 = 1024;

/// Static allocation budget for the shadow atlas array texture.
///
/// Layers are handed out by the extraction pass (a directional light claims one
/// per cascade, a point light six) up to `max_layers`; `resolution` is the
/// shared square edge length of every layer.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShadowAtlasConfig {
    /// Number of array layers the atlas texture is allocated with.
    pub max_layers: u32,
    /// Square edge resolution (texels) of each layer.
    pub resolution: u32,
}

impl Default for ShadowAtlasConfig {
    fn default() -> Self {
        Self {
            max_layers: DEFAULT_SHADOW_ATLAS_LAYERS,
            resolution: DEFAULT_SHADOW_ATLAS_RESOLUTION,
        }
    }
}

impl ShadowAtlasConfig {
    /// Builds a config, clamping both dimensions to at least one so the texture
    /// descriptor is always valid.
    pub(crate) fn new(max_layers: u32, resolution: u32) -> Self {
        Self {
            max_layers: max_layers.max(1),
            resolution: resolution.max(1),
        }
    }

    /// The `UV` size of one texel (`1 / resolution`), the value shaders use to
    /// scale their `PCF` / `PCSS` tap offsets.
    pub(crate) fn texel_uv_size(&self) -> [f32; 2] {
        let inv = 1.0 / self.resolution as f32;
        [inv, inv]
    }
}

/// The shadow depth atlas: one `R32Float` array texture, its array view, and a
/// linear sampler.  Rebuilt only when [`ShadowAtlasConfig`] changes.
#[derive(Resource)]
pub(crate) struct ShadowAtlas {
    config: ShadowAtlasConfig,
    view: TextureView,
    sampler: Sampler,
    /// One single-layer `D2` view per array layer, used as the colour render
    /// target when the depth pass fills that layer.  Index `layer` addresses
    /// the same global layer the resolve pass samples through `view`.
    layer_views: Vec<TextureView>,
    /// Transient hardware depth-stencil target shared by every layer's depth
    /// draw; cleared to `1.0` at the start of each pass and never sampled.
    depth_view: TextureView,
}

impl FromWorld for ShadowAtlas {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        Self::create(device, ShadowAtlasConfig::default())
    }
}

impl ShadowAtlas {
    /// Allocates the array texture, its `D2Array` view, and the sampler for the
    /// given config.
    fn create(device: &RenderDevice, config: ShadowAtlasConfig) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("prism shadow atlas"),
            size: Extent3d {
                width: config.resolution,
                height: config.resolution,
                depth_or_array_layers: config.max_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SHADOW_ATLAS_FORMAT,
            // RENDER_ATTACHMENT: the depth-raster pass draws into each layer.
            // TEXTURE_BINDING: the resolve pass samples the whole array.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("prism shadow atlas view"),
            format: Some(SHADOW_ATLAS_FORMAT),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(config.max_layers),
            usage: None,
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("prism shadow atlas sampler"),
            // Out-of-range taps are resolved to the "far/lit" value inside the
            // shader by an explicit UV bounds check, so edge clamping here is
            // only a defensive fallback.
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        // One single-layer colour view per array layer so the depth pass can
        // target a specific layer as a render attachment (array views cannot be
        // bound as colour attachments directly).
        let mut layer_views = Vec::with_capacity(config.max_layers as usize);
        for layer in 0..config.max_layers {
            layer_views.push(texture.create_view(&TextureViewDescriptor {
                label: Some("prism shadow atlas layer"),
                format: Some(SHADOW_ATLAS_FORMAT),
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                usage: None,
            }));
        }
        // A single depth-stencil target reused for every layer's pass: shadow
        // views are rendered one at a time, so one depth buffer at the shared
        // resolution suffices and is cleared per pass.
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("prism shadow atlas depth"),
            size: Extent3d {
                width: config.resolution,
                height: config.resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SHADOW_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            label: Some("prism shadow atlas depth view"),
            format: Some(SHADOW_DEPTH_FORMAT),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });
        Self {
            config,
            view,
            sampler,
            layer_views,
            depth_view,
        }
    }

    /// Reallocates the atlas if `config` differs from the live allocation.
    /// Returns `true` when a rebuild happened so the bind group can invalidate.
    pub(crate) fn ensure(&mut self, device: &RenderDevice, config: ShadowAtlasConfig) -> bool {
        if self.config == config {
            return false;
        }
        *self = Self::create(device, config);
        true
    }

    /// The `D2Array` view the resolve pass binds.
    pub(crate) fn view(&self) -> &TextureView {
        &self.view
    }

    /// The linear sampler the resolve pass binds alongside the array view.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// The single-layer colour view for `layer`, the depth pass's render
    /// target, or `None` if `layer` is outside the allocated range.
    pub(crate) fn layer_view(&self, layer: u32) -> Option<&TextureView> {
        self.layer_views.get(layer as usize)
    }

    /// The shared transient depth-stencil view every layer's depth draw tests
    /// against.
    pub(crate) fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }
}

/// Per-frame CPU staging of the shadow records before they are packed into the
/// GPU buffers.  The extraction pass fills these from the reference cascade /
/// atlas planners; [`ShadowGpuBuffers::rebuild`] copies them onto the GPU.
#[derive(Resource, Default)]
pub(crate) struct ExtractedShadows {
    /// One record per shadow-casting directional light this frame.
    pub directionals: Vec<GpuDirectionalShadow>,
    /// One record per shadow-casting point light this frame.
    pub points: Vec<GpuPointShadow>,
    /// One record per shadow-casting spot light this frame.
    pub spots: Vec<GpuSpotShadow>,
    /// The frame header: live slot counts and the shared atlas resolution.
    pub globals: GpuShadowGlobals,
    /// One entry per atlas layer that must be filled this frame: which layer to
    /// render into and the per-view matrix the depth pass binds.  Produced by
    /// [`plan_shadow_depth_draws`](prism_render_shading::plan_shadow_depth_draws)
    /// during extraction, consumed by the depth pass.
    pub depth_draws: Vec<prism_render_shading::ShadowDepthDraw>,
}

/// The three storage buffers mirroring [`ExtractedShadows`] onto the GPU.
#[derive(Resource)]
pub(crate) struct ShadowGpuBuffers {
    directionals: RawBufferVec<GpuDirectionalShadow>,
    points: RawBufferVec<GpuPointShadow>,
    spots: RawBufferVec<GpuSpotShadow>,
    globals: RawBufferVec<GpuShadowGlobals>,
    version: u32,
}

impl FromWorld for ShadowGpuBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut directionals = RawBufferVec::new(BufferUsages::STORAGE);
        directionals.set_label(Some("prism directional shadows"));
        let mut points = RawBufferVec::new(BufferUsages::STORAGE);
        points.set_label(Some("prism point shadows"));
        let mut spots = RawBufferVec::new(BufferUsages::STORAGE);
        spots.set_label(Some("prism spot shadows"));
        let mut globals = RawBufferVec::new(BufferUsages::STORAGE);
        globals.set_label(Some("prism shadow globals"));
        Self {
            directionals,
            points,
            spots,
            globals,
            version: 1,
        }
    }
}

impl ShadowGpuBuffers {
    /// Repacks the extracted shadows into the parallel storage arrays.  The
    /// globals record always holds exactly one element so the slot counts stay
    /// authoritative even when both shadow arrays are empty.
    pub(crate) fn rebuild(&mut self, shadows: &ExtractedShadows) {
        self.directionals.clear();
        self.points.clear();
        self.spots.clear();
        self.globals.clear();

        let directional_len = shadows.directionals.len().min(MAX_SHADOW_DIRECTIONALS);
        let point_len = shadows.points.len().min(MAX_SHADOW_POINTS);
        let spot_len = shadows.spots.len().min(MAX_SHADOW_SPOTS);
        self.directionals
            .extend(shadows.directionals[..directional_len].iter().copied());
        self.points
            .extend(shadows.points[..point_len].iter().copied());
        self.spots
            .extend(shadows.spots[..spot_len].iter().copied());

        // Keep the header counts authoritative against the clamped arrays so the
        // shader never reads past a populated slot even if extraction overfills.
        let mut globals = shadows.globals;
        globals.directional_count = globals.directional_count.min(directional_len as u32);
        globals.point_count = globals.point_count.min(point_len as u32);
        globals.spot_count = globals.spot_count.min(spot_len as u32);
        self.globals.push(globals);

        self.version = self.version.wrapping_add(1).max(1);
    }

    /// Streams the packed arrays to the GPU.  Empty shadow arrays are padded
    /// with a single disabled element so the storage bindings are never
    /// zero-sized; the shader skips them via their `enabled == 0` flag.
    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.directionals.is_empty() {
            self.directionals.push(GpuDirectionalShadow::default());
        }
        if self.points.is_empty() {
            self.points.push(GpuPointShadow::default());
        }
        if self.spots.is_empty() {
            self.spots.push(GpuSpotShadow::default());
        }
        if self.globals.is_empty() {
            self.globals.push(GpuShadowGlobals::default());
        }
        self.directionals.write_buffer(device, queue);
        self.points.write_buffer(device, queue);
        self.spots.write_buffer(device, queue);
        self.globals.write_buffer(device, queue);
    }

    /// Monotonic version bumped on every rebuild, for bind-group caching.
    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    /// The four storage buffers once they have been uploaded at least once.
    pub(crate) fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.directionals.buffer()?,
            self.points.buffer()?,
            self.spots.buffer()?,
            self.globals.buffer()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_config_clamps_dimensions_to_at_least_one() {
        let config = ShadowAtlasConfig::new(0, 0);
        assert_eq!(config.max_layers, 1);
        assert_eq!(config.resolution, 1);
    }

    #[test]
    fn atlas_config_texel_uv_size_is_inverse_resolution() {
        let config = ShadowAtlasConfig::new(4, 2048);
        assert_eq!(config.texel_uv_size(), [1.0 / 2048.0, 1.0 / 2048.0]);
    }

    #[test]
    fn rebuild_packs_both_arrays_and_single_globals_record() {
        let mut world = World::new();
        let mut buffers = ShadowGpuBuffers::from_world(&mut world);
        let mut shadows = ExtractedShadows::default();
        shadows.directionals.push(GpuDirectionalShadow::default());
        shadows.points.push(GpuPointShadow::default());
        shadows.points.push(GpuPointShadow::default());
        shadows.spots.push(GpuSpotShadow::default());
        shadows.globals.directional_count = 1;
        shadows.globals.point_count = 2;
        shadows.globals.spot_count = 1;
        shadows.globals.atlas_resolution = 1024;
        buffers.rebuild(&shadows);

        assert_eq!(buffers.directionals.values().len(), 1);
        assert_eq!(buffers.points.values().len(), 2);
        assert_eq!(buffers.spots.values().len(), 1);
        assert_eq!(buffers.globals.values().len(), 1);
        assert_eq!(buffers.globals.values()[0].point_count, 2);
        assert_eq!(buffers.globals.values()[0].spot_count, 1);
        assert_eq!(buffers.globals.values()[0].atlas_resolution, 1024);
    }

    #[test]
    fn rebuild_bumps_version_monotonically() {
        let mut world = World::new();
        let mut buffers = ShadowGpuBuffers::from_world(&mut world);
        let before = buffers.version();
        buffers.rebuild(&ExtractedShadows::default());
        assert_ne!(buffers.version(), before);
        // The globals record is always present even with no shadow casters.
        assert_eq!(buffers.globals.values().len(), 1);
    }
}
