//! Global 2D altitude/sun-zenith LUT allocation.
use bevy_ecs::prelude::*;
use bevy_render::{render_resource::*, renderer::RenderDevice};
pub(crate) const LUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub(crate) const LUT_WIDTH: u32 = 256;
pub(crate) const LUT_HEIGHT: u32 = 128;
#[derive(Resource)]
#[expect(dead_code, reason = "owns the GPU allocation backing the view")]
pub(crate) struct SkyMultiscatterLut {
    texture: Texture,
    view: TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}
impl SkyMultiscatterLut {
    pub(crate) fn view(&self) -> &TextureView {
        &self.view
    }
}
pub(crate) fn init_sky_multiscatter_lut(mut commands: Commands, device: Res<RenderDevice>) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("prism sky multiple-scattering LUT"),
        size: Extent3d {
            width: LUT_WIDTH,
            height: LUT_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: LUT_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    commands.insert_resource(SkyMultiscatterLut {
        texture,
        view,
        width: LUT_WIDTH,
        height: LUT_HEIGHT,
    });
}
