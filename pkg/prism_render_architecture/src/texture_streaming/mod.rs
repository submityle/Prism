//! Texture mip and virtual-texture streaming.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TexturePageKey {
    pub texture: u32,
    pub mip: u8,
    pub layer: u16,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSemantic {
    Color,
    Normal,
    RoughnessMetalAo,
    Height,
    Mask,
    Hdr,
}
