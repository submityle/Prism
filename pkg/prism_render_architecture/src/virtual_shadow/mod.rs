//! Virtual shadow-map feature boundary.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShadowPageKey {
    pub light: u32,
    pub level: u16,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VirtualShadowSettings {
    pub physical_pages: u32,
    pub page_size: u16,
    pub max_clip_levels: u8,
}
