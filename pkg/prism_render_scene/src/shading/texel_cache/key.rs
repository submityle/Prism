//! Stable object/texture-space identity for a shade-cache page.

/// Which view-independent response is stored in a shade-cache page.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum ShadeChannel {
    /// Diffuse/irradiance contribution for a physically based surface.
    LitDiffuse = 0,
    /// Ramp-ready diffuse contribution for a stylized surface.
    StylizedDiffuse = 1,
    /// View-independent subsurface contribution.
    Subsurface = 2,
    /// Emissive contribution.
    Emission = 3,
}

/// A page in object/texture space.
///
/// `material_epoch` is part of the identity so a re-authored material cannot
/// accidentally consume an old page whose coordinates happen to match.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShadePageKey {
    /// Stable render-object identity.
    pub object_id: u32,
    /// Stable material identity.
    pub material_id: u32,
    /// Material revision/epoch captured when the page was requested.
    pub material_epoch: u64,
    /// Mesh face or chart layer.
    pub face: u16,
    /// Mip level in object/texture space; zero is finest.
    pub mip: u8,
    /// Cached response channel.
    pub channel: ShadeChannel,
    /// Page coordinate in the chart's x direction.
    pub page_x: u16,
    /// Page coordinate in the chart's y direction.
    pub page_y: u16,
}

impl ShadePageKey {
    /// Returns the coarser covering key `delta` mips above this page.
    #[must_use]
    pub const fn coarsen(self, delta: u8) -> Self {
        Self {
            object_id: self.object_id,
            material_id: self.material_id,
            material_epoch: self.material_epoch,
            face: self.face,
            mip: self.mip.saturating_add(delta),
            channel: self.channel,
            page_x: if delta >= u16::BITS as u8 {
                0
            } else {
                self.page_x >> delta
            },
            page_y: if delta >= u16::BITS as u8 {
                0
            } else {
                self.page_y >> delta
            },
        }
    }
}
