//! Backend-neutral lighting-channel and NPR light-layer routing decision math.
//!
//! Every AAA renderer needs a cheap, deterministic answer to two questions
//! *before* it evaluates an expensive BSDF for a light/surface pair:
//!
//! 1. **Does this light touch this object at all?** — Unreal's *Lighting
//!    Channels*: each light and each primitive carries a small channel bitmask
//!    and the light only affects the primitive when their masks intersect. This
//!    is a pure visibility gate shared by *every* illumination model — PBR,
//!    NPR and hybrid alike — because it is decided on the primitive/light
//!    pair, not on the shading maths.
//! 2. **Which stylized accumulation bin does its contribution land in?** — the
//!    NPR extension: a stylized front splits lighting into *layers* (key / fill
//!    / rim / custom) so the composite can ramp, tint or outline each band
//!    independently. A light carries a layer bitmask and feeds every bin whose
//!    bit is set. PBR simply ignores the layer axis (all lights collapse into
//!    the single lit accumulator), which is exactly why the same routing record
//!    serves PBR, NPR and hybrid: the data is identical, only the resolution
//!    differs.
//!
//! Keeping this as backend-neutral integer math lets the CPU golden and the
//! `shaders/light_routing.wesl` GPU twin evaluate the *same* culling
//! decision, and lets the clustered-forward cull ([`crate::cluster`])
//! refine a cluster's per-light bitset by channel with no floating-point work.

/// Number of lighting channels. Chosen to pack into a single G-buffer byte /
/// stencil nibble on the GPU side; Unreal exposes three, we generalise to a
/// byte so custom render passes have room without widening storage.
pub const MAX_LIGHTING_CHANNELS: u32 = 8;

/// Number of NPR light layers (stylized accumulation bins). Four covers the
/// canonical key / fill / rim / custom split; the mask still packs into a
/// nibble alongside the channel byte.
pub const MAX_LIGHT_LAYERS: u32 = 4;

/// A lighting-channel membership bitmask.
///
/// Bit `i` set means the light or primitive participates in channel
/// `i`. A light affects a primitive when their masks share at least one
/// bit. The default is channel `0` only, matching Unreal where
/// unconfigured lights and primitives all sit on the first channel and
/// therefore interact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightingChannelMask(u32);

impl Default for LightingChannelMask {
    fn default() -> Self {
        // Channel 0 set: the neutral "everything talks to everything" state.
        Self(1)
    }
}

impl LightingChannelMask {
    /// Valid-bit window for [`MAX_LIGHTING_CHANNELS`] channels.
    const VALID: u32 = (1u32 << MAX_LIGHTING_CHANNELS) - 1;

    /// An empty mask that participates in no channel (affects nothing).
    #[must_use]
    pub const fn none() -> Self {
        Self(0)
    }

    /// Builds a mask from raw bits, discarding any above
    /// [`MAX_LIGHTING_CHANNELS`].
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & Self::VALID)
    }

    /// Builds a mask with the single channel `index` set; out-of-range
    /// indices yield the empty mask so a bad authoring value fails safe (light
    /// touches nothing) rather than silently aliasing channel 0.
    #[must_use]
    pub fn from_index(index: u32) -> Self {
        if index < MAX_LIGHTING_CHANNELS {
            Self(1u32 << index)
        } else {
            Self(0)
        }
    }

    /// The raw channel bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether channel `index` is a member of this mask.
    #[must_use]
    pub fn contains(self, index: u32) -> bool {
        index < MAX_LIGHTING_CHANNELS && (self.0 & (1u32 << index)) != 0
    }

    /// Whether the mask participates in no channel at all.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns a copy with channel `index` added.
    #[must_use]
    pub fn with_channel(self, index: u32) -> Self {
        Self(self.0 | Self::from_index(index).0)
    }

    /// Returns a copy with channel `index` removed.
    #[must_use]
    pub fn without_channel(self, index: u32) -> Self {
        Self(self.0 & !Self::from_index(index).0)
    }

    /// Union (bits set in either mask).
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection (bits set in both masks).
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Whether this light mask affects a primitive carrying `primitive`:
    /// true iff they share at least one channel.
    #[must_use]
    pub const fn affects(self, primitive: Self) -> bool {
        (self.0 & primitive.0) != 0
    }
}

/// An NPR light-layer routing bitmask (stylized accumulation bins).
///
/// Bit `i` set means the light contributes to layer `i` (0 = key,
/// 1 = fill, 2 = rim, 3 = custom by convention, though the composite is free
/// to assign meaning). A light with no layer bits set contributes to no
/// stylized bin; the default routes to the key layer only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightLayerMask(u32);

impl Default for LightLayerMask {
    fn default() -> Self {
        // Key layer (0) only.
        Self(1)
    }
}

impl LightLayerMask {
    const VALID: u32 = (1u32 << MAX_LIGHT_LAYERS) - 1;

    /// A mask that feeds no stylized layer.
    #[must_use]
    pub const fn none() -> Self {
        Self(0)
    }

    /// A mask feeding every stylized layer (the PBR-equivalent "all bins"
    /// state used when a stylized front wants a light in every band).
    #[must_use]
    pub const fn all() -> Self {
        Self(Self::VALID)
    }

    /// Builds a mask from raw bits, discarding any above
    /// [`MAX_LIGHT_LAYERS`].
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & Self::VALID)
    }

    /// Builds a mask with the single layer `index` set; out-of-range
    /// indices yield the empty mask (fails safe, contributes to nothing).
    #[must_use]
    pub fn from_index(index: u32) -> Self {
        if index < MAX_LIGHT_LAYERS {
            Self(1u32 << index)
        } else {
            Self(0)
        }
    }

    /// The raw layer bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether the light feeds layer `index`.
    #[must_use]
    pub fn contains(self, index: u32) -> bool {
        index < MAX_LIGHT_LAYERS && (self.0 & (1u32 << index)) != 0
    }

    /// Whether the light feeds no layer.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The routing record a light carries: which primitives it may touch
/// (channels) and, for stylized fronts, which accumulation bins it feeds
/// (layers). PBR and hybrid fronts read [`channels`](Self::channels) and
/// ignore [`layers`](Self::layers); NPR reads both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightRouting {
    /// Visibility gate shared by every illumination model.
    pub channels: LightingChannelMask,
    /// Stylized accumulation-bin routing; ignored by pure PBR.
    pub layers: LightLayerMask,
}

impl LightRouting {
    /// PBR / hybrid visibility gate: does this light touch a primitive on
    /// `primitive_channels`? Decided purely on channel intersection, so
    /// it is identical across illumination models.
    #[must_use]
    pub const fn lights_primitive(self, primitive_channels: LightingChannelMask) -> bool {
        self.channels.affects(primitive_channels)
    }

    /// NPR resolution: does this light contribute to the stylized bin
    /// `active_layer` *for* a primitive on `primitive_channels`?
    /// Both gates must pass — the channel visibility gate and the layer
    /// routing gate — so a stylized key/fill/rim pass only sums the lights
    /// authored into that band and visible to the object.
    #[must_use]
    pub fn contributes_to_layer(
        self,
        primitive_channels: LightingChannelMask,
        active_layer: u32,
    ) -> bool {
        self.channels.affects(primitive_channels) && self.layers.contains(active_layer)
    }
}

/// Refines a cluster's active-light bitset by lighting channel.
///
/// The clustered-forward cull ([`crate::cluster`]) hands the resolve pass
/// a `u64` bitset whose bit `i` marks light `i` (within the
/// cluster's light slice) as spatially reaching the froxel. This clears every
/// bit whose light does not share a channel with the shaded primitive, so the
/// BSDF loop skips lights that are near but on a different channel. Bits beyond
/// `channel_masks.len()` are cleared (no routing record => cannot be
/// proven to affect the primitive).
#[must_use]
pub fn cull_lights_by_channel(
    active: u64,
    channel_masks: &[LightingChannelMask],
    primitive: LightingChannelMask,
) -> u64 {
    let mut remaining = active;
    let mut result: u64 = 0;
    while remaining != 0 {
        let bit = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let index = bit as usize;
        let keep = channel_masks
            .get(index)
            .is_some_and(|mask| mask.affects(primitive));
        if keep {
            result |= 1u64 << bit;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_channel_masks_interact() {
        // Unconfigured light and primitive both sit on channel 0 and interact.
        let light = LightingChannelMask::default();
        let prim = LightingChannelMask::default();
        assert!(light.affects(prim));
        assert!(!light.is_empty());
        assert_eq!(light.bits(), 1);
    }

    #[test]
    fn disjoint_channels_do_not_interact() {
        let light = LightingChannelMask::from_index(1);
        let prim = LightingChannelMask::from_index(2);
        assert!(!light.affects(prim));
        assert!(!prim.affects(light));
    }

    #[test]
    fn shared_channel_makes_them_interact() {
        let light = LightingChannelMask::from_index(1).with_channel(3);
        let prim = LightingChannelMask::from_index(3);
        assert!(light.affects(prim));
        assert!(light.contains(1) && light.contains(3));
        assert!(!light.contains(2));
    }

    #[test]
    fn empty_mask_affects_nothing() {
        let empty = LightingChannelMask::none();
        assert!(empty.is_empty());
        assert!(!empty.affects(LightingChannelMask::default()));
        assert!(!LightingChannelMask::default().affects(empty));
    }

    #[test]
    fn out_of_range_channel_fails_safe() {
        // An index at/above MAX must not alias channel 0 or any valid bit.
        let mask = LightingChannelMask::from_index(MAX_LIGHTING_CHANNELS);
        assert!(mask.is_empty());
        assert!(!mask.contains(0));
        assert!(!LightingChannelMask::default().contains(MAX_LIGHTING_CHANNELS));
    }

    #[test]
    fn from_bits_discards_high_bits() {
        // Bits above the channel window are masked away, keeping storage tight.
        let mask = LightingChannelMask::from_bits(0xFFFF_FFFF);
        assert_eq!(mask.bits(), (1u32 << MAX_LIGHTING_CHANNELS) - 1);
    }

    #[test]
    fn with_and_without_channel_are_inverses() {
        let base = LightingChannelMask::from_index(2);
        let added = base.with_channel(5);
        assert!(added.contains(2) && added.contains(5));
        let removed = added.without_channel(5);
        assert_eq!(removed, base);
    }

    #[test]
    fn union_and_intersection_behave() {
        let a = LightingChannelMask::from_index(0).with_channel(1);
        let b = LightingChannelMask::from_index(1).with_channel(2);
        assert_eq!(a.union(b).bits(), 0b0111);
        assert_eq!(a.intersection(b), LightingChannelMask::from_index(1));
    }

    #[test]
    fn pbr_gate_ignores_layers() {
        // Two lights differing only in layer routing gate identically for PBR.
        let prim = LightingChannelMask::default();
        let key_only = LightRouting {
            channels: LightingChannelMask::default(),
            layers: LightLayerMask::from_index(0),
        };
        let rim_only = LightRouting {
            channels: LightingChannelMask::default(),
            layers: LightLayerMask::from_index(2),
        };
        assert!(key_only.lights_primitive(prim));
        assert!(rim_only.lights_primitive(prim));
    }

    #[test]
    fn npr_layer_routing_splits_bins() {
        let prim = LightingChannelMask::default();
        let rim = LightRouting {
            channels: LightingChannelMask::default(),
            layers: LightLayerMask::from_index(2),
        };
        // Contributes to the rim bin (2) but not the key (0) or fill (1) bins.
        assert!(rim.contributes_to_layer(prim, 2));
        assert!(!rim.contributes_to_layer(prim, 0));
        assert!(!rim.contributes_to_layer(prim, 1));
    }

    #[test]
    fn layer_routing_still_respects_channel_gate() {
        // A light on the right layer but wrong channel contributes to nothing.
        let prim = LightingChannelMask::from_index(4);
        let wrong_channel = LightRouting {
            channels: LightingChannelMask::from_index(1),
            layers: LightLayerMask::all(),
        };
        assert!(!wrong_channel.contributes_to_layer(prim, 0));
        assert!(!wrong_channel.lights_primitive(prim));
    }

    #[test]
    fn all_layers_feeds_every_bin() {
        let prim = LightingChannelMask::default();
        let everywhere = LightRouting {
            channels: LightingChannelMask::default(),
            layers: LightLayerMask::all(),
        };
        for layer in 0..MAX_LIGHT_LAYERS {
            assert!(everywhere.contributes_to_layer(prim, layer));
        }
        assert!(!everywhere.contributes_to_layer(prim, MAX_LIGHT_LAYERS));
    }

    #[test]
    fn default_routing_is_channel0_key_layer() {
        let r = LightRouting::default();
        assert!(r.lights_primitive(LightingChannelMask::default()));
        assert!(r.contributes_to_layer(LightingChannelMask::default(), 0));
        assert!(!r.contributes_to_layer(LightingChannelMask::default(), 1));
    }

    #[test]
    fn cull_keeps_only_channel_matching_lights() {
        // Cluster reports lights 0,1,2 spatially reaching the froxel; only the
        // ones sharing a channel with the primitive survive.
        let prim = LightingChannelMask::from_index(1);
        let masks = [
            LightingChannelMask::from_index(0), // no share -> culled
            LightingChannelMask::from_index(1), // share -> kept
            LightingChannelMask::from_index(1).with_channel(2), // share -> kept
        ];
        let active = 0b111u64;
        let culled = cull_lights_by_channel(active, &masks, prim);
        assert_eq!(culled, 0b110u64);
    }

    #[test]
    fn cull_clears_bits_without_a_routing_record() {
        // A bit set beyond the mask slice cannot be proven to affect the
        // primitive, so it is culled (fail safe).
        let prim = LightingChannelMask::default();
        let masks = [LightingChannelMask::default()];
        let active = 0b101u64; // light 0 has a record, light 2 does not
        let culled = cull_lights_by_channel(active, &masks, prim);
        assert_eq!(culled, 0b001u64);
    }

    #[test]
    fn cull_of_empty_set_is_empty() {
        let masks = [LightingChannelMask::default()];
        assert_eq!(cull_lights_by_channel(0, &masks, LightingChannelMask::default()), 0);
    }

    #[test]
    fn cull_preserves_high_bit_indices() {
        // Light at bit 63 sharing a channel must survive the cull.
        let prim = LightingChannelMask::from_index(3);
        let mut masks = vec![LightingChannelMask::none(); 64];
        masks[63] = LightingChannelMask::from_index(3);
        let active = 1u64 << 63;
        assert_eq!(cull_lights_by_channel(active, &masks, prim), 1u64 << 63);
    }
}
