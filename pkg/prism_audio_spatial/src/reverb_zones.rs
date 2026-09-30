//! Reverb zones and game-defined auxiliary sends: the control-rate layer that
//! decides *how much* of each source is routed to which environmental
//! reverb/effect return bus based on where the listener stands in the world.
//!
//! # The model
//!
//! A [`ReverbZone`] is a world-space region (an axis-aligned box or a sphere)
//! bound to one auxiliary return bus ([`AuxBusId`]) with a base *send level*.
//! When the listener is inside the zone the source's wet path is routed to that
//! bus at the base level; as the listener leaves, the send fades smoothly to
//! zero across a *blend band* just outside the region, so walking from a cave
//! into open air cross-fades the cave reverb out instead of cutting it.
//!
//! A [`ReverbZoneField`] borrows a slice of zones and answers the single
//! control-rate question a mixer needs: "given the listener's position right
//! now, which buses are active and at what send level?" It writes the answer
//! into a caller-owned, fixed-capacity buffer of [`AuxSend`]s, so the whole
//! query is allocation free.
//!
//! Per-source coupling is deliberately factored out: the *environment* decides
//! the zone send (a function of the listener), while each *source* scales it by
//! its own [`wet_gain`](crate::spatializer::SpatialParams::wet_gain) (a function
//! of that source's distance and occlusion). Combine the two with
//! [`source_send_gain`].
//!
//! # Why the listener drives it
//!
//! Environmental reverb models the space the *listener* occupies, so the send
//! set is a property of the listener's position, not each emitter's. This is
//! the classic "game-defined aux send" split: one environment query per
//! control block feeds every voice, and each voice only contributes its own
//! wet scale on top. It keeps the per-source cost to a single multiply.
//!
//! # Overlap policy
//!
//! Overlapping zones that target the *same* bus collapse to the **strongest**
//! effective send (the dominant environment wins rather than double-sending).
//! Zones targeting *different* buses each contribute their own send, up to
//! [`MAX_AUX_SENDS`]; if more distinct buses are active than the buffer holds,
//! the weakest send is dropped so the loudest environments survive.
//!
//! # Control rate, not audio rate
//!
//! [`ReverbZoneField::resolve`] is a pure function evaluated once per control
//! block. It allocates nothing, locks nothing, and cannot panic, so it is safe
//! to call from a device callback thread. It holds no audio state.
//!
//! # Determinism
//!
//! All distance and blend math routes through [`bevy_math::ops`] (libm-backed)
//! rather than `f32` intrinsics, and the smoothstep blend uses only
//! polynomial arithmetic, so a given world configuration resolves to
//! bit-identical sends across targets and can be golden-compared.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. The design (listener-driven
//! environmental sends, a smoothstep transition band, and the same-bus/max
//! overlap rule) is implemented from standard, publicly documented
//! game-audio routing concepts (game-defined auxiliary sends, submix sends,
//! and snapshot-style reverb regions).

use prism_audio_core::math::{MIN_AUDIBLE_GAIN, Sample};

use bevy_math::{Vec3, ops};

/// Identifier of an auxiliary return bus (a reverb or effect return in the
/// mixer graph). Opaque; the mixer owns the mapping to an actual bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AuxBusId(pub u32);

/// The maximum number of distinct auxiliary sends resolved simultaneously.
///
/// A single listener rarely stands in more than a few overlapping environments
/// at once; this bound keeps [`ReverbZoneField::resolve`] fixed-cost and
/// allocation free. Callers size their output buffer to this.
pub const MAX_AUX_SENDS: usize = 4;

/// A resolved routing of a source's wet path to one auxiliary bus.
///
/// `level` is the *environmental* send level in `[0, 1]` before per-source
/// scaling; combine it with a source's
/// [`wet_gain`](crate::spatializer::SpatialParams::wet_gain) via
/// [`source_send_gain`] to get the final per-source send.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AuxSend {
    /// Destination auxiliary return bus.
    pub bus: AuxBusId,
    /// Environmental send level in `[0, 1]`.
    pub level: Sample,
}

/// The spatial extent of a [`ReverbZone`].
///
/// Both shapes expose a signed distance (negative inside, positive outside)
/// so the transition band is defined uniformly regardless of shape.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ZoneShape {
    /// Axis-aligned box centred at `center` with non-negative half-extents.
    Box {
        /// World-space centre of the box.
        center: Vec3,
        /// Non-negative half-extents along each axis.
        half_extents: Vec3,
    },
    /// Sphere centred at `center` with a non-negative `radius`.
    Sphere {
        /// World-space centre of the sphere.
        center: Vec3,
        /// Non-negative radius.
        radius: f32,
    },
}

impl ZoneShape {
    /// Signed distance from `point` to the shape surface: negative inside,
    /// zero on the surface, positive outside (Euclidean distance to the
    /// nearest surface point for the sphere; the standard exterior box SDF
    /// otherwise).
    #[must_use]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        match *self {
            ZoneShape::Box {
                center,
                half_extents,
            } => {
                let he = half_extents.max(Vec3::ZERO);
                let d = (point - center).abs() - he;
                let outside = d.max(Vec3::ZERO);
                let outside_len = ops::sqrt(outside.dot(outside));
                let inside = d.x.max(d.y).max(d.z).min(0.0);
                outside_len + inside
            }
            ZoneShape::Sphere { center, radius } => {
                let v = point - center;
                ops::sqrt(v.dot(v)) - radius.max(0.0)
            }
        }
    }

    /// Returns `true` when `point` lies inside or on the shape surface.
    #[must_use]
    pub fn contains(&self, point: Vec3) -> bool {
        self.signed_distance(point) <= 0.0
    }
}

/// A world-space region that routes the listener's environment to one
/// auxiliary reverb/effect bus.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReverbZone {
    /// Destination auxiliary return bus.
    pub bus: AuxBusId,
    /// Spatial extent of the zone.
    pub shape: ZoneShape,
    /// Base send level in `[0, 1]` applied when the listener is fully inside.
    pub send_level: Sample,
    /// Width in metres of the transition band *outside* the shape over which
    /// the send fades from `send_level` to zero. Zero gives a hard edge.
    pub blend_distance: f32,
}

impl ReverbZone {
    /// Builds a box-shaped reverb zone. `send_level` is clamped to `[0, 1]`;
    /// `half_extents` and `blend_distance` are clamped to be non-negative.
    #[must_use]
    pub fn box_zone(
        bus: AuxBusId,
        center: Vec3,
        half_extents: Vec3,
        send_level: Sample,
        blend_distance: f32,
    ) -> Self {
        Self {
            bus,
            shape: ZoneShape::Box {
                center,
                half_extents: half_extents.max(Vec3::ZERO),
            },
            send_level: send_level.clamp(0.0, 1.0),
            blend_distance: blend_distance.max(0.0),
        }
    }

    /// Builds a sphere-shaped reverb zone. `send_level` is clamped to `[0, 1]`;
    /// `radius` and `blend_distance` are clamped to be non-negative.
    #[must_use]
    pub fn sphere_zone(
        bus: AuxBusId,
        center: Vec3,
        radius: f32,
        send_level: Sample,
        blend_distance: f32,
    ) -> Self {
        Self {
            bus,
            shape: ZoneShape::Sphere {
                center,
                radius: radius.max(0.0),
            },
            send_level: send_level.clamp(0.0, 1.0),
            blend_distance: blend_distance.max(0.0),
        }
    }

    /// The membership weight in `[0, 1]` for a listener at `point`: `1` fully
    /// inside, smoothly falling to `0` at `blend_distance` beyond the surface
    /// via a cubic smoothstep (zero first derivative at both ends, so the
    /// cross-fade has no audible kink).
    #[must_use]
    pub fn weight(&self, point: Vec3) -> Sample {
        let s = self.shape.signed_distance(point);
        if s <= 0.0 {
            return 1.0;
        }
        let b = self.blend_distance;
        if b <= 0.0 || s >= b {
            return 0.0;
        }
        let t = s / b;
        // Smoothstep falls 0 -> 1; invert it for a 1 -> 0 fade as we exit.
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// The effective environmental send level in `[0, 1]` for a listener at
    /// `point`: the base [`send_level`](Self::send_level) scaled by the
    /// membership [`weight`](Self::weight).
    #[must_use]
    pub fn effective_send(&self, point: Vec3) -> Sample {
        self.send_level * self.weight(point)
    }
}

/// A borrowed set of reverb zones queried against a single listener position.
///
/// Holds no audio state and owns nothing; construct one per control block over
/// the currently loaded zones and call [`resolve`](Self::resolve).
#[derive(Debug, Clone, Copy)]
pub struct ReverbZoneField<'a> {
    /// The active reverb zones to consider.
    pub zones: &'a [ReverbZone],
}

impl<'a> ReverbZoneField<'a> {
    /// Wraps a slice of zones.
    #[must_use]
    pub fn new(zones: &'a [ReverbZone]) -> Self {
        Self { zones }
    }

    /// Resolves the active auxiliary sends for `listener_pos` into `out`,
    /// returning how many entries were written (at most `out.len()` and
    /// [`MAX_AUX_SENDS`]).
    ///
    /// Zones targeting the same bus collapse to the strongest effective send;
    /// distinct buses each get an entry. When more distinct buses are active
    /// than `out` can hold, the weakest send is dropped so the loudest
    /// environments survive. Sends below [`MIN_AUDIBLE_GAIN`] are ignored.
    ///
    /// This is a pure function: no allocation, no locks, no panics.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Vec3;
    /// # use prism_audio_spatial::reverb_zones::{
    /// #     AuxBusId, AuxSend, ReverbZone, ReverbZoneField,
    /// # };
    /// let cave = ReverbZone::sphere_zone(AuxBusId(7), Vec3::ZERO, 5.0, 0.8, 2.0);
    /// let zones = [cave];
    /// let field = ReverbZoneField::new(&zones);
    /// let mut out = [AuxSend { bus: AuxBusId(0), level: 0.0 }; 4];
    /// // Dead centre: full send to bus 7.
    /// let n = field.resolve(Vec3::ZERO, &mut out);
    /// assert_eq!(n, 1);
    /// assert_eq!(out[0].bus, AuxBusId(7));
    /// assert!((out[0].level - 0.8).abs() < 1e-6);
    /// // Well outside the blend band: nothing.
    /// assert_eq!(field.resolve(Vec3::new(100.0, 0.0, 0.0), &mut out), 0);
    /// ```
    pub fn resolve(&self, listener_pos: Vec3, out: &mut [AuxSend]) -> usize {
        let cap = out.len().min(MAX_AUX_SENDS);
        if cap == 0 {
            return 0;
        }
        let mut n = 0usize;
        for zone in self.zones {
            let level = zone.effective_send(listener_pos);
            if level <= MIN_AUDIBLE_GAIN {
                continue;
            }
            if let Some(slot) = out[..n].iter_mut().find(|s| s.bus == zone.bus) {
                // Same bus already present: keep the dominant environment.
                if level > slot.level {
                    slot.level = level;
                }
                continue;
            }
            if n < cap {
                out[n] = AuxSend {
                    bus: zone.bus,
                    level,
                };
                n += 1;
                continue;
            }
            // Buffer full and a new bus: evict the weakest if this is louder.
            let mut weakest = 0usize;
            for i in 1..n {
                if out[i].level < out[weakest].level {
                    weakest = i;
                }
            }
            if level > out[weakest].level {
                out[weakest] = AuxSend {
                    bus: zone.bus,
                    level,
                };
            }
        }
        n
    }
}

/// Combines an environmental send level with a source's own wet scale into the
/// final per-source send gain, clamped to `[0, 1]`.
///
/// `zone_send` comes from [`ReverbZoneField::resolve`] (a property of the
/// listener's environment); `source_wet_gain` is the source's
/// [`wet_gain`](crate::spatializer::SpatialParams::wet_gain) (a property of the
/// source's distance and occlusion). The two multiply.
///
/// # Examples
///
/// ```
/// # use prism_audio_spatial::reverb_zones::source_send_gain;
/// assert!((source_send_gain(0.8, 0.5) - 0.4).abs() < 1e-6);
/// assert_eq!(source_send_gain(1.5, 1.5), 1.0); // clamped
/// ```
#[must_use]
pub fn source_send_gain(zone_send: Sample, source_wet_gain: Sample) -> Sample {
    (zone_send * source_wet_gain).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus(id: u32) -> AuxBusId {
        AuxBusId(id)
    }

    #[test]
    fn sphere_signed_distance_sign() {
        let s = ZoneShape::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        assert!(s.signed_distance(Vec3::ZERO) < 0.0);
        assert!((s.signed_distance(Vec3::new(2.0, 0.0, 0.0))).abs() < 1e-6);
        assert!((s.signed_distance(Vec3::new(5.0, 0.0, 0.0)) - 3.0).abs() < 1e-6);
        assert!(s.contains(Vec3::new(1.0, 0.0, 0.0)));
        assert!(!s.contains(Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn box_signed_distance_inside_and_outside() {
        let b = ZoneShape::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        // Inside: negative, equals distance to nearest face.
        assert!((b.signed_distance(Vec3::ZERO) + 1.0).abs() < 1e-6);
        // On a face.
        assert!(b.signed_distance(Vec3::new(1.0, 0.0, 0.0)).abs() < 1e-6);
        // Outside along one axis.
        assert!((b.signed_distance(Vec3::new(3.0, 0.0, 0.0)) - 2.0).abs() < 1e-6);
        // Outside a corner: Euclidean distance to the corner.
        let corner = b.signed_distance(Vec3::new(2.0, 2.0, 2.0));
        assert!((corner - ops::sqrt(3.0)).abs() < 1e-5);
    }

    #[test]
    fn negative_inputs_are_clamped() {
        let z = ReverbZone::sphere_zone(bus(1), Vec3::ZERO, -4.0, 2.0, -1.0);
        assert!(z.send_level <= 1.0 && z.send_level >= 0.0);
        assert!(z.blend_distance >= 0.0);
        if let ZoneShape::Sphere { radius, .. } = z.shape {
            assert!(radius >= 0.0);
        } else {
            panic!("expected sphere");
        }
    }

    #[test]
    fn weight_is_one_inside_zero_far_and_monotone_in_band() {
        let z = ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 2.0, 1.0, 4.0);
        assert!((z.weight(Vec3::ZERO) - 1.0).abs() < 1e-6);
        assert!((z.weight(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);
        // Halfway through the band.
        let mid = z.weight(Vec3::new(4.0, 0.0, 0.0));
        assert!(mid > 0.0 && mid < 1.0);
        // Beyond the band.
        assert!(z.weight(Vec3::new(6.0, 0.0, 0.0)).abs() < 1e-6);
        assert!(z.weight(Vec3::new(100.0, 0.0, 0.0)).abs() < 1e-6);
        // Monotone decreasing across the band.
        let a = z.weight(Vec3::new(3.0, 0.0, 0.0));
        let b = z.weight(Vec3::new(5.0, 0.0, 0.0));
        assert!(a > mid && mid > b);
    }

    #[test]
    fn hard_edge_when_blend_zero() {
        let z = ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 2.0, 0.5, 0.0);
        assert!((z.weight(Vec3::new(1.9, 0.0, 0.0)) - 1.0).abs() < 1e-6);
        assert!(z.weight(Vec3::new(2.1, 0.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn effective_send_scales_with_level_and_weight() {
        let z = ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 2.0, 0.6, 4.0);
        assert!((z.effective_send(Vec3::ZERO) - 0.6).abs() < 1e-6);
        let mid = z.effective_send(Vec3::new(4.0, 0.0, 0.0));
        assert!(mid > 0.0 && mid < 0.6);
        assert!(z.effective_send(Vec3::new(10.0, 0.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn resolve_single_zone() {
        let zones = [ReverbZone::sphere_zone(bus(7), Vec3::ZERO, 5.0, 0.8, 2.0)];
        let field = ReverbZoneField::new(&zones);
        let mut out = [AuxSend { bus: bus(0), level: 0.0 }; MAX_AUX_SENDS];
        let n = field.resolve(Vec3::ZERO, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].bus, bus(7));
        assert!((out[0].level - 0.8).abs() < 1e-6);
    }

    #[test]
    fn resolve_outside_all_zones_is_empty() {
        let zones = [ReverbZone::sphere_zone(bus(7), Vec3::ZERO, 1.0, 0.8, 1.0)];
        let field = ReverbZoneField::new(&zones);
        let mut out = [AuxSend { bus: bus(0), level: 0.0 }; MAX_AUX_SENDS];
        assert_eq!(field.resolve(Vec3::new(50.0, 0.0, 0.0), &mut out), 0);
    }

    #[test]
    fn same_bus_collapses_to_strongest() {
        let zones = [
            ReverbZone::sphere_zone(bus(3), Vec3::ZERO, 10.0, 0.3, 1.0),
            ReverbZone::sphere_zone(bus(3), Vec3::ZERO, 10.0, 0.9, 1.0),
        ];
        let field = ReverbZoneField::new(&zones);
        let mut out = [AuxSend { bus: bus(0), level: 0.0 }; MAX_AUX_SENDS];
        let n = field.resolve(Vec3::ZERO, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].bus, bus(3));
        assert!((out[0].level - 0.9).abs() < 1e-6);
    }

    #[test]
    fn distinct_buses_each_get_a_send() {
        let zones = [
            ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 10.0, 0.5, 1.0),
            ReverbZone::sphere_zone(bus(2), Vec3::ZERO, 10.0, 0.7, 1.0),
        ];
        let field = ReverbZoneField::new(&zones);
        let mut out = [AuxSend { bus: bus(0), level: 0.0 }; MAX_AUX_SENDS];
        let n = field.resolve(Vec3::ZERO, &mut out);
        assert_eq!(n, 2);
        let mut buses = [out[0].bus, out[1].bus];
        buses.sort();
        assert_eq!(buses, [bus(1), bus(2)]);
    }

    #[test]
    fn overflow_keeps_loudest_sends() {
        // Five distinct buses at strictly increasing levels; the weakest must
        // be evicted so the four loudest survive.
        let zones = [
            ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 10.0, 0.1, 1.0),
            ReverbZone::sphere_zone(bus(2), Vec3::ZERO, 10.0, 0.2, 1.0),
            ReverbZone::sphere_zone(bus(3), Vec3::ZERO, 10.0, 0.3, 1.0),
            ReverbZone::sphere_zone(bus(4), Vec3::ZERO, 10.0, 0.4, 1.0),
            ReverbZone::sphere_zone(bus(5), Vec3::ZERO, 10.0, 0.5, 1.0),
        ];
        let field = ReverbZoneField::new(&zones);
        let mut out = [AuxSend { bus: bus(0), level: 0.0 }; MAX_AUX_SENDS];
        let n = field.resolve(Vec3::ZERO, &mut out);
        assert_eq!(n, MAX_AUX_SENDS);
        // Bus 1 (level 0.1) must have been dropped.
        assert!(out.iter().take(n).all(|s| s.bus != bus(1)));
        assert!(out.iter().take(n).any(|s| s.bus == bus(5)));
    }

    #[test]
    fn zero_capacity_buffer_writes_nothing() {
        let zones = [ReverbZone::sphere_zone(bus(1), Vec3::ZERO, 10.0, 0.9, 1.0)];
        let field = ReverbZoneField::new(&zones);
        let mut out: [AuxSend; 0] = [];
        assert_eq!(field.resolve(Vec3::ZERO, &mut out), 0);
    }

    #[test]
    fn source_send_gain_multiplies_and_clamps() {
        assert!((source_send_gain(0.8, 0.5) - 0.4).abs() < 1e-6);
        assert!((source_send_gain(0.0, 1.0)).abs() < 1e-6);
        assert_eq!(source_send_gain(2.0, 2.0), 1.0);
    }
}
