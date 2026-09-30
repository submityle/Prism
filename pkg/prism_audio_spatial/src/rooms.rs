//! Rooms and portals: routing sound between acoustically distinct volumes.
//!
//! Games are built from rooms joined by doors, windows, and corridors. A sound
//! made in one room reaches a listener in another by two mechanisms:
//!
//! * **Through the walls** -- attenuated by each partition's transmission loss
//!   (the muffled thump of music through a closed door), and
//! * **Through the openings** -- an open door acts as a secondary aperture: the
//!   sound appears to radiate *from the doorway*, arrives later (it took the
//!   detour), and its loudness depends on how far the door is open and how
//!   squarely it faces the source and the listener.
//!
//! This module models both. It defines the geometry ([`Room`] volumes and the
//! [`Portal`] apertures that connect them), the membership test that places a
//! point in a room ([`room_of`]), and a [`RoomNetwork`] that implements the
//! crate's [`PropagationBackend`] so the whole rooms-and-portals result flows
//! through the same propagation -> occlusion -> spatialiser pipeline as every
//! other arrival.
//!
//! # Layering
//!
//! A [`RoomNetwork`] is a [`PropagationBackend`]: given a [`Listener`] and an
//! [`Emitter`] it emits a bounded set of [`PropagationPath`]s (a direct
//! through-wall transmission plus one arrival per connecting portal) and a
//! [`PropagationSummary`] whose `direct` factors feed [`crate::occlusion`].
//! It reuses [`AcousticMaterial`] for both wall transmission and the closed
//! leaf of a door, and the listener-local direction convention from
//! [`crate::geometry`], so nothing about coordinates or materials is redefined
//! here.
//!
//! # The model (all classic geometry / acoustics)
//!
//! * **Membership.** A [`Room`] is an axis-aligned box. [`room_of`] returns the
//!   *innermost* (smallest) room containing a point, so a closet inside a hall
//!   wins over the hall. A point in no room is "outside" (`None`), a single
//!   shared open-air volume.
//! * **Through-wall transmission.** When listener and source are in different
//!   rooms the direct line of sight crosses the partitions between them; its
//!   gain is the product of the crossed walls' [`transmission_gain`]s.
//! * **Portal coupling.** An open door radiates most strongly along its normal
//!   and less at a glancing angle. This is the standard **Fresnel-Kirchhoff
//!   obliquity factor** `(1 + cos theta) / 2` applied on both the source and
//!   listener sides of the aperture (see [`obliquity_factor`]), multiplied by
//!   how far the door is open ([`Portal::transmission_gain`], which blends the
//!   closed leaf's material transmission up to a fully-open unity). The
//!   aperture point the wave squeezes through is the point on the doorway
//!   rectangle nearest the source ([`Portal::closest_point`]), which also sets
//!   the direction the listener perceives and the (longer) detour delay.
//!
//! Spectral colouring of the muffled through-wall sound is deliberately left to
//! the transmission model (broadband gain, as in [`crate::propagation`]); this
//! module does not fabricate a wall low-pass corner.
//!
//! # Control rate, not audio rate
//!
//! Like every [`PropagationBackend`], [`RoomNetwork::query`] runs at control
//! rate. Its pure geometry helpers ([`room_of`], [`Room::contains`],
//! [`Portal::closest_point`], [`portal_coupling_gain`], [`obliquity_factor`])
//! allocate nothing, lock nothing, and cannot panic; every square root and
//! division routes through [`bevy_math::ops`] so the result is deterministic
//! and golden-comparable.
//!
//! # Provenance
//!
//! The obliquity factor is the textbook Fresnel-Kirchhoff diffraction obliquity
//! term; transmission loss is the standard partition definition; the
//! innermost-volume membership and nearest-point-on-rectangle are elementary
//! geometry. This module is engine-agnostic and contains **no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, or Steam Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics and geometry
//! knowledge.

use bevy_math::{Vec3, ops};
use prism_audio_core::math::{MIN_AUDIBLE_GAIN, Sample};

use crate::doppler::SPEED_OF_SOUND_MPS;
use crate::geometry::{Emitter, Listener};
use crate::occlusion::OcclusionFactors;
use crate::propagation::{
    AcousticMaterial, FULL_BAND_CUTOFF_HZ, MAX_PROPAGATION_PATHS, PathKind, PropagationBackend,
    PropagationPath, PropagationSummary,
};

/// Squared-length threshold below which a vector is treated as degenerate and
/// replaced by a fallback direction.
const DEGENERATE_EPSILON: Sample = 1.0e-12;

/// A stable handle identifying a [`Room`] within a [`RoomNetwork`].
///
/// Plain description data; the caller assigns ids from its own scene model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RoomId(pub u32);

/// An axis-aligned acoustic volume.
///
/// A room is a box (`center` +/- `half_extents`) tagged with the transmission
/// behaviour of its enclosing walls. Membership is tested with
/// [`Self::contains`]; when boxes nest, [`room_of`] resolves ties by the
/// smaller [`Self::volume`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Room {
    /// Stable identity of this room.
    pub id: RoomId,
    /// World-space centre of the box, in metres.
    pub center: Vec3,
    /// Non-negative half-extents of the box along each world axis, in metres.
    pub half_extents: Vec3,
    /// Acoustic behaviour of the walls enclosing this room (transmission loss
    /// governs how much of a sound leaks straight through to the outside).
    pub wall: AcousticMaterial,
}

impl Room {
    /// Builds a room from its identity, box, and wall material.
    #[inline]
    #[must_use]
    pub const fn new(id: RoomId, center: Vec3, half_extents: Vec3, wall: AcousticMaterial) -> Self {
        Self {
            id,
            center,
            half_extents,
            wall,
        }
    }

    /// Returns `true` when `point` lies inside (or on the surface of) the box.
    #[inline]
    #[must_use]
    pub fn contains(&self, point: Vec3) -> bool {
        let d = point - self.center;
        d.x.abs() <= self.half_extents.x
            && d.y.abs() <= self.half_extents.y
            && d.z.abs() <= self.half_extents.z
    }

    /// The enclosed volume, in cubic metres. Used by [`room_of`] to prefer the
    /// innermost of several containing rooms.
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Sample {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }
}

/// A rectangular opening connecting two rooms (a door, window, or archway).
///
/// The aperture is centred at `center`, faces along `normal`, and is spanned by
/// tangents derived from `up`; `half_width`/`half_height` are its half-extents
/// in those tangent directions. `openness` in `[0, 1]` is how far it is open
/// (0 = fully shut, sound only leaks through the leaf's `material`; 1 = wide
/// open). `front` and `back` name the rooms on either side (`None` == outside).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Portal {
    /// World-space centre of the aperture, in metres.
    pub center: Vec3,
    /// Aperture facing direction (need not be normalised; re-normalised on use).
    pub normal: Vec3,
    /// Reference "up" tangent; orthogonalised against `normal` on use.
    pub up: Vec3,
    /// Half-width of the aperture along its right tangent, in metres.
    pub half_width: Sample,
    /// Half-height of the aperture along its up tangent, in metres.
    pub half_height: Sample,
    /// Room on the `+normal` side (`None` == outside).
    pub front: Option<RoomId>,
    /// Room on the `-normal` side (`None` == outside).
    pub back: Option<RoomId>,
    /// How far the portal is open, in `[0, 1]`.
    pub openness: Sample,
    /// Acoustic behaviour of the closed leaf (governs leak when shut).
    pub material: AcousticMaterial,
}

impl Portal {
    /// Builds a portal. `openness` is clamped to `[0, 1]`.
    #[inline]
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a portal is a flat aperture description; bundling its fields into a sub-struct would not reduce the caller burden"
    )]
    pub fn new(
        center: Vec3,
        normal: Vec3,
        up: Vec3,
        half_width: Sample,
        half_height: Sample,
        front: Option<RoomId>,
        back: Option<RoomId>,
        openness: Sample,
        material: AcousticMaterial,
    ) -> Self {
        Self {
            center,
            normal,
            up,
            half_width: half_width.max(0.0),
            half_height: half_height.max(0.0),
            front,
            back,
            openness: openness.clamp(0.0, 1.0),
            material,
        }
    }

    /// The area of the aperture, in square metres.
    #[inline]
    #[must_use]
    pub fn area(&self) -> Sample {
        4.0 * self.half_width * self.half_height
    }

    /// Returns `true` when this portal joins rooms `a` and `b` (in either
    /// order). Sides are compared including `None` (outside).
    #[inline]
    #[must_use]
    pub fn connects(&self, a: Option<RoomId>, b: Option<RoomId>) -> bool {
        (self.front == a && self.back == b) || (self.front == b && self.back == a)
    }

    /// The room on the opposite side of `room`, or `None` when this portal does
    /// not touch `room`.
    #[inline]
    #[must_use]
    pub fn other_side(&self, room: Option<RoomId>) -> Option<Option<RoomId>> {
        if self.front == room {
            Some(self.back)
        } else if self.back == room {
            Some(self.front)
        } else {
            None
        }
    }

    /// The linear gain of sound coupling through the aperture *before* the
    /// directional obliquity term: the closed leaf's transmission blended up to
    /// unity as the door opens.
    ///
    /// `openness == 0` returns the leaf material's [`transmission_gain`], and
    /// `openness == 1` returns `1.0`.
    ///
    /// [`transmission_gain`]: AcousticMaterial::transmission_gain
    #[inline]
    #[must_use]
    pub fn transmission_gain(&self) -> Sample {
        let closed = self.material.transmission_gain();
        closed + (1.0 - closed) * self.openness.clamp(0.0, 1.0)
    }

    /// The orthonormal `(right, up)` tangent basis of the aperture plane.
    ///
    /// `up` is orthogonalised against the (re-normalised) `normal`; if the two
    /// are degenerate an arbitrary but stable perpendicular is chosen so the
    /// basis is always well defined.
    #[must_use]
    fn basis(&self) -> (Vec3, Vec3) {
        let n = normalize_or(self.normal, Vec3::NEG_Z);
        let up_raw = self.up - n * self.up.dot(n);
        let up = normalize_or(up_raw, any_perpendicular(n));
        let right = n.cross(up);
        (right, up)
    }

    /// The point on the aperture rectangle nearest `point`.
    ///
    /// `point` is projected onto the aperture plane and clamped to the
    /// rectangle, so a source off to one side of a wide doorway couples through
    /// the near edge rather than the geometric centre.
    #[must_use]
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        let (right, up) = self.basis();
        let d = point - self.center;
        let u = d.dot(right).clamp(-self.half_width, self.half_width);
        let v = d.dot(up).clamp(-self.half_height, self.half_height);
        self.center + right * u + up * v
    }
}

/// The **Fresnel-Kirchhoff obliquity factor** `(1 + cos theta) / 2` for a wave
/// meeting an aperture at angle `theta` from its normal, where `cos_theta` is
/// the cosine of that angle.
///
/// The magnitude of `cos_theta` is used (the aperture radiates on both faces),
/// clamped to `[0, 1]`, so the result ranges from `1` at normal incidence down
/// to `0.5` at grazing incidence.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::rooms::obliquity_factor;
/// assert!((obliquity_factor(1.0) - 1.0).abs() < 1e-6);
/// assert!((obliquity_factor(0.0) - 0.5).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn obliquity_factor(cos_theta: Sample) -> Sample {
    let c = cos_theta.abs().min(1.0);
    0.5 * (1.0 + c)
}

/// The linear gain of a source coupling through `portal` to a listener,
/// combining the door's openness ([`Portal::transmission_gain`]) with the
/// [`obliquity_factor`] on the source and listener sides of the aperture.
///
/// The aperture point used is [`Portal::closest_point`] of the emitter, so an
/// off-axis source couples through the nearest part of the opening. The result
/// is clamped to `[0, 1]`.
#[must_use]
pub fn portal_coupling_gain(listener_pos: Vec3, emitter_pos: Vec3, portal: &Portal) -> Sample {
    let normal = normalize_or(portal.normal, Vec3::NEG_Z);
    let aperture = portal.closest_point(emitter_pos);

    let to_aperture = normalize_or(aperture - emitter_pos, normal);
    let to_listener = normalize_or(listener_pos - aperture, normal);

    let obliquity = obliquity_factor(to_aperture.dot(normal)) * obliquity_factor(to_listener.dot(normal));
    (portal.transmission_gain() * obliquity).clamp(0.0, 1.0)
}

/// Returns the innermost (smallest-volume) room containing `point`, or `None`
/// when `point` lies in no room (outside).
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::propagation::AcousticMaterial;
/// use prism_audio_spatial::rooms::{Room, RoomId, room_of};
///
/// let hall = Room::new(RoomId(0), Vec3::ZERO, Vec3::splat(10.0), AcousticMaterial::OPEN);
/// let closet = Room::new(RoomId(1), Vec3::ZERO, Vec3::splat(1.0), AcousticMaterial::OPEN);
/// let rooms = [hall, closet];
/// // The origin is inside both; the smaller closet wins.
/// assert_eq!(room_of(Vec3::ZERO, &rooms), Some(RoomId(1)));
/// // Far outside both boxes.
/// assert_eq!(room_of(Vec3::splat(100.0), &rooms), None);
/// ```
#[must_use]
pub fn room_of(point: Vec3, rooms: &[Room]) -> Option<RoomId> {
    let mut best: Option<(RoomId, Sample)> = None;
    for room in rooms {
        if room.contains(point) {
            let v = room.volume();
            let replace = match best {
                Some((_, best_v)) => v < best_v,
                None => true,
            };
            if replace {
                best = Some((room.id, v));
            }
        }
    }
    best.map(|(id, _)| id)
}

/// A scene of [`Room`]s joined by [`Portal`]s, exposed as a
/// [`PropagationBackend`].
///
/// [`Self::query`] resolves how a source in one room reaches a listener in
/// another: a direct through-wall transmission arrival plus one arrival per
/// connecting portal, bounded by [`MAX_PROPAGATION_PATHS`] and the caller's
/// buffer. When both endpoints share a room (or both are outside) it degrades
/// to a single open direct path, matching [`crate::propagation::FreeFieldBackend`].
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::geometry::{Emitter, Listener};
/// use prism_audio_spatial::propagation::{
///     AcousticMaterial, PropagationBackend, PropagationPath,
/// };
/// use prism_audio_spatial::rooms::{Portal, Room, RoomId, RoomNetwork};
///
/// // Two rooms side by side along +X, joined by an open door on the shared wall.
/// let brick = AcousticMaterial::new(40.0, 0.2);
/// let rooms = [
///     Room::new(RoomId(0), Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
///     Room::new(RoomId(1), Vec3::new(5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
/// ];
/// let door = Portal::new(
///     Vec3::ZERO, Vec3::X, Vec3::Y, 0.5, 1.0,
///     Some(RoomId(1)), Some(RoomId(0)), 1.0, AcousticMaterial::OPEN,
/// );
/// let portals = [door];
/// let net = RoomNetwork::new(&rooms, &portals);
///
/// let listener = Listener::default(); // at the origin-ish room 0 boundary
/// let listener = Listener { position: Vec3::new(-4.0, 0.0, 0.0), ..listener };
/// let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
///
/// let mut paths = [PropagationPath::SILENT; 4];
/// let summary = net.query(&listener, &emitter, &mut paths);
/// // A through-wall arrival plus the open-door arrival.
/// assert_eq!(summary.path_count, 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RoomNetwork<'a> {
    /// The rooms of the scene.
    pub rooms: &'a [Room],
    /// The portals joining the rooms.
    pub portals: &'a [Portal],
}

impl<'a> RoomNetwork<'a> {
    /// Builds a network over borrowed room and portal slices.
    #[inline]
    #[must_use]
    pub fn new(rooms: &'a [Room], portals: &'a [Portal]) -> Self {
        Self { rooms, portals }
    }

    /// The innermost room containing `point` (see [`room_of`]).
    #[inline]
    #[must_use]
    pub fn room_of(&self, point: Vec3) -> Option<RoomId> {
        room_of(point, self.rooms)
    }

    /// The transmission gain of the wall enclosing `room` (`1.0` for outside or
    /// an unknown id).
    #[must_use]
    fn room_wall_gain(&self, room: Option<RoomId>) -> Sample {
        match room {
            Some(id) => self
                .rooms
                .iter()
                .find(|r| r.id == id)
                .map(|r| r.wall.transmission_gain())
                .unwrap_or(1.0),
            None => 1.0,
        }
    }
}

impl PropagationBackend for RoomNetwork<'_> {
    fn query(
        &self,
        listener: &Listener,
        emitter: &Emitter,
        paths: &mut [PropagationPath],
    ) -> PropagationSummary {
        if paths.is_empty() {
            return PropagationSummary {
                direct: OcclusionFactors::OPEN,
                path_count: 0,
            };
        }

        let local = listener.localize(emitter);
        let listener_room = self.room_of(listener.position);
        let emitter_room = self.room_of(emitter.position);

        // Same acoustic volume (including both outside): one open direct path.
        if listener_room == emitter_room {
            paths[0] = PropagationPath {
                kind: PathKind::Direct,
                delay_seconds: local.distance / SPEED_OF_SOUND_MPS,
                gain: 1.0,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                direction: local.direction,
            };
            return PropagationSummary {
                direct: OcclusionFactors::OPEN,
                path_count: 1,
            };
        }

        // Different rooms: the direct line of sight crosses the enclosing walls
        // of both endpoints.
        let wall_gain =
            self.room_wall_gain(listener_room) * self.room_wall_gain(emitter_room);
        let mut count = 0usize;
        paths[count] = PropagationPath {
            kind: PathKind::Transmission,
            delay_seconds: local.distance / SPEED_OF_SOUND_MPS,
            gain: wall_gain.clamp(0.0, 1.0),
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            direction: local.direction,
        };
        count += 1;

        // One arrival per portal joining the two rooms.
        let max_paths = paths.len().min(MAX_PROPAGATION_PATHS);
        for portal in self.portals {
            if count >= max_paths {
                break;
            }
            if !portal.connects(listener_room, emitter_room) {
                continue;
            }
            let gain = portal_coupling_gain(listener.position, emitter.position, portal);
            if gain <= MIN_AUDIBLE_GAIN {
                continue;
            }
            let aperture = portal.closest_point(emitter.position);
            let leg_in = distance(emitter.position, aperture);
            let leg_out = distance(aperture, listener.position);
            let direction = listener
                .localize(&Emitter::point(aperture, Vec3::ZERO))
                .direction;
            paths[count] = PropagationPath {
                kind: PathKind::Transmission,
                delay_seconds: (leg_in + leg_out) / SPEED_OF_SOUND_MPS,
                gain,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                direction,
            };
            count += 1;
        }

        // The direct line of sight is blocked by the walls; report the residual
        // blocking so the occlusion model can duck the direct and wet paths.
        let block = (1.0 - wall_gain).clamp(0.0, 1.0);
        PropagationSummary {
            direct: OcclusionFactors::new(block, block),
            path_count: count,
        }
    }
}

/// Euclidean distance between two points, via [`bevy_math::ops`].
#[inline]
#[must_use]
fn distance(a: Vec3, b: Vec3) -> Sample {
    let d = a - b;
    ops::sqrt(d.dot(d))
}

/// Normalises `v`, or returns `fallback` when `v` is degenerate.
#[inline]
#[must_use]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.dot(v);
    if len_sq <= DEGENERATE_EPSILON {
        fallback
    } else {
        v / ops::sqrt(len_sq)
    }
}

/// An arbitrary unit vector perpendicular to the (assumed unit) `n`.
#[inline]
#[must_use]
fn any_perpendicular(n: Vec3) -> Vec3 {
    // Cross with whichever axis is least aligned with `n` to avoid degeneracy.
    let axis = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::X
    } else if n.y.abs() <= n.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    normalize_or(n.cross(axis), Vec3::Y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Quat;

    const EPS: Sample = 1.0e-5;

    fn room(id: u32, center: Vec3, he: Vec3) -> Room {
        Room::new(RoomId(id), center, he, AcousticMaterial::new(40.0, 0.2))
    }

    #[test]
    fn contains_inside_boundary_outside() {
        let r = room(0, Vec3::ZERO, Vec3::splat(2.0));
        assert!(r.contains(Vec3::ZERO));
        assert!(r.contains(Vec3::new(2.0, -2.0, 2.0))); // on the surface
        assert!(!r.contains(Vec3::new(2.001, 0.0, 0.0)));
    }

    #[test]
    fn room_of_prefers_innermost() {
        let hall = room(0, Vec3::ZERO, Vec3::splat(10.0));
        let closet = room(1, Vec3::ZERO, Vec3::splat(1.0));
        let rooms = [hall, closet];
        assert_eq!(room_of(Vec3::ZERO, &rooms), Some(RoomId(1)));
        // Inside the hall but outside the closet.
        assert_eq!(room_of(Vec3::new(5.0, 0.0, 0.0), &rooms), Some(RoomId(0)));
    }

    #[test]
    fn room_of_outside_is_none() {
        let rooms = [room(0, Vec3::ZERO, Vec3::splat(1.0))];
        assert_eq!(room_of(Vec3::splat(50.0), &rooms), None);
    }

    #[test]
    fn portal_connects_symmetric() {
        let p = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            1.0,
            AcousticMaterial::OPEN,
        );
        assert!(p.connects(Some(RoomId(0)), Some(RoomId(1))));
        assert!(p.connects(Some(RoomId(1)), Some(RoomId(0))));
        assert!(!p.connects(Some(RoomId(0)), Some(RoomId(2))));
        assert_eq!(p.other_side(Some(RoomId(0))), Some(Some(RoomId(1))));
        assert_eq!(p.other_side(Some(RoomId(2))), None);
    }

    #[test]
    fn portal_transmission_gain_blends_openness() {
        let closed_mat = AcousticMaterial::new(20.0, 0.0); // ~0.1 linear
        let mut p = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            None,
            None,
            0.0,
            closed_mat,
        );
        assert!((p.transmission_gain() - closed_mat.transmission_gain()).abs() < EPS);
        p.openness = 1.0;
        assert!((p.transmission_gain() - 1.0).abs() < EPS);
        p.openness = 0.5;
        let expected = 0.5 * (closed_mat.transmission_gain() + 1.0);
        assert!((p.transmission_gain() - expected).abs() < EPS);
    }

    #[test]
    fn portal_new_clamps_openness_and_extents() {
        let p = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            -1.0,
            2.0,
            None,
            None,
            5.0,
            AcousticMaterial::OPEN,
        );
        assert_eq!(p.openness, 1.0);
        assert_eq!(p.half_width, 0.0);
        assert!((p.area() - 0.0).abs() < EPS);
    }

    #[test]
    fn closest_point_clamps_to_rectangle() {
        // Aperture in the x=0 plane, facing +X, 1 wide (y) and 2 tall (z-ish).
        let p = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Z,
            0.5,
            1.0,
            None,
            None,
            1.0,
            AcousticMaterial::OPEN,
        );
        // Point straight in front projects to the centre.
        let c = p.closest_point(Vec3::new(3.0, 0.0, 0.0));
        assert!((c - Vec3::ZERO).length() < EPS);
        // Point far to the side clamps to the near edge.
        let c = p.closest_point(Vec3::new(3.0, 10.0, 0.0));
        // right tangent = normal x up = X x Z = -Y, so +Y offset clamps to -0.5 along right.
        assert!(c.y.abs() <= 0.5 + EPS);
        assert!(c.z.abs() <= 1.0 + EPS);
    }

    #[test]
    fn obliquity_bounds() {
        assert!((obliquity_factor(1.0) - 1.0).abs() < EPS);
        assert!((obliquity_factor(-1.0) - 1.0).abs() < EPS);
        assert!((obliquity_factor(0.0) - 0.5).abs() < EPS);
        assert!((obliquity_factor(2.0) - 1.0).abs() < EPS); // clamped
    }

    #[test]
    fn coupling_stronger_on_axis_than_oblique() {
        let door = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            1.0,
            AcousticMaterial::OPEN,
        );
        let on_axis = portal_coupling_gain(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            &door,
        );
        let oblique = portal_coupling_gain(
            Vec3::new(-5.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            &door,
        );
        assert!(on_axis > oblique);
        assert!(on_axis <= 1.0 && oblique >= 0.0);
    }

    #[test]
    fn coupling_closed_matches_material_on_axis() {
        let mat = AcousticMaterial::new(20.0, 0.0);
        let door = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            0.0,
            mat,
        );
        // On axis both obliquity factors are ~1, so coupling == material gain.
        let g = portal_coupling_gain(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            &door,
        );
        assert!((g - mat.transmission_gain()).abs() < 1e-3);
    }

    fn two_room_scene() -> ([Room; 2], [Portal; 1]) {
        let brick = AcousticMaterial::new(40.0, 0.2);
        let rooms = [
            Room::new(RoomId(0), Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
            Room::new(RoomId(1), Vec3::new(5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
        ];
        let door = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            1.0,
            AcousticMaterial::OPEN,
        );
        (rooms, [door])
    }

    #[test]
    fn same_room_single_direct_path() {
        let (rooms, portals) = two_room_scene();
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-6.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(-4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 1);
        assert_eq!(paths[0].kind, PathKind::Direct);
        assert!((paths[0].gain - 1.0).abs() < EPS);
        assert_eq!(s.direct, OcclusionFactors::OPEN);
    }

    #[test]
    fn both_outside_single_direct_path() {
        let (rooms, portals) = two_room_scene();
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-50.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 1);
        assert_eq!(paths[0].kind, PathKind::Direct);
    }

    #[test]
    fn different_rooms_wall_plus_portal() {
        let (rooms, portals) = two_room_scene();
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-4.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 2);
        assert_eq!(paths[0].kind, PathKind::Transmission); // through wall
        assert_eq!(paths[1].kind, PathKind::Transmission); // through door
        // The open door is far louder than the muffled through-wall path.
        assert!(paths[1].gain > paths[0].gain);
        // Direct line of sight is blocked -> non-open factors.
        assert!(s.direct.direct_factor() > 0.0);
    }

    #[test]
    fn portal_path_detours_and_bends() {
        // Door offset in +Z so the through-door arrival is longer and bent.
        let brick = AcousticMaterial::new(40.0, 0.2);
        let rooms = [
            Room::new(RoomId(0), Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
            Room::new(RoomId(1), Vec3::new(5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
        ];
        let door = Portal::new(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            1.0,
            AcousticMaterial::OPEN,
        );
        let portals = [door];
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-4.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 2);
        // Detour via the offset door is longer than the straight wall path.
        assert!(paths[1].delay_seconds > paths[0].delay_seconds);
        // Perceived direction is toward the door (+Z component), not straight (-X).
        assert!(paths[1].direction.z.abs() > EPS);
    }

    #[test]
    fn different_rooms_no_portal_only_wall() {
        let brick = AcousticMaterial::new(40.0, 0.2);
        let rooms = [
            Room::new(RoomId(0), Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
            Room::new(RoomId(1), Vec3::new(5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
        ];
        let portals: [Portal; 0] = [];
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-4.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 1);
        assert_eq!(paths[0].kind, PathKind::Transmission);
        assert!(paths[0].gain < 1.0);
    }

    #[test]
    fn empty_buffer_reports_no_paths() {
        let (rooms, portals) = two_room_scene();
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths: [PropagationPath; 0] = [];
        let s = net.query(&listener, &emitter, &mut paths);
        assert_eq!(s.path_count, 0);
    }

    #[test]
    fn path_count_is_bounded_by_buffer() {
        // Many connecting portals, tiny buffer: never overrun.
        let brick = AcousticMaterial::new(40.0, 0.2);
        let rooms = [
            Room::new(RoomId(0), Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
            Room::new(RoomId(1), Vec3::new(5.0, 0.0, 0.0), Vec3::splat(5.0), brick),
        ];
        let door = Portal::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0.5,
            1.0,
            Some(RoomId(1)),
            Some(RoomId(0)),
            1.0,
            AcousticMaterial::OPEN,
        );
        let portals = [door, door, door, door, door];
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-4.0, 0.0, 0.0),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 3];
        let s = net.query(&listener, &emitter, &mut paths);
        assert!(s.path_count <= 3);
    }

    #[test]
    fn rotated_listener_keeps_directions_unit() {
        let (rooms, portals) = two_room_scene();
        let net = RoomNetwork::new(&rooms, &portals);
        let listener = Listener {
            position: Vec3::new(-4.0, 0.0, 0.0),
            orientation: Quat::from_rotation_y(1.2),
            ..Listener::default()
        };
        let emitter = Emitter::point(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; 4];
        let s = net.query(&listener, &emitter, &mut paths);
        for path in &paths[..s.path_count] {
            assert!((path.direction.length() - 1.0).abs() < 1e-4);
            assert!(path.gain >= 0.0 && path.gain <= 1.0);
            assert!(path.delay_seconds >= 0.0);
        }
    }
}
