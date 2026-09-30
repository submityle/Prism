//! Multi-hop portal routing: pathfinding sound across chains of connected rooms.
//!
//! [`crate::rooms`] resolves a source and a listener that are in the *same*
//! room (a direct arrival) or in two rooms joined by a *single* shared portal.
//! It does not, however, route sound that must pass through an intermediate
//! room: a source in room `C`, a listener in room `A`, coupled only through
//! `C -> B -> A`. This module fills that gap by treating the scene as a graph
//! -- **rooms are nodes, portals are edges** -- and enumerating the bounded set
//! of acyclic paths from the source room to the listener room.
//!
//! Each discovered path is reduced to a compact [`RoutedPath`]: the ordered
//! portals it squeezes through, the accumulated detour distance and delay, and
//! the product of the per-portal coupling gains times the end-to-end distance
//! attenuation. The caller feeds these taps to a delay-line renderer just as it
//! would the single-hop arrivals from [`crate::rooms`].
//!
//! # The model (all classic geometry / graph search)
//!
//! * **Graph.** A node is an `Option<RoomId>` (with `None` the shared open-air
//!   "outside" volume). A portal is an undirected edge between the two rooms it
//!   touches; [`Portal::other_side`] walks it. The source and listener rooms
//!   come from [`room_of`].
//! * **Search.** A depth-bounded, acyclic depth-first search enumerates every
//!   path from the source room to the listener room whose portal count does not
//!   exceed `min(max_hops, MAX_PORTAL_HOPS)`. A path never revisits a room, so
//!   the search always terminates and cannot loop. Portals are visited in
//!   ascending index order, so the enumeration is deterministic and
//!   golden-comparable.
//! * **Geometry.** A path's waypoint chain is
//!   `source -> aperture_1 -> aperture_2 -> ... -> listener`, where each
//!   aperture is [`Portal::closest_point`] of the previous waypoint (the nearest
//!   point on that doorway rectangle). The path length is the sum of the
//!   consecutive segment lengths; the delay is that length divided by the speed
//!   of sound.
//! * **Gain.** The path gain is the product of each portal's
//!   [`portal_coupling_gain`] (openness times the Fresnel-Kirchhoff obliquity on
//!   both faces) evaluated between its incoming and outgoing waypoints, times a
//!   `1 / max(distance, MIN_DISTANCE_METRES)` spreading term. More hops and
//!   longer detours therefore attenuate monotonically.
//! * **Same room.** When source and listener share a room the result is a single
//!   zero-hop [`RoutedPath`] (`hop_count == 0`) carrying the direct distance,
//!   delay, and spreading gain; no portal detour is produced (any such detour
//!   would have to leave and re-enter the source room, which cycle avoidance
//!   forbids).
//! * **Overflow.** When more paths are found than the output slice (capped at
//!   [`MAX_ROUTED_PATHS`]) can hold, the strongest by total gain are kept and the
//!   weakest evicted (streaming top-k).
//!
//! # Control rate, not audio rate
//!
//! [`route_portals`] runs at control rate whenever the scene topology or the
//! source/listener rooms change. It is **allocation free, lock free, and panic
//! free**: the search state (visited-room path, portal-index stack) lives in
//! fixed-size stack arrays bounded by [`MAX_PORTAL_HOPS`], room and portal
//! counts are safely truncated to [`MAX_ROOMS`] / [`MAX_PORTALS`], and every
//! square root and division routes through [`bevy_math::ops`] for deterministic,
//! golden-comparable output. Delays are accumulated by integer stepping rather
//! than a float-to-int cast.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**. It is implemented purely from publicly documented acoustics (the
//! Fresnel-Kirchhoff aperture obliquity and inverse-distance spreading reused
//! from [`crate::rooms`]) and elementary bounded graph search (acyclic
//! depth-first enumeration of shortest paths), with no proprietary algorithm.

use bevy_math::{Vec3, ops};
use prism_audio_core::math::Sample;

use crate::geometry::{Emitter, Listener};
use crate::rooms::{Portal, Room, RoomId, portal_coupling_gain, room_of};

/// Maximum number of rooms the search will consider (extra rooms are ignored).
pub const MAX_ROOMS: usize = 64;

/// Maximum number of portals the search will consider (extra portals are
/// ignored).
pub const MAX_PORTALS: usize = 128;

/// Maximum number of portals a single routed path may traverse (the search
/// depth bound). Kept small so the fixed-size search state stays on the stack
/// and within the serde array limit.
pub const MAX_PORTAL_HOPS: usize = 4;

/// Maximum number of routed paths returned by a single query.
pub const MAX_ROUTED_PATHS: usize = 32;

/// Default speed of sound in dry air at room temperature (metres per second).
pub const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Largest representable delay in whole samples (matches the crate's other
/// delay-line modules); accumulated distances that exceed it are clamped.
const MAX_DELAY_SAMPLES: usize = 1 << 18;

/// Distances below this (metres) are clamped before the spreading division to
/// avoid a blow-up when waypoints coincide.
const MIN_DISTANCE_METRES: Sample = 0.1;

/// One portal traversal within a [`RoutedPath`].
///
/// `portal_index` is the index into the caller's portal slice, and `aperture`
/// is the world-space point on that doorway the wave passes through (the point
/// nearest the incoming waypoint).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PortalHop {
    /// Index of the traversed portal in the caller's portal slice.
    pub portal_index: usize,
    /// World-space aperture point the wave squeezes through on this hop.
    pub aperture: Vec3,
}

impl PortalHop {
    /// A placeholder hop used to initialise fixed-size storage.
    const EMPTY: Self = Self { portal_index: 0, aperture: Vec3::ZERO };
}

/// One routed arrival: an ordered chain of portal hops with the accumulated
/// geometry and gain.
///
/// `hop_count` is the number of valid entries in `hops` (a prefix). A
/// `hop_count` of `0` is the same-room direct arrival, in which case `hops`
/// carries no meaningful data. `total_distance` is the summed waypoint-chain
/// length (metres), `delay_samples` its integer path delay, and `total_gain`
/// the product of the per-portal couplings times the distance attenuation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RoutedPath {
    /// Number of valid entries in `hops` (`0` for the same-room direct path).
    pub hop_count: usize,
    /// The ordered portal hops (only the first `hop_count` are meaningful).
    pub hops: [PortalHop; MAX_PORTAL_HOPS],
    /// Product of the per-portal coupling gains times distance attenuation.
    pub total_gain: Sample,
    /// Summed waypoint-chain length in metres.
    pub total_distance: Sample,
    /// Integer path delay in whole samples.
    pub delay_samples: usize,
}

impl RoutedPath {
    /// A silent placeholder path (used to initialise a caller's output slice).
    pub const SILENT: Self = Self {
        hop_count: 0,
        hops: [PortalHop::EMPTY; MAX_PORTAL_HOPS],
        total_gain: 0.0,
        total_distance: 0.0,
        delay_samples: 0,
    };
}

impl Default for RoutedPath {
    #[inline]
    fn default() -> Self {
        Self::SILENT
    }
}

/// The Euclidean distance between two points, routed through [`bevy_math::ops`]
/// for determinism.
#[inline]
#[must_use]
fn distance(a: Vec3, b: Vec3) -> Sample {
    let d = b - a;
    ops::sqrt(d.dot(d))
}

/// Converts a delay in seconds to whole samples by integer stepping (no
/// float-to-int cast), rounding to nearest and clamping to [`MAX_DELAY_SAMPLES`].
#[must_use]
fn seconds_to_samples(seconds: Sample, sample_rate: Sample) -> usize {
    let cap = MAX_DELAY_SAMPLES as Sample;
    let exact = (seconds * sample_rate).max(0.0).min(cap);
    let mut n: usize = 0;
    let mut acc: Sample = 0.0;
    while acc + 1024.0 <= exact {
        acc += 1024.0;
        n += 1024;
    }
    while acc + 1.0 <= exact {
        acc += 1.0;
        n += 1;
    }
    if exact - acc >= 0.5 {
        n += 1;
    }
    n
}

/// Mutable state threaded through the recursive search.
struct Search<'a> {
    portals: &'a [Portal],
    emitter_pos: Vec3,
    listener_pos: Vec3,
    listener_room: Option<RoomId>,
    max_hops: usize,
    sample_rate: Sample,
    sound_speed: Sample,
    out: &'a mut [RoutedPath],
    cap: usize,
    count: usize,
    portal_count: usize,
    hop_portals: [usize; MAX_PORTAL_HOPS],
    visited: [Option<RoomId>; MAX_PORTAL_HOPS + 1],
    visited_len: usize,
}

impl Search<'_> {
    /// Returns `true` when `room` already lies on the current path.
    fn already_visited(&self, room: Option<RoomId>) -> bool {
        let mut i = 0;
        while i < self.visited_len {
            if self.visited[i] == room {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Recursively extends the current path from `current` at portal depth
    /// `depth`, emitting a [`RoutedPath`] whenever the listener room is reached.
    fn walk(&mut self, current: Option<RoomId>, depth: usize) {
        // Reaching the listener room after at least one hop completes a path.
        if depth > 0 && current == self.listener_room {
            self.emit(depth);
            return;
        }
        if depth >= self.max_hops {
            return;
        }
        let mut p = 0;
        while p < self.portal_count {
            if let Some(other) = self.portals[p].other_side(current)
                && !self.already_visited(other)
            {
                self.hop_portals[depth] = p;
                self.visited[self.visited_len] = other;
                self.visited_len += 1;
                self.walk(other, depth + 1);
                self.visited_len -= 1;
            }
            p += 1;
        }
    }

    /// Reduces the current portal chain of length `depth` to a [`RoutedPath`]
    /// and inserts it into the output using streaming top-k by gain.
    fn emit(&mut self, depth: usize) {
        let mut hops = [PortalHop::EMPTY; MAX_PORTAL_HOPS];
        let mut prev = self.emitter_pos;
        let mut total_distance: Sample = 0.0;
        let mut gain: Sample = 1.0;

        let mut i = 0;
        while i < depth {
            let idx = self.hop_portals[i];
            let portal = &self.portals[idx];
            let aperture = portal.closest_point(prev);
            // The outgoing waypoint is the next aperture, or the listener at the
            // end of the chain.
            let next = if i + 1 < depth {
                let next_portal = &self.portals[self.hop_portals[i + 1]];
                next_portal.closest_point(aperture)
            } else {
                self.listener_pos
            };
            total_distance += distance(prev, aperture);
            gain *= portal_coupling_gain(next, prev, portal);
            hops[i] = PortalHop { portal_index: idx, aperture };
            prev = aperture;
            i += 1;
        }
        // Final leg from the last aperture to the listener.
        total_distance += distance(prev, self.listener_pos);

        let spread = 1.0 / total_distance.max(MIN_DISTANCE_METRES);
        gain *= spread;
        let speed = self.sound_speed.max(1.0);
        let delay_samples = seconds_to_samples(total_distance / speed, self.sample_rate);

        let path = RoutedPath {
            hop_count: depth,
            hops,
            total_gain: gain,
            total_distance,
            delay_samples,
        };
        self.insert(path);
    }

    /// Inserts `path`, keeping the strongest [`cap`](Self::cap) paths by gain.
    fn insert(&mut self, path: RoutedPath) {
        if self.cap == 0 {
            return;
        }
        if self.count < self.cap {
            self.out[self.count] = path;
            self.count += 1;
            return;
        }
        // Full: replace the weakest if the newcomer is louder.
        let mut min_index = 0;
        let mut min_gain = self.out[0].total_gain;
        let mut i = 1;
        while i < self.cap {
            if self.out[i].total_gain < min_gain {
                min_gain = self.out[i].total_gain;
                min_index = i;
            }
            i += 1;
        }
        if path.total_gain > min_gain {
            self.out[min_index] = path;
        }
    }
}

/// Enumerates the bounded set of multi-hop portal paths from the source room to
/// the listener room, writing them into `out` and returning the number written.
///
/// The source room is [`room_of`] the emitter, the listener room is `room_of`
/// the listener. When both are the same room a single zero-hop direct path is
/// written. Otherwise an acyclic, depth-bounded search enumerates paths whose
/// portal count does not exceed `min(max_hops, MAX_PORTAL_HOPS)`; the strongest
/// `min(out.len(), MAX_ROUTED_PATHS)` by gain are kept.
///
/// This is a control-rate routine: it allocates nothing, locks nothing, and
/// cannot panic. Degenerate inputs (empty rooms or portals, coincident
/// source/listener, a disconnected topology, or over-limit counts) return
/// safely.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::geometry::{Emitter, Listener};
/// use prism_audio_spatial::propagation::AcousticMaterial;
/// use prism_audio_spatial::rooms::{Portal, Room, RoomId};
/// use prism_audio_spatial::portal_graph::{route_portals, RoutedPath, DEFAULT_SOUND_SPEED};
///
/// // A linear chain A(-10) - B(0) - C(10) joined by two open doors.
/// let rooms = [
///     Room::new(RoomId(0), Vec3::new(-10.0, 0.0, 0.0), Vec3::splat(5.0), AcousticMaterial::OPEN),
///     Room::new(RoomId(1), Vec3::new(0.0, 0.0, 0.0), Vec3::splat(5.0), AcousticMaterial::OPEN),
///     Room::new(RoomId(2), Vec3::new(10.0, 0.0, 0.0), Vec3::splat(5.0), AcousticMaterial::OPEN),
/// ];
/// let portals = [
///     Portal::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, Vec3::Y, 1.0, 1.0,
///         Some(RoomId(1)), Some(RoomId(0)), 1.0, AcousticMaterial::OPEN),
///     Portal::new(Vec3::new(5.0, 0.0, 0.0), Vec3::X, Vec3::Y, 1.0, 1.0,
///         Some(RoomId(2)), Some(RoomId(1)), 1.0, AcousticMaterial::OPEN),
/// ];
/// let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
/// let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
///
/// let mut out = [RoutedPath::SILENT; 8];
/// let n = route_portals(&rooms, &portals, &listener, &emitter, 4, 48_000.0, DEFAULT_SOUND_SPEED, &mut out);
/// assert_eq!(n, 1);
/// assert_eq!(out[0].hop_count, 2);
/// ```
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a routing query needs the full scene, endpoints, and audio format"
)]
pub fn route_portals(
    rooms: &[Room],
    portals: &[Portal],
    listener: &Listener,
    emitter: &Emitter,
    max_hops: usize,
    sample_rate: Sample,
    sound_speed: Sample,
    out: &mut [RoutedPath],
) -> usize {
    let cap = out.len().min(MAX_ROUTED_PATHS);
    if cap == 0 {
        return 0;
    }

    let source_room = room_of(emitter.position, rooms);
    let listener_room = room_of(listener.position, rooms);

    // Same room (including both outside): a single zero-hop direct arrival.
    if source_room == listener_room {
        let dist = distance(emitter.position, listener.position);
        let spread = 1.0 / dist.max(MIN_DISTANCE_METRES);
        let speed = sound_speed.max(1.0);
        let delay_samples = seconds_to_samples(dist / speed, sample_rate);
        out[0] = RoutedPath {
            hop_count: 0,
            hops: [PortalHop::EMPTY; MAX_PORTAL_HOPS],
            total_gain: spread,
            total_distance: dist,
            delay_samples,
        };
        return 1;
    }

    let effective_hops = max_hops.min(MAX_PORTAL_HOPS);
    if effective_hops == 0 {
        return 0;
    }
    let portal_count = portals.len().min(MAX_PORTALS);

    let mut search = Search {
        portals,
        emitter_pos: emitter.position,
        listener_pos: listener.position,
        listener_room,
        max_hops: effective_hops,
        sample_rate,
        sound_speed,
        out,
        cap,
        count: 0,
        portal_count,
        hop_portals: [0; MAX_PORTAL_HOPS],
        visited: [None; MAX_PORTAL_HOPS + 1],
        visited_len: 0,
    };
    // Seed the visited set with the source room so the search never returns to
    // it (cycle avoidance and the same-room degenerate case).
    search.visited[0] = source_room;
    search.visited_len = 1;
    search.walk(source_room, 0);
    search.count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::AcousticMaterial;

    const FS: Sample = 48_000.0;

    fn open_room(id: u32, center: Vec3) -> Room {
        Room::new(RoomId(id), center, Vec3::splat(5.0), AcousticMaterial::OPEN)
    }

    fn open_door(center: Vec3, a: u32, b: u32) -> Portal {
        Portal::new(
            center,
            Vec3::X,
            Vec3::Y,
            1.0,
            1.0,
            Some(RoomId(a)),
            Some(RoomId(b)),
            1.0,
            AcousticMaterial::OPEN,
        )
    }

    /// A(-10) - B(0) - C(10) linear chain with doors at x = -5 and x = +5.
    fn chain_scene() -> ([Room; 3], [Portal; 2]) {
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(0.0, 0.0, 0.0)),
            open_room(2, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals = [
            open_door(Vec3::new(-5.0, 0.0, 0.0), 1, 0),
            open_door(Vec3::new(5.0, 0.0, 0.0), 2, 1),
        ];
        (rooms, portals)
    }

    #[test]
    fn two_hop_chain_found_in_order() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].hop_count, 2);
        // Source is in room C, so the first portal traversed is the C<->B door
        // (index 1), then the B<->A door (index 0).
        assert_eq!(out[0].hops[0].portal_index, 1);
        assert_eq!(out[0].hops[1].portal_index, 0);
    }

    #[test]
    fn detour_longer_than_straight_line() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        let straight = distance(emitter.position, listener.position);
        // Doors are on the direct axis here, so the detour equals the straight
        // line; a moved door makes it strictly longer (see next test).
        assert!(out[0].total_distance >= straight - 1e-3);
    }

    #[test]
    fn offset_doors_bend_the_path() {
        // Move both doors off the x-axis so the chain must bend.
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(0.0, 0.0, 0.0)),
            open_room(2, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals = [
            open_door(Vec3::new(-5.0, 0.0, 3.0), 1, 0),
            open_door(Vec3::new(5.0, 0.0, 3.0), 2, 1),
        ];
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        let straight = distance(emitter.position, listener.position);
        // The offset doors force a detour strictly longer than the straight line.
        assert!(out[0].total_distance > straight + 0.1);
        assert!(out[0].delay_samples > 0);
    }

    #[test]
    fn same_room_is_zero_hop_direct() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(9.5, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(10.5, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].hop_count, 0);
        let dist = distance(emitter.position, listener.position);
        assert!((out[0].total_distance - dist).abs() < 1e-4);
    }

    #[test]
    fn disconnected_rooms_yield_no_path() {
        // Two rooms, no portal between them.
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals: [Portal; 0] = [];
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 0);
    }

    #[test]
    fn cycle_is_not_revisited() {
        // A triangle A-B-C with three portals: A<->B, B<->C, C<->A. From C to A
        // there are two acyclic routes (direct C<->A, and C<->B<->A) but no path
        // may revisit a room, so the search terminates.
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(0.0, 0.0, 10.0)),
            open_room(2, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals = [
            open_door(Vec3::new(-5.0, 0.0, 5.0), 1, 0), // A<->B
            open_door(Vec3::new(5.0, 0.0, 5.0), 2, 1),  // B<->C
            open_door(Vec3::new(0.0, 0.0, 0.0), 2, 0),  // C<->A
        ];
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        // The one-hop direct route and the two-hop detour: exactly two paths.
        assert_eq!(n, 2);
        for path in &out[..n] {
            assert!(path.hop_count >= 1 && path.hop_count <= 2);
        }
    }

    #[test]
    fn max_hops_clamp_blocks_long_route() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        // The only route needs 2 hops; allowing only 1 finds nothing.
        let n = route_portals(&rooms, &portals, &listener, &emitter, 1, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 0);
    }

    #[test]
    fn more_hops_attenuate_more() {
        // Compare a 1-hop and a 2-hop route to the same listener via a triangle.
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(0.0, 0.0, 10.0)),
            open_room(2, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals = [
            open_door(Vec3::new(-5.0, 0.0, 5.0), 1, 0),
            open_door(Vec3::new(5.0, 0.0, 5.0), 2, 1),
            open_door(Vec3::new(0.0, 0.0, 0.0), 2, 0),
        ];
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 8];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 2);
        // Find the 1-hop and 2-hop paths and confirm the shorter one is louder.
        let one = out[..n].iter().find(|p| p.hop_count == 1).copied();
        let two = out[..n].iter().find(|p| p.hop_count == 2).copied();
        let one = one.unwrap_or(RoutedPath::SILENT);
        let two = two.unwrap_or(RoutedPath::SILENT);
        assert!(one.total_gain > two.total_gain);
        assert!(one.total_distance < two.total_distance);
    }

    #[test]
    fn overflow_keeps_strongest() {
        // Triangle yields two paths; a single-slot output must keep the louder.
        let rooms = [
            open_room(0, Vec3::new(-10.0, 0.0, 0.0)),
            open_room(1, Vec3::new(0.0, 0.0, 10.0)),
            open_room(2, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let portals = [
            open_door(Vec3::new(-5.0, 0.0, 5.0), 1, 0),
            open_door(Vec3::new(5.0, 0.0, 5.0), 2, 1),
            open_door(Vec3::new(0.0, 0.0, 0.0), 2, 0),
        ];
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut full = [RoutedPath::SILENT; 8];
        let nf = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut full);
        assert_eq!(nf, 2);
        let strongest = full[..nf]
            .iter()
            .map(|p| p.total_gain)
            .fold(0.0_f32, f32::max);
        let mut one = [RoutedPath::SILENT; 1];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut one);
        assert_eq!(n, 1);
        assert!((one[0].total_gain - strongest).abs() < 1e-6);
    }

    #[test]
    fn deterministic_portal_order() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut a = [RoutedPath::SILENT; 8];
        let mut b = [RoutedPath::SILENT; 8];
        let na = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut a);
        let nb = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut b);
        assert_eq!(na, nb);
        assert_eq!(a[..na], b[..nb]);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let empty_rooms: [Room; 0] = [];
        let empty_portals: [Portal; 0] = [];
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::ZERO, Vec3::ZERO);
        // Empty scene: both endpoints are outside -> one zero-hop direct path.
        let mut out = [RoutedPath::SILENT; 4];
        let n = route_portals(&empty_rooms, &empty_portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].hop_count, 0);

        // Zero-length output slice.
        let mut none: [RoutedPath; 0] = [];
        let n0 = route_portals(&empty_rooms, &empty_portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut none);
        assert_eq!(n0, 0);

        // Zero sample rate and zero sound speed must not divide by zero/panic.
        let (rooms, portals) = chain_scene();
        let l2 = Listener { position: Vec3::new(-9.0, 0.0, 0.0), ..Listener::default() };
        let e2 = Emitter::point(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO);
        let mut out2 = [RoutedPath::SILENT; 8];
        let n2 = route_portals(&rooms, &portals, &l2, &e2, 4, 0.0, 0.0, &mut out2);
        assert_eq!(n2, 1);
    }

    #[test]
    fn coincident_source_listener_is_finite() {
        let (rooms, portals) = chain_scene();
        let listener = Listener { position: Vec3::new(10.0, 0.0, 0.0), ..Listener::default() };
        let emitter = Emitter::point(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO);
        let mut out = [RoutedPath::SILENT; 4];
        let n = route_portals(&rooms, &portals, &listener, &emitter, 4, FS, DEFAULT_SOUND_SPEED, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].hop_count, 0);
        assert!(out[0].total_gain.is_finite());
        assert!(out[0].total_gain > 0.0);
    }
}
