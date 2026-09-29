//! Simulation-space and large-world relocation contracts for the Ember
//! particle engine (design §26).
//!
//! An emitter declares the *simulation space* its particles live in (the
//! [`SimSpace`] enum owned by the parent module): `Local` particles follow the
//! owning transform (a hand-held torch), `World` particles detach into world
//! space the instant they spawn (a trail left behind), and `Hybrid` particles
//! spawn in local space but then update in world space. This module spells out,
//! on the `CPU` reference side, exactly which coordinate frame *spawn* and
//! *update* run in for each space and how a particle's position and velocity
//! are carried between the local and world frames.
//!
//! It also models **origin rebasing** ("floating origin"): an open world that
//! stretches for kilometres cannot be simulated directly in absolute world
//! coordinates, because `fp32` loses mantissa bits as magnitude grows and
//! particle motion visibly quantizes far from the origin. The engine instead
//! simulates relative to a periodically re-centred origin (snapped to the
//! chunk grid) so the numbers fed to the integrator stay small. This file
//! provides the rebase trigger, the offset computation, the position-fix-up
//! contract, the invariance of velocity under a rebase, and an order-of-
//! magnitude `fp32` `ULP` estimator that quantifies the precision loss the
//! rebase exists to avoid.
//!
//! Finally it models the **two-layer chunk coordinate** (an integer chunk
//! index plus an intra-chunk `fp32` offset) used to address positions in the
//! large world, and how a rebase interacts correctly with all three simulation
//! spaces at once.
//!
//! All math reuses the hand-rolled [`Vec3`] (add / sub / scale / mul / dot);
//! rotations are expressed through a caller-supplied orthonormal basis rather
//! than any trigonometric call, and the only "special" float operation reached
//! for is the `sqrt` already inside `Vec3`. No transcendental function is used,
//! keeping this reference bit-reproducible against a future `GPU` kernel.

use super::{SimSpace, Vec3, EPS_LEN_SQ};

/// Where a simulation space physically *stores* its particle positions.
///
/// This is the coordinate frame the pooled position attribute is expressed in
/// once a particle is live, and therefore the frame the integrator reads and
/// writes each update step.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StorageSpace {
    /// Positions are stored relative to the owning transform; the world-space
    /// position is reconstructed by applying the transform at render time.
    Local,
    /// Positions are stored directly in world space and are independent of the
    /// owning transform after spawn.
    World,
}

/// The resolved spawn/update coordinate contract for one [`SimSpace`].
///
/// Returned by [`plan`]. It answers the two questions the emission and
/// simulation stages need: does a freshly spawned particle's local position
/// have to be baked into world space at spawn time, and which frame does every
/// subsequent update step run in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimSpacePlan {
    /// `true` when spawn positions/velocities (authored in the emitter's local
    /// frame) must be transformed into world space *once* at spawn, because
    /// the particle is thereafter stored and updated in world space.
    pub spawn_bake_to_world: bool,
    /// The frame the pooled position attribute is stored and updated in.
    pub storage: StorageSpace,
}

/// Resolves the spawn/update coordinate contract for a simulation space.
///
/// - `Local`: spawn *and* update in the local frame; nothing is baked to world
///   because the transform is applied at render time (a hand-held torch).
/// - `World`: the local spawn sample is baked to world once, then every update
///   runs in world space so the particle stays put as its emitter moves away.
/// - `Hybrid`: identical storage/update to `World` (baked at spawn, updated in
///   world), but the spawn *distribution* is authored in the local frame so it
///   inherits the emitter's placement and orientation before detaching.
#[must_use]
pub fn plan(space: SimSpace) -> SimSpacePlan {
    match space {
        SimSpace::Local => SimSpacePlan {
            spawn_bake_to_world: false,
            storage: StorageSpace::Local,
        },
        SimSpace::World | SimSpace::Hybrid => SimSpacePlan {
            spawn_bake_to_world: true,
            storage: StorageSpace::World,
        },
    }
}

/// The storage frame a simulation space uses (a convenience over [`plan`]).
#[must_use]
pub fn storage_space(space: SimSpace) -> StorageSpace {
    plan(space).storage
}

/// A rigid transform: a translation plus an orthonormal rotation basis.
///
/// The rotation is carried as the three world-space images of the local unit
/// axes (`right` maps local `+x`, `up` maps local `+y`, `forward` maps local
/// `+z`). Callers construct these basis vectors from whatever orientation
/// source they own (a quaternion, a matrix, a pre-baked lookup); this contract
/// layer never derives them from angles, so no trigonometric function is
/// reached for here. When the basis is orthonormal the inverse is the
/// transpose, which is why [`TransformFrame::inverse_transform_point`] can undo
/// [`TransformFrame::transform_point`] with three dot products and no division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformFrame {
    /// World-space position of the local origin.
    pub origin: Vec3,
    /// World-space image of the local `+x` axis.
    pub right: Vec3,
    /// World-space image of the local `+y` axis.
    pub up: Vec3,
    /// World-space image of the local `+z` axis.
    pub forward: Vec3,
}

impl TransformFrame {
    /// The identity frame: origin at zero, axes unchanged.
    pub const IDENTITY: Self = Self {
        origin: Vec3::ZERO,
        right: Vec3::new(1.0, 0.0, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
        forward: Vec3::new(0.0, 0.0, 1.0),
    };

    /// Transforms a *point* from local space to world space.
    ///
    /// `world = origin + right·x + up·y + forward·z`, expressed purely with
    /// `Vec3` scale/add so it stays a plain linear combination (no matrix type,
    /// no transcendental call).
    #[must_use]
    pub fn transform_point(self, local: Vec3) -> Vec3 {
        self.origin
            .add(self.right.scale(local.x))
            .add(self.up.scale(local.y))
            .add(self.forward.scale(local.z))
    }

    /// Transforms a *direction* (velocity, offset) from local to world space.
    ///
    /// Identical to [`Self::transform_point`] without the translation, which is
    /// exactly what a velocity needs: it rotates but is not re-centred.
    #[must_use]
    pub fn transform_direction(self, local: Vec3) -> Vec3 {
        self.right
            .scale(local.x)
            .add(self.up.scale(local.y))
            .add(self.forward.scale(local.z))
    }

    /// Transforms a *point* from world space back to local space.
    ///
    /// Uses the transpose (dot with each basis vector) of the rotation, which
    /// is the exact inverse only for an orthonormal basis; the constructor
    /// contract requires one, so no re-normalization or division is performed.
    #[must_use]
    pub fn inverse_transform_point(self, world: Vec3) -> Vec3 {
        let d = world.sub(self.origin);
        Vec3::new(d.dot(self.right), d.dot(self.up), d.dot(self.forward))
    }

    /// Transforms a *direction* from world space back to local space.
    #[must_use]
    pub fn inverse_transform_direction(self, world: Vec3) -> Vec3 {
        Vec3::new(
            world.dot(self.right),
            world.dot(self.up),
            world.dot(self.forward),
        )
    }
}

/// Configuration for the origin-rebasing (floating-origin) trigger.
///
/// The simulation is re-centred whenever the camera drifts further than
/// `threshold` from the current origin, before `fp32` precision at that
/// distance becomes visible. A larger threshold rebases less often (cheaper,
/// but coarser far-field precision); a smaller one rebases more often.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RebaseConfig {
    /// Distance from the current origin at which a rebase is triggered.
    pub threshold: f32,
}

impl RebaseConfig {
    /// Builds a config from a positive trigger distance.
    #[must_use]
    pub const fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Returns `true` when the camera has drifted past the trigger distance.
    ///
    /// Compares squared magnitudes so no `sqrt` is needed and no `fp32`
    /// equality is performed.
    #[must_use]
    pub fn should_rebase(self, camera: Vec3) -> bool {
        camera.length_squared() > self.threshold * self.threshold
    }
}

/// A two-layer world coordinate: an integer chunk index plus an intra-chunk
/// `fp32` offset.
///
/// Storing a position as `(chunk, offset)` keeps the `fp32` part bounded to a
/// single chunk (so its `ULP` stays tiny regardless of how far the chunk is
/// from the absolute origin), while the integer index carries the large-scale
/// magnitude exactly. This is the durable, precision-stable representation the
/// large world addresses positions with; the small `offset` is what the
/// integrator actually simulates on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkCoord {
    /// Integer chunk index along each axis.
    pub chunk: [i32; 3],
    /// Position within the chunk, in `[0, chunk_size)` per axis after a
    /// [`ChunkGrid::split`].
    pub offset: Vec3,
}

/// The uniform chunk lattice used to split and compose world positions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkGrid {
    /// Edge length of a cubic chunk, in world units.
    pub chunk_size: f32,
}

impl ChunkGrid {
    /// Builds a grid from a positive chunk edge length.
    #[must_use]
    pub const fn new(chunk_size: f32) -> Self {
        Self { chunk_size }
    }

    /// Splits an absolute world position into `(chunk, offset)`.
    ///
    /// The chunk index is the floored quotient and the offset is the remainder,
    /// so the offset always lands in `[0, chunk_size)`. `floor` is an ordinary
    /// rounding operation, not a transcendental call.
    #[must_use]
    pub fn split(self, world: Vec3) -> ChunkCoord {
        let cs = self.chunk_size;
        let fx = (world.x / cs).floor();
        let fy = (world.y / cs).floor();
        let fz = (world.z / cs).floor();
        let offset = Vec3::new(world.x - fx * cs, world.y - fy * cs, world.z - fz * cs);
        ChunkCoord {
            chunk: [fx as i32, fy as i32, fz as i32],
            offset,
        }
    }

    /// Composes a chunk coordinate back into an *absolute* world position.
    ///
    /// Far from the origin this deliberately re-introduces the `fp32`
    /// magnitude the split was hiding, so it is for display/debug or a final
    /// bake, never for the inner simulation loop; prefer
    /// [`Self::compose_relative`] there.
    #[must_use]
    pub fn compose(self, coord: ChunkCoord) -> Vec3 {
        let cs = self.chunk_size;
        Vec3::new(
            coord.chunk[0] as f32 * cs,
            coord.chunk[1] as f32 * cs,
            coord.chunk[2] as f32 * cs,
        )
        .add(coord.offset)
    }

    /// Composes a chunk coordinate into a position *relative to a reference
    /// chunk* (typically the camera's chunk / the current rebased origin).
    ///
    /// Only the small integer chunk delta contributes, so the result stays near
    /// the origin and keeps full `fp32` precision — this is the value handed to
    /// the integrator.
    #[must_use]
    pub fn compose_relative(self, coord: ChunkCoord, reference: [i32; 3]) -> Vec3 {
        let cs = self.chunk_size;
        let dx = (coord.chunk[0] - reference[0]) as f32;
        let dy = (coord.chunk[1] - reference[1]) as f32;
        let dz = (coord.chunk[2] - reference[2]) as f32;
        Vec3::new(dx * cs, dy * cs, dz * cs).add(coord.offset)
    }

    /// Snaps an arbitrary position down to its enclosing chunk origin.
    ///
    /// Used to compute a rebase offset that is a whole number of chunks, so a
    /// rebase never disturbs the intra-chunk `offset` of any [`ChunkCoord`].
    #[must_use]
    pub fn snap_to_chunk(self, pos: Vec3) -> Vec3 {
        let cs = self.chunk_size;
        Vec3::new(
            (pos.x / cs).floor() * cs,
            (pos.y / cs).floor() * cs,
            (pos.z / cs).floor() * cs,
        )
    }
}

/// Computes the chunk-aligned rebase offset for a new camera position.
///
/// The returned vector is the amount subtracted from every world-space
/// position during the rebase; snapping it to the chunk grid keeps chunk
/// indices and intra-chunk offsets consistent across the relocation.
#[must_use]
pub fn rebase_offset(camera: Vec3, grid: ChunkGrid) -> Vec3 {
    grid.snap_to_chunk(camera)
}

/// Returns `true` when a rebase offset is large enough to bother applying.
///
/// A near-zero offset (camera still inside the current origin chunk) can be
/// skipped; the squared length is compared against [`EPS_LEN_SQ`] so no `fp32`
/// equality is used.
#[must_use]
pub fn offset_is_significant(offset: Vec3) -> bool {
    offset.length_squared() > EPS_LEN_SQ
}

/// Applies a rebase offset to a single world-space *position*.
///
/// Every world-stored particle position is translated by `-offset` so the new
/// origin sits where the offset points. This is a pure translation, so all
/// relative geometry between particles is preserved exactly.
#[must_use]
pub fn apply_rebase_position(world_pos: Vec3, offset: Vec3) -> Vec3 {
    world_pos.sub(offset)
}

/// Returns a velocity unchanged across a rebase.
///
/// A rebase is a translation of the coordinate origin, and velocity is a
/// *difference* of positions, so the common offset cancels: velocities (and
/// accelerations, forces, and every other direction quantity) are invariant.
/// This helper exists to make that invariance an explicit, testable part of
/// the contract rather than an unstated assumption.
#[must_use]
pub fn apply_rebase_velocity(world_vel: Vec3) -> Vec3 {
    world_vel
}

/// Rebases one particle's position according to its simulation space.
///
/// The offset is applied only to particles stored in world space
/// (`SimSpace::World` and the post-spawn `SimSpace::Hybrid`). `SimSpace::Local`
/// positions are relative to the owning transform, so they are left untouched:
/// the transform's own `origin` is what gets rebased, which keeps every stored
/// local position valid without any per-particle work. Velocity is invariant
/// in every space (see [`apply_rebase_velocity`]).
#[must_use]
pub fn rebase_particle(space: SimSpace, pos: Vec3, offset: Vec3) -> Vec3 {
    match storage_space(space) {
        StorageSpace::World => apply_rebase_position(pos, offset),
        StorageSpace::Local => pos,
    }
}

/// Order-of-magnitude `fp32` `ULP` (unit in the last place) at `value`.
///
/// The spacing between adjacent `fp32` numbers near magnitude `m` is roughly
/// `m · 2^-23`, so a position `1_000_000` units from the origin can only be
/// represented to within ~`0.06` world units — motion smaller than that
/// quantizes away. This estimator extracts the binary exponent from the `fp32`
/// bit pattern (pure integer ops, no `log`) and rebuilds the corresponding
/// power of two, giving the step size the origin rebase is designed to keep
/// small. Zero and subnormal inputs report the smallest positive subnormal.
#[must_use]
pub fn fp32_ulp(value: f32) -> f32 {
    // Clear the sign bit rather than calling `abs`, keeping this pure bit math.
    let mag_bits = value.to_bits() & 0x7fff_ffff;
    let exp = (mag_bits >> 23) & 0xff;
    if exp == 0 {
        // Zero or subnormal: the step is the smallest positive subnormal.
        return f32::from_bits(1);
    }
    // ULP magnitude is 2^(exp - 127 - 23); its biased exponent is exp - 23.
    let ulp_exp = exp as i32 - 23;
    if ulp_exp <= 0 {
        return f32::from_bits(1);
    }
    f32::from_bits((ulp_exp as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `+90°`-about-`z` orthonormal basis, built without any trig call:
    /// local `+x` maps to world `+y`, local `+y` maps to world `-x`.
    fn rotated_frame(origin: Vec3) -> TransformFrame {
        TransformFrame {
            origin,
            right: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::new(-1.0, 0.0, 0.0),
            forward: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    fn approx(a: Vec3, b: Vec3) -> bool {
        a.distance_squared(b) < 1e-6
    }

    #[test]
    fn plan_matches_space_semantics() {
        assert_eq!(
            plan(SimSpace::Local),
            SimSpacePlan {
                spawn_bake_to_world: false,
                storage: StorageSpace::Local,
            }
        );
        assert_eq!(
            plan(SimSpace::World),
            SimSpacePlan {
                spawn_bake_to_world: true,
                storage: StorageSpace::World,
            }
        );
        // Hybrid stores/updates like World but is authored in local at spawn.
        assert_eq!(plan(SimSpace::Hybrid).storage, StorageSpace::World);
        assert!(plan(SimSpace::Hybrid).spawn_bake_to_world);
    }

    #[test]
    fn identity_frame_is_a_no_op() {
        let p = Vec3::new(3.0, -4.0, 5.0);
        assert_eq!(TransformFrame::IDENTITY.transform_point(p), p);
        assert_eq!(TransformFrame::IDENTITY.transform_direction(p), p);
        assert_eq!(TransformFrame::IDENTITY.inverse_transform_point(p), p);
    }

    #[test]
    fn point_round_trips_through_local_and_world() {
        let frame = rotated_frame(Vec3::new(100.0, 0.0, 0.0));
        let local = Vec3::new(1.0, 2.0, 3.0);
        let world = frame.transform_point(local);
        // Explicit expected world position for the +90° rotation about z.
        assert!(approx(world, Vec3::new(98.0, 1.0, 3.0)));
        assert!(approx(frame.inverse_transform_point(world), local));
    }

    #[test]
    fn direction_round_trips_and_ignores_translation() {
        let frame = rotated_frame(Vec3::new(500.0, -20.0, 7.0));
        let vel = Vec3::new(2.0, 0.0, -1.0);
        let world = frame.transform_direction(vel);
        // A direction is unaffected by the frame's origin.
        assert!(approx(
            world,
            rotated_frame(Vec3::ZERO).transform_direction(vel)
        ));
        assert!(approx(frame.inverse_transform_direction(world), vel));
    }

    #[test]
    fn rebase_trigger_respects_threshold() {
        let cfg = RebaseConfig::new(1000.0);
        assert!(!cfg.should_rebase(Vec3::new(0.0, 0.0, 500.0)));
        assert!(!cfg.should_rebase(Vec3::new(600.0, 600.0, 0.0)));
        assert!(cfg.should_rebase(Vec3::new(0.0, 0.0, 2000.0)));
    }

    #[test]
    fn rebase_preserves_relative_positions() {
        let grid = ChunkGrid::new(16.0);
        let p1 = Vec3::new(1000.0, 0.0, 0.0);
        let p2 = Vec3::new(1005.0, 3.0, 0.0);
        let offset = rebase_offset(p1, grid);
        let r1 = apply_rebase_position(p1, offset);
        let r2 = apply_rebase_position(p2, offset);
        // The gap between the two particles is identical before and after.
        assert!(approx(r1.sub(r2), p1.sub(p2)));
        // The rebased positions are much closer to the origin than the inputs.
        assert!(r1.length_squared() < p1.length_squared());
    }

    #[test]
    fn rebase_leaves_velocity_untouched() {
        let vel = Vec3::new(-3.0, 12.0, 0.5);
        assert_eq!(apply_rebase_velocity(vel), vel);
    }

    #[test]
    fn offset_significance_uses_eps() {
        assert!(!offset_is_significant(Vec3::ZERO));
        assert!(offset_is_significant(Vec3::new(16.0, 0.0, 0.0)));
    }

    #[test]
    fn chunk_split_then_compose_is_identity() {
        let grid = ChunkGrid::new(16.0);
        let world = Vec3::new(40.0, -3.0, 0.0);
        let coord = grid.split(world);
        // Floored index and in-range offset.
        assert_eq!(coord.chunk, [2, -1, 0]);
        assert!(coord.offset.x >= 0.0 && coord.offset.x < 16.0);
        assert!(coord.offset.y >= 0.0 && coord.offset.y < 16.0);
        assert!(approx(grid.compose(coord), world));
    }

    #[test]
    fn compose_relative_stays_small_for_distant_chunks() {
        let grid = ChunkGrid::new(16.0);
        let coord = ChunkCoord {
            chunk: [100_000, 0, 0],
            offset: Vec3::new(4.0, 0.0, 0.0),
        };
        // Referenced against its own chunk, only the intra-chunk offset remains.
        let rel = grid.compose_relative(coord, [100_000, 0, 0]);
        assert!(approx(rel, Vec3::new(4.0, 0.0, 0.0)));
        // Absolute composition, by contrast, is enormous.
        assert!(grid.compose(coord).length_squared() > rel.length_squared());
    }

    #[test]
    fn rebase_handles_all_three_spaces() {
        let pos = Vec3::new(1000.0, 0.0, 0.0);
        let offset = Vec3::new(1000.0, 0.0, 0.0);
        // World and Hybrid are world-stored: both translate by -offset.
        assert!(approx(
            rebase_particle(SimSpace::World, pos, offset),
            Vec3::ZERO
        ));
        assert!(approx(
            rebase_particle(SimSpace::Hybrid, pos, offset),
            Vec3::ZERO
        ));
        // Local positions are relative to the transform and are left untouched.
        assert!(approx(rebase_particle(SimSpace::Local, pos, offset), pos));
        assert_eq!(storage_space(SimSpace::Local), StorageSpace::Local);
        assert_eq!(storage_space(SimSpace::World), StorageSpace::World);
        assert_eq!(storage_space(SimSpace::Hybrid), StorageSpace::World);
    }

    #[test]
    fn fp32_ulp_grows_with_distance_from_origin() {
        let near = fp32_ulp(1.0);
        let far = fp32_ulp(1.0e6);
        // Precision degrades with magnitude: the far step is far coarser.
        assert!(far > near);
        // Both are strictly positive, and zero reports the subnormal floor.
        assert!(near > 0.0);
        assert!(fp32_ulp(0.0) > 0.0);
        // The step just above 1.0 is the classic 2^-23 spacing.
        assert!(approx(
            Vec3::new(near, 0.0, 0.0),
            Vec3::new(f32::from_bits((104_u32) << 23), 0.0, 0.0)
        ));
    }
}
