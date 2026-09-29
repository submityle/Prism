//! Clustered forward light-sampling `DataInterface` for particle shading — the
//! deterministic `CPU` reference for "a particle samples its incident light from
//! a pre-binned punctual-light list" (design §8.3 *Scene* category, consumed by
//! the §17 lighting closures).
//!
//! # What this module is
//!
//! A froxel (frustum-voxel) grid partitions the view frustum into screen-space
//! tiles along X/Y and depth *slices* along the view Z axis. A light-culling
//! pass (out of scope here) assigns each punctual light to the clusters it
//! touches and packs the assignments into a compact `offset + count` index
//! list. At shade time a particle maps its view-space position to a single
//! cluster, reads that cluster's short light slice, and accumulates diffuse
//! response. This mirrors the clustered-forward rig in `bevy_light`, the
//! Frostbite / DOOM 2016 clustered-forward renderers, and the light-sampling
//! blocks of Unity's `VFX Graph` — without reusing any of their code.
//!
//! # Depth slicing: linear / explicit, never logarithmic
//!
//! Production clustered renderers usually distribute depth slices
//! *logarithmically* (`slice = floor(k * ln(z/near))`) so slices grow with
//! distance. This crate is a dependency-free, transcendental-free contract: the
//! natural logarithm is banned (see `clippy.toml`). We therefore expose two
//! transcendental-free distributions — a uniform linear split between the near
//! and far planes, and an author-supplied array of explicit monotonic slice
//! boundaries. The linear split trades some far-field slice utilization for bit-
//! reproducibility; when an author wants log-like behavior they precompute the
//! boundary array offline (where a logarithm is allowed) and hand the finished
//! `f32` edges to [`ClusterGrid::with_boundaries`]. No runtime `ln`/`exp` call
//! is ever made here.
//!
//! # Orthogonality to the sibling scene `DataInterface`s
//!
//! - `raytrace.rs` is the *ray-traced* scene lighting `DI`: it answers incident
//!   light and soft-shadow visibility by casting rays against `bevy_solari`
//!   geometry. This module answers the *same shading question* from a rasterizer-
//!   friendly binned light list; the two are alternative light-sampling sources
//!   feeding one shading closure, not competitors.
//! - `shading.rs` is the *router*: it decides which shading model (`Unlit` /
//!   `PBR` / `NPR` / custom) runs and therefore whether clustered lighting is
//!   even sampled. It never bins lights itself.
//! - `collision.rs` answers particle-versus-environment *contact*, an unrelated
//!   spatial query; it shares only the froxel-style spatial-hashing intuition,
//!   not the light data.
//!
//! # Numerics
//!
//! All math is spelled out on the shared hand-rolled [`Vec3`]. The only
//! irrational operation is `sqrt` (via [`Vec3::length`] /
//! [`Vec3::normalize_or_zero`]); every distance/spot falloff is a polynomial or
//! a `dot` compared against a stored cosine threshold, and every division is
//! guarded by [`EPS`]. Empty clusters, absent lights, and out-of-frustum
//! particles all fall back to [`Vec3::ZERO`] rather than producing `NaN` or
//! panicking, so the reference stays bit-reproducible against a future `GPU`
//! clustered-shading kernel.

use alloc::vec::Vec;

use super::Vec3;

/// Shared floating-point tolerance for this module.
///
/// Used both to guard divisions (denominators are floored to `EPS`) and as the
/// slack for the `#[cfg(test)]` approximate-equality assertions. It is
/// deliberately looser than the crate's `EPS_LEN_SQ` squared-length epsilon
/// because it also bounds the accumulated polynomial round-off of a full
/// lighting sum.
pub const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Light representation (design §8.3, §17).
// ---------------------------------------------------------------------------

/// Whether a [`PunctualLight`] radiates in every direction or is confined to a
/// cone.
///
/// This is the discriminator that decides whether the cone-angle falloff is
/// evaluated; it holds no floating-point payload, so it derives `Eq` and `Hash`
/// for use as a map key or in exhaustive match arms.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LightKind {
    /// An omnidirectional point light; the cone test is skipped.
    Point,
    /// A spot light constrained to a cone; the cone test applies.
    Spot,
}

/// One punctual (point or spot) light in the clustered light table.
///
/// The cone thresholds are stored as *cosines* — `cos_inner` for the fully lit
/// inner cone and `cos_outer` for the outer cutoff — because obtaining an angle
/// from a `dot` would require `acos`, a banned transcendental. Since cosine is
/// monotonically decreasing on `[0, pi]`, a direction is inside the cone exactly
/// when its axis-alignment cosine is **greater** than the threshold cosine, and
/// `cos_inner >= cos_outer` for a well-formed cone. Colour and intensity are
/// pre-multiplied into a single radiance vector (`color * intensity`) so the
/// hot accumulation loop performs no extra scalar multiply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PunctualLight {
    /// World/view-space position of the light (must match the space the grid
    /// and particle positions use).
    pub position: Vec3,
    /// Cone axis for a [`LightKind::Spot`], pointing from the light toward the
    /// lit region. Ignored for [`LightKind::Point`]; normalized robustly at use.
    pub direction: Vec3,
    /// Radiance = base colour times scalar intensity, pre-multiplied.
    pub color_intensity: Vec3,
    /// Influence radius / range `r`; beyond it the windowed falloff is exactly
    /// zero, which is also the cluster-culling radius.
    pub range: f32,
    /// Cosine of the inner cone half-angle (full brightness at/above this).
    pub cos_inner: f32,
    /// Cosine of the outer cone half-angle (zero brightness at/below this).
    pub cos_outer: f32,
    /// Point vs. spot discriminator.
    pub kind: LightKind,
}

impl PunctualLight {
    /// Builds an omnidirectional point light.
    ///
    /// The cone cosines are set to `-1` (the cosine of `pi`), so every direction
    /// trivially satisfies the inside-cone test and the cone falloff is a
    /// constant `1`, even before [`LightKind::Point`] short-circuits it.
    #[must_use]
    pub fn point(position: Vec3, color_intensity: Vec3, range: f32) -> Self {
        Self {
            position,
            direction: Vec3::new(0.0, 0.0, 1.0),
            color_intensity,
            range,
            cos_inner: -1.0,
            cos_outer: -1.0,
            kind: LightKind::Point,
        }
    }

    /// Builds a spot light from an axis and its inner/outer cone cosines.
    ///
    /// The caller passes cosines directly (for example a precomputed
    /// `cos(half_angle)`); this constructor performs no trigonometry. The two
    /// thresholds are ordered so `cos_inner >= cos_outer` regardless of the
    /// argument order, keeping the falloff window non-degenerate.
    #[must_use]
    pub fn spot(
        position: Vec3,
        direction: Vec3,
        color_intensity: Vec3,
        range: f32,
        cos_inner: f32,
        cos_outer: f32,
    ) -> Self {
        let (lo, hi) = if cos_inner >= cos_outer {
            (cos_outer, cos_inner)
        } else {
            (cos_inner, cos_outer)
        };
        Self {
            position,
            direction: direction.normalize_or_zero(),
            color_intensity,
            range,
            cos_inner: hi,
            cos_outer: lo,
            kind: LightKind::Spot,
        }
    }

    /// Returns `true` when the light has a usable (strictly positive) range.
    ///
    /// A non-positive range means the influence sphere is empty, so the light
    /// contributes nothing and the accumulation loop can skip it.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.range > EPS
    }

    /// The squared influence radius, precomputed for the culling comparison so
    /// the hot loop compares squared distances and avoids a `sqrt` on misses.
    #[must_use]
    pub fn range_squared(self) -> f32 {
        self.range * self.range
    }
}

// ---------------------------------------------------------------------------
// Froxel cluster grid (design §8.3).
// ---------------------------------------------------------------------------

/// A froxel cluster coordinate: a tile column, a tile row, and a depth slice.
///
/// All three fields are integer indices, so this derives `Eq`/`Hash` and can key
/// a debug map of cluster occupancy. It is produced by
/// [`ClusterGrid::cluster_coord`] and flattened by
/// [`ClusterGrid::linear_index`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClusterCoord {
    /// Tile column in `0..tile_count_x`.
    pub x: u32,
    /// Tile row in `0..tile_count_y`.
    pub y: u32,
    /// Depth slice in `0..slice_count`.
    pub z: u32,
}

/// The froxel grid mapping view-space positions to clusters.
///
/// The grid works in a right-handed *view space* whose camera sits at the origin
/// looking down **+Z**, so a point's view depth is simply its `z` and increases
/// with distance. Screen tiling uses a symmetric perspective frustum described by
/// two half-extent *slopes* (`slope_x`, `slope_y`), i.e. `tan(fov/2)` values the
/// author computes once — this module never calls `tan`. At view depth `z` the
/// frustum half-width is `z * slope_x`, so the normalized screen coordinate is
/// `view.x / (z * slope_x)` (division guarded by [`EPS`]). Depth slices are
/// delimited by [`ClusterGrid::slice_boundaries`], a monotonically increasing
/// array of `slice_count + 1` view-depth edges (linear or explicit; never
/// logarithmic — see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterGrid {
    tile_count_x: u32,
    tile_count_y: u32,
    slope_x: f32,
    slope_y: f32,
    slice_boundaries: Vec<f32>,
}

impl ClusterGrid {
    /// Builds a grid with **linearly** spaced depth slices between `near` and
    /// `far`.
    ///
    /// This is the transcendental-free stand-in for logarithmic slicing: the
    /// `slice_count + 1` boundaries are evenly spaced in view depth. Degenerate
    /// inputs are clamped to a valid minimum (at least one tile per axis, at
    /// least one slice, `far` pushed above `near` by [`EPS`]) so the constructor
    /// always yields a usable grid rather than panicking.
    #[must_use]
    pub fn linear(
        tile_count_x: u32,
        tile_count_y: u32,
        slice_count: u32,
        near: f32,
        far: f32,
        slope_x: f32,
        slope_y: f32,
    ) -> Self {
        let tile_count_x = tile_count_x.max(1);
        let tile_count_y = tile_count_y.max(1);
        let slice_count = slice_count.max(1);
        let near = near.max(EPS);
        let far = if far > near + EPS { far } else { near + EPS };
        let span = far - near;
        let inv_slices = 1.0 / (slice_count as f32);
        let mut slice_boundaries = Vec::with_capacity((slice_count + 1) as usize);
        let mut i = 0u32;
        while i <= slice_count {
            let t = (i as f32) * inv_slices;
            slice_boundaries.push(near + span * t);
            i += 1;
        }
        Self {
            tile_count_x,
            tile_count_y,
            slope_x: slope_x.max(EPS),
            slope_y: slope_y.max(EPS),
            slice_boundaries,
        }
    }

    /// Builds a grid from an author-supplied array of monotonic depth
    /// boundaries.
    ///
    /// `boundaries` must be sorted ascending and hold at least two entries; the
    /// slice count is `boundaries.len() - 1`. This is the escape hatch for
    /// "log-like" slicing computed offline: the author bakes the edges (where a
    /// logarithm is permitted) and passes the finished `f32` array here, so no
    /// transcendental runs at simulation time. Any non-increasing input edge is
    /// nudged up by [`EPS`] to keep the boundary array strictly monotonic, and a
    /// too-short array is padded to a single valid slice, so a lookup can never
    /// divide by a zero-width slice.
    #[must_use]
    pub fn with_boundaries(
        tile_count_x: u32,
        tile_count_y: u32,
        slope_x: f32,
        slope_y: f32,
        boundaries: &[f32],
    ) -> Self {
        let mut slice_boundaries = Vec::with_capacity(boundaries.len().max(2));
        if boundaries.len() < 2 {
            let base = boundaries.first().copied().unwrap_or(EPS).max(EPS);
            slice_boundaries.push(base);
            slice_boundaries.push(base + EPS);
        } else {
            let mut prev = boundaries[0].max(EPS);
            slice_boundaries.push(prev);
            let mut i = 1usize;
            while i < boundaries.len() {
                let next = if boundaries[i] > prev + EPS {
                    boundaries[i]
                } else {
                    prev + EPS
                };
                slice_boundaries.push(next);
                prev = next;
                i += 1;
            }
        }
        Self {
            tile_count_x: tile_count_x.max(1),
            tile_count_y: tile_count_y.max(1),
            slope_x: slope_x.max(EPS),
            slope_y: slope_y.max(EPS),
            slice_boundaries,
        }
    }

    /// Number of tile columns along screen X.
    #[must_use]
    pub fn tile_count_x(&self) -> u32 {
        self.tile_count_x
    }

    /// Number of tile rows along screen Y.
    #[must_use]
    pub fn tile_count_y(&self) -> u32 {
        self.tile_count_y
    }

    /// Number of depth slices along view Z.
    #[must_use]
    pub fn slice_count(&self) -> u32 {
        // The array always holds at least two edges, so this never underflows.
        (self.slice_boundaries.len() as u32) - 1
    }

    /// Total number of clusters (`tile_x * tile_y * slice_count`).
    ///
    /// This is the exact length a companion [`ClusterLightList`] must have one
    /// `offset`/`count` pair for.
    #[must_use]
    pub fn cluster_count(&self) -> u32 {
        self.tile_count_x * self.tile_count_y * self.slice_count()
    }

    /// The monotonic view-depth slice boundaries (`slice_count + 1` entries).
    #[must_use]
    pub fn slice_boundaries(&self) -> &[f32] {
        &self.slice_boundaries
    }

    /// The near-plane view depth (first slice boundary).
    #[must_use]
    pub fn near(&self) -> f32 {
        self.slice_boundaries[0]
    }

    /// The far-plane view depth (last slice boundary).
    #[must_use]
    pub fn far(&self) -> f32 {
        self.slice_boundaries[self.slice_boundaries.len() - 1]
    }

    /// Maps a view depth `z` to its depth slice, or `None` when `z` is in front
    /// of the near plane or behind the far plane.
    ///
    /// The lookup is a linear scan of the boundary array (slice counts are small,
    /// typically 16–32); it returns the slice `i` with
    /// `boundaries[i] <= z < boundaries[i + 1]`, and treats the far plane itself
    /// as belonging to the last slice.
    #[must_use]
    pub fn depth_slice(&self, z: f32) -> Option<u32> {
        if z < self.near() - EPS || z > self.far() + EPS {
            return None;
        }
        let slices = self.slice_count();
        let mut i = 0u32;
        while i < slices {
            let lo = self.slice_boundaries[i as usize];
            let hi = self.slice_boundaries[(i + 1) as usize];
            if z >= lo - EPS && z <= hi + EPS {
                return Some(i);
            }
            i += 1;
        }
        // Numerically at/just past the far edge: clamp to the last slice.
        Some(slices - 1)
    }

    /// Maps a normalized screen coordinate in `[-1, 1]` to a tile index in
    /// `0..count`, or `None` when it falls outside the frustum.
    ///
    /// The mapping is `(ndc + 1) * 0.5 * count`, floored; the half-open right
    /// edge (`ndc == 1`) is folded back into the last tile so a point exactly on
    /// the frustum boundary is still classified.
    fn ndc_to_tile(ndc: f32, count: u32) -> Option<u32> {
        if !(-1.0 - EPS..=1.0 + EPS).contains(&ndc) {
            return None;
        }
        let scaled = (ndc + 1.0) * 0.5 * (count as f32);
        let tile = scaled.floor();
        if tile < 0.0 {
            Some(0)
        } else if tile >= (count as f32) {
            Some(count - 1)
        } else {
            Some(tile as u32)
        }
    }

    /// Maps a view-space position to its cluster coordinate, or `None` when the
    /// point lies outside the frustum (behind the near plane, beyond the far
    /// plane, or off-screen).
    ///
    /// The perspective divide uses the per-axis frustum slope at the point's
    /// depth; both divisions are guarded because the depth is already clamped to
    /// at least the near plane (`>= EPS`).
    #[must_use]
    pub fn cluster_coord(&self, view_pos: Vec3) -> Option<ClusterCoord> {
        let z = view_pos.z;
        let slice = self.depth_slice(z)?;
        // Depth is valid here, so `z` is at least the near plane and positive.
        let half_w = z * self.slope_x;
        let half_h = z * self.slope_y;
        if half_w <= EPS || half_h <= EPS {
            return None;
        }
        let ndc_x = view_pos.x / half_w;
        let ndc_y = view_pos.y / half_h;
        let tx = Self::ndc_to_tile(ndc_x, self.tile_count_x)?;
        let ty = Self::ndc_to_tile(ndc_y, self.tile_count_y)?;
        Some(ClusterCoord {
            x: tx,
            y: ty,
            z: slice,
        })
    }

    /// Flattens a cluster coordinate to a linear cluster index, or `None` when
    /// any component is out of range.
    ///
    /// The layout is slice-major then row-major:
    /// `z * (tile_x * tile_y) + y * tile_x + x`, matching the packing a `GPU`
    /// light-culling pass would write.
    #[must_use]
    pub fn linear_index(&self, coord: ClusterCoord) -> Option<u32> {
        if coord.x >= self.tile_count_x
            || coord.y >= self.tile_count_y
            || coord.z >= self.slice_count()
        {
            return None;
        }
        let per_slice = self.tile_count_x * self.tile_count_y;
        Some(coord.z * per_slice + coord.y * self.tile_count_x + coord.x)
    }

    /// Convenience: view-space position straight to a linear cluster index, or
    /// `None` when the point is outside the frustum.
    #[must_use]
    pub fn cluster_index(&self, view_pos: Vec3) -> Option<u32> {
        let coord = self.cluster_coord(view_pos)?;
        self.linear_index(coord)
    }
}

// ---------------------------------------------------------------------------
// Per-cluster light index list (design §8.3).
// ---------------------------------------------------------------------------

/// A compact per-cluster light index list using `offset + count` slices into a
/// shared flat index buffer.
///
/// This is the exact storage a clustered light-culling pass emits: `offsets[c]`
/// and `counts[c]` locate cluster `c`'s run inside `indices`, and each entry of
/// that run is an index into the light table. Packing every cluster's lights
/// contiguously keeps the shade-time read to one slice with no per-cluster heap
/// indirection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterLightList {
    offsets: Vec<u32>,
    counts: Vec<u32>,
    indices: Vec<u32>,
}

impl ClusterLightList {
    /// An empty list sized for `cluster_count` clusters, all with zero lights.
    ///
    /// This is the safe default for a frame with no lights: every
    /// [`ClusterLightList::lights_for`] returns an empty slice, so accumulation
    /// falls back to [`Vec3::ZERO`].
    #[must_use]
    pub fn empty(cluster_count: u32) -> Self {
        let n = cluster_count as usize;
        let mut offsets = Vec::with_capacity(n);
        let mut counts = Vec::with_capacity(n);
        offsets.resize(n, 0);
        counts.resize(n, 0);
        Self {
            offsets,
            counts,
            indices: Vec::new(),
        }
    }

    /// Builds a list from a per-cluster slice of light-index runs.
    ///
    /// `per_cluster[c]` is the (already culled) set of light-table indices that
    /// touch cluster `c`. The runs are concatenated into the flat index buffer in
    /// cluster order and the matching `offset`/`count` pairs are recorded, so the
    /// result is the packed form a `GPU` pass would produce.
    #[must_use]
    pub fn from_per_cluster(per_cluster: &[&[u32]]) -> Self {
        let n = per_cluster.len();
        let mut offsets = Vec::with_capacity(n);
        let mut counts = Vec::with_capacity(n);
        let total: usize = per_cluster.iter().map(|run| run.len()).sum();
        let mut indices = Vec::with_capacity(total);
        for run in per_cluster {
            offsets.push(indices.len() as u32);
            counts.push(run.len() as u32);
            indices.extend_from_slice(run);
        }
        Self {
            offsets,
            counts,
            indices,
        }
    }

    /// Number of clusters this list describes.
    #[must_use]
    pub fn cluster_count(&self) -> u32 {
        self.offsets.len() as u32
    }

    /// Total number of packed light indices across all clusters.
    #[must_use]
    pub fn total_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Returns cluster `cluster_index`'s light-index slice, or an empty slice for
    /// an out-of-range cluster or a cluster with no lights.
    ///
    /// The bounds are validated against both the `offset/count` metadata and the
    /// flat buffer length, so a malformed run can never produce an out-of-bounds
    /// read — it degrades to an empty slice instead.
    #[must_use]
    pub fn lights_for(&self, cluster_index: u32) -> &[u32] {
        let c = cluster_index as usize;
        if c >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[c] as usize;
        let count = self.counts[c] as usize;
        let end = start.saturating_add(count);
        if start >= self.indices.len() || end > self.indices.len() {
            return &[];
        }
        &self.indices[start..end]
    }
}

// ---------------------------------------------------------------------------
// Falloff primitives (polynomial only — no transcendentals).
// ---------------------------------------------------------------------------

/// Clamps `x` to the unit interval `[0, 1]`.
#[must_use]
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The classic Hermite `smoothstep` between `edge0` and `edge1`.
///
/// Evaluates `t * t * (3 - 2 * t)` on the clamped, normalized parameter, giving
/// zero-derivative endpoints without any transcendental call. When the two edges
/// are within [`EPS`] of each other the interpolation collapses to a hard step
/// at `edge1` (denominator guarded), so a degenerate cone or window never
/// divides by zero.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() <= EPS {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = clamp01((x - edge0) / denom);
    t * t * (3.0 - 2.0 * t)
}

/// The windowed inverse-square distance attenuation used by clustered forward
/// rigs, expressed purely as polynomials.
///
/// The physical term is `1 / (d^2 + eps)`; the window
/// `(1 - (d / r)^4)^2`, clamped to `[0, 1]`, smoothly pulls the contribution to
/// exactly zero at the influence radius `r` so a light's cluster footprint stays
/// bounded. `(d / r)^4` is computed as `(d2 / r2)^2` from squared quantities, so
/// the whole factor needs a single guarded division and no `sqrt`. Returns `0`
/// for a non-positive range or for `d2` at/beyond `r2`.
#[must_use]
pub fn distance_attenuation(distance_squared: f32, range: f32) -> f32 {
    if range <= EPS {
        return 0.0;
    }
    let r2 = range * range;
    if distance_squared >= r2 {
        return 0.0;
    }
    let ratio2 = distance_squared / r2;
    let ratio4 = ratio2 * ratio2;
    let window = clamp01(1.0 - ratio4);
    let window_sq = window * window;
    let inv_sq = 1.0 / (distance_squared + EPS);
    window_sq * inv_sq
}

/// The spot-cone angular attenuation from an alignment cosine and the light's
/// stored inner/outer cone cosines.
///
/// `cos_angle` is `dot(cone_axis, light_to_point)` with both unit length. Because
/// cosine decreases with angle, full brightness is at or above `cos_inner` and
/// zero at or below `cos_outer`; between them we apply [`smoothstep`] for a soft
/// penumbra. This is the `dot`-versus-stored-cosine test that replaces an
/// `acos`, so no transcendental is evaluated.
#[must_use]
pub fn spot_attenuation(cos_angle: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    smoothstep(cos_outer, cos_inner, cos_angle)
}

// ---------------------------------------------------------------------------
// Lighting accumulation (design §8.3, §17).
// ---------------------------------------------------------------------------

/// Accumulates the diffuse response of a single light at a surface point.
///
/// Returns the light's contribution as `radiance * n_dot_l * distance_falloff *
/// spot_falloff`, or [`Vec3::ZERO`] when the light is inactive, the point is
/// outside the influence radius, the surface faces away, or the point sits
/// inside the spot's outer cutoff. The `normal` is normalized robustly; a zero
/// normal is treated as unlit (`n_dot_l == 0`) rather than yielding `NaN`.
#[must_use]
pub fn shade_point_light(light: PunctualLight, surface_pos: Vec3, normal: Vec3) -> Vec3 {
    if !light.is_active() {
        return Vec3::ZERO;
    }
    let to_light = light.position.sub(surface_pos);
    let dist_sq = to_light.length_squared();
    if dist_sq >= light.range_squared() {
        return Vec3::ZERO;
    }
    let l = to_light.normalize_or_zero();
    if l.length_squared() <= EPS {
        // Particle sits on the light; no well-defined direction, treat as unlit.
        return Vec3::ZERO;
    }
    let n = normal.normalize_or_zero();
    let n_dot_l = n.dot(l).max(0.0);
    if n_dot_l <= EPS {
        return Vec3::ZERO;
    }
    let atten = distance_attenuation(dist_sq, light.range);
    if atten <= EPS {
        return Vec3::ZERO;
    }
    let spot = match light.kind {
        LightKind::Point => 1.0,
        LightKind::Spot => {
            // The cone axis points toward the lit region; `-l` is the direction
            // from the light to the surface point.
            let cos_angle = light.direction.dot(l.scale(-1.0));
            spot_attenuation(cos_angle, light.cos_inner, light.cos_outer)
        }
    };
    if spot <= EPS {
        return Vec3::ZERO;
    }
    light.color_intensity.scale(n_dot_l * atten * spot)
}

/// Accumulates diffuse lighting for a particle from every light binned into its
/// cluster.
///
/// The particle position is expected in the same view space the [`ClusterGrid`]
/// uses. The routine maps the position to a cluster, reads that cluster's light
/// slice from `list`, and sums [`shade_point_light`] over the referenced entries
/// of `lights`. Every failure mode — the particle outside the frustum, an empty
/// or out-of-range cluster, a light index past the end of the table — is a safe
/// [`Vec3::ZERO`] fallback, so the function never panics and never returns
/// `NaN`.
#[must_use]
pub fn accumulate_lighting(
    grid: &ClusterGrid,
    list: &ClusterLightList,
    lights: &[PunctualLight],
    particle_pos: Vec3,
    normal: Vec3,
) -> Vec3 {
    let Some(cluster) = grid.cluster_index(particle_pos) else {
        return Vec3::ZERO;
    };
    let mut sum = Vec3::ZERO;
    for &light_index in list.lights_for(cluster) {
        let idx = light_index as usize;
        if idx >= lights.len() {
            continue;
        }
        sum = sum.add(shade_point_light(lights[idx], particle_pos, normal));
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for the approximate-equality checks in this module's
    /// tests; equal to the public [`EPS`] tolerance scaled up for accumulated
    /// polynomial round-off.
    const TEST_EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn unit_grid() -> ClusterGrid {
        // 4x4 tiles, 4 linear slices from near=1 to far=5, 45-degree-ish frustum.
        ClusterGrid::linear(4, 4, 4, 1.0, 5.0, 1.0, 1.0)
    }

    #[test]
    fn grid_dimensions_and_counts() {
        let grid = unit_grid();
        assert_eq!(grid.tile_count_x(), 4);
        assert_eq!(grid.tile_count_y(), 4);
        assert_eq!(grid.slice_count(), 4);
        assert_eq!(grid.cluster_count(), 4 * 4 * 4);
        assert!(approx(grid.near(), 1.0));
        assert!(approx(grid.far(), 5.0));
        // Five boundaries for four slices, evenly spaced by 1.0.
        assert_eq!(grid.slice_boundaries().len(), 5);
        assert!(approx(grid.slice_boundaries()[1], 2.0));
        assert!(approx(grid.slice_boundaries()[3], 4.0));
    }

    #[test]
    fn degenerate_grid_inputs_are_clamped() {
        // Zero tiles/slices and far <= near must not panic and must stay valid.
        let grid = ClusterGrid::linear(0, 0, 0, 2.0, 1.0, 0.0, 0.0);
        assert_eq!(grid.tile_count_x(), 1);
        assert_eq!(grid.tile_count_y(), 1);
        assert_eq!(grid.slice_count(), 1);
        assert!(grid.far() > grid.near());
        assert_eq!(grid.cluster_count(), 1);
    }

    #[test]
    fn depth_slice_mapping_is_monotonic() {
        let grid = unit_grid();
        // near..far split into [1,2),[2,3),[3,4),[4,5].
        assert_eq!(grid.depth_slice(1.0), Some(0));
        assert_eq!(grid.depth_slice(1.5), Some(0));
        assert_eq!(grid.depth_slice(2.5), Some(1));
        assert_eq!(grid.depth_slice(3.5), Some(2));
        assert_eq!(grid.depth_slice(4.9), Some(3));
        assert_eq!(grid.depth_slice(5.0), Some(3));
        // Out of frustum in depth.
        assert_eq!(grid.depth_slice(0.5), None);
        assert_eq!(grid.depth_slice(6.0), None);
    }

    #[test]
    fn explicit_boundaries_preserve_slices() {
        // Author-baked "log-like" edges; strictly increasing already.
        let grid = ClusterGrid::with_boundaries(2, 2, 1.0, 1.0, &[1.0, 2.0, 4.0, 8.0]);
        assert_eq!(grid.slice_count(), 3);
        assert!(approx(grid.near(), 1.0));
        assert!(approx(grid.far(), 8.0));
        assert_eq!(grid.depth_slice(1.5), Some(0));
        assert_eq!(grid.depth_slice(3.0), Some(1));
        assert_eq!(grid.depth_slice(7.0), Some(2));
    }

    #[test]
    fn explicit_boundaries_repair_non_monotonic_input() {
        // A flat/decreasing run must be nudged strictly increasing, never divide
        // by a zero-width slice.
        let grid = ClusterGrid::with_boundaries(1, 1, 1.0, 1.0, &[1.0, 1.0, 0.5]);
        assert_eq!(grid.slice_count(), 2);
        let b = grid.slice_boundaries();
        assert!(b[1] > b[0]);
        assert!(b[2] > b[1]);
    }

    #[test]
    fn explicit_boundaries_pad_short_input() {
        let grid = ClusterGrid::with_boundaries(1, 1, 1.0, 1.0, &[3.0]);
        assert_eq!(grid.slice_count(), 1);
        assert!(grid.far() > grid.near());
    }

    #[test]
    fn cluster_coord_center_and_corners() {
        let grid = unit_grid();
        // On the view axis at depth 1.5 -> center tiles, slice 0.
        let center = grid.cluster_coord(Vec3::new(0.0, 0.0, 1.5)).unwrap();
        assert_eq!(center.x, 2);
        assert_eq!(center.y, 2);
        assert_eq!(center.z, 0);
        // Far-left of the frustum at that depth: ndc_x = -1 -> tile 0.
        let half_w = 1.5; // z * slope_x
        let left = grid.cluster_coord(Vec3::new(-half_w, 0.0, 1.5)).unwrap();
        assert_eq!(left.x, 0);
        // Far-right edge -> last tile.
        let right = grid.cluster_coord(Vec3::new(half_w, 0.0, 1.5)).unwrap();
        assert_eq!(right.x, 3);
    }

    #[test]
    fn cluster_coord_out_of_frustum_is_none() {
        let grid = unit_grid();
        // Behind near plane.
        assert!(grid.cluster_coord(Vec3::new(0.0, 0.0, 0.2)).is_none());
        // Beyond far plane.
        assert!(grid.cluster_coord(Vec3::new(0.0, 0.0, 9.0)).is_none());
        // Off-screen horizontally (well past the frustum half-width at z=2).
        assert!(grid.cluster_coord(Vec3::new(100.0, 0.0, 2.0)).is_none());
    }

    #[test]
    fn linear_index_layout_and_bounds() {
        let grid = unit_grid();
        let per_slice = 16;
        assert_eq!(
            grid.linear_index(ClusterCoord { x: 0, y: 0, z: 0 }),
            Some(0)
        );
        assert_eq!(
            grid.linear_index(ClusterCoord { x: 3, y: 3, z: 0 }),
            Some(15)
        );
        assert_eq!(
            grid.linear_index(ClusterCoord { x: 0, y: 0, z: 1 }),
            Some(per_slice)
        );
        assert_eq!(
            grid.linear_index(ClusterCoord { x: 1, y: 2, z: 3 }),
            Some(3 * per_slice + 2 * 4 + 1)
        );
        // Out-of-range coordinate rejected.
        assert_eq!(grid.linear_index(ClusterCoord { x: 4, y: 0, z: 0 }), None);
        assert_eq!(grid.linear_index(ClusterCoord { x: 0, y: 0, z: 4 }), None);
    }

    #[test]
    fn light_list_slices_are_correct() {
        // Three clusters: [10, 11], [], [12].
        let c0: &[u32] = &[10, 11];
        let c1: &[u32] = &[];
        let c2: &[u32] = &[12];
        let list = ClusterLightList::from_per_cluster(&[c0, c1, c2]);
        assert_eq!(list.cluster_count(), 3);
        assert_eq!(list.total_indices(), 3);
        assert_eq!(list.lights_for(0), &[10, 11]);
        assert_eq!(list.lights_for(1), &[] as &[u32]);
        assert_eq!(list.lights_for(2), &[12]);
    }

    #[test]
    fn light_list_out_of_range_cluster_is_empty() {
        let list = ClusterLightList::empty(2);
        assert_eq!(list.cluster_count(), 2);
        assert_eq!(list.total_indices(), 0);
        assert_eq!(list.lights_for(0), &[] as &[u32]);
        // Past the end -> empty, not a panic.
        assert_eq!(list.lights_for(99), &[] as &[u32]);
    }

    #[test]
    fn distance_attenuation_window_shape() {
        // Zero distance -> dominated by 1/eps, but window == 1 there.
        let near0 = distance_attenuation(0.0, 4.0);
        assert!(near0 > 0.0);
        // At the influence radius the window forces exactly zero.
        assert!(approx(distance_attenuation(16.0, 4.0), 0.0));
        assert!(approx(distance_attenuation(20.0, 4.0), 0.0));
        // Monotonic decrease from a small distance to a larger one.
        let a = distance_attenuation(1.0, 4.0);
        let b = distance_attenuation(4.0, 4.0);
        assert!(a > b);
        // Non-positive range -> zero.
        assert!(approx(distance_attenuation(1.0, 0.0), 0.0));
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!(approx(smoothstep(0.0, 1.0, -1.0), 0.0));
        assert!(approx(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx(smoothstep(0.0, 1.0, 0.5), 0.5));
        assert!(approx(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx(smoothstep(0.0, 1.0, 2.0), 1.0));
        // Degenerate edges collapse to a hard step, no divide-by-zero.
        assert!(approx(smoothstep(1.0, 1.0, 1.5), 1.0));
        assert!(approx(smoothstep(1.0, 1.0, 0.5), 0.0));
    }

    #[test]
    fn spot_attenuation_cone_falloff() {
        // Inner cos 0.9, outer cos 0.7.
        // Inside the inner cone -> full.
        assert!(approx(spot_attenuation(0.95, 0.9, 0.7), 1.0));
        // Outside the outer cone -> zero.
        assert!(approx(spot_attenuation(0.6, 0.9, 0.7), 0.0));
        // Midway in cosine space -> partial, strictly between 0 and 1.
        let mid = spot_attenuation(0.8, 0.9, 0.7);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn single_point_light_diffuse() {
        // Light one unit above the surface, surface normal up: N.L == 1.
        let light = PunctualLight::point(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0), 10.0);
        let surface = Vec3::new(0.0, 0.0, -1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let out = shade_point_light(light, surface, normal);
        // radiance * 1 * attenuation(dist_sq=1, r=10).
        let atten = distance_attenuation(1.0, 10.0);
        let expected = Vec3::new(2.0, 2.0, 2.0).scale(atten);
        assert!(approx_vec(out, expected));
        assert!(out.x > 0.0);
    }

    #[test]
    fn point_light_backface_is_dark() {
        let light = PunctualLight::point(Vec3::ZERO, Vec3::splat(5.0), 10.0);
        let surface = Vec3::new(0.0, 0.0, -1.0);
        // Normal facing away from the light.
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let out = shade_point_light(light, surface, normal);
        assert!(approx_vec(out, Vec3::ZERO));
    }

    #[test]
    fn point_light_outside_range_is_dark() {
        let light = PunctualLight::point(Vec3::ZERO, Vec3::splat(5.0), 2.0);
        let surface = Vec3::new(0.0, 0.0, -10.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        assert!(approx_vec(
            shade_point_light(light, surface, normal),
            Vec3::ZERO
        ));
    }

    #[test]
    fn spot_light_angle_falloff() {
        // Spot at origin aimed down -Z toward a surface below it.
        let axis = Vec3::new(0.0, 0.0, -1.0);
        let light = PunctualLight::spot(
            Vec3::ZERO,
            axis,
            Vec3::splat(4.0),
            10.0,
            0.95, // cos inner
            0.80, // cos outer
        );
        // Directly on-axis: point straight below the light, normal facing up.
        let on_axis = Vec3::new(0.0, 0.0, -2.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let lit = shade_point_light(light, on_axis, normal);
        assert!(lit.x > 0.0);
        // Off to the side far enough to leave the cone: axis alignment drops
        // below cos_outer, so the spot term is zero.
        let off_axis = Vec3::new(5.0, 0.0, -0.2);
        let dark = shade_point_light(light, off_axis, normal);
        assert!(approx_vec(dark, Vec3::ZERO));
    }

    #[test]
    fn spot_constructor_orders_cosines() {
        // Pass the cosines in the "wrong" order; constructor must reorder.
        let light = PunctualLight::spot(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::splat(1.0),
            5.0,
            0.7, // given as inner but is actually the smaller (outer) cosine
            0.9,
        );
        assert!(light.cos_inner >= light.cos_outer);
        assert!(approx(light.cos_inner, 0.9));
        assert!(approx(light.cos_outer, 0.7));
    }

    #[test]
    fn accumulate_gathers_cluster_lights() {
        let grid = unit_grid();
        // Particle near the view axis at depth 1.5 -> a specific cluster.
        let particle = Vec3::new(0.0, 0.0, 1.5);
        let cluster = grid.cluster_index(particle).unwrap();

        // Two lights straddling the particle; place their table indices in the
        // particle's cluster.
        let lights = vec![
            PunctualLight::point(Vec3::new(0.0, 0.0, 1.0), Vec3::splat(3.0), 8.0),
            PunctualLight::point(Vec3::new(0.0, 0.0, 2.0), Vec3::splat(3.0), 8.0),
        ];

        // Build a per-cluster list with the two lights only in `cluster`.
        let mut per_cluster: Vec<Vec<u32>> = Vec::new();
        per_cluster.resize(grid.cluster_count() as usize, Vec::new());
        per_cluster[cluster as usize] = vec![0, 1];
        let refs: Vec<&[u32]> = per_cluster.iter().map(Vec::as_slice).collect();
        let list = ClusterLightList::from_per_cluster(&refs);

        let normal = Vec3::new(0.0, 0.0, -1.0); // faces toward light 0 (in front)
        let out = accumulate_lighting(&grid, &list, &lights, particle, normal);
        assert!(out.x > 0.0);

        // The sum must equal the two contributions computed independently.
        let manual = shade_point_light(lights[0], particle, normal)
            .add(shade_point_light(lights[1], particle, normal));
        assert!(approx_vec(out, manual));
    }

    #[test]
    fn accumulate_empty_cluster_returns_zero() {
        let grid = unit_grid();
        let particle = Vec3::new(0.0, 0.0, 1.5);
        // Correctly sized list, but every cluster is empty.
        let list = ClusterLightList::empty(grid.cluster_count());
        let lights = vec![PunctualLight::point(Vec3::ZERO, Vec3::splat(9.0), 20.0)];
        let out = accumulate_lighting(&grid, &list, &lights, particle, Vec3::new(0.0, 0.0, -1.0));
        assert!(approx_vec(out, Vec3::ZERO));
    }

    #[test]
    fn accumulate_out_of_frustum_returns_zero() {
        let grid = unit_grid();
        let list = ClusterLightList::empty(grid.cluster_count());
        let lights = vec![PunctualLight::point(Vec3::ZERO, Vec3::splat(9.0), 20.0)];
        // Behind the near plane -> no cluster -> zero.
        let behind = Vec3::new(0.0, 0.0, 0.1);
        assert!(approx_vec(
            accumulate_lighting(&grid, &list, &lights, behind, Vec3::new(0.0, 0.0, -1.0)),
            Vec3::ZERO
        ));
    }

    #[test]
    fn accumulate_ignores_out_of_range_light_index() {
        let grid = unit_grid();
        let particle = Vec3::new(0.0, 0.0, 1.5);
        let cluster = grid.cluster_index(particle).unwrap();
        // Reference a light index (5) that does not exist in the table.
        let mut per_cluster: Vec<Vec<u32>> = Vec::new();
        per_cluster.resize(grid.cluster_count() as usize, Vec::new());
        per_cluster[cluster as usize] = vec![5];
        let refs: Vec<&[u32]> = per_cluster.iter().map(Vec::as_slice).collect();
        let list = ClusterLightList::from_per_cluster(&refs);
        // Empty light table -> the dangling index is skipped, no panic.
        let lights: Vec<PunctualLight> = Vec::new();
        let out = accumulate_lighting(&grid, &list, &lights, particle, Vec3::new(0.0, 0.0, -1.0));
        assert!(approx_vec(out, Vec3::ZERO));
    }

    #[test]
    fn no_nan_from_coincident_light_and_surface() {
        // Light exactly on the surface point -> direction undefined; must be zero.
        let light = PunctualLight::point(Vec3::new(1.0, 1.0, 1.0), Vec3::splat(5.0), 10.0);
        let out = shade_point_light(light, Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(approx_vec(out, Vec3::ZERO));
        assert!(!out.x.is_nan() && !out.y.is_nan() && !out.z.is_nan());
    }

    #[test]
    fn zero_normal_is_unlit_not_nan() {
        let light = PunctualLight::point(Vec3::ZERO, Vec3::splat(5.0), 10.0);
        let out = shade_point_light(light, Vec3::new(0.0, 0.0, -2.0), Vec3::ZERO);
        assert!(approx_vec(out, Vec3::ZERO));
    }
}
