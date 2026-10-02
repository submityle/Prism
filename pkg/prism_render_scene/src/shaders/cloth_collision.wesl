//! GPU compute twins of the cloth body-collision and self-collision passes.
//!
//! Compute twins of the CPU golden reference
//! `prism_render_architecture::cloth::collision` — `resolve_body_collisions`
//! (analytic body proxies) and `resolve_self_collision` (uniform spatial hash).
//! These are the `ClothKernel::BodyCollision`, `ClothKernel::SelfCollisionHashBuild`
//! and `ClothKernel::SelfCollisionResolve` dispatches; the stable `WESL` entry
//! point names emitted by `prism_render_architecture::cloth::gpu::kernels` are
//! `cloth_body_collision`, `cloth_self_collision_hash_build`,
//! `cloth_self_collision_resolve` and `cloth_backstop`, matched byte-for-byte by
//! the entry points below. Their bind-group layouts differ from the solver kernels in
//! `cloth_sim.wesl`, which is why they live in this sibling module.
//!
//! The sandbox has no `GPU`, so this shader is validated by naga compilation
//! only (via the resolve `shader_tests`); no dispatch is executed here.
//!
//! ## Determinism note (honest `GPU` trade-off)
//!
//! The CPU `resolve_self_collision` buckets particles into an ordered
//! `BTreeMap`, walks cells and buckets in ascending order, and applies each
//! pair correction in place (Gauss-Seidel). That ordering makes the CPU pass
//! bit-reproducible. A massively parallel `GPU` pass cannot reproduce that exact
//! sequential order: the hash build uses `atomicExchange` (whose interleaving is
//! unspecified) and the resolve step reads neighbour positions that peer
//! invocations may already be updating. This kernel therefore implements a
//! **Jacobi-style parallel approximation** — every particle accumulates its own
//! separation from all in-range neighbours and writes only its own slot — which
//! is the standard production `GPU` self-collision model (`Chaos` Cloth,
//! `NvCloth`) rather than a bit-parity mirror of the CPU Gauss-Seidel pass. The
//! body-collision kernel, by contrast, *is* a bit-parity mirror: each particle's
//! collider projection is independent, so parallelism changes nothing.
//!
//! ## Binding budget
//!
//! `cloth_body_collision` now uses a 3-storage + 1-uniform layout
//! (`positions`, `colliders`, `prev_positions`, `params`): the `prev_positions`
//! buffer (binding 3) supplies the frame-start reference the tangential Coulomb
//! friction pass measures slip against, matching the CPU
//! `resolve_body_collisions_with_friction`. The host descriptor
//! `ClothKernel::BodyCollision` in `prism_render_architecture::cloth::gpu::kernels`
//! now declares `storage_buffers: 3` and `GpuClothBodyParams` carries the
//! `friction` coefficient, so the host binds binding 3 and drives the uniform
//! directly: a non-zero `friction` reproduces the CPU Coulomb slip damping
//! bit-for-bit, and `friction == 0` collapses to the frictionless path. Per-particle painted backstops
//! (`collision::apply_backstop`) run as their own `ClothKernel::Backstop`
//! dispatch (`cloth_backstop` below) on a matching 2-storage + 1-uniform layout
//! (`positions`, `backstops`, `params`), so they are a real bit-parity kernel
//! rather than being fabricated into the body-collision budget.
//! `cloth_self_collision_hash_build` / `_resolve` honour the 3-storage +
//! 1-uniform contract (`positions`, `cell_table`, `particle_next`, `params`).

/// Squared-length floor below which two points are treated as coincident, so a
/// separating direction is never normalised from ~0. Mirrors the CPU
/// `EPS_LEN_SQ` shared across the cloth subsystem.
const CLOTH_COL_EPS_LEN_SQ: f32 = 1.0e-12;

/// Linked-list terminator for the spatial-hash cell chains: an index this large
/// can never be a valid particle, so it marks "end of chain" / "empty cell".
const CLOTH_COL_SENTINEL: u32 = 0xffffffffu;

/// Analytic body-collider tag: solid sphere. Mirrors `BodyCollider::Sphere`.
const CLOTH_COLLIDER_SPHERE: u32 = 0u;
/// Analytic body-collider tag: solid capsule. Mirrors `BodyCollider::Capsule`.
const CLOTH_COLLIDER_CAPSULE: u32 = 1u;
/// Analytic body-collider tag: half-space. Mirrors `BodyCollider::HalfSpace`.
const CLOTH_COLLIDER_HALF_SPACE: u32 = 2u;

/// One analytic body-collision proxy. A tagged union mirroring the CPU
/// `prism_render_architecture::cloth::collision::BodyCollider`; `kind` selects
/// how the remaining fields are read:
///
/// - `Sphere`   : `a` = centre, `radius` = radius.
/// - `Capsule`  : `a` = segment endpoint `p0`, `b` = segment endpoint `p1`,
///   `radius` = inflation radius.
/// - `HalfSpace`: `a` = plane normal (need not be unit), `radius` = plane
///   offset measured against the normal.
///
/// Scalar `f32` fields (rather than `vec3`) keep the record at a flat 32-byte
/// `std430` stride with no interior padding surprises.
struct ClothCollider {
    /// Selects the collider variant (`CLOTH_COLLIDER_*`).
    kind: u32,
    /// First point: sphere centre / capsule `p0` / half-space normal `x`.
    ax: f32,
    /// First point `y`.
    ay: f32,
    /// First point `z`.
    az: f32,
    /// Second point (capsule `p1` only, else ignored) `x`.
    bx: f32,
    /// Second point `y`.
    by: f32,
    /// Second point `z`.
    bz: f32,
    /// Radius (sphere / capsule) or signed offset (half-space).
    radius: f32,
}

/// One spatial-hash cell header. `head` is the index of the first particle in
/// the cell's linked list (`CLOTH_COL_SENTINEL` = empty); `count` is the cell
/// occupancy, maintained for diagnostics. Two `u32` = the 8-byte
/// `prism_render_architecture::cloth::gpu::buffers::HASH_CELL_STRIDE`. (The host
/// `start offset + count` phrasing describes a counting-sort compaction; a true
/// prefix-sum compaction would need an extra scan kernel that the current
/// `ClothKernel` set does not declare, so the single available build kernel uses
/// an atomic linked-list build instead — `head` reuses the first word.)
struct ClothHashCell {
    /// Index of the first particle chained into this cell, or the sentinel.
    head: atomic<u32>,
    /// Number of particles chained into this cell.
    count: atomic<u32>,
}

/// Uniform parameters for the body-collision dispatch.
struct ClothBodyParams {
    /// Number of particles bounding the per-particle dispatch.
    particle_count: u32,
    /// Number of colliders every free particle is projected against, in order.
    collider_count: u32,
    /// Cloth-side Coulomb friction coefficient (`mu`, clamped to `0..=1` on
    /// read). Mirrors the CPU `resolve_body_collisions_with_friction` `friction`
    /// argument sourced from `FabricMaterial::friction`. A host that has not yet
    /// wired friction leaves this zero, which is exactly frictionless.
    friction: f32,
    /// Trailing pad word so the record keeps a flat 16-byte stride.
    pad: f32,
}

/// Uniform parameters for the self-collision hash build and resolve dispatches.
struct ClothSelfParams {
    /// Number of particles bounding the per-particle dispatches.
    particle_count: u32,
    /// Number of hash buckets in `cell_table` (the modulus for `cell_hash`).
    table_size: u32,
    /// Uniform grid cell edge, world units (assumed `> 0`; matches CPU
    /// `cell_size`).
    cell_size: f32,
    /// Separation distance below which a particle pair is pushed apart
    /// (matches CPU `thickness`).
    thickness: f32,
}

// ---------------------------------------------------------------------------
// Body collision (`cloth_body_collision`)
// ---------------------------------------------------------------------------

// Particle state: `xyz` = world position, `w` = inverse mass (`<= 0` = pinned).
@group(0) @binding(0) var<storage, read_write> body_positions: array<vec4<f32>>;
// Analytic colliders applied, in array order, to every free particle.
@group(0) @binding(1) var<storage, read> colliders: array<ClothCollider>;
// Body-collision dispatch bounds.
@group(0) @binding(2) var<uniform> body_params: ClothBodyParams;
// Frame-start positions (`xyz`; `w` ignored) the friction pass measures each
// particle's tangential slip against. Mirrors the CPU `prev_positions` slice.
@group(0) @binding(3) var<storage, read> body_prev_positions: array<vec4<f32>>;

/// Projects `pos` out to the surface of the sphere `(center, radius)`. Mirrors
/// the CPU `project_out_of_sphere`: a non-positive radius or an exterior point
/// is untouched, and a point coincident with the centre escapes along `+Y` for
/// a deterministic, non-`NaN` result.
fn cloth_project_out_of_sphere(pos: vec3<f32>, center: vec3<f32>, radius: f32) -> vec3<f32> {
    if radius <= 0.0 {
        return pos;
    }
    let delta = pos - center;
    let dist_sq = dot(delta, delta);
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= CLOTH_COL_EPS_LEN_SQ {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = delta * (1.0 / sqrt(dist_sq));
    return center + dir * radius;
}

/// Projects `pos` onto the plane `dot(normal, x) == offset` when it lies on the
/// infeasible side (`dot(normal, pos) < offset`). Mirrors the CPU
/// `project_out_of_half_space`: the normal need not be unit (the correction
/// divides by its squared length) and a (near) zero normal is inert.
fn cloth_project_out_of_half_space(pos: vec3<f32>, normal: vec3<f32>, offset: f32) -> vec3<f32> {
    let len_sq = dot(normal, normal);
    if len_sq <= CLOTH_COL_EPS_LEN_SQ {
        return pos;
    }
    let signed = dot(normal, pos) - offset;
    if signed >= 0.0 {
        return pos;
    }
    let t = -signed / len_sq;
    return pos + normal * t;
}

/// Returns the point on segment `p0`..`p1` closest to `pos`. Mirrors the CPU
/// `closest_point_on_segment`: the parameter is clamped to `[0, 1]` (turning
/// capsule caps into hemispheres) and a zero-length segment degenerates to
/// `p0`.
fn cloth_closest_point_on_segment(p0: vec3<f32>, p1: vec3<f32>, pos: vec3<f32>) -> vec3<f32> {
    let axis = p1 - p0;
    let len_sq = dot(axis, axis);
    if len_sq <= CLOTH_COL_EPS_LEN_SQ {
        return p0;
    }
    let t = clamp(dot(pos - p0, axis) / len_sq, 0.0, 1.0);
    return p0 + axis * t;
}

/// Projects `pos` out of one collider. Mirrors `BodyCollider::project`: the
/// capsule reduces to a sphere about the closest point on its segment axis.
fn cloth_project_collider(col: ClothCollider, pos: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(col.ax, col.ay, col.az);
    if col.kind == CLOTH_COLLIDER_SPHERE {
        return cloth_project_out_of_sphere(pos, a, col.radius);
    }
    if col.kind == CLOTH_COLLIDER_CAPSULE {
        let b = vec3<f32>(col.bx, col.by, col.bz);
        let closest = cloth_closest_point_on_segment(a, b, pos);
        return cloth_project_out_of_sphere(pos, closest, col.radius);
    }
    // Half-space: `a` is the normal, `radius` is the offset.
    return cloth_project_out_of_half_space(pos, a, col.radius);
}

/// Removes up to `friction * depth` of tangential slip from a projected
/// particle. Bit-parity mirror of the CPU `damp_tangential_slip`: given the
/// frame-start position `prev`, the post-projection position `projected`, and
/// the collision `correction` (`projected - pre_projection`, whose length is the
/// penetration depth and whose direction is the outward contact normal), it
/// splits the frame displacement `projected - prev` against that normal and
/// shortens the tangential part by `min(||tangential||, friction * depth)`.
/// Static friction (slip `<= friction * depth`) sticks; dynamic friction caps
/// the removed slip linearly. A non-positive `friction`, a (near) zero depth, or
/// a (near) zero tangential slip returns `projected` untouched, and every
/// normalisation divides by a magnitude already checked against
/// `CLOTH_COL_EPS_LEN_SQ`, so the result is always finite.
fn cloth_damp_tangential_slip(
    prev: vec3<f32>,
    projected: vec3<f32>,
    correction: vec3<f32>,
    friction: f32,
) -> vec3<f32> {
    if friction <= 0.0 {
        return projected;
    }
    let depth_sq = dot(correction, correction);
    if depth_sq <= CLOTH_COL_EPS_LEN_SQ {
        return projected;
    }
    let depth = sqrt(depth_sq);
    let normal = correction * (1.0 / depth);
    let disp = projected - prev;
    let normal_amount = dot(disp, normal);
    let tangential = disp - normal * normal_amount;
    let tan_sq = dot(tangential, tangential);
    if tan_sq <= CLOTH_COL_EPS_LEN_SQ {
        return projected;
    }
    let tan_len = sqrt(tan_sq);
    // scale = min(mu * ||push|| / ||tangent||, 1.0): saturates at 1 inside the
    // static cone (full cancellation) and shrinks the slide by mu*push in the
    // dynamic regime. Same op order as the golden apply_coulomb_friction's
    // `(mu * normal_push / tan_len).min(1.0)` then `tangent.scale(scale)`, so
    // the GPU friction matches the CPU golden bit-for-bit (one fmul, one fdiv,
    // one min, one fmul) instead of the earlier reciprocal-mul reformulation.
    let scale = min(friction * depth / tan_len, 1.0);
    return projected - tangential * scale;
}

/// Body-collision stage: one invocation per particle projects a free particle
/// out of every collider in order, damping its tangential slip against each
/// contact with Coulomb friction as it goes. Bit-parity mirror of the CPU
/// `resolve_body_collisions_with_friction`: friction is applied per collider
/// (the projected position after collider `i` feeds collider `i + 1`), and the
/// tangential slide is always measured against the frame-start `prev` position,
/// not the running position, so the multi-collider result matches the CPU
/// golden exactly rather than only for a single contact. Pinned particles are
/// skipped, an empty collider set is a no-op, and `friction` is clamped to
/// `0..=1` with `0` exactly frictionless (`cloth_damp_tangential_slip` returns
/// the raw projection unchanged, so the frictionless path is bit-identical to a
/// plain sequential projection).
@compute @workgroup_size(64)
fn cloth_body_collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= body_params.particle_count {
        return;
    }
    // Pinned particles (inverse mass `<= 0`) never move.
    if body_positions[p].w <= 0.0 {
        return;
    }
    let mu = clamp(body_params.friction, 0.0, 1.0);
    let prev = body_prev_positions[p].xyz;
    var pos = body_positions[p].xyz;
    for (var i = 0u; i < body_params.collider_count; i = i + 1u) {
        let before = pos;
        let projected = cloth_project_collider(colliders[i], before);
        // Per-collider correction: length = this collider's push-out depth,
        // direction = its outward contact normal. Damp the frame's tangential
        // slip (measured from the frame-start `prev`) against each contact in
        // turn, matching the CPU golden's per-collider friction application.
        let correction = projected - before;
        pos = cloth_damp_tangential_slip(prev, projected, correction, mu);
    }
    body_positions[p] = vec4<f32>(pos, body_positions[p].w);
}

// ---------------------------------------------------------------------------
// Self collision (`cloth_self_collision_hash_build` / `_resolve`)
// ---------------------------------------------------------------------------

// Particle state shared by both self-collision kernels.
@group(0) @binding(0) var<storage, read_write> self_positions: array<vec4<f32>>;
// Per-bucket linked-list heads + occupancy counts.
@group(0) @binding(1) var<storage, read_write> cell_table: array<ClothHashCell>;
// Per-particle "next in chain" pointer, forming each bucket's linked list.
@group(0) @binding(2) var<storage, read_write> particle_next: array<u32>;
// Self-collision dispatch bounds and grid parameters.
@group(0) @binding(3) var<uniform> self_params: ClothSelfParams;

/// Maps a world position to its integer spatial-hash cell. Mirrors the CPU
/// `cell_of` (`floor(pos / cell_size)`); the `i32` cast is not saturating on the
/// `GPU`, but the subsequent hash folds any value into `[0, table_size)`.
fn cloth_cell_of(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let inv = 1.0 / cell_size;
    return vec3<i32>(floor(pos * inv));
}

/// Folds an integer cell coordinate into a bucket index in `[0, table_size)`.
/// A large-prime multiply-xor spatial hash (the standard uniform-grid hash used
/// by `Chaos` Cloth / `NvCloth`); collisions only cost redundant distance tests,
/// which the exact-cell recheck in the resolve pass then rejects.
fn cloth_cell_hash(cell: vec3<i32>, table_size: u32) -> u32 {
    let x = bitcast<u32>(cell.x) * 73856093u;
    let y = bitcast<u32>(cell.y) * 19349663u;
    let z = bitcast<u32>(cell.z) * 83492791u;
    return (x ^ y ^ z) % table_size;
}

/// Hash-build stage: one invocation per particle atomically prepends the
/// particle to its cell's linked list. This is a flat per-particle launch
/// (`@workgroup_size(64)`, one invocation per `gid.x`), matching the
/// `ClothKernel::SelfCollisionHashBuild` contract's `DispatchDomain::Particle`
/// shape — a multi-axis tile would map several invocations onto the same
/// `gid.x` and insert each particle into its chain more than once. The host
/// must clear `cell_table` heads to `CLOTH_COL_SENTINEL` (a `0xff` byte-fill)
/// and reset the counts before this dispatch, mirroring the CPU rebuild of a
/// fresh `BTreeMap` each call.
@compute @workgroup_size(64)
fn cloth_self_collision_hash_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One invocation per particle: the linear `gid.x` is the particle index.
    let p = gid.x;
    if p >= self_params.particle_count {
        return;
    }
    let cell = cloth_cell_of(self_positions[p].xyz, self_params.cell_size);
    let bucket = cloth_cell_hash(cell, self_params.table_size);
    // Prepend: `next[p]` takes the old head, the head becomes `p`.
    let prev_head = atomicExchange(&cell_table[bucket].head, p);
    particle_next[p] = prev_head;
    atomicAdd(&cell_table[bucket].count, 1u);
}

/// Resolve stage: one invocation per particle accumulates its separation from
/// every in-range neighbour across the 27-cell neighbourhood and writes only
/// its own slot (Jacobi). Mirrors the CPU `resolve_pair` separation law — the
/// inverse-mass split `w_self / (w_self + w_other)`, the `thickness` target, and
/// the index-signed `X`-axis fallback for coincident points — but applies it in parallel
/// (see the module determinism note). Pinned neighbours (`w_other == 0`) push a
/// free particle by the full penetration; a pinned self never moves.
@compute @workgroup_size(64)
fn cloth_self_collision_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= self_params.particle_count {
        return;
    }
    let pos_self = self_positions[p].xyz;
    let w_self = max(self_positions[p].w, 0.0);
    // A pinned particle cannot move; skip the neighbour walk entirely.
    if w_self <= 0.0 {
        return;
    }

    let thickness = self_params.thickness;
    let thickness_sq = thickness * thickness;
    let base_cell = cloth_cell_of(pos_self, self_params.cell_size);
    var delta_sum = vec3<f32>(0.0, 0.0, 0.0);

    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = base_cell + vec3<i32>(dx, dy, dz);
                let bucket = cloth_cell_hash(neighbor_cell, self_params.table_size);
                var j = atomicLoad(&cell_table[bucket].head);
                // Walk the bucket's linked list; a bounded step count guards
                // against a corrupt (cyclic) chain so the loop always halts.
                for (var step = 0u; step < self_params.particle_count; step = step + 1u) {
                    if j == CLOTH_COL_SENTINEL {
                        break;
                    }
                    let cur = j;
                    j = particle_next[cur];
                    if cur == p {
                        continue;
                    }
                    // Hash collisions can chain foreign cells into this bucket;
                    // only act on neighbours whose true cell is the one we are
                    // visiting, so each pair is counted at most once per cell.
                    let other_pos = self_positions[cur].xyz;
                    let other_cell = cloth_cell_of(other_pos, self_params.cell_size);
                    if other_cell.x != neighbor_cell.x
                        || other_cell.y != neighbor_cell.y
                        || other_cell.z != neighbor_cell.z {
                        continue;
                    }
                    let diff = pos_self - other_pos;
                    let dist_sq = dot(diff, diff);
                    if dist_sq >= thickness_sq {
                        continue;
                    }
                    let w_other = max(self_positions[cur].w, 0.0);
                    let w_sum = w_self + w_other;
                    if w_sum <= 0.0 {
                        continue;
                    }
                    let share = w_self / w_sum;
                    if dist_sq <= CLOTH_COL_EPS_LEN_SQ {
                        // Coincident: no separating direction is defined, so
                        // split along the fixed `X` axis exactly like the CPU
                        // golden `resolve_pair`. There the lower-indexed
                        // particle takes the `-X` half and the higher-indexed
                        // one the `+X` half of the full-`thickness` push; this
                        // Jacobi pass reproduces that per particle by signing
                        // the axis on the index comparison, so two coincident
                        // free particles separate symmetrically instead of both
                        // drifting the same way and staying coincident.
                        var axis_sign = 1.0;
                        if p < cur {
                            axis_sign = -1.0;
                        }
                        delta_sum = delta_sum
                            + vec3<f32>(axis_sign * thickness * share, 0.0, 0.0);
                    } else {
                        let dist = sqrt(dist_sq);
                        let penetration = thickness - dist;
                        let dir = diff / dist;
                        delta_sum = delta_sum + dir * (penetration * share);
                    }
                }
            }
        }
    }

    self_positions[p] = vec4<f32>(pos_self + delta_sum, self_positions[p].w);
}

// ---------------------------------------------------------------------------
// Painted backstop (`cloth_backstop`)
// ---------------------------------------------------------------------------

/// One painted-backstop plane. A bit-parity twin of the CPU golden
/// `prism_render_architecture::cloth::collision::Backstop`, packed into two
/// 16-byte `std430` slots so the record stride matches the host `BACKSTOP_STRIDE`
/// (32 bytes): slot 0 is `origin.xyz` plus the scalar `distance`, slot 1 is
/// `normal.xyz` plus one pad word.
struct ClothBackstop {
    /// Anchor point on the skinned surface (`x`).
    ox: f32,
    /// Anchor `y`.
    oy: f32,
    /// Anchor `z`.
    oz: f32,
    /// How far behind `origin` (along `-normal`) the particle may travel before
    /// it is pushed back onto the limiting plane.
    distance: f32,
    /// Outward plane normal (need not be unit) `x`.
    nx: f32,
    /// Normal `y`.
    ny: f32,
    /// Normal `z`.
    nz: f32,
    /// Trailing pad word so the record occupies exactly two `vec4` slots.
    pad: f32,
}

/// Uniform parameters for the backstop dispatch.
struct ClothBackstopParams {
    /// Number of particles (and paired backstop records) the pass runs over.
    particle_count: u32,
}

// Particle state: `xyz` = world position, `w` = inverse mass (`<= 0` = pinned).
@group(0) @binding(0) var<storage, read_write> backstop_positions: array<vec4<f32>>;
// `backstops[i]` constrains particle `i`.
@group(0) @binding(1) var<storage, read> backstops: array<ClothBackstop>;
// Backstop dispatch bounds.
@group(0) @binding(2) var<uniform> backstop_params: ClothBackstopParams;

/// Clamps `pos` to the front side of one backstop plane. Bit-parity mirror of
/// the CPU `apply_backstop`: a (near) zero normal is inert (no defined plane),
/// and a particle at signed distance `s < -distance` is pushed forward along
/// the unit normal onto the limiting plane; otherwise `pos` is untouched.
fn cloth_apply_backstop(pos: vec3<f32>, b: ClothBackstop) -> vec3<f32> {
    let normal = vec3<f32>(b.nx, b.ny, b.nz);
    let len_sq = dot(normal, normal);
    if len_sq <= CLOTH_COL_EPS_LEN_SQ {
        return pos;
    }
    let n = normal * (1.0 / sqrt(len_sq));
    let origin = vec3<f32>(b.ox, b.oy, b.oz);
    let s = dot(n, pos - origin);
    let min_s = -b.distance;
    if s < min_s {
        return pos + n * (min_s - s);
    }
    return pos;
}

/// Backstop stage: one invocation per particle pushes a free particle onto the
/// front side of its painted backstop plane. Mirrors the CPU
/// `resolve_backstops` (pinned particles are skipped; the pass is a no-op when
/// no backstop records are resident).
@compute @workgroup_size(64)
fn cloth_backstop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= backstop_params.particle_count {
        return;
    }
    // Pinned particles (inverse mass `<= 0`) never move.
    if backstop_positions[p].w <= 0.0 {
        return;
    }
    let pos = backstop_positions[p].xyz;
    let clamped = cloth_apply_backstop(pos, backstops[p]);
    backstop_positions[p] = vec4<f32>(clamped, backstop_positions[p].w);
}
