//! Quadric-error-metric (`QEM`) edge-collapse mesh decimation for the `CPU`
//! golden path.
//!
//! Shipping a mesh at a single resolution wastes both memory and shading work:
//! a hero asset authored at a million triangles should degrade gracefully to a
//! few thousand when it covers a handful of pixels. The classic automatic way
//! to build those lower levels of detail is Garland & Heckbert's *quadric error
//! metric* edge collapse (SIGGRAPH '97): repeatedly merge the vertex pair whose
//! removal perturbs the surface least, measured by a per-vertex quadratic form
//! that accumulates the squared distance to every face plane incident on that
//! vertex.
//!
//! ## The quadric
//!
//! Each triangle defines a plane `n · x + d = 0` with a unit normal `n` and
//! `d = -n · p0`. The squared distance of a point `x` to that plane is
//! `(n · x + d)^2`, which expands to the homogeneous quadratic form
//! `vᵀ K v` with `v = [x, y, z, 1]` and `K = [a b c d]ᵀ [a b c d]` (the `4 × 4`
//! outer product of the plane coefficients). A vertex's quadric is the sum of
//! the `K` of every face touching it, so `vᵀ Q v` reads back the total squared
//! planar error of placing that vertex at `v`. Summing two endpoints' quadrics
//! gives the error of the merged vertex, and minimizing it is a `3 × 3` linear
//! solve.
//!
//! ## The greedy loop
//!
//! Every surviving edge is scored by the error at its *optimal* merged
//! position and pushed onto a min-heap. The cheapest edge is collapsed, the
//! surviving vertex inherits the summed quadric and the solved position, the
//! other endpoint is retired, degenerate faces are dropped, and the affected
//! edges are rescored and re-pushed. Stale heap entries are discarded lazily by
//! comparing a per-vertex version stamp, so no priority-decrease bookkeeping is
//! needed. The loop stops once the live triangle count reaches the target or no
//! legal collapse remains.
//!
//! ## Boundary preservation
//!
//! Open boundary edges would otherwise shrink inward, because only one face
//! constrains them. Following Garland & Heckbert, each boundary edge adds a
//! heavily weighted *virtual* plane perpendicular to its single incident face
//! and passing through the edge; this pins the silhouette so holes and open
//! borders keep their shape as the interior simplifies.
//!
//! ## Numerics
//!
//! The accumulation and the linear solve run in `f64` to resist the
//! catastrophic cancellation that `f32` quadric sums are prone to; only the
//! final positions are narrowed back to `f32`. The whole module stays within
//! the golden-path float policy — the only non-arithmetic operation is the
//! `sqrt` used to normalize plane normals, so there are no transcendental
//! calls.

use alloc::collections::BinaryHeap;
use core::cmp::Reverse;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Failure modes of [`decimate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimationError {
    /// The requested triangle target was zero; a mesh must keep at least one
    /// triangle to remain a surface.
    ZeroTarget,
    /// Rebuilding the simplified [`TriangleMesh`] failed. Carries the
    /// propagated [`TriangleMeshError`]; by construction the compacted pools
    /// are valid, so this is not expected in practice.
    Rebuild(TriangleMeshError),
}

impl core::fmt::Display for DecimationError {
    /// Formats the error for human-readable diagnostics.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroTarget => {
                write!(f, "decimation target must be at least one triangle")
            }
            Self::Rebuild(err) => {
                write!(f, "failed to rebuild decimated mesh: {err}")
            }
        }
    }
}

impl std::error::Error for DecimationError {}

/// Weight applied to the virtual boundary-preserving planes, relative to the
/// unit face planes. A large multiplier makes moving a boundary vertex off its
/// silhouette dramatically more expensive than ordinary interior error, so
/// open borders stay put until the interior is exhausted.
const BOUNDARY_WEIGHT: f64 = 1_000.0;

/// Threshold below which the `3 × 3` quadric system is treated as singular and
/// the solver falls back to sampling the edge endpoints and midpoint. Scaled by
/// the matrix magnitude at the call site, this is a relative tolerance.
const SINGULAR_EPSILON: f64 = 1.0e-12;

/// A symmetric `4 × 4` error quadric stored as its ten unique upper-triangle
/// entries, indexed as the outer product of the plane coefficients
/// `[a, b, c, d]`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Quadric {
    /// `a·a` — the `(1, 1)` entry.
    aa: f64,
    /// `a·b` — the `(1, 2)` entry.
    ab: f64,
    /// `a·c` — the `(1, 3)` entry.
    ac: f64,
    /// `a·d` — the `(1, 4)` entry.
    ad: f64,
    /// `b·b` — the `(2, 2)` entry.
    bb: f64,
    /// `b·c` — the `(2, 3)` entry.
    bc: f64,
    /// `b·d` — the `(2, 4)` entry.
    bd: f64,
    /// `c·c` — the `(3, 3)` entry.
    cc: f64,
    /// `c·d` — the `(3, 4)` entry.
    cd: f64,
    /// `d·d` — the `(4, 4)` entry.
    dd: f64,
}

impl Quadric {
    /// The all-zero quadric, the additive identity for accumulation.
    const ZERO: Self = Self {
        aa: 0.0,
        ab: 0.0,
        ac: 0.0,
        ad: 0.0,
        bb: 0.0,
        bc: 0.0,
        bd: 0.0,
        cc: 0.0,
        cd: 0.0,
        dd: 0.0,
    };

    /// Builds the quadric `weight · ([a b c d]ᵀ [a b c d])` from the plane
    /// coefficients `(a, b, c, d)`.
    fn from_plane(a: f64, b: f64, c: f64, d: f64, weight: f64) -> Self {
        Self {
            aa: weight * a * a,
            ab: weight * a * b,
            ac: weight * a * c,
            ad: weight * a * d,
            bb: weight * b * b,
            bc: weight * b * c,
            bd: weight * b * d,
            cc: weight * c * c,
            cd: weight * c * d,
            dd: weight * d * d,
        }
    }

    /// Adds `other` into `self` entry-wise (quadrics compose by summation).
    fn add_assign(&mut self, other: &Self) {
        self.aa += other.aa;
        self.ab += other.ab;
        self.ac += other.ac;
        self.ad += other.ad;
        self.bb += other.bb;
        self.bc += other.bc;
        self.bd += other.bd;
        self.cc += other.cc;
        self.cd += other.cd;
        self.dd += other.dd;
    }

    /// Evaluates the quadratic form `vᵀ Q v` at the point `v = [x, y, z, 1]`,
    /// i.e. the accumulated squared planar error of placing a vertex there.
    fn error_at(&self, p: [f64; 3]) -> f64 {
        let [x, y, z] = p;
        self.aa * x * x
            + 2.0 * self.ab * x * y
            + 2.0 * self.ac * x * z
            + 2.0 * self.ad * x
            + self.bb * y * y
            + 2.0 * self.bc * y * z
            + 2.0 * self.bd * y
            + self.cc * z * z
            + 2.0 * self.cd * z
            + self.dd
    }

    /// Solves for the error-minimizing position by inverting the leading
    /// `3 × 3` block against the negated `[ad, bd, cd]` column.
    ///
    /// Returns `None` when that block is numerically singular (a flat or
    /// symmetric neighbourhood with no unique minimizer), signalling the caller
    /// to fall back to sampling the endpoints and midpoint.
    fn optimal_position(&self) -> Option<[f64; 3]> {
        // Rows of the symmetric 3x3 block.
        let m = [
            [self.aa, self.ab, self.ac],
            [self.ab, self.bb, self.bc],
            [self.ac, self.bc, self.cc],
        ];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let scale = self.aa.abs() + self.bb.abs() + self.cc.abs() + 1.0;
        if det.abs() <= SINGULAR_EPSILON * scale * scale * scale {
            return None;
        }
        // Right-hand side is the negated fourth column of the quadric.
        let rhs = [-self.ad, -self.bd, -self.cd];
        let inv_det = 1.0 / det;
        // Cramer's rule: replace each column with `rhs` and divide by `det`.
        let x = (rhs[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (rhs[1] * m[2][2] - m[1][2] * rhs[2])
            + m[0][2] * (rhs[1] * m[2][1] - m[1][1] * rhs[2]))
            * inv_det;
        let y = (m[0][0] * (rhs[1] * m[2][2] - m[1][2] * rhs[2])
            - rhs[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * rhs[2] - rhs[1] * m[2][0]))
            * inv_det;
        let z = (m[0][0] * (m[1][1] * rhs[2] - rhs[1] * m[2][1])
            - m[0][1] * (m[1][0] * rhs[2] - rhs[1] * m[2][0])
            + rhs[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            * inv_det;
        Some([x, y, z])
    }
}

/// A pending edge collapse queued on the priority heap.
#[derive(Clone, Copy, Debug)]
struct Collapse {
    /// Error at the solved merged position; the heap orders by this ascending.
    cost: f64,
    /// Surviving vertex index (the merged vertex keeps this slot).
    keep: u32,
    /// Retired vertex index (folded into `keep`).
    drop: u32,
    /// Version stamp of `keep` when this entry was created; a mismatch on pop
    /// means the entry is stale.
    keep_version: u32,
    /// Version stamp of `drop` when this entry was created.
    drop_version: u32,
    /// The solved merged position, carried so pops avoid re-solving.
    target: [f64; 3],
}

impl PartialEq for Collapse {
    /// Equality compares only the ordering key (the cost bits).
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
    }
}

impl Eq for Collapse {}

impl PartialOrd for Collapse {
    /// Delegates to the total order over `cost`.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Collapse {
    /// Orders by `cost` using the IEEE total order so `NaN` never corrupts the
    /// heap; wrapped in [`Reverse`] at push time to make the heap a min-heap.
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cost.total_cmp(&other.cost)
    }
}

/// Mutable working state of one decimation run.
struct Decimator {
    /// Vertex positions in `f64`, indexed by vertex id.
    positions: Vec<[f64; 3]>,
    /// Optional per-vertex normals (empty when the input had none).
    normals: Vec<[f64; 3]>,
    /// Optional per-vertex `UV`s (empty when the input had none).
    uvs: Vec<[f64; 2]>,
    /// Accumulated error quadric per vertex.
    quadrics: Vec<Quadric>,
    /// `true` once a vertex has been folded into another and retired.
    removed_vertex: Vec<bool>,
    /// Monotonic version stamp per vertex; bumped whenever a vertex's quadric
    /// or position changes so stale heap entries can be detected.
    version: Vec<u32>,
    /// Face vertex triples; retired faces are marked in `removed_face`.
    faces: Vec<[u32; 3]>,
    /// `true` once a face has collapsed to zero area and been dropped.
    removed_face: Vec<bool>,
    /// For each vertex, the ids of the faces that currently reference it.
    incident: Vec<Vec<u32>>,
    /// Count of faces still live, used for the termination test.
    live_faces: usize,
}

impl Decimator {
    /// Builds the working state from an input mesh: narrows attributes to
    /// `f64`, seeds per-face quadrics (plus boundary planes), and records the
    /// vertex→face adjacency.
    fn new(mesh: &TriangleMesh) -> Self {
        let positions: Vec<[f64; 3]> = mesh
            .positions()
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let normals: Vec<[f64; 3]> = mesh
            .normals()
            .iter()
            .map(|n| [n[0] as f64, n[1] as f64, n[2] as f64])
            .collect();
        let uvs: Vec<[f64; 2]> = mesh
            .uvs()
            .iter()
            .map(|t| [t[0] as f64, t[1] as f64])
            .collect();
        let faces: Vec<[u32; 3]> = mesh.indices().to_vec();

        let vertex_count = positions.len();
        let mut quadrics = vec![Quadric::ZERO; vertex_count];
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
        let mut removed_face = vec![false; faces.len()];
        let mut live_faces = 0usize;

        for (face_id, tri) in faces.iter().enumerate() {
            let [i0, i1, i2] = *tri;
            // Guard against pre-existing degenerate triangles.
            if i0 == i1 || i1 == i2 || i0 == i2 {
                removed_face[face_id] = true;
                continue;
            }
            live_faces += 1;
            let p0 = positions[i0 as usize];
            let p1 = positions[i1 as usize];
            let p2 = positions[i2 as usize];
            if let Some((a, b, c, d)) = plane_of(p0, p1, p2) {
                let kp = Quadric::from_plane(a, b, c, d, 1.0);
                quadrics[i0 as usize].add_assign(&kp);
                quadrics[i1 as usize].add_assign(&kp);
                quadrics[i2 as usize].add_assign(&kp);
            }
            incident[i0 as usize].push(face_id as u32);
            incident[i1 as usize].push(face_id as u32);
            incident[i2 as usize].push(face_id as u32);
        }

        add_boundary_quadrics(&faces, &removed_face, &positions, &mut quadrics);

        Self {
            positions,
            normals,
            uvs,
            quadrics,
            removed_vertex: vec![false; vertex_count],
            version: vec![0; vertex_count],
            faces,
            removed_face,
            incident,
            live_faces,
        }
    }

    /// Returns the sorted, de-duplicated set of live neighbour vertices of
    /// `vertex`, derived from its incident faces.
    fn neighbours(&self, vertex: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for &face_id in &self.incident[vertex as usize] {
            if self.removed_face[face_id as usize] {
                continue;
            }
            for &other in &self.faces[face_id as usize] {
                if other != vertex && !self.removed_vertex[other as usize] {
                    out.push(other);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Scores the collapse of edge `(a, b)`: solves the merged quadric for the
    /// optimal position (falling back to endpoints/midpoint when singular) and
    /// returns a stamped [`Collapse`].
    fn score_edge(&self, a: u32, b: u32) -> Collapse {
        let mut q = self.quadrics[a as usize];
        q.add_assign(&self.quadrics[b as usize]);
        let (target, cost) = match q.optimal_position() {
            Some(opt) => (opt, q.error_at(opt)),
            None => {
                let pa = self.positions[a as usize];
                let pb = self.positions[b as usize];
                let mid = [
                    0.5 * (pa[0] + pb[0]),
                    0.5 * (pa[1] + pb[1]),
                    0.5 * (pa[2] + pb[2]),
                ];
                let mut best = (pa, q.error_at(pa));
                let eb = q.error_at(pb);
                if eb < best.1 {
                    best = (pb, eb);
                }
                let em = q.error_at(mid);
                if em < best.1 {
                    best = (mid, em);
                }
                best
            }
        };
        Collapse {
            cost,
            keep: a,
            drop: b,
            keep_version: self.version[a as usize],
            drop_version: self.version[b as usize],
            target,
        }
    }

    /// Returns `true` when a popped [`Collapse`] is still valid: both endpoints
    /// alive and their version stamps unchanged since the entry was queued.
    fn is_fresh(&self, c: &Collapse) -> bool {
        !self.removed_vertex[c.keep as usize]
            && !self.removed_vertex[c.drop as usize]
            && self.version[c.keep as usize] == c.keep_version
            && self.version[c.drop as usize] == c.drop_version
    }

    /// Applies the collapse `drop → keep` at `target`: moves and re-attributes
    /// the surviving vertex, inherits the dropped quadric and faces, retires
    /// degenerate faces, and bumps the surviving version stamp.
    fn apply(&mut self, keep: u32, drop: u32, target: [f64; 3]) {
        let ki = keep as usize;
        let di = drop as usize;

        // Interpolate attributes along the edge toward `target`.
        let (t, _) = project_param(self.positions[ki], self.positions[di], target);
        if !self.normals.is_empty() {
            self.normals[ki] = lerp_normalized(self.normals[ki], self.normals[di], t);
        }
        if !self.uvs.is_empty() {
            let a = self.uvs[ki];
            let b = self.uvs[di];
            self.uvs[ki] = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        }

        self.positions[ki] = target;
        let dropped_quadric = self.quadrics[di];
        self.quadrics[ki].add_assign(&dropped_quadric);
        self.removed_vertex[di] = true;

        // Re-point the dropped vertex's faces at `keep`, retiring any that
        // become degenerate, and merge its surviving faces into `keep`.
        let drop_faces = std::mem::take(&mut self.incident[di]);
        for face_id in drop_faces {
            let fi = face_id as usize;
            if self.removed_face[fi] {
                continue;
            }
            for slot in &mut self.faces[fi] {
                if *slot == drop {
                    *slot = keep;
                }
            }
            let [a, b, c] = self.faces[fi];
            if a == b || b == c || a == c {
                self.removed_face[fi] = true;
                self.live_faces -= 1;
            } else if !self.incident[ki].contains(&face_id) {
                self.incident[ki].push(face_id);
            }
        }

        self.version[ki] += 1;
    }

    /// Compacts the surviving vertices and faces into a fresh [`TriangleMesh`],
    /// remapping indices and narrowing positions/attributes back to `f32`.
    fn into_mesh(self) -> Result<TriangleMesh, DecimationError> {
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let has_normals = !self.normals.is_empty();
        let has_uvs = !self.uvs.is_empty();

        for (vid, removed) in self.removed_vertex.iter().enumerate() {
            if *removed {
                continue;
            }
            remap[vid] = positions.len() as u32;
            let p = self.positions[vid];
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            if has_normals {
                let n = self.normals[vid];
                normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
            }
            if has_uvs {
                let t = self.uvs[vid];
                uvs.push([t[0] as f32, t[1] as f32]);
            }
        }

        let mut indices = Vec::new();
        for (fid, tri) in self.faces.iter().enumerate() {
            if self.removed_face[fid] {
                continue;
            }
            let [a, b, c] = *tri;
            let (ra, rb, rc) =
                (remap[a as usize], remap[b as usize], remap[c as usize]);
            // A live face can only reference live, remapped vertices.
            if ra == u32::MAX || rb == u32::MAX || rc == u32::MAX {
                continue;
            }
            if ra == rb || rb == rc || ra == rc {
                continue;
            }
            indices.push([ra, rb, rc]);
        }

        TriangleMesh::new(positions, normals, uvs, indices)
            .map_err(DecimationError::Rebuild)
    }
}

/// Simplifies `mesh` down toward `target_triangles` live triangles using
/// greedy quadric-error-metric edge collapses.
///
/// The result keeps at least `target_triangles` triangles when the topology
/// allows further collapses; a mesh already at or below the target is returned
/// essentially unchanged (coincident/degenerate faces are still cleaned up).
/// Positions, normals, and `UV`s of merged vertices are placed at the
/// error-minimizing point and interpolated along the collapsed edge.
///
/// # Errors
///
/// Returns [`DecimationError::ZeroTarget`] when `target_triangles == 0`, and
/// [`DecimationError::Rebuild`] if the compacted mesh fails validation (not
/// expected for well-formed input).
pub fn decimate(
    mesh: &TriangleMesh,
    target_triangles: usize,
) -> Result<TriangleMesh, DecimationError> {
    if target_triangles == 0 {
        return Err(DecimationError::ZeroTarget);
    }

    let mut state = Decimator::new(mesh);
    if state.live_faces <= target_triangles {
        return state.into_mesh();
    }

    // Seed the heap with every unique live edge, scored once.
    let mut heap: BinaryHeap<Reverse<Collapse>> = BinaryHeap::new();
    let vertex_count = state.positions.len();
    for v in 0..vertex_count as u32 {
        if state.removed_vertex[v as usize] {
            continue;
        }
        for n in state.neighbours(v) {
            // Push each undirected edge once (from its lower-indexed endpoint).
            if v < n {
                heap.push(Reverse(state.score_edge(v, n)));
            }
        }
    }

    while state.live_faces > target_triangles {
        let Some(Reverse(collapse)) = heap.pop() else {
            break;
        };
        if !state.is_fresh(&collapse) {
            continue;
        }
        let keep = collapse.keep;
        let drop = collapse.drop;
        state.apply(keep, drop, collapse.target);

        // Rescore every edge now incident on the surviving vertex.
        for n in state.neighbours(keep) {
            heap.push(Reverse(state.score_edge(keep, n)));
        }
    }

    state.into_mesh()
}

/// Returns the normalized plane `(a, b, c, d)` of triangle `(p0, p1, p2)`, or
/// `None` for a degenerate (zero-area) triangle whose normal cannot be
/// normalized.
fn plane_of(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Option<(f64, f64, f64, f64)> {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len <= 0.0 {
        return None;
    }
    let inv = 1.0 / len;
    let a = n[0] * inv;
    let b = n[1] * inv;
    let c = n[2] * inv;
    let d = -(a * p0[0] + b * p0[1] + c * p0[2]);
    Some((a, b, c, d))
}

/// Adds a heavily weighted virtual plane for every open boundary edge (an edge
/// referenced by exactly one live face), pinning the surface silhouette so
/// holes and borders resist inward collapse.
fn add_boundary_quadrics(
    faces: &[[u32; 3]],
    removed_face: &[bool],
    positions: &[[f64; 3]],
    quadrics: &mut [Quadric],
) {
    use std::collections::HashMap;

    // Count how many live faces reference each undirected edge, and remember
    // one face normal per edge for the perpendicular virtual plane.
    let mut edge_use: HashMap<(u32, u32), (u32, [f64; 3])> = HashMap::new();
    for (fid, tri) in faces.iter().enumerate() {
        if removed_face[fid] {
            continue;
        }
        let [i0, i1, i2] = *tri;
        let plane = plane_of(
            positions[i0 as usize],
            positions[i1 as usize],
            positions[i2 as usize],
        );
        let Some((a, b, c, _)) = plane else {
            continue;
        };
        let face_normal = [a, b, c];
        for &(u, v) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_use
                .entry(key)
                .and_modify(|e| e.0 += 1)
                .or_insert((1, face_normal));
        }
    }

    for (&(u, v), &(count, face_normal)) in &edge_use {
        if count != 1 {
            continue;
        }
        let pu = positions[u as usize];
        let pv = positions[v as usize];
        let edge = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
        // Virtual plane normal: perpendicular to both the edge and the face
        // normal, i.e. in the face plane but across the boundary edge.
        let n = [
            edge[1] * face_normal[2] - edge[2] * face_normal[1],
            edge[2] * face_normal[0] - edge[0] * face_normal[2],
            edge[0] * face_normal[1] - edge[1] * face_normal[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len <= 0.0 {
            continue;
        }
        let inv = 1.0 / len;
        let a = n[0] * inv;
        let b = n[1] * inv;
        let c = n[2] * inv;
        let d = -(a * pu[0] + b * pu[1] + c * pu[2]);
        let kp = Quadric::from_plane(a, b, c, d, BOUNDARY_WEIGHT);
        quadrics[u as usize].add_assign(&kp);
        quadrics[v as usize].add_assign(&kp);
    }
}

/// Projects `target` onto the segment `[a, b]`, returning the clamped parameter
/// `t ∈ [0, 1]` (fraction from `a` toward `b`) and the squared segment length.
fn project_param(a: [f64; 3], b: [f64; 3], target: [f64; 3]) -> (f64, f64) {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if len_sq <= 0.0 {
        return (0.0, 0.0);
    }
    let at = [target[0] - a[0], target[1] - a[1], target[2] - a[2]];
    let dot = at[0] * ab[0] + at[1] * ab[1] + at[2] * ab[2];
    ((dot / len_sq).clamp(0.0, 1.0), len_sq)
}

/// Linearly interpolates two normals by `t` and renormalizes, falling back to
/// the first (then `+Z`) when the blend cancels to zero length.
fn lerp_normalized(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let n = [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        return [n[0] * inv, n[1] * inv, n[2] * inv];
    }
    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if la > 0.0 {
        let inv = 1.0 / la;
        return [a[0] * inv, a[1] * inv, a[2] * inv];
    }
    [0.0, 0.0, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

    /// Builds a flat `grid × grid` quad lattice on the `z = 0` plane, split into
    /// two triangles per cell, with no normals or `UV`s.
    fn planar_grid(grid: usize) -> TriangleMesh {
        let mut positions = Vec::new();
        for j in 0..=grid {
            for i in 0..=grid {
                positions.push([i as f32, j as f32, 0.0]);
            }
        }
        let stride = (grid + 1) as u32;
        let mut indices = Vec::new();
        for j in 0..grid as u32 {
            for i in 0..grid as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.push([a, b, c]);
                indices.push([b, d, c]);
            }
        }
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// A unit cube centered at the origin as a closed 12-triangle mesh.
    fn unit_cube() -> TriangleMesh {
        let positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let indices = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn zero_target_is_rejected() {
        let mesh = planar_grid(2);
        assert_eq!(decimate(&mesh, 0), Err(DecimationError::ZeroTarget));
    }

    #[test]
    fn below_target_returns_cleaned_mesh() {
        let mesh = planar_grid(1); // two triangles
        let out = decimate(&mesh, 100).unwrap();
        assert_eq!(out.triangle_count(), 2);
    }

    #[test]
    fn planar_grid_reaches_target() {
        let mesh = planar_grid(8); // 128 triangles
        let out = decimate(&mesh, 16).unwrap();
        assert!(out.triangle_count() <= 16, "got {}", out.triangle_count());
        assert!(out.triangle_count() >= 2);
    }

    #[test]
    fn decimation_reduces_triangle_count() {
        let mesh = planar_grid(6); // 72 triangles
        let before = mesh.triangle_count();
        let out = decimate(&mesh, 24).unwrap();
        assert!(out.triangle_count() < before);
    }

    #[test]
    fn planar_grid_stays_coplanar() {
        let mesh = planar_grid(8);
        let out = decimate(&mesh, 10).unwrap();
        // Every surviving vertex must remain on the z = 0 plane: QEM has zero
        // in-plane error, so optimal positions never leave it.
        for p in out.positions() {
            assert!(p[2].abs() < 1.0e-4, "z drifted to {}", p[2]);
        }
    }

    #[test]
    fn decimated_plane_is_still_hittable() {
        let mesh = planar_grid(8);
        let out = decimate(&mesh, 12).unwrap();
        let bvh = TriangleMeshBvh::build(out);
        // Fire a ray straight down at an off-seam interior point.
        let ray = Ray::infinite([3.53, 4.47, 5.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray);
        assert!(hit.is_some(), "ray missed the decimated plane");
        let hit = hit.unwrap();
        assert!((hit.position[2]).abs() < 1.0e-3);
    }

    #[test]
    fn cube_decimation_preserves_watertight_hits() {
        let mesh = unit_cube();
        // The cube is already minimal at 12 triangles; ask for fewer and verify
        // the simplified hull is still a solid the ray can hit.
        let out = decimate(&mesh, 8).unwrap();
        assert!(out.triangle_count() <= 12);
        assert!(out.triangle_count() >= 2);
        let bvh = TriangleMeshBvh::build(out);
        let ray = Ray::infinite([0.37, 0.19, 5.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_some());
    }

    #[test]
    fn boundary_corners_are_preserved() {
        let mesh = planar_grid(8);
        let out = decimate(&mesh, 8).unwrap();
        // The four outer corners anchor the silhouette; boundary quadrics
        // should keep all of them present in the simplified mesh.
        let corners = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]];
        for corner in corners {
            let found = out.positions().iter().any(|p| {
                (p[0] - corner[0]).abs() < 1.0e-3 && (p[1] - corner[1]).abs() < 1.0e-3
            });
            assert!(found, "corner {corner:?} was collapsed away");
        }
    }

    #[test]
    fn normals_and_uvs_survive_decimation() {
        let grid = 6;
        let base = planar_grid(grid);
        let normals = vec![[0.0, 0.0, 1.0]; base.vertex_count()];
        let uvs: Vec<[f32; 2]> = base
            .positions()
            .iter()
            .map(|p| [p[0] / grid as f32, p[1] / grid as f32])
            .collect();
        let mesh = TriangleMesh::new(
            base.positions().to_vec(),
            normals,
            uvs,
            base.indices().to_vec(),
        )
        .unwrap();
        let out = decimate(&mesh, 16).unwrap();
        assert!(out.has_normals());
        assert!(out.has_uvs());
        assert_eq!(out.normals().len(), out.vertex_count());
        assert_eq!(out.uvs().len(), out.vertex_count());
        for n in out.normals() {
            assert!((n[2] - 1.0).abs() < 1.0e-4, "normal tilted: {n:?}");
        }
    }

    #[test]
    fn pre_existing_degenerate_faces_are_dropped() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![
            [0, 1, 2],
            [1, 1, 2], // degenerate
            [1, 3, 2],
        ];
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap();
        let out = decimate(&mesh, 100).unwrap();
        assert_eq!(out.triangle_count(), 2);
    }

    #[test]
    fn quadric_error_is_zero_on_defining_plane() {
        // A plane through the origin with +Z normal: points on z = 0 have no
        // error, points off it grow quadratically.
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(q.error_at([3.0, -2.0, 0.0]).abs() < 1.0e-12);
        assert!((q.error_at([0.0, 0.0, 2.0]) - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn optimal_position_recovers_plane_intersection() {
        // Three mutually perpendicular planes meeting at (1, 2, 3) force a
        // unique minimizer there.
        let mut q = Quadric::from_plane(1.0, 0.0, 0.0, -1.0, 1.0);
        q.add_assign(&Quadric::from_plane(0.0, 1.0, 0.0, -2.0, 1.0));
        q.add_assign(&Quadric::from_plane(0.0, 0.0, 1.0, -3.0, 1.0));
        let p = q.optimal_position().expect("non-singular");
        assert!((p[0] - 1.0).abs() < 1.0e-9);
        assert!((p[1] - 2.0).abs() < 1.0e-9);
        assert!((p[2] - 3.0).abs() < 1.0e-9);
    }

    #[test]
    fn optimal_position_singular_returns_none() {
        // A single plane leaves two translational degrees of freedom, so the
        // 3x3 block is singular and no unique minimizer exists.
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(q.optimal_position().is_none());
    }
}
