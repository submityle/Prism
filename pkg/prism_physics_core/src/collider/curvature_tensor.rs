//! Per-vertex principal curvature *directions* via the Rusinkiewicz tensor.
//!
//! [`estimate_curvature`](crate::collider::estimate_curvature) reports curvature
//! *magnitudes* (mean, Gaussian, principal values) but not the directions in
//! which the surface bends. Direction-aware curvature drives anisotropic
//! remeshing, feature-aligned decimation, grain/scratch mapping and anisotropic
//! friction frames, so cooking pipelines want the full second-fundamental-form
//! tensor, not just its invariants.
//!
//! This module implements Rusinkiewicz's robust estimator ("Estimating
//! Curvatures and Their Derivatives on Triangle Meshes", 3DPVT 2004): for each
//! face it fits the second fundamental form from the variation of the vertex
//! normals along the three edges, then accumulates each face tensor into its
//! vertices using Voronoi corner-area weights, re-projecting the tensor into a
//! stable per-vertex tangent frame. Diagonalising the accumulated 2x2 tensor
//! yields the two principal curvatures and their orthonormal tangent
//! directions.
//!
//! It reuses the existing angle-weighted vertex normals
//! ([`vertex_normals`](crate::collider::vertex_normals)) and is otherwise pure
//! triangle-mesh geometry with no coupling to the collision pipeline. The
//! algorithm is a published, independently reimplemented method; nothing here
//! is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::vertex_normals::vertex_normals;

/// Cross-product length below which a triangle is treated as degenerate.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// A symmetric 2x2 curvature tensor expressed in a tangent frame `(u, v)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tensor2 {
    /// `d/du` of the normal, projected onto `u`.
    ku: f32,
    /// Mixed `uv` term.
    kuv: f32,
    /// `d/dv` of the normal, projected onto `v`.
    kv: f32,
}

/// Principal curvatures and directions at one vertex.
///
/// `dir1` is the tangent direction of the principal curvature with the larger
/// magnitude (`k1`), `dir2` the smaller (`k2`); `dir1`, `dir2` and `normal`
/// form a right-handed orthonormal frame. A flat or ill-defined vertex reports
/// zero curvatures with a zero `normal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalCurvature {
    /// Principal curvature with the larger magnitude.
    pub k1: f32,
    /// Principal curvature with the smaller magnitude.
    pub k2: f32,
    /// Tangent direction associated with [`PrincipalCurvature::k1`].
    pub dir1: Vec3,
    /// Tangent direction associated with [`PrincipalCurvature::k2`].
    pub dir2: Vec3,
    /// Unit surface normal at the vertex (zero when undefined).
    pub normal: Vec3,
}

impl PrincipalCurvature {
    /// A flat / undefined vertex.
    const FLAT: Self = Self {
        k1: 0.0,
        k2: 0.0,
        dir1: Vec3::ZERO,
        dir2: Vec3::ZERO,
        normal: Vec3::ZERO,
    };

    /// Mean curvature `H = (k1 + k2) / 2`.
    #[must_use]
    pub fn mean(&self) -> f32 {
        0.5 * (self.k1 + self.k2)
    }

    /// Gaussian curvature `K = k1 * k2`.
    #[must_use]
    pub fn gaussian(&self) -> f32 {
        self.k1 * self.k2
    }
}

/// Per-vertex principal-curvature report, indexed against the input vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureTensorReport {
    /// One entry per input vertex.
    pub curvatures: Vec<PrincipalCurvature>,
}

impl CurvatureTensorReport {
    /// Number of vertices covered by the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.curvatures.len()
    }

    /// Whether the report is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.curvatures.is_empty()
    }
}

/// Estimates per-vertex principal curvatures and their directions.
///
/// Returns `None` when `vertices` or `indices` is empty, or when vertex normals
/// cannot be computed. The output has exactly one entry per input vertex;
/// vertices with no finite incident area are reported as
/// [`PrincipalCurvature::FLAT`].
#[must_use]
pub fn estimate_curvature_tensor(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
) -> Option<CurvatureTensorReport> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let normals = vertex_normals(vertices, indices)?;
    let n = vertices.len();

    // Pass 1: Voronoi corner areas and point areas.
    let (corner_areas, point_areas) = corner_and_point_areas(vertices, indices, n);

    // Pass 2: initialise a stable tangent frame per vertex from an incident
    // edge crossed with the vertex normal (arbitrary but deterministic).
    let (mut pdir1, mut pdir2) = initial_frames(vertices, indices, &normals, n);

    // Pass 3: fit each face tensor and accumulate it into its vertices.
    let mut tensors = vec![Tensor2::default(); n];
    for (face_idx, tri) in indices.iter().enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let p = [vertices[i0], vertices[i1], vertices[i2]];
        let nn = [normals[i0], normals[i1], normals[i2]];

        let Some((t, b, face_tensor)) = fit_face_tensor(&p, &nn) else {
            continue;
        };

        let corner = corner_areas[face_idx];
        let verts = [i0, i1, i2];
        for (j, &vj) in verts.iter().enumerate() {
            let area = point_areas[vj];
            if area <= 0.0 {
                continue;
            }
            let projected = proj_curv(t, b, face_tensor, pdir1[vj], pdir2[vj]);
            let wt = corner[j] / area;
            tensors[vj].ku += wt * projected.ku;
            tensors[vj].kuv += wt * projected.kuv;
            tensors[vj].kv += wt * projected.kv;
        }
    }

    // Pass 4: diagonalise.
    let mut curvatures = Vec::with_capacity(n);
    for v in 0..n {
        let normal = normals[v];
        if normal.length_squared() <= 0.0 || point_areas[v] <= 0.0 {
            curvatures.push(PrincipalCurvature::FLAT);
            continue;
        }
        curvatures.push(diagonalize(pdir1[v], pdir2[v], tensors[v], normal));
    }

    // `pdir1`/`pdir2` are consumed as scratch frames; drop explicitly for
    // clarity (they are not part of the output).
    pdir1.clear();
    pdir2.clear();

    Some(CurvatureTensorReport { curvatures })
}

/// Computes Voronoi corner areas (`[a0, a1, a2]` per face) and the accumulated
/// point area per vertex, following Meyer et al. via Rusinkiewicz's layout.
fn corner_and_point_areas(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    vertex_count: usize,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let mut corner_areas = vec![[0.0_f32; 3]; indices.len()];
    let mut point_areas = vec![0.0_f32; vertex_count];

    for (face_idx, tri) in indices.iter().enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            continue;
        }
        let p0 = vertices[i0];
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        // e[j] = p[prev] - p[next]: e0 = p2 - p1, e1 = p0 - p2, e2 = p1 - p0.
        let e = [p2 - p1, p0 - p2, p1 - p0];
        let cross = e[0].cross(e[1]);
        let cross_len = cross.length();
        if cross_len <= DEGENERATE_EPSILON {
            continue;
        }
        let area = 0.5 * cross_len;
        let l2 = [
            e[0].length_squared(),
            e[1].length_squared(),
            e[2].length_squared(),
        ];
        let ew = [
            l2[0] * (l2[1] + l2[2] - l2[0]),
            l2[1] * (l2[2] + l2[0] - l2[1]),
            l2[2] * (l2[0] + l2[1] - l2[2]),
        ];
        let mut ca = [0.0_f32; 3];
        if ew[0] <= 0.0 {
            ca[1] = -0.25 * l2[2] * area / e[0].dot(e[2]);
            ca[2] = -0.25 * l2[1] * area / e[0].dot(e[1]);
            ca[0] = area - ca[1] - ca[2];
        } else if ew[1] <= 0.0 {
            ca[2] = -0.25 * l2[0] * area / e[1].dot(e[0]);
            ca[0] = -0.25 * l2[2] * area / e[1].dot(e[2]);
            ca[1] = area - ca[2] - ca[0];
        } else if ew[2] <= 0.0 {
            ca[0] = -0.25 * l2[1] * area / e[2].dot(e[1]);
            ca[1] = -0.25 * l2[0] * area / e[2].dot(e[0]);
            ca[2] = area - ca[0] - ca[1];
        } else {
            let ewscale = 0.5 * area / (ew[0] + ew[1] + ew[2]);
            for j in 0..3 {
                ca[j] = ewscale * (ew[(j + 1) % 3] + ew[(j + 2) % 3]);
            }
        }
        corner_areas[face_idx] = ca;
        point_areas[i0] += ca[0];
        point_areas[i1] += ca[1];
        point_areas[i2] += ca[2];
    }

    (corner_areas, point_areas)
}

/// Builds an initial per-vertex tangent frame `(pdir1, pdir2)` from an incident
/// edge crossed with the vertex normal.
fn initial_frames(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    normals: &[Vec3],
    vertex_count: usize,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut seed = vec![Vec3::ZERO; vertex_count];
    for tri in indices {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            continue;
        }
        seed[i0] = vertices[i1] - vertices[i0];
        seed[i1] = vertices[i2] - vertices[i1];
        seed[i2] = vertices[i0] - vertices[i2];
    }

    let mut pdir1 = vec![Vec3::ZERO; vertex_count];
    let mut pdir2 = vec![Vec3::ZERO; vertex_count];
    for v in 0..vertex_count {
        let normal = normals[v];
        if normal.length_squared() <= 0.0 {
            continue;
        }
        let mut u = seed[v].cross(normal).normalize_or_zero();
        if u.length_squared() <= 0.0 {
            // Seed edge was parallel to the normal; pick an arbitrary tangent.
            u = fallback_tangent(normal);
        }
        pdir1[v] = u;
        pdir2[v] = normal.cross(u).normalize_or_zero();
    }
    (pdir1, pdir2)
}

/// An arbitrary unit tangent perpendicular to `normal`.
fn fallback_tangent(normal: Vec3) -> Vec3 {
    let reference = if normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    normal.cross(reference).normalize_or_zero()
}

/// Fits the second fundamental form of one triangle from the variation of its
/// vertex normals along the edges. Returns the face tangent frame `(t, b)` and
/// the fitted tensor, or `None` for a degenerate / singular face.
fn fit_face_tensor(p: &[Vec3; 3], n: &[Vec3; 3]) -> Option<(Vec3, Vec3, Tensor2)> {
    let e = [p[2] - p[1], p[0] - p[2], p[1] - p[0]];
    let t = e[0].normalize_or_zero();
    if t.length_squared() <= 0.0 {
        return None;
    }
    let face_normal = e[0].cross(e[1]);
    if face_normal.length_squared() <= DEGENERATE_EPSILON {
        return None;
    }
    let b = face_normal.cross(t).normalize_or_zero();
    if b.length_squared() <= 0.0 {
        return None;
    }

    // Normal equations (symmetric 3x3) for the least-squares fit of (a, b, c).
    let mut w00 = 0.0_f64;
    let mut w01 = 0.0_f64;
    let mut w22 = 0.0_f64;
    let mut m = [0.0_f64; 3];
    #[expect(
        clippy::needless_range_loop,
        reason = "loop drives modular (j+1, j+2) edge/normal indexing"
    )]
    for j in 0..3 {
        let prev = (j + 2) % 3;
        let next = (j + 1) % 3;
        let u = f64::from(e[j].dot(t));
        let v = f64::from(e[j].dot(b));
        w00 += u * u;
        w01 += u * v;
        w22 += v * v;
        let dn = n[prev] - n[next];
        let dnu = f64::from(dn.dot(t));
        let dnv = f64::from(dn.dot(b));
        m[0] += dnu * u;
        m[1] += dnu * v + dnv * u;
        m[2] += dnv * v;
    }
    let w = [[w00, w01, 0.0], [w01, w00 + w22, w01], [0.0, w01, w22]];
    let x = solve_sym3(w, m)?;
    Some((
        t,
        b,
        Tensor2 {
            ku: x[0] as f32,
            kuv: x[1] as f32,
            kv: x[2] as f32,
        },
    ))
}

/// Solves `w * x = m` for a 3x3 system via Gaussian elimination with partial
/// pivoting. Returns `None` when the system is singular.
#[expect(
    clippy::needless_range_loop,
    reason = "Gaussian elimination indexes 2D matrix rows and columns by position"
)]
fn solve_sym3(mut w: [[f64; 3]; 3], mut m: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        // Partial pivot.
        let mut pivot = col;
        let mut best = w[col][col].abs();
        for r in (col + 1)..3 {
            let v = w[r][col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best <= 1.0e-20 {
            return None;
        }
        if pivot != col {
            w.swap(col, pivot);
            m.swap(col, pivot);
        }
        let diag = w[col][col];
        for r in (col + 1)..3 {
            let factor = w[r][col] / diag;
            if factor != 0.0 {
                for c in col..3 {
                    w[r][c] -= factor * w[col][c];
                }
                m[r] -= factor * m[col];
            }
        }
    }
    // Back substitution.
    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut sum = m[i];
        for c in (i + 1)..3 {
            sum -= w[i][c] * x[c];
        }
        x[i] = sum / w[i][i];
    }
    Some(x)
}

/// Rotates the tangent frame `(old_u, old_v)` so its implied normal aligns with
/// `new_norm`, returning the rotated `(u, v)`.
fn rot_coord_sys(old_u: Vec3, old_v: Vec3, new_norm: Vec3) -> (Vec3, Vec3) {
    let old_norm = old_u.cross(old_v);
    let ndot = old_norm.dot(new_norm);
    if ndot <= -1.0 {
        return (-old_u, -old_v);
    }
    let perp = new_norm - ndot * old_norm;
    let dperp = (old_norm + new_norm) / (1.0 + ndot);
    let new_u = old_u - dperp * old_u.dot(perp);
    let new_v = old_v - dperp * old_v.dot(perp);
    (new_u, new_v)
}

/// Re-expresses a curvature tensor given in frame `(old_u, old_v)` in the new
/// frame `(new_u, new_v)` (first rotating the new frame onto the old normal).
fn proj_curv(old_u: Vec3, old_v: Vec3, tensor: Tensor2, new_u: Vec3, new_v: Vec3) -> Tensor2 {
    let (r_new_u, r_new_v) = rot_coord_sys(new_u, new_v, old_u.cross(old_v));
    let u1 = r_new_u.dot(old_u);
    let v1 = r_new_u.dot(old_v);
    let u2 = r_new_v.dot(old_u);
    let v2 = r_new_v.dot(old_v);
    Tensor2 {
        ku: tensor.ku * u1 * u1 + tensor.kuv * (2.0 * u1 * v1) + tensor.kv * v1 * v1,
        kuv: tensor.ku * u1 * u2 + tensor.kuv * (u1 * v2 + u2 * v1) + tensor.kv * v1 * v2,
        kv: tensor.ku * u2 * u2 + tensor.kuv * (2.0 * u2 * v2) + tensor.kv * v2 * v2,
    }
}

/// Diagonalises the accumulated vertex tensor to principal curvatures and
/// directions, with the larger-magnitude curvature reported as `k1`.
fn diagonalize(old_u: Vec3, old_v: Vec3, tensor: Tensor2, new_norm: Vec3) -> PrincipalCurvature {
    let (r_old_u, r_old_v) = rot_coord_sys(old_u, old_v, new_norm);
    let (mut c, mut s, mut tt) = (1.0_f32, 0.0_f32, 0.0_f32);
    if tensor.kuv != 0.0 {
        // One Jacobi rotation diagonalises the symmetric 2x2 tensor.
        let h = 0.5 * (tensor.kv - tensor.ku) / tensor.kuv;
        let root = f64::from(1.0 + h * h).sqrt() as f32;
        tt = if h < 0.0 {
            1.0 / (h - root)
        } else {
            1.0 / (h + root)
        };
        c = 1.0 / (f64::from(1.0 + tt * tt).sqrt() as f32);
        s = tt * c;
    }
    let mut k1 = tensor.ku - tt * tensor.kuv;
    let mut k2 = tensor.kv + tt * tensor.kuv;

    let (dir1, dir2);
    if k1.abs() >= k2.abs() {
        dir1 = c * r_old_u - s * r_old_v;
    } else {
        core::mem::swap(&mut k1, &mut k2);
        dir1 = s * r_old_u + c * r_old_v;
    }
    let dir1 = dir1.normalize_or_zero();
    dir2 = new_norm.cross(dir1).normalize_or_zero();

    PrincipalCurvature {
        k1,
        k2,
        dir1,
        dir2,
        normal: new_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grid in the xy-plane with `(res+1)^2` vertices.
    fn plane_grid(res: usize, size: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for j in 0..=res {
            for i in 0..=res {
                let x = (i as f32 / res as f32 - 0.5) * size;
                let y = (j as f32 / res as f32 - 0.5) * size;
                verts.push(Vec3::new(x, y, 0.0));
            }
        }
        let stride = (res + 1) as u32;
        let mut inds = Vec::new();
        for j in 0..res as u32 {
            for i in 0..res as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                inds.push([a, b, c]);
                inds.push([b, d, c]);
            }
        }
        (verts, inds)
    }

    /// A `UV` sphere of radius `r` with shared vertices (poles duplicated per
    /// column, which is fine: we only test the interior band).
    fn uv_sphere(stacks: usize, slices: usize, r: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for i in 0..=stacks {
            let phi = core::f64::consts::PI * (i as f64) / (stacks as f64);
            let (sp, cp) = (phi.sin(), phi.cos());
            for j in 0..=slices {
                let theta = 2.0 * core::f64::consts::PI * (j as f64) / (slices as f64);
                let x = (r as f64 * sp * theta.cos()) as f32;
                let y = (r as f64 * sp * theta.sin()) as f32;
                let z = (r as f64 * cp) as f32;
                verts.push(Vec3::new(x, y, z));
            }
        }
        let stride = (slices + 1) as u32;
        let mut inds = Vec::new();
        for i in 0..stacks as u32 {
            for j in 0..slices as u32 {
                let a = i * stride + j;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                inds.push([a, c, b]);
                inds.push([b, c, d]);
            }
        }
        (verts, inds)
    }

    /// An open cylinder (lateral surface only) of radius `r`, height `h`, axis z.
    fn cylinder(rings: usize, slices: usize, r: f32, h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for i in 0..=rings {
            let z = (i as f32 / rings as f32 - 0.5) * h;
            for j in 0..=slices {
                let theta = 2.0 * core::f64::consts::PI * (j as f64) / (slices as f64);
                let x = (r as f64 * theta.cos()) as f32;
                let y = (r as f64 * theta.sin()) as f32;
                verts.push(Vec3::new(x, y, z));
            }
        }
        let stride = (slices + 1) as u32;
        let mut inds = Vec::new();
        for i in 0..rings as u32 {
            for j in 0..slices as u32 {
                let a = i * stride + j;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                inds.push([a, b, c]);
                inds.push([b, d, c]);
            }
        }
        (verts, inds)
    }

    #[test]
    fn rejects_empty_input() {
        assert!(estimate_curvature_tensor(&[], &[]).is_none());
        assert!(estimate_curvature_tensor(&[Vec3::ZERO], &[]).is_none());
    }

    #[test]
    fn plane_has_zero_curvature() {
        let (v, i) = plane_grid(8, 4.0);
        let report = estimate_curvature_tensor(&v, &i).unwrap();
        assert_eq!(report.len(), v.len());
        // Interior vertices should read as flat.
        let stride = 9usize;
        for j in 1..8 {
            for col in 1..8 {
                let c = report.curvatures[j * stride + col];
                assert!(c.k1.abs() < 1e-3, "k1 = {}", c.k1);
                assert!(c.k2.abs() < 1e-3, "k2 = {}", c.k2);
            }
        }
    }

    #[test]
    fn sphere_has_isotropic_curvature() {
        let r = 2.0_f32;
        let (v, i) = uv_sphere(40, 60, r);
        let report = estimate_curvature_tensor(&v, &i).unwrap();
        let expected = 1.0 / r;
        // Sample the equatorial band, away from the pole columns.
        let stride = 61usize;
        let mut tested = 0;
        for stack in 15..=25 {
            for slice in 5..56 {
                let c = report.curvatures[stack * stride + slice];
                assert!(
                    (c.k1.abs() - expected).abs() < 0.15 * expected,
                    "k1 = {}, expected ~{expected}",
                    c.k1
                );
                assert!(
                    (c.k2.abs() - expected).abs() < 0.15 * expected,
                    "k2 = {}, expected ~{expected}",
                    c.k2
                );
                // Directions must be tangent (perpendicular to the normal).
                assert!(c.dir1.dot(c.normal).abs() < 1e-2);
                assert!(c.dir2.dot(c.normal).abs() < 1e-2);
                tested += 1;
            }
        }
        assert!(tested > 0);
    }

    #[test]
    fn cylinder_is_anisotropic_with_axial_flat_direction() {
        let r = 1.5_f32;
        let (v, i) = cylinder(30, 60, r, 6.0);
        let report = estimate_curvature_tensor(&v, &i).unwrap();
        let expected = 1.0 / r;
        let stride = 61usize;
        let mut tested = 0;
        for ring in 10..=20 {
            for slice in 5..56 {
                let c = report.curvatures[ring * stride + slice];
                // Larger-magnitude curvature is the circumferential bend ~1/r.
                assert!(
                    (c.k1.abs() - expected).abs() < 0.2 * expected,
                    "k1 = {}, expected ~{expected}",
                    c.k1
                );
                // Smaller-magnitude curvature is ~0 along the axis.
                assert!(c.k2.abs() < 0.2 * expected, "k2 = {}", c.k2);
                // dir1 (strong bend) is circumferential: ~no z component.
                assert!(c.dir1.z.abs() < 0.2, "dir1 = {:?}", c.dir1);
                // dir2 (flat) runs along the axis: ~unit z component.
                assert!(c.dir2.z.abs() > 0.8, "dir2 = {:?}", c.dir2);
                tested += 1;
            }
        }
        assert!(tested > 0);
    }

    #[test]
    fn directions_are_orthonormal() {
        let (v, i) = uv_sphere(24, 36, 1.0);
        let report = estimate_curvature_tensor(&v, &i).unwrap();
        for c in &report.curvatures {
            if c.normal.length_squared() <= 0.0 {
                continue;
            }
            assert!((c.dir1.length() - 1.0).abs() < 1e-3);
            assert!((c.dir2.length() - 1.0).abs() < 1e-3);
            assert!(c.dir1.dot(c.dir2).abs() < 1e-2);
        }
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = uv_sphere(16, 24, 1.0);
        let a = estimate_curvature_tensor(&v, &i).unwrap();
        let b = estimate_curvature_tensor(&v, &i).unwrap();
        assert_eq!(a, b);
    }
}
