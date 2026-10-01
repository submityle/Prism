//! Host-side deterministic preparation for the cloth aero kernel.
//!
//! The `GPU` kernel does the floating-point drag/lift arithmetic, but the parts
//! that must stay bit-identical to the [`prism_physics_core`] golden are built
//! here on the host: triangle filtering (out-of-range faces dropped exactly as
//! `apply_aero_forces` skips them), the per-triangle wind (the steady field plus
//! the integer-hashed [`turbulence_offset`](prism_physics_core::turbulence_offset)
//! jitter), and the per-vertex incidence `CSR` in ascending triangle order so
//! each vertex folds its faces in the golden's reduction order. Performing the
//! integer work on the host keeps the only `GPU` floating-point work the force
//! itself, which the parity suite checks within a tight tolerance.
//!
//! # Provenance
//!
//! Plain host bookkeeping around the standard per-triangle aerodynamics model.
//! No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{turbulence_offset, AeroParams, WindField};

use super::{ClothAeroParams, ClothAeroTriangle, Real};

/// A fully flattened, upload-ready snapshot of one cloth aero pass.
///
/// Every field is plain-old-data laid out for direct upload; the four-wide
/// packing (`[_; 4]`) matches the `vec4` strides the kernel binds.
pub struct ClothAeroPrep {
    /// Number of addressable particles (min of positions / velocity / inverse
    /// mass column lengths).
    pub vertex_count: u32,
    /// Number of retained (in-range) triangles.
    pub triangle_count: u32,
    /// Sanitized drag coefficient threaded to the kernel.
    pub drag: f32,
    /// Sanitized lift coefficient threaded to the kernel.
    pub lift: f32,
    /// Sanitized air density (`<= 0` selects the linear model).
    pub air_density: f32,
    /// Frozen positions, padded to `vec4` (`w` unused).
    pub positions: Vec<[f32; 4]>,
    /// Frozen velocities, padded to `vec4` (`w` carried through untouched).
    pub velocities: Vec<[f32; 4]>,
    /// Index-aligned inverse masses (`0` marks a pinned particle).
    pub inverse_masses: Vec<f32>,
    /// Retained triangle corner indices, padded to `vec4<u32>`.
    pub triangles: Vec<[u32; 4]>,
    /// Per-triangle wind (steady field plus turbulence), padded to `vec4`.
    pub triangle_winds: Vec<[f32; 4]>,
    /// `CSR` offsets into [`vert_tris`](Self::vert_tris), length
    /// `vertex_count + 1`.
    pub vert_offsets: Vec<u32>,
    /// Incident retained-triangle indices per vertex, ascending by triangle so
    /// each vertex folds its faces in the golden's reduction order. A triangle
    /// that references a vertex on more than one corner appears once per corner.
    pub vert_tris: Vec<u32>,
}

/// Builds the upload-ready [`ClothAeroPrep`] for one Jacobi aero pass, or
/// returns [`None`] when the pass is a no-op.
///
/// The build mirrors `prism_physics_core`'s `apply_aero_forces` guards exactly:
/// it returns [`None`] (the `GPU` path then leaves velocities untouched) when
/// `dt` is non-positive or non-finite, when the particle set is empty, when
/// there are no triangles, or when the `velocities`/`inverse_masses` columns are
/// not index-aligned with `positions`. Out-of-range triangles are dropped. The
/// wind field and coefficients are sanitized once up front, matching the golden.
#[must_use]
pub fn build(
    positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[Real],
    triangles: &[ClothAeroTriangle],
    params: ClothAeroParams,
    dt: Real,
) -> Option<ClothAeroPrep> {
    let count = positions.len();
    if dt <= 0.0
        || !dt.is_finite()
        || count == 0
        || triangles.is_empty()
        || velocities.len() != count
        || inverse_masses.len() != count
    {
        return None;
    }

    // Sanitize the field and coefficients exactly as the golden does, so the
    // per-triangle wind and the drag/lift the kernel reads already match.
    let field = WindField::new(Vec3::from_array(params.velocity), params.turbulence).sanitized();
    let aero = AeroParams::new(params.drag, params.lift)
        .with_air_density(params.air_density)
        .sanitized();

    let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);

    // Retain only in-range triangles, baking each one's wind (steady field plus
    // the deterministic turbulence jitter) in the process.
    let mut kept: Vec<[u32; 4]> = Vec::with_capacity(triangles.len());
    let mut winds: Vec<[f32; 4]> = Vec::with_capacity(triangles.len());
    for tri in triangles {
        let [i0, i1, i2] = tri.indices();
        if i0 >= count_u32 || i1 >= count_u32 || i2 >= count_u32 {
            continue;
        }
        let wind = field.velocity + turbulence_offset([i0, i1, i2], field.turbulence);
        kept.push([i0, i1, i2, 0]);
        winds.push([wind.x, wind.y, wind.z, 0.0]);
    }
    if kept.is_empty() {
        return None;
    }

    // Per-vertex incidence: walking retained triangles ascending gives each
    // vertex its faces in ascending triangle order (the golden's fold order). A
    // triangle that lands on a vertex twice (degenerate) is pushed twice, so the
    // gather reproduces the sequential pass's per-corner accumulation exactly.
    let mut per_vertex: Vec<Vec<u32>> = Vec::with_capacity(count);
    per_vertex.resize_with(count, Vec::new);
    for (ti, corners) in kept.iter().enumerate() {
        for &v in &corners[..3] {
            per_vertex[v as usize].push(ti as u32);
        }
    }
    let mut vert_offsets: Vec<u32> = Vec::with_capacity(count + 1);
    let mut vert_tris: Vec<u32> = Vec::new();
    vert_offsets.push(0);
    for entries in &per_vertex {
        vert_tris.extend_from_slice(entries);
        vert_offsets.push(u32::try_from(vert_tris.len()).unwrap_or(u32::MAX));
    }

    let positions_packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let velocities_packed: Vec<[f32; 4]> =
        velocities.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();

    Some(ClothAeroPrep {
        vertex_count: count_u32,
        triangle_count: u32::try_from(kept.len()).unwrap_or(u32::MAX),
        drag: aero.drag,
        lift: aero.lift,
        air_density: aero.air_density,
        positions: positions_packed,
        velocities: velocities_packed,
        inverse_masses: inverse_masses.to_vec(),
        triangles: kept,
        triangle_winds: winds,
        vert_offsets,
        vert_tris,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(i0: u32, i1: u32, i2: u32) -> ClothAeroTriangle {
        ClothAeroTriangle::new(i0, i1, i2)
    }

    fn unit_params() -> ClothAeroParams {
        ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.0)
    }

    #[test]
    fn non_positive_or_mismatched_inputs_are_no_ops() {
        let positions = alloc::vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let vel = alloc::vec![Vec3::ZERO; 3];
        let im = alloc::vec![1.0_f32; 3];
        let tris = alloc::vec![tri(0, 1, 2)];
        assert!(build(&positions, &vel, &im, &tris, unit_params(), 0.0).is_none());
        assert!(build(&positions, &vel, &im, &tris, unit_params(), Real::NAN).is_none());
        assert!(build(&positions, &vel, &im, &[], unit_params(), 1.0).is_none());
        // Mismatched column lengths.
        let short_vel = alloc::vec![Vec3::ZERO; 2];
        assert!(build(&positions, &short_vel, &im, &tris, unit_params(), 1.0).is_none());
    }

    #[test]
    fn all_out_of_range_triangles_is_a_no_op() {
        let positions = alloc::vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let vel = alloc::vec![Vec3::ZERO; 3];
        let im = alloc::vec![1.0_f32; 3];
        let tris = alloc::vec![tri(0, 1, 9)];
        assert!(build(&positions, &vel, &im, &tris, unit_params(), 1.0).is_none());
    }

    #[test]
    fn incidence_csr_frames_every_corner_in_ascending_order() {
        // Two triangles sharing the edge (1,2).
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let vel = alloc::vec![Vec3::ZERO; 4];
        let im = alloc::vec![1.0_f32; 4];
        let tris = alloc::vec![tri(0, 1, 2), tri(1, 3, 2)];
        let prep = build(&positions, &vel, &im, &tris, unit_params(), 1.0).expect("non-degenerate");

        assert_eq!(prep.vertex_count, 4);
        assert_eq!(prep.triangle_count, 2);
        assert_eq!(prep.vert_offsets.first(), Some(&0));
        assert_eq!(
            prep.vert_offsets.last(),
            Some(&(prep.vert_tris.len() as u32))
        );
        // 2 triangles * 3 corners = 6 incidence entries total.
        assert_eq!(prep.vert_tris.len(), 6);
        // Vertex 1 is in both triangles.
        let lo = prep.vert_offsets[1] as usize;
        let hi = prep.vert_offsets[2] as usize;
        assert_eq!(&prep.vert_tris[lo..hi], &[0, 1]);
        // Every vertex's incidence list is ascending.
        for w in prep.vert_offsets.windows(2) {
            let list = &prep.vert_tris[w[0] as usize..w[1] as usize];
            for p in list.windows(2) {
                assert!(p[0] < p[1], "incidence must be ascending by triangle");
            }
        }
    }

    #[test]
    fn turbulence_bakes_into_per_triangle_wind() {
        let positions = alloc::vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let vel = alloc::vec![Vec3::ZERO; 3];
        let im = alloc::vec![1.0_f32; 3];
        let tris = alloc::vec![tri(0, 1, 2)];
        let turbulent = ClothAeroParams::new([0.0, 0.0, 2.0], 0.5, 1.0, 0.0);
        let prep = build(&positions, &vel, &im, &tris, turbulent, 1.0).expect("non-degenerate");
        let expected = Vec3::new(0.0, 0.0, 2.0) + turbulence_offset([0, 1, 2], 0.5);
        assert_eq!(
            prep.triangle_winds[0],
            [expected.x, expected.y, expected.z, 0.0]
        );
    }
}
