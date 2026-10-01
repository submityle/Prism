//! The `CPU` golden twin for the cloth self-collision kernel.
//!
//! The authoritative Jacobi golden lives in [`prism_physics_core`]
//! (`resolve_self_collision_virtual_jacobi` and its augment variant); rather
//! than copy that arithmetic and risk it drifting, this twin *delegates* to it
//! and returns the applied positions of a single Jacobi pass. The GPU kernel in
//! [`super::gpu`] produces the same applied positions, so the parity suite
//! compares the two within a tight tolerance.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! all-pairs brute-force reference that re-derives the own-slot accumulate /
//! barycentric-scatter result without the uniform hash, closing the loop from
//! first principles (no fake parity).
//!
//! # Provenance
//!
//! The virtual-particle technique is the published `NvCloth` method; the Jacobi
//! own-slot reformulation is standard parallel position-based dynamics. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_core::{
    resolve_self_collision_virtual_augment_jacobi, resolve_self_collision_virtual_jacobi,
    VirtualParticle,
};

use super::ClothSelfCollisionScope;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs one Jacobi virtual-particle self-collision pass on the `CPU`, returning
/// the applied positions.
///
/// This is the golden twin of [`super::gpu::GpuClothSelfCollision::solve`]:
/// [`ClothSelfCollisionScope::All`] resolves every sample pair (a self-contained
/// tier), while [`ClothSelfCollisionScope::VirtualOnly`] skips real-vs-real
/// pairs so it augments a friction point-to-point pass without stripping its
/// tangential friction. A non-positive `cell_size`/`thickness`, a mismatched
/// `inverse_masses` length, or fewer than two samples, leaves `positions`
/// unchanged.
#[must_use]
pub fn cpu_cloth_self_collision_jacobi(
    positions: &[Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
    scope: ClothSelfCollisionScope,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    match scope {
        ClothSelfCollisionScope::All => resolve_self_collision_virtual_jacobi(
            &mut out,
            inverse_masses,
            virtuals,
            cell_size,
            thickness,
        ),
        ClothSelfCollisionScope::VirtualOnly => resolve_self_collision_virtual_augment_jacobi(
            &mut out,
            inverse_masses,
            virtuals,
            cell_size,
            thickness,
        ),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::{generate_virtual_particles, VirtualParticlePattern};

    /// An independent all-pairs brute-force Jacobi reference, with no uniform
    /// hash: for `cell_size >= thickness` the golden's 27-cell neighbourhood
    /// captures exactly the same penetrating pairs, so the two must agree.
    fn brute_force_jacobi(
        positions: &[Vec3],
        inverse_masses: &[Real],
        virtuals: &[VirtualParticle],
        thickness: Real,
        scope: ClothSelfCollisionScope,
    ) -> Vec<Vec3> {
        let real_count = positions.len();
        // Samples: real particles, then in-range virtuals.
        let mut verts: Vec<[u32; 3]> = Vec::new();
        let mut weights: Vec<[Real; 3]> = Vec::new();
        for i in 0..real_count {
            verts.push([i as u32, i as u32, i as u32]);
            weights.push([1.0, 0.0, 0.0]);
        }
        for vp in virtuals {
            if vp.verts.iter().all(|&v| (v as usize) < real_count) {
                verts.push(vp.verts);
                weights.push(vp.weights);
            }
        }
        let n = verts.len();
        if n < 2 {
            return positions.to_vec();
        }

        let eps_len_sq = 1.0e-12_f32;
        let thickness_sq = thickness * thickness;
        let position = |s: usize| -> Vec3 {
            let mut p = Vec3::ZERO;
            for k in 0..3 {
                let w = weights[s][k];
                if w != 0.0 {
                    p += positions[verts[s][k] as usize] * w;
                }
            }
            p
        };
        let eff = |s: usize| -> Real {
            let mut e = 0.0;
            for k in 0..3 {
                let w = weights[s][k];
                if w != 0.0 {
                    e += w * w * inverse_masses[verts[s][k] as usize].max(0.0);
                }
            }
            e
        };
        let shares = |a: usize, b: usize| -> bool {
            for ka in 0..3 {
                if weights[a][ka] <= 0.0 {
                    continue;
                }
                for kb in 0..3 {
                    if weights[b][kb] <= 0.0 {
                        continue;
                    }
                    if verts[a][ka] == verts[b][kb] {
                        return true;
                    }
                }
            }
            false
        };

        // Phase 1: each sample's own half-correction, from the frozen snapshot.
        let mut sample_dp = vec![Vec3::ZERO; n];
        for a in 0..n {
            let mut acc = Vec3::ZERO;
            for b in 0..n {
                if b == a {
                    continue;
                }
                if scope == ClothSelfCollisionScope::VirtualOnly
                    && a < real_count
                    && b < real_count
                {
                    continue;
                }
                if shares(a, b) {
                    continue;
                }
                let pa = position(a);
                let pb = position(b);
                let delta = pb - pa;
                let dist_sq = delta.length_squared();
                if dist_sq >= thickness_sq {
                    continue;
                }
                let wa = eff(a);
                let wb = eff(b);
                let w_sum = wa + wb;
                if w_sum <= 0.0 {
                    continue;
                }
                let (dir, penetration) = if dist_sq <= eps_len_sq {
                    (Vec3::new(1.0, 0.0, 0.0), thickness)
                } else {
                    let dist = dist_sq.sqrt();
                    (delta / dist, thickness - dist)
                };
                acc += dir * (-penetration * (wa / w_sum));
            }
            sample_dp[a] = acc;
        }

        // Phase 2: barycentric scatter onto the real vertices.
        let mut out = positions.to_vec();
        for a in 0..n {
            let dp = sample_dp[a];
            if dp == Vec3::ZERO {
                continue;
            }
            let e = eff(a);
            if e <= 0.0 {
                continue;
            }
            for k in 0..3 {
                let w = weights[a][k];
                if w == 0.0 {
                    continue;
                }
                let j = verts[a][k] as usize;
                let im = inverse_masses[j].max(0.0);
                if im <= 0.0 {
                    continue;
                }
                out[j] += dp * (w * im / e);
            }
        }
        out
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() <= 1.0e-5 * a.length().max(1.0)
    }

    #[test]
    fn golden_matches_brute_force_full_scope() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(4.0, 0.0, 1.0),
            Vec3::new(0.0, 4.0, 1.0),
        ];
        let im = vec![1.0; 6];
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );
        // cell_size >= thickness so the hash captures the brute-force pair set.
        let (cell, thick) = (2.0_f32, 1.5_f32);
        let golden = cpu_cloth_self_collision_jacobi(
            &positions,
            &im,
            &virtuals,
            cell,
            thick,
            ClothSelfCollisionScope::All,
        );
        let brute =
            brute_force_jacobi(&positions, &im, &virtuals, thick, ClothSelfCollisionScope::All);
        for (g, b) in golden.iter().zip(brute.iter()) {
            assert!(close(*g, *b), "golden {g:?} vs brute {b:?}");
        }
    }

    #[test]
    fn golden_matches_brute_force_virtual_only_scope() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.3, 1.3, 0.3),
        ];
        let im = vec![1.0; 4];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        let (cell, thick) = (2.0_f32, 1.5_f32);
        let golden = cpu_cloth_self_collision_jacobi(
            &positions,
            &im,
            &virtuals,
            cell,
            thick,
            ClothSelfCollisionScope::VirtualOnly,
        );
        let brute = brute_force_jacobi(
            &positions,
            &im,
            &virtuals,
            thick,
            ClothSelfCollisionScope::VirtualOnly,
        );
        for (g, b) in golden.iter().zip(brute.iter()) {
            assert!(close(*g, *b), "golden {g:?} vs brute {b:?}");
        }
    }

    #[test]
    fn no_op_inputs_leave_positions_unchanged() {
        let positions = vec![Vec3::ZERO, Vec3::X];
        let im = vec![1.0, 1.0];
        let out = cpu_cloth_self_collision_jacobi(
            &positions,
            &im,
            &[],
            0.0,
            0.2,
            ClothSelfCollisionScope::All,
        );
        assert_eq!(out, positions);
    }
}
