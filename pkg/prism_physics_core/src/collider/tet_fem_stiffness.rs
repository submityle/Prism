//! Per-element linear-elastic stiffness matrix of a constant-strain
//! tetrahedron (`CST`).
//!
//! Implicit finite-element (`FEM`) integrators, modal (eigenvalue) analysis and
//! stiffness-warping solvers all need the 12x12 element stiffness matrix `Ke`
//! that relates the twelve nodal displacement degrees of freedom of a
//! tetrahedron to the twelve nodal forces they induce under small-strain linear
//! elasticity. For a constant-strain tetrahedron the strain is constant over
//! the element, so the matrix has the closed form
//!
//! ```text
//! Ke = V * B^T * D * B
//! ```
//!
//! where
//!
//! * `V` is the element's (unsigned) rest volume,
//! * `B` is the 6x12 strain-displacement matrix assembled from the four
//!   constant shape-function gradients, and
//! * `D` is the 6x6 isotropic constitutive (material) matrix in `Voigt`
//!   notation with engineering shear strains.
//!
//! The strain and stress are stored in `Voigt` order
//! `[exx, eyy, ezz, gxy, gyz, gzx]`, where the last three components are the
//! engineering shear strains (`gxy = du_x/dy + du_y/dx`, and so on).
//!
//! This module contains only material-independent linear algebra driven by the
//! rest-pose [`TetFemElement`] basis and a scalar material; it holds no
//! simulation state and performs no time integration, so it is fully decoupled
//! from any solver. The co-rotational / polar-decomposition post-processing a
//! warped solver layers on top lives elsewhere (see the crate-level `svd3`
//! helpers); this module is strictly the linear part. All of it is standard
//! linear elasticity; nothing here is derived from Unreal Engine source.

use super::tet_fem_basis::TetFemElement;

/// An isotropic linear-elastic material described by its engineering
/// constants.
///
/// The pair (`young_modulus`, `poisson_ratio`) is the most common way to
/// specify an isotropic solid; the Lamé parameters used by the constitutive
/// matrix are derived from them via [`IsotropicElasticity::lame`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotropicElasticity {
    /// Young's modulus `E`, the stiffness under uniaxial stress. Must be
    /// strictly positive.
    pub young_modulus: f32,
    /// Poisson's ratio `nu`, the ratio of transverse contraction to axial
    /// extension. Physically admissible (and numerically well posed) values lie
    /// in the open interval `(-1, 0.5)`; the incompressible limit `nu = 0.5` is
    /// excluded because the Lamé parameter `lambda` diverges there.
    pub poisson_ratio: f32,
}

impl IsotropicElasticity {
    /// Creates a material from Young's modulus `e` and Poisson's ratio `nu`.
    ///
    /// Returns `None` unless `e > 0` and `nu` lies strictly inside `(-1, 0.5)`.
    /// Non-finite inputs are rejected because the ordering comparisons are
    /// false for `NaN`.
    #[must_use]
    pub fn new(e: f32, nu: f32) -> Option<Self> {
        if e > 0.0 && nu > -1.0 && nu < 0.5 {
            Some(Self {
                young_modulus: e,
                poisson_ratio: nu,
            })
        } else {
            None
        }
    }

    /// The Lamé parameters `(lambda, mu)` of this material.
    ///
    /// `lambda = E * nu / ((1 + nu) * (1 - 2 nu))` and `mu = E / (2 (1 + nu))`,
    /// where `mu` is the shear modulus.
    #[must_use]
    pub fn lame(&self) -> (f32, f32) {
        let e = self.young_modulus;
        let nu = self.poisson_ratio;
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        (lambda, mu)
    }

    /// The 6x6 isotropic constitutive matrix `D` in `Voigt` notation with
    /// engineering shear strains, mapping the strain vector
    /// `[exx, eyy, ezz, gxy, gyz, gzx]` to the stress vector in the same order.
    ///
    /// The upper-left 3x3 block has `lambda + 2 mu` on its diagonal and
    /// `lambda` off-diagonal; the lower-right 3x3 block is `mu` times the
    /// identity (one factor of `mu` per engineering shear component).
    #[must_use]
    pub fn constitutive_matrix(&self) -> [[f32; 6]; 6] {
        let (lambda, mu) = self.lame();
        let diag = lambda + 2.0 * mu;
        [
            [diag, lambda, lambda, 0.0, 0.0, 0.0],
            [lambda, diag, lambda, 0.0, 0.0, 0.0],
            [lambda, lambda, diag, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ]
    }
}

impl Default for IsotropicElasticity {
    /// A soft rubber-like default (`E = 1e6`, `nu = 0.4`).
    fn default() -> Self {
        Self {
            young_modulus: 1.0e6,
            poisson_ratio: 0.4,
        }
    }
}

/// The 12x12 linear stiffness matrix of a single tetrahedral element.
///
/// The twelve degrees of freedom are ordered node-major as
/// `[u_x0, u_y0, u_z0, u_x1, ..., u_z3]`: component `3 i + c` is the `c`-th
/// Cartesian displacement of local node `i`. The matrix is symmetric and
/// positive semi-definite with a six-dimensional null space (three rigid
/// translations and three infinitesimal rigid rotations).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetStiffness {
    /// The dense 12x12 stiffness entries, row-major.
    pub k: [[f32; 12]; 12],
}

impl TetStiffness {
    /// The stiffness entry coupling degree of freedom `i` to degree of freedom
    /// `j`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.k[i][j]
    }

    /// The nodal forces `K u` produced by the nodal displacement vector `u`.
    #[must_use]
    pub fn apply(&self, u: &[f32; 12]) -> [f32; 12] {
        let mut out = [0.0f32; 12];
        for (out_i, krow) in out.iter_mut().zip(self.k.iter()) {
            let mut acc = 0.0f64;
            for (&uj, &kij) in u.iter().zip(krow.iter()) {
                acc += f64::from(kij) * f64::from(uj);
            }
            *out_i = acc as f32;
        }
        out
    }

    /// The elastic strain energy `0.5 * u^T K u` stored at displacement `u`.
    ///
    /// This is non-negative for every `u` because `K` is positive
    /// semi-definite.
    #[must_use]
    pub fn energy(&self, u: &[f32; 12]) -> f32 {
        let ku = self.apply(u);
        let mut acc = 0.0f64;
        for (&ui, &kui) in u.iter().zip(ku.iter()) {
            acc += f64::from(ui) * f64::from(kui);
        }
        (0.5 * acc) as f32
    }
}

/// Builds the 12x12 linear stiffness matrix `Ke = V * B^T * D * B` of one
/// constant-strain tetrahedron.
///
/// `element` supplies the rest-pose shape-function gradients and rest volume;
/// `material` supplies the isotropic constitutive matrix `D`. The matrix is
/// scaled by the *unsigned* rest volume `|V|`, so an inverted rest tetrahedron
/// (negative signed volume) still yields a physically positive stiffness.
///
/// The accumulation is carried out in `f64` and rounded to `f32` on output to
/// limit cancellation error in the triple product.
#[must_use]
pub fn element_stiffness(element: &TetFemElement, material: &IsotropicElasticity) -> TetStiffness {
    // Material matrix D (6x6), computed in f64.
    let e = f64::from(material.young_modulus);
    let nu = f64::from(material.poisson_ratio);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let diag = lambda + 2.0 * mu;
    let d: [[f64; 6]; 6] = [
        [diag, lambda, lambda, 0.0, 0.0, 0.0],
        [lambda, diag, lambda, 0.0, 0.0, 0.0],
        [lambda, lambda, diag, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, mu, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, mu, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, mu],
    ];

    // Strain-displacement matrix B (6x12) from the four shape gradients.
    let mut b = [[0.0f64; 12]; 6];
    for (i, g) in element.shape_gradients.iter().enumerate() {
        let bx = f64::from(g.x);
        let by = f64::from(g.y);
        let bz = f64::from(g.z);
        let x = 3 * i;
        let y = 3 * i + 1;
        let z = 3 * i + 2;
        // exx = du_x/dx
        b[0][x] = bx;
        // eyy = du_y/dy
        b[1][y] = by;
        // ezz = du_z/dz
        b[2][z] = bz;
        // gxy = du_x/dy + du_y/dx
        b[3][x] = by;
        b[3][y] = bx;
        // gyz = du_y/dz + du_z/dy
        b[4][y] = bz;
        b[4][z] = by;
        // gzx = du_z/dx + du_x/dz
        b[5][x] = bz;
        b[5][z] = bx;
    }

    // db = D * B (6x12).
    let mut db = [[0.0f64; 12]; 6];
    for r in 0..6 {
        for c in 0..12 {
            let mut acc = 0.0f64;
            for k in 0..6 {
                acc += d[r][k] * b[k][c];
            }
            db[r][c] = acc;
        }
    }

    // Ke = V * B^T * db (12x12), V the unsigned rest volume.
    let volume = f64::from(element.rest_volume.abs());
    let mut k = [[0.0f32; 12]; 12];
    for a in 0..12 {
        for c in 0..12 {
            let mut acc = 0.0f64;
            for r in 0..6 {
                acc += b[r][a] * db[r][c];
            }
            k[a][c] = (volume * acc) as f32;
        }
    }

    TetStiffness { k }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Vertices of the canonical unit (reference) tetrahedron.
    const UNIT: [Vec3; 4] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    /// A generic skewed tetrahedron (non-degenerate, no axis-aligned faces).
    const SKEW: [Vec3; 4] = [
        Vec3::new(0.1, -0.2, 0.3),
        Vec3::new(1.3, 0.2, -0.1),
        Vec3::new(-0.2, 1.1, 0.4),
        Vec3::new(0.3, 0.5, 1.4),
    ];

    fn unit_element() -> TetFemElement {
        TetFemElement::from_rest(UNIT[0], UNIT[1], UNIT[2], UNIT[3], 1e-12).unwrap()
    }

    fn skew_element() -> TetFemElement {
        TetFemElement::from_rest(SKEW[0], SKEW[1], SKEW[2], SKEW[3], 1e-12).unwrap()
    }

    #[test]
    fn constitutive_matrix_matches_lame_special_case() {
        // E = 1, nu = 0  ->  lambda = 0, mu = 0.5, so D = diag(1,1,1,.5,.5,.5).
        let mat = IsotropicElasticity::new(1.0, 0.0).unwrap();
        let (lambda, mu) = mat.lame();
        assert!(lambda.abs() <= 1e-7, "lambda = {lambda}");
        assert!((mu - 0.5).abs() <= 1e-7, "mu = {mu}");
        let d = mat.constitutive_matrix();
        let expected = [1.0f32, 1.0, 1.0, 0.5, 0.5, 0.5];
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { expected[i] } else { 0.0 };
                assert!(
                    (d[i][j] - want).abs() <= 1e-6,
                    "D[{i}][{j}] = {} want {want}",
                    d[i][j]
                );
            }
        }
    }

    #[test]
    fn stiffness_is_symmetric() {
        let elem = skew_element();
        let mat = IsotropicElasticity::new(2.5e4, 0.3).unwrap();
        let s = element_stiffness(&elem, &mat);
        let scale = {
            let mut m = 0.0f32;
            for i in 0..12 {
                for j in 0..12 {
                    m = m.max(s.get(i, j).abs());
                }
            }
            m
        };
        assert!(scale > 0.0);
        for i in 0..12 {
            for j in 0..12 {
                let diff = (s.get(i, j) - s.get(j, i)).abs();
                assert!(diff <= 1e-4 * scale, "asymmetry at ({i},{j}) = {diff}");
            }
        }
    }

    #[test]
    fn rigid_translation_is_in_the_null_space() {
        let elem = skew_element();
        let mat = IsotropicElasticity::new(1.0e3, 0.33).unwrap();
        let s = element_stiffness(&elem, &mat);
        // Translate every node by the same vector t = (0.7, -0.4, 1.2).
        let mut u = [0.0f32; 12];
        for i in 0..4 {
            u[3 * i] = 0.7;
            u[3 * i + 1] = -0.4;
            u[3 * i + 2] = 1.2;
        }
        let f = s.apply(&u);
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() <= 1e-2, "translation force[{i}] = {fi}");
        }
        assert!(s.energy(&u).abs() <= 1e-3, "translation energy nonzero");
    }

    #[test]
    fn infinitesimal_rotation_is_in_the_null_space() {
        let elem = unit_element();
        let mat = IsotropicElasticity::new(1.0, 0.25).unwrap();
        let s = element_stiffness(&elem, &mat);
        // u_i = omega x x_i with omega = (0,0,1): a linearized rigid rotation.
        let omega = Vec3::new(0.0, 0.0, 1.0);
        let mut u = [0.0f32; 12];
        for (i, x) in UNIT.iter().enumerate() {
            let w = omega.cross(*x);
            u[3 * i] = w.x;
            u[3 * i + 1] = w.y;
            u[3 * i + 2] = w.z;
        }
        let f = s.apply(&u);
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() <= 1e-5, "rotation force[{i}] = {fi}");
        }
        assert!(s.energy(&u).abs() <= 1e-6, "rotation energy nonzero");
    }

    #[test]
    fn energy_is_non_negative() {
        let elem = skew_element();
        let mat = IsotropicElasticity::new(5.0e3, 0.2).unwrap();
        let s = element_stiffness(&elem, &mat);
        // A spread of pseudo-random displacement vectors.
        let seeds = [
            [
                0.3f32, -0.1, 0.2, 0.5, 0.9, -0.4, -0.7, 0.1, 0.6, 0.2, -0.3, 0.8,
            ],
            [
                1.0, 0.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1, 0.0,
            ],
            [
                -0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.05, -0.05, 0.5,
            ],
        ];
        for u in &seeds {
            let energy = s.energy(u);
            assert!(energy >= -1e-4, "negative energy {energy}");
        }
        // A pure stretch must store strictly positive energy.
        let mut stretch = [0.0f32; 12];
        stretch[3] = 0.1; // move node 1 along +x
        assert!(s.energy(&stretch) > 1e-6, "stretch energy not positive");
    }

    #[test]
    fn stiffness_scales_linearly_with_youngs_modulus() {
        let elem = skew_element();
        let m1 = IsotropicElasticity::new(1.0e3, 0.3).unwrap();
        let m2 = IsotropicElasticity::new(3.0e3, 0.3).unwrap();
        let s1 = element_stiffness(&elem, &m1);
        let s2 = element_stiffness(&elem, &m2);
        for i in 0..12 {
            for j in 0..12 {
                let a = s1.get(i, j);
                let b = s2.get(i, j);
                let diff = (b - 3.0 * a).abs();
                assert!(diff <= 1e-3 * (1.0 + a.abs()), "non-linear at ({i},{j})");
            }
        }
    }

    #[test]
    fn stiffness_scales_linearly_with_volume() {
        // Scaling the whole tetrahedron by s multiplies Ke by s:
        // V ~ s^3, gradients ~ 1/s, so V * B^T D B ~ s^3 * s^-2 = s.
        let s_fac = 2.0f32;
        let base = skew_element();
        let scaled = TetFemElement::from_rest(
            SKEW[0] * s_fac,
            SKEW[1] * s_fac,
            SKEW[2] * s_fac,
            SKEW[3] * s_fac,
            1e-12,
        )
        .unwrap();
        let mat = IsotropicElasticity::new(4.2e3, 0.28).unwrap();
        let sb = element_stiffness(&base, &mat);
        let ss = element_stiffness(&scaled, &mat);
        for i in 0..12 {
            for j in 0..12 {
                let a = sb.get(i, j);
                let b = ss.get(i, j);
                let diff = (b - s_fac * a).abs();
                assert!(
                    diff <= 1e-3 * (1.0 + a.abs()),
                    "volume scaling off at ({i},{j}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn invalid_materials_are_rejected() {
        assert!(IsotropicElasticity::new(0.0, 0.3).is_none());
        assert!(IsotropicElasticity::new(-1.0, 0.3).is_none());
        assert!(IsotropicElasticity::new(1.0, 0.5).is_none());
        assert!(IsotropicElasticity::new(1.0, 0.6).is_none());
        assert!(IsotropicElasticity::new(1.0, -1.0).is_none());
        assert!(IsotropicElasticity::new(1.0, -1.5).is_none());
        assert!(IsotropicElasticity::new(f32::NAN, 0.3).is_none());
        assert!(IsotropicElasticity::new(1.0, f32::NAN).is_none());
        // A representative admissible material is accepted.
        assert!(IsotropicElasticity::new(1.0, 0.49).is_some());
    }

    #[test]
    fn element_stiffness_is_deterministic() {
        let elem = skew_element();
        let mat = IsotropicElasticity::new(7.0e3, 0.31).unwrap();
        let a = element_stiffness(&elem, &mat);
        let b = element_stiffness(&elem, &mat);
        assert_eq!(a, b);
    }
}
