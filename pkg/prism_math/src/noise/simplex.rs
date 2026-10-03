//! Simplex noise (2D/3D), deterministic given a seed.
//!
//! Simplex noise (Ken Perlin, 2001; this implementation follows Stefan
//! Gustavson's public-domain formulation) uses a simplicial grid to avoid the
//! directional axis-aligned artifacts of classic Perlin noise and to scale
//! better to higher dimensions. Output is in approximately `[-1, 1]`.

use crate::noise::perm::Permutation;

/// Simplex gradient-noise generator.
#[derive(Clone)]
pub struct Simplex {
    perm: Permutation,
}

/// The 12 edge-midpoint gradients of a cube, shared by the 2D/3D kernels.
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

#[inline]
fn dot2(g: [f32; 3], x: f32, y: f32) -> f32 {
    g[0] * x + g[1] * y
}

#[inline]
fn dot3(g: [f32; 3], x: f32, y: f32, z: f32) -> f32 {
    g[0] * x + g[1] * y + g[2] * z
}

impl Simplex {
    /// Create a generator from a seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { perm: Permutation::new(seed) }
    }

    #[inline]
    fn grad_index(&self, i: i32) -> usize {
        (self.perm.hash(i) as usize) % 12
    }

    /// Sample 2D simplex noise at `(xin, yin)`.
    pub fn get2(&self, xin: f32, yin: f32) -> f32 {
        // Skew/unskew factors for the 2D simplex grid.
        const F2: f32 = 0.366_025_42; // 0.5*(sqrt(3)-1)
        const G2: f32 = 0.211_324_87; // (3-sqrt(3))/6

        let s = (xin + yin) * F2;
        let i = (xin + s).floor();
        let j = (yin + s).floor();
        let t = (i + j) * G2;
        let x0 = xin - (i - t);
        let y0 = yin - (j - t);

        // Which simplex (triangle) are we in?
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f32 + G2;
        let y1 = y0 - j1 as f32 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let ii = i as i32;
        let jj = j as i32;
        let gi0 = self.grad_index(ii + i32::from(self.perm.hash(jj)));
        let gi1 = self.grad_index(ii + i1 + i32::from(self.perm.hash(jj + j1)));
        let gi2 = self.grad_index(ii + 1 + i32::from(self.perm.hash(jj + 1)));

        let n0 = Self::corner2(x0, y0, GRAD3[gi0]);
        let n1 = Self::corner2(x1, y1, GRAD3[gi1]);
        let n2 = Self::corner2(x2, y2, GRAD3[gi2]);

        // Scale to roughly [-1, 1].
        70.0 * (n0 + n1 + n2)
    }

    #[inline]
    fn corner2(x: f32, y: f32, g: [f32; 3]) -> f32 {
        let t = 0.5 - x * x - y * y;
        if t < 0.0 {
            0.0
        } else {
            let t2 = t * t;
            t2 * t2 * dot2(g, x, y)
        }
    }

    /// Sample 3D simplex noise at `(xin, yin, zin)`.
    pub fn get3(&self, xin: f32, yin: f32, zin: f32) -> f32 {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;

        let s = (xin + yin + zin) * F3;
        let i = (xin + s).floor();
        let j = (yin + s).floor();
        let k = (zin + s).floor();
        let t = (i + j + k) * G3;
        let x0 = xin - (i - t);
        let y0 = yin - (j - t);
        let z0 = zin - (k - t);

        // Determine simplex corner traversal order.
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };

        let x1 = x0 - i1 as f32 + G3;
        let y1 = y0 - j1 as f32 + G3;
        let z1 = z0 - k1 as f32 + G3;
        let x2 = x0 - i2 as f32 + 2.0 * G3;
        let y2 = y0 - j2 as f32 + 2.0 * G3;
        let z2 = z0 - k2 as f32 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let ii = i as i32;
        let jj = j as i32;
        let kk = k as i32;

        let h = |a: i32, b: i32, c: i32| i32::from(self.perm.hash(a + i32::from(self.perm.hash(b + i32::from(self.perm.hash(c))))));
        let gi0 = (h(ii, jj, kk) as usize) % 12;
        let gi1 = (h(ii + i1, jj + j1, kk + k1) as usize) % 12;
        let gi2 = (h(ii + i2, jj + j2, kk + k2) as usize) % 12;
        let gi3 = (h(ii + 1, jj + 1, kk + 1) as usize) % 12;

        let n0 = Self::corner3(x0, y0, z0, GRAD3[gi0]);
        let n1 = Self::corner3(x1, y1, z1, GRAD3[gi1]);
        let n2 = Self::corner3(x2, y2, z2, GRAD3[gi2]);
        let n3 = Self::corner3(x3, y3, z3, GRAD3[gi3]);

        32.0 * (n0 + n1 + n2 + n3)
    }

    #[inline]
    fn corner3(x: f32, y: f32, z: f32, g: [f32; 3]) -> f32 {
        let t = 0.6 - x * x - y * y - z * z;
        if t < 0.0 {
            0.0
        } else {
            let t2 = t * t;
            t2 * t2 * dot3(g, x, y, z)
        }
    }
}
