//! Perlin "improved" gradient noise (2D/3D), deterministic given a seed.

use crate::noise::perm::Permutation;

/// Perlin gradient-noise generator. Output is in approximately `[-1, 1]` and is
/// `C^1`-continuous. Construct with a seed for a reproducible field.
#[derive(Clone)]
pub struct Perlin {
    perm: Permutation,
}

/// 6t^5 - 15t^4 + 10t^3: Ken Perlin's quintic fade (zero 1st/2nd derivatives
/// at the endpoints, for seamless interpolation).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// 2D gradient dotted with the distance vector (eight evenly-spaced directions).
#[inline]
fn grad2(hash: u8, x: f32, y: f32) -> f32 {
    const H: f32 = core::f32::consts::FRAC_1_SQRT_2;
    // Unit-length gradients; the field is scaled by `SQRT_2` so the result
    // lands in `[-1, 1]`.
    const G: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (H, H),
        (-H, H),
        (H, -H),
        (-H, -H),
    ];
    let (gx, gy) = G[(hash & 7) as usize];
    gx * x + gy * y
}

/// 3D gradient dotted with the distance vector (Perlin's 12 edge gradients).
#[inline]
fn grad3(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let u = if h & 1 == 0 { u } else { -u };
    let v = if h & 2 == 0 { v } else { -v };
    u + v
}

impl Perlin {
    /// Scale applied to raw 2D output so it fills `[-1, 1]`.
    const SCALE2: f32 = core::f32::consts::SQRT_2;
    /// Scale applied to raw 3D output. This "improved noise" 3D kernel is
    /// already bounded within `[-1, 1]` (empirical peak `~0.9`), so unlike the
    /// 2D path it needs no inflation; the constant is kept for symmetry.
    const SCALE3: f32 = 1.0;

    /// Create a generator from a seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { perm: Permutation::new(seed) }
    }

    /// Sample 2D noise at `(x, y)`.
    #[inline]
    pub fn get2(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor();
        let yi = y.floor();
        let xf = x - xi;
        let yf = y - yi;
        let (xi, yi) = (xi as i32, yi as i32);

        let p = &self.perm;
        let a = i32::from(p.hash(xi)) + yi;
        let b = i32::from(p.hash(xi + 1)) + yi;

        let u = fade(xf);
        let v = fade(yf);

        let aa = p.hash(a);
        let ab = p.hash(a + 1);
        let ba = p.hash(b);
        let bb = p.hash(b + 1);

        let x1 = lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u);
        let x2 = lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u);
        lerp(x1, x2, v) * Self::SCALE2
    }

    /// Sample 3D noise at `(x, y, z)`.
    #[inline]
    pub fn get3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor();
        let yi = y.floor();
        let zi = z.floor();
        let xf = x - xi;
        let yf = y - yi;
        let zf = z - zi;
        let (xi, yi, zi) = (xi as i32, yi as i32, zi as i32);

        let p = &self.perm;
        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let a = i32::from(p.hash(xi)) + yi;
        let b = i32::from(p.hash(xi + 1)) + yi;
        let aa = i32::from(p.hash(a)) + zi;
        let ab = i32::from(p.hash(a + 1)) + zi;
        let ba = i32::from(p.hash(b)) + zi;
        let bb = i32::from(p.hash(b + 1)) + zi;

        let g = |idx: i32, dx: f32, dy: f32, dz: f32| grad3(p.hash(idx), dx, dy, dz);

        let x1 = lerp(g(aa, xf, yf, zf), g(ba, xf - 1.0, yf, zf), u);
        let x2 = lerp(g(ab, xf, yf - 1.0, zf), g(bb, xf - 1.0, yf - 1.0, zf), u);
        let y1 = lerp(x1, x2, v);

        let x3 = lerp(g(aa + 1, xf, yf, zf - 1.0), g(ba + 1, xf - 1.0, yf, zf - 1.0), u);
        let x4 =
            lerp(g(ab + 1, xf, yf - 1.0, zf - 1.0), g(bb + 1, xf - 1.0, yf - 1.0, zf - 1.0), u);
        let y2 = lerp(x3, x4, v);

        lerp(y1, y2, w) * Self::SCALE3
    }
}
