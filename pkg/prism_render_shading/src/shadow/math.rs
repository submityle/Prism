//! Column-major 4x4 matrix and small vector helpers used by the shadow
//! projection.  Kept dependency-free and restricted to the arithmetic that maps
//! one-to-one onto WESL builtins (`mat4x4<f32> * vec4<f32>`, `/`, `sqrt`) so the
//! CPU golden stays byte-for-byte in step with the future `shadow.wesl` twin.
//!
//! Matrices are stored **column-major** as `[f32; 16]`, matching WGSL's
//! `mat4x4<f32>` memory layout and `glam::Mat4::to_cols_array`, so a matrix
//! uploaded to the GPU and the one fed to this reference are byte-identical.

/// Column-major 4x4 matrix: element `(row, col)` lives at index `col * 4 + row`.
pub type Mat4 = [f32; 16];

/// Multiplies the column-major `matrix` by the homogeneous point
/// `(point, 1.0)`, returning the full `vec4` (perspective divide not applied).
pub fn transform_point(matrix: &Mat4, point: [f32; 3]) -> [f32; 4] {
    let [x, y, z] = point;
    [
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
    ]
}

/// Multiplies the column-major `matrix` by the homogeneous direction
/// `(direction, 0.0)`.  The translation column is ignored, as for a normal or
/// light vector.
pub fn transform_direction(matrix: &Mat4, direction: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = direction;
    [
        matrix[0] * x + matrix[4] * y + matrix[8] * z,
        matrix[1] * x + matrix[5] * y + matrix[9] * z,
        matrix[2] * x + matrix[6] * y + matrix[10] * z,
    ]
}

/// Euclidean length of a 3-vector.
pub(crate) fn length3(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

/// Column-major 4x4 matrix product `a * b` (same convention as
/// [`transform_point`]: element `(row, col)` at index `col * 4 + row`).
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0_f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0_f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Right-handed look-at **view** matrix (column-major), mapping world space
/// into a camera/light eye space that looks from `eye` toward `center` with the
/// given `up`.  The view looks down its local `-z`, matching `glam::Mat4::
/// look_at_rh` and the perspective/orthographic conventions used across wgpu.
pub fn look_at_rh(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize3(sub3(center, eye));
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot3(s, eye),
        -dot3(u, eye),
        dot3(f, eye),
        1.0,
    ]
}

/// Right-handed orthographic projection (column-major) mapping the box
/// `[left, right] x [bottom, top] x [-near, -far]` (view space, looking down
/// `-z`) into wgpu clip space with `z` in `[0, 1]`.  Mirrors
/// `glam::Mat4::orthographic_rh` so an uploaded projection and this reference
/// are byte-identical.
pub fn orthographic_rh_01(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let rcp_width = (right - left).recip();
    let rcp_height = (top - bottom).recip();
    let r = (near - far).recip();
    [
        rcp_width + rcp_width,
        0.0,
        0.0,
        0.0,
        0.0,
        rcp_height + rcp_height,
        0.0,
        0.0,
        0.0,
        0.0,
        r,
        0.0,
        -(left + right) * rcp_width,
        -(top + bottom) * rcp_height,
        r * near,
        1.0,
    ]
}

/// Dot product of two 3-vectors.
pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b`.
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a x b`.
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes `value`; returns a zero vector for degenerate input (callers in
/// this module always feed well-conditioned frustum geometry).
pub(crate) fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(value, value);
    if len_sq > 1.0e-20 {
        let inv = len_sq.sqrt().recip();
        [value[0] * inv, value[1] * inv, value[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A column-major translation matrix must move a point by its 4th column.
    #[test]
    fn transform_point_applies_translation_column() {
        let mut m = identity();
        m[12] = 2.0;
        m[13] = -3.0;
        m[14] = 5.0;
        assert_eq!(transform_point(&m, [1.0, 1.0, 1.0]), [3.0, -2.0, 6.0, 1.0]);
    }

    /// A direction ignores the translation column (w = 0).
    #[test]
    fn transform_direction_ignores_translation() {
        let mut m = identity();
        m[12] = 10.0;
        m[13] = 10.0;
        m[14] = 10.0;
        assert_eq!(transform_direction(&m, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }


    fn identity() -> Mat4 {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m
    }
}
