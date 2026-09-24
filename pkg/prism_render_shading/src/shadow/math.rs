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
