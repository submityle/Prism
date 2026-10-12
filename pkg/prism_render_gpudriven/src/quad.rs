//! The shared quad geometry the GPU-driven scene draw and the naive per-object
//! baseline both rasterize.
//!
//! One instance is a single triangle-strip quad. The corner order is fixed
//! (bit 0 selects x, bit 1 selects y) and mirrored verbatim by both WGSL
//! shaders, so the CPU reference can predict exactly where an instance lands on
//! screen.

use crate::draw::INDICES_PER_INSTANCE;

/// Vertices in one quad: the size of one index block.
pub const QUAD_VERTICES: u32 = 4;

/// The quad's half-extent in world units.
///
/// It matches the instance AABB half-extent ([`crate::scene::DEFAULT_HALF_EXTENT`])
/// so a drawn quad fills exactly the volume the frustum cull classified.
pub const QUAD_HALF_EXTENT: f32 = 0.05;

// The index table is addressed in blocks of `INDICES_PER_INSTANCE`, so the
// geometry must have exactly that many vertices.
const _: () = assert!(QUAD_VERTICES == INDICES_PER_INSTANCE);

/// The `(x, y)` corner offsets in `[0, 1]` for a triangle-strip corner index.
///
/// Corner `c` has `x = c & 1` and `y = c >> 1`: the strip order `0, 1, 2, 3` is
/// `(0,0) (1,0) (0,1) (1,1)`.
#[must_use]
pub fn corner_offset(corner: u32) -> [f32; 2] {
    [
        f32::from((corner & 1) as u8),
        f32::from((corner >> 1) as u8),
    ]
}

/// The four world-space XY corners of an instance quad centred on `center`.
#[must_use]
pub fn corners(center: [f32; 2]) -> [[f32; 2]; QUAD_VERTICES as usize] {
    let mut corners = [[0.0f32; 2]; QUAD_VERTICES as usize];
    for (index, corner) in corners.iter_mut().enumerate() {
        let offset = corner_offset(index as u32);
        corner[0] = center[0] + (offset[0] - 0.5) * (QUAD_HALF_EXTENT * 2.0);
        corner[1] = center[1] + (offset[1] - 0.5) * (QUAD_HALF_EXTENT * 2.0);
    }
    corners
}

/// The indices of one triangle-strip quad, shared by every instance.
pub const QUAD_STRIP_INDICES: [u16; QUAD_VERTICES as usize] = [0, 1, 2, 3];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_offsets_follow_the_strip_order() {
        assert_eq!(corner_offset(0), [0.0, 0.0]);
        assert_eq!(corner_offset(1), [1.0, 0.0]);
        assert_eq!(corner_offset(2), [0.0, 1.0]);
        assert_eq!(corner_offset(3), [1.0, 1.0]);
    }

    #[test]
    fn corners_bracket_the_center_symmetrically() {
        let corners = corners([1.0, -2.0]);
        let min_x = corners.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
        let max_x = corners
            .iter()
            .map(|c| c[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max_x - min_x - QUAD_HALF_EXTENT * 2.0).abs() < 1e-6);
        assert!(((min_x + max_x) * 0.5 - 1.0).abs() < 1e-6);
    }
}
