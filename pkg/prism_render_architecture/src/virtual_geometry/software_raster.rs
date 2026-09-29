//! Deterministic vis-buffer software rasterizer (CPU golden standard).
//!
//! Virtualized geometry routes near sub-pixel clusters through a compute
//! software rasterizer that writes a 64-bit *visibility buffer* rather than
//! shaded pixels. The GPU compute path is backend-owned, but its rasterization
//! math must be pinned to a single, backend-independent reference so the
//! compute twin can be validated bit-for-bit. This module is that reference: a
//! watertight, top-left-rule triangle rasterizer with reversed-Z depth
//! compositing, expressed entirely on the CPU with no GPU or math-crate
//! dependency.
//!
//! # Vis-buffer layout
//!
//! Each pixel is a `u64` packing a depth key in the high 32 bits and an opaque
//! payload (typically a packed cluster/triangle id) in the low 32 bits:
//!
//! ```text
//! bit 63                              32 31                            0
//! +-------------------------------------+------------------------------+
//! |            depth key (u32)          |        payload (u32)         |
//! +-------------------------------------+------------------------------+
//! ```
//!
//! Depth uses **reversed-Z**: input depth is in `[0, 1]` with `1.0` nearest.
//! [`encode_depth`] maps it so nearer surfaces produce a *larger* key, which
//! lets the whole `u64` be composited with a single `max` comparison. A cleared
//! pixel is `0` (the farthest possible key), so the first covering triangle
//! always wins over the clear value.
//!
//! # Coverage and watertightness
//!
//! Coverage uses the signed edge function with a strict top-left fill rule so
//! that a shared edge between two adjacent triangles is owned by exactly one of
//! them: no double-covered pixels, no cracks. Depth is interpolated with
//! screen-space-linear barycentrics, matching a hardware depth buffer for the
//! NDC depth values fed in here.

/// Screen-space vertex fed to the rasterizer.
///
/// `pos` is in pixel space with a y-down origin (pixel centers live at
/// `(x + 0.5, y + 0.5)`), and `depth` is the reversed-Z NDC depth in `[0, 1]`
/// with `1.0` nearest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenVertex {
    /// Pixel-space position, y-down.
    pub pos: [f32; 2],
    /// Reversed-Z depth in `[0, 1]`, `1.0` nearest.
    pub depth: f32,
}

impl ScreenVertex {
    /// Builds a screen vertex from a pixel-space position and reversed-Z depth.
    #[must_use]
    pub const fn new(pos: [f32; 2], depth: f32) -> Self {
        Self { pos, depth }
    }
}

/// Packs a depth key and payload into one vis-buffer word.
#[must_use]
pub const fn pack_vis(depth_key: u32, payload: u32) -> u64 {
    ((depth_key as u64) << 32) | (payload as u64)
}

/// Extracts the depth key from a vis-buffer word.
#[must_use]
pub const fn vis_depth(packed: u64) -> u32 {
    (packed >> 32) as u32
}

/// Extracts the payload from a vis-buffer word.
#[must_use]
pub const fn vis_payload(packed: u64) -> u32 {
    (packed & 0xFFFF_FFFF) as u32
}

/// Encodes reversed-Z depth in `[0, 1]` into a compositing key.
///
/// The input is clamped to `[0, 1]`; `1.0` (nearest) maps to `u32::MAX` and
/// `0.0` (farthest) maps to `0`. An `f64` intermediate avoids the precision
/// loss of scaling near `u32::MAX` in `f32`.
#[must_use]
pub fn encode_depth(depth: f32) -> u32 {
    let clamped = f64::from(depth.clamp(0.0, 1.0));
    (clamped * f64::from(u32::MAX)) as u32
}

/// Signed edge function of point `p` against the directed edge `a -> b`.
///
/// Positive when `p` is to the interior side of a positive-area (counter-
/// clockwise in this y-down space) triangle edge.
#[must_use]
#[inline]
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Returns `true` when the directed edge `a -> b` is a top or left edge.
///
/// Under the y-down, positive-area convention used here a *top* edge is
/// horizontal and pointing in `+x` (`dy == 0 && dx > 0`) and a *left* edge
/// points upward (`dy < 0`). A pixel exactly on such an edge is considered
/// covered, which makes shared edges belong to exactly one triangle.
#[must_use]
#[inline]
fn is_top_left(a: [f32; 2], b: [f32; 2]) -> bool {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    dy < 0.0 || (dy == 0.0 && dx > 0.0)
}

/// A dense visibility buffer of packed depth/payload words.
#[derive(Clone, Debug)]
pub struct VisBuffer {
    width: u32,
    height: u32,
    pixels: Vec<u64>,
}

impl VisBuffer {
    /// Allocates a cleared vis-buffer of the given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![0; len],
        }
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Clears every pixel back to the farthest key with an empty payload.
    pub fn clear(&mut self) {
        self.pixels.iter_mut().for_each(|p| *p = 0);
    }

    /// Returns the packed word at `(x, y)`, or `0` when out of bounds.
    #[must_use]
    pub fn at(&self, x: u32, y: u32) -> u64 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Read-only view of the packed pixel storage in row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[u64] {
        &self.pixels
    }

    /// Composites a candidate word at `(x, y)` keeping the nearest surface.
    ///
    /// Because depth occupies the high bits and reversed-Z makes nearer
    /// surfaces larger, the nearest surface is simply the maximum `u64`.
    #[inline]
    fn composite(&mut self, x: u32, y: u32, candidate: u64) {
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        let slot = &mut self.pixels[idx];
        if candidate > *slot {
            *slot = candidate;
        }
    }
}

/// Rasterizes one triangle into `buffer`, compositing by nearest depth.
///
/// Vertices may be supplied in either winding: a negative-area triangle has its
/// last two vertices (position and depth together) swapped so the interior test
/// runs against a positive area. Degenerate (zero-area) triangles are skipped.
/// When `cull_back` is set, triangles whose *original* winding is back-facing
/// (non-positive area before any swap) are skipped entirely.
///
/// `payload` is written verbatim into the low 32 bits of every covered pixel.
/// The scan is clamped to the buffer bounds, so vertices off-screen never index
/// out of range.
pub fn rasterize_triangle(
    buffer: &mut VisBuffer,
    vertices: [ScreenVertex; 3],
    payload: u32,
    cull_back: bool,
) {
    let [v0, mut v1, mut v2] = vertices;

    let raw_area = edge(v0.pos, v1.pos, v2.pos);
    if raw_area == 0.0 {
        // Degenerate triangle: no coverage.
        return;
    }
    if cull_back && raw_area < 0.0 {
        // Back-facing under the original winding.
        return;
    }
    if raw_area < 0.0 {
        core::mem::swap(&mut v1, &mut v2);
    }
    let area = edge(v0.pos, v1.pos, v2.pos);
    // After the swap the area is strictly positive.
    let inv_area = 1.0 / area;

    // Screen-space bounding box, clamped to the buffer.
    let min_x = v0.pos[0].min(v1.pos[0]).min(v2.pos[0]);
    let max_x = v0.pos[0].max(v1.pos[0]).max(v2.pos[0]);
    let min_y = v0.pos[1].min(v1.pos[1]).min(v2.pos[1]);
    let max_y = v0.pos[1].max(v1.pos[1]).max(v2.pos[1]);

    let x_start = clamp_floor(min_x, buffer.width);
    let x_end = clamp_ceil(max_x, buffer.width);
    let y_start = clamp_floor(min_y, buffer.height);
    let y_end = clamp_ceil(max_y, buffer.height);

    // Edge v1->v2 owns e0/bary0, v2->v0 owns e1/bary1, v0->v1 owns e2/bary2.
    let tl0 = is_top_left(v1.pos, v2.pos);
    let tl1 = is_top_left(v2.pos, v0.pos);
    let tl2 = is_top_left(v0.pos, v1.pos);

    for y in y_start..y_end {
        for x in x_start..x_end {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let e0 = edge(v1.pos, v2.pos, p);
            let e1 = edge(v2.pos, v0.pos, p);
            let e2 = edge(v0.pos, v1.pos, p);

            let inside = covers(e0, tl0) && covers(e1, tl1) && covers(e2, tl2);
            if !inside {
                continue;
            }

            let b0 = e0 * inv_area;
            let b1 = e1 * inv_area;
            let b2 = e2 * inv_area;
            let depth = b0 * v0.depth + b1 * v1.depth + b2 * v2.depth;

            let candidate = pack_vis(encode_depth(depth), payload);
            buffer.composite(x, y, candidate);
        }
    }
}

/// Fill test for one edge value under the top-left rule.
#[must_use]
#[inline]
fn covers(edge_value: f32, top_left: bool) -> bool {
    edge_value > 0.0 || (edge_value == 0.0 && top_left)
}

/// Floors a coordinate and clamps it into `[0, limit]` as a start index.
#[must_use]
#[inline]
fn clamp_floor(value: f32, limit: u32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    let floored = value.floor() as u32;
    floored.min(limit)
}

/// Ceils a coordinate to an exclusive end index clamped into `[0, limit]`.
#[must_use]
#[inline]
fn clamp_ceil(value: f32, limit: u32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    let ceiled = value.ceil() as u32;
    ceiled.min(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
        ScreenVertex::new([x, y], depth)
    }

    #[test]
    fn pack_round_trips_depth_and_payload() {
        let packed = pack_vis(0xDEAD_BEEF, 0x0BAD_F00D);
        assert_eq!(vis_depth(packed), 0xDEAD_BEEF);
        assert_eq!(vis_payload(packed), 0x0BAD_F00D);
    }

    #[test]
    fn nearer_reversed_z_packs_larger() {
        // depth 1.0 is nearest -> largest key -> largest word.
        let near = pack_vis(encode_depth(1.0), 0);
        let far = pack_vis(encode_depth(0.0), u32::MAX);
        assert!(near > far, "nearest surface must dominate the composite");
        assert_eq!(encode_depth(1.0), u32::MAX);
        assert_eq!(encode_depth(0.0), 0);
    }

    #[test]
    fn edge_sign_matches_orientation() {
        // Reference CCW (positive-area) triangle in y-down space.
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 4.0];
        assert!(edge(a, b, c) > 0.0);
        // A point clearly outside the a->b edge is negative.
        assert!(edge(a, b, [1.0, -1.0]) < 0.0);
    }

    #[test]
    fn top_left_classification() {
        // Top edge: horizontal pointing +x.
        assert!(is_top_left([0.0, 0.0], [4.0, 0.0]));
        // Left edge: pointing up (dy < 0).
        assert!(is_top_left([0.0, 4.0], [0.0, 0.0]));
        // Diagonal descending edge is neither.
        assert!(!is_top_left([0.0, 0.0], [4.0, 4.0]));
        // Bottom edge pointing -x is neither.
        assert!(!is_top_left([4.0, 4.0], [0.0, 4.0]));
    }

    #[test]
    fn watertight_shared_diagonal_covers_each_pixel_once() {
        // Split a 4x4 quad along the (0,0)-(4,4) diagonal into two triangles
        // and confirm every interior pixel is covered by exactly one of them.
        let mut a = VisBuffer::new(4, 4);
        let mut b = VisBuffer::new(4, 4);
        // Triangle A: (0,0),(4,0),(4,4); triangle B: (0,0),(4,4),(0,4).
        rasterize_triangle(
            &mut a,
            [sv(0.0, 0.0, 0.5), sv(4.0, 0.0, 0.5), sv(4.0, 4.0, 0.5)],
            1,
            false,
        );
        rasterize_triangle(
            &mut b,
            [sv(0.0, 0.0, 0.5), sv(4.0, 4.0, 0.5), sv(0.0, 4.0, 0.5)],
            2,
            false,
        );
        for y in 0..4 {
            for x in 0..4 {
                let hit_a = vis_payload(a.at(x, y)) == 1;
                let hit_b = vis_payload(b.at(x, y)) == 2;
                assert!(
                    hit_a ^ hit_b,
                    "pixel ({x},{y}) must be covered exactly once, got a={hit_a} b={hit_b}"
                );
            }
        }
    }

    #[test]
    fn nearer_triangle_wins_regardless_of_draw_order() {
        let far = [sv(0.0, 0.0, 0.2), sv(4.0, 0.0, 0.2), sv(0.0, 4.0, 0.2)];
        let near = [sv(0.0, 0.0, 0.9), sv(4.0, 0.0, 0.9), sv(0.0, 4.0, 0.9)];

        let mut near_first = VisBuffer::new(4, 4);
        rasterize_triangle(&mut near_first, near, 7, false);
        rasterize_triangle(&mut near_first, far, 3, false);

        let mut far_first = VisBuffer::new(4, 4);
        rasterize_triangle(&mut far_first, far, 3, false);
        rasterize_triangle(&mut far_first, near, 7, false);

        // Pixel (0,0) center (0.5,0.5) is strictly inside both triangles.
        assert_eq!(vis_payload(near_first.at(0, 0)), 7);
        assert_eq!(vis_payload(far_first.at(0, 0)), 7);
        assert_eq!(near_first.at(0, 0), far_first.at(0, 0));
    }

    #[test]
    fn back_face_culling_skips_reversed_winding() {
        // Clockwise (negative-area) winding of the reference triangle.
        let cw = [sv(0.0, 0.0, 0.5), sv(0.0, 4.0, 0.5), sv(4.0, 0.0, 0.5)];

        let mut culled = VisBuffer::new(4, 4);
        rasterize_triangle(&mut culled, cw, 5, true);
        assert!(culled.pixels().iter().all(|&p| p == 0), "back face must be culled");

        // Without culling the same winding still rasterizes (after reorder).
        let mut kept = VisBuffer::new(4, 4);
        rasterize_triangle(&mut kept, cw, 5, false);
        assert_eq!(vis_payload(kept.at(0, 0)), 5);
    }

    #[test]
    fn degenerate_triangle_writes_nothing() {
        let mut buffer = VisBuffer::new(4, 4);
        let line = [sv(0.0, 0.0, 0.5), sv(2.0, 2.0, 0.5), sv(4.0, 4.0, 0.5)];
        rasterize_triangle(&mut buffer, line, 9, false);
        assert!(buffer.pixels().iter().all(|&p| p == 0));
    }

    #[test]
    fn offscreen_vertices_do_not_panic_and_clamp() {
        let mut buffer = VisBuffer::new(4, 4);
        // Triangle mostly off the top-left, only clipping into the buffer.
        let tri = [
            sv(-8.0, -8.0, 0.5),
            sv(6.0, -2.0, 0.5),
            sv(-2.0, 6.0, 0.5),
        ];
        rasterize_triangle(&mut buffer, tri, 4, false);
        // Out-of-range queries return the cleared value rather than panicking.
        assert_eq!(buffer.at(99, 99), 0);
    }

    #[test]
    fn centroid_depth_is_vertex_mean() {
        // A triangle whose centroid lands on a pixel center yields the mean
        // reversed-Z depth of its three vertices.
        let mut buffer = VisBuffer::new(8, 8);
        // Centroid of (1.5,1.5),(7.5,1.5),(1.5,7.5) is (3.5,3.5) -> pixel (3,3).
        let tri = [sv(1.5, 1.5, 0.9), sv(7.5, 1.5, 0.6), sv(1.5, 7.5, 0.3)];
        rasterize_triangle(&mut buffer, tri, 1, false);
        let key = vis_depth(buffer.at(3, 3));
        let expected = encode_depth((0.9 + 0.6 + 0.3) / 3.0);
        // f32 barycentric accumulation and the f32 vertex-mean round
        // slightly differently before the u32::MAX-scaled encode, so allow
        // a sub-ULP-of-f32-depth tolerance across the full 32-bit range.
        let diff = key.abs_diff(expected);
        assert!(diff <= 1024, "centroid depth key {key} vs expected {expected}");
    }
}
