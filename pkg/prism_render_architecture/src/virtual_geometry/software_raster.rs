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
//! [`encode_depth`] takes the raw IEEE bit pattern of that depth
//! (`bitcast<u32>(z)` in the compute twin), which is monotonic for
//! non-negative floats, so nearer surfaces produce a *larger* key and the
//! whole `u64` composites with a single `max` comparison. A cleared pixel is
//! `0` (the farthest possible key), so the first covering triangle always wins
//! over the clear value.
//!
//! The payload sub-divides into `(cluster_id << 7) | triangle_id`
//! ([`pack_cluster_triangle`], [`cluster_of`], [`triangle_of`]): 7 low bits for
//! the per-cluster triangle index (at most 128 triangles per cluster) and the
//! high bits for the cluster id, matching the shipping meshlet raster layout so
//! this reference and the GPU twin decode identical ids.
//!
//! # Coverage and watertightness
//!
//! Coverage uses the signed edge function with a strict top-left fill rule so
//! that a shared edge between two adjacent triangles is owned by exactly one of
//! them: no double-covered pixels, no cracks. This is the watertight end state
//! the shipping compute shader still tracks as a TODO, so this reference pins
//! the correct behavior the twin is validated against. Depth is interpolated
//! with screen-space-linear barycentrics, matching a hardware depth buffer for
//! the NDC depth values fed in here.

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

/// Projects a world-space vertex through a clip matrix into a [`ScreenVertex`].
///
/// This mirrors the shipping meshlet software-raster vertex path exactly, so
/// the CPU reference covers the full transform and not just the screen-space
/// fill. The GPU twin's per-vertex projection stage is diffed against this:
///
/// ```text
/// clip  = clip_from_world * vec4(world, 1)   // column-major, WGSL `m * v`
/// ndc   = clip.xyz / clip.w                  // perspective divide
/// uv    = ndc.xy * vec2(0.5, -0.5) + 0.5     // ndc_to_uv
/// pos   = uv * viewport                      // viewport = (width, height)
/// depth = ndc.z                              // reversed-Z, 1.0 nearest
/// ```
///
/// `clip_from_world` is stored **column-major** to match a WGSL
/// `mat4x4<f32>`: `clip_from_world[c]` is column `c`, so the product is
/// `clip[r] = sum over c of clip_from_world[c][r] * world_h[c]`. The result is
/// stored as `ScreenVertex.pos` in y-down pixel space, ready to feed
/// [`rasterize_triangle`] and [`rasterize_cluster`] directly.
///
/// Returns `None` when the vertex is on or behind the camera plane
/// (`clip.w <= 0`), where the perspective divide is undefined. The shipping
/// shader relies on upstream near-plane culling to exclude these; the reference
/// makes that precondition explicit instead of emitting a garbage projection.
#[must_use]
pub fn project_vertex(
    clip_from_world: &[[f32; 4]; 4],
    world_pos: [f32; 3],
    viewport: [f32; 2],
) -> Option<ScreenVertex> {
    let world_h = [world_pos[0], world_pos[1], world_pos[2], 1.0];
    // Column-major matrix-vector product, matching WGSL `clip_from_world * v`.
    let clip = [
        clip_from_world[0][0] * world_h[0]
            + clip_from_world[1][0] * world_h[1]
            + clip_from_world[2][0] * world_h[2]
            + clip_from_world[3][0] * world_h[3],
        clip_from_world[0][1] * world_h[0]
            + clip_from_world[1][1] * world_h[1]
            + clip_from_world[2][1] * world_h[2]
            + clip_from_world[3][1] * world_h[3],
        clip_from_world[0][2] * world_h[0]
            + clip_from_world[1][2] * world_h[1]
            + clip_from_world[2][2] * world_h[2]
            + clip_from_world[3][2] * world_h[3],
        clip_from_world[0][3] * world_h[0]
            + clip_from_world[1][3] * world_h[1]
            + clip_from_world[2][3] * world_h[2]
            + clip_from_world[3][3] * world_h[3],
    ];
    if clip[3] <= 0.0 {
        return None;
    }
    let inv_w = 1.0 / clip[3];
    let ndc = [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w];
    // `ndc_to_uv`: flips y so uv is y-down, then scales into the viewport.
    let u = ndc[0] * 0.5 + 0.5;
    let v = ndc[1] * -0.5 + 0.5;
    Some(ScreenVertex::new([u * viewport[0], v * viewport[1]], ndc[2]))
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
/// This mirrors the shipping GPU contract exactly: the depth is the raw IEEE
/// bit pattern of the reversed-Z NDC depth (`bitcast<u32>(z)` in the compute
/// twin). For non-negative floats the bit pattern is monotonic, so `1.0`
/// (nearest) yields the largest key and `0.0` (farthest) yields `0`, which is
/// also the cleared value. The input is clamped to `[0, 1]` so a stray
/// negative depth (sign bit set) can never masquerade as the nearest surface.
#[must_use]
pub fn encode_depth(depth: f32) -> u32 {
    depth.clamp(0.0, 1.0).to_bits()
}

/// Number of low bits of a vis-buffer payload reserved for the triangle id.
///
/// A cluster holds at most 128 triangles (one per rasterizer thread), so 7
/// bits address every triangle and the remaining high bits carry the cluster
/// id. This matches the shipping `packed_ids = (cluster_id << 7) | triangle_id`
/// layout so the CPU reference and GPU twin agree bit-for-bit.
pub const CLUSTER_TRIANGLE_BITS: u32 = 7;

/// Mask selecting the triangle-id field of a packed payload.
const TRIANGLE_ID_MASK: u32 = (1 << CLUSTER_TRIANGLE_BITS) - 1;

/// Packs a cluster id and triangle id into one vis-buffer payload.
///
/// The triangle id is masked to [`CLUSTER_TRIANGLE_BITS`] bits; callers must
/// keep `triangle_id < 128`. The cluster id occupies the high bits.
#[must_use]
pub const fn pack_cluster_triangle(cluster_id: u32, triangle_id: u32) -> u32 {
    (cluster_id << CLUSTER_TRIANGLE_BITS) | (triangle_id & TRIANGLE_ID_MASK)
}

/// Extracts the cluster id from a packed payload.
#[must_use]
pub const fn cluster_of(payload: u32) -> u32 {
    payload >> CLUSTER_TRIANGLE_BITS
}

/// Extracts the triangle id from a packed payload.
#[must_use]
pub const fn triangle_of(payload: u32) -> u32 {
    payload & TRIANGLE_ID_MASK
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

/// Maximum triangles a single cluster may hold.
///
/// The payload reserves [`CLUSTER_TRIANGLE_BITS`] bits for the triangle index,
/// so triangle ids past this count would alias earlier triangles. Cluster build
/// guarantees this bound (one triangle per rasterizer thread); the reference
/// enforces it so a malformed cluster surfaces instead of silently aliasing.
pub const MAX_CLUSTER_TRIANGLES: usize = 1 << CLUSTER_TRIANGLE_BITS;

/// Rasterizes an indexed cluster into `buffer`, mirroring the GPU dispatch.
///
/// This is the cluster-granularity companion to [`rasterize_triangle`]: it is
/// the CPU reference a per-cluster compute dispatch is diffed against. Each
/// entry of `triangles` is a triple of indices into `vertices`; triangle `i`
/// writes the payload [`pack_cluster_triangle`]`(cluster_id, i)`, so the
/// vis-buffer decodes back to the same cluster/triangle ids the twin produces.
///
/// Triangles are processed in order with nearest-depth compositing, identical
/// to issuing each through [`rasterize_triangle`]. A triple that indexes past
/// `vertices` is skipped rather than panicking, so a malformed index list is
/// inert. Only the first [`MAX_CLUSTER_TRIANGLES`] triangles are rasterized;
/// any beyond that would alias triangle ids and are dropped.
pub fn rasterize_cluster(
    buffer: &mut VisBuffer,
    vertices: &[ScreenVertex],
    triangles: &[[u32; 3]],
    cluster_id: u32,
    cull_back: bool,
) {
    let count = triangles.len().min(MAX_CLUSTER_TRIANGLES);
    for (triangle_id, tri) in triangles[..count].iter().enumerate() {
        let [i0, i1, i2] = *tri;
        let (Some(&v0), Some(&v1), Some(&v2)) = (
            vertices.get(i0 as usize),
            vertices.get(i1 as usize),
            vertices.get(i2 as usize),
        ) else {
            // Out-of-range index: skip this triangle, never index out of bounds.
            continue;
        };
        let payload = pack_cluster_triangle(cluster_id, triangle_id as u32);
        rasterize_triangle(buffer, [v0, v1, v2], payload, cull_back);
    }
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
    fn cluster_triangle_payload_round_trips() {
        // Mirrors the shipping `(cluster_id << 7) | triangle_id` layout.
        let payload = pack_cluster_triangle(0x0012_3456, 127);
        assert_eq!(cluster_of(payload), 0x0012_3456);
        assert_eq!(triangle_of(payload), 127);
        // Triangle field is exactly 7 bits, cluster occupies the rest.
        assert_eq!(pack_cluster_triangle(1, 0), 1 << CLUSTER_TRIANGLE_BITS);
        assert_eq!(triangle_of(pack_cluster_triangle(0, 0x7F)), 0x7F);
    }

    #[test]
    fn nearer_reversed_z_packs_larger() {
        // depth 1.0 is nearest -> largest key -> largest word.
        let near = pack_vis(encode_depth(1.0), 0);
        let far = pack_vis(encode_depth(0.0), u32::MAX);
        assert!(near > far, "nearest surface must dominate the composite");
        // Matches bitcast<u32>(reversed-Z NDC depth) in the GPU twin.
        assert_eq!(encode_depth(1.0), 1.0f32.to_bits());
        assert_eq!(encode_depth(0.0), 0);
        // Monotonic across the visible range: nearer depth -> larger key.
        assert!(encode_depth(0.75) > encode_depth(0.25));
        // Out-of-range negative depth cannot masquerade as nearest.
        assert_eq!(encode_depth(-1.0), 0);
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
        // f32 barycentric accumulation and the f32 vertex-mean round to
        // slightly different bit patterns, so compare the depth keys within a
        // few ULPs rather than exactly.
        let diff = key.abs_diff(expected);
        assert!(diff <= 16, "centroid depth key {key} vs expected {expected}");
    }

    #[test]
    fn cluster_writes_per_triangle_payloads() {
        // A two-triangle cluster: each covered pixel decodes to the shared
        // cluster id and its own triangle id.
        let verts = [
            sv(0.5, 0.5, 0.5),
            sv(6.5, 0.5, 0.5),
            sv(0.5, 6.5, 0.5),
            sv(9.5, 9.5, 0.5),
        ];
        // Triangle 0 near the origin, triangle 1 near the far corner.
        let tris = [[0, 1, 2], [3, 1, 2]];
        let mut buffer = VisBuffer::new(10, 10);
        rasterize_cluster(&mut buffer, &verts, &tris, 42, false);
        // Pixel (0,0) is strictly inside triangle 0 only.
        let p0 = vis_payload(buffer.at(0, 0));
        assert_eq!(cluster_of(p0), 42);
        assert_eq!(triangle_of(p0), 0);
    }

    #[test]
    fn cluster_matches_individual_triangle_dispatch() {
        // Rasterizing a cluster equals issuing each triangle by hand with the
        // packed cluster/triangle payload, in order.
        let verts = [
            sv(0.0, 0.0, 0.4),
            sv(8.0, 0.0, 0.4),
            sv(8.0, 8.0, 0.4),
            sv(0.0, 8.0, 0.4),
        ];
        let tris = [[0, 1, 2], [0, 2, 3]];

        let mut via_cluster = VisBuffer::new(8, 8);
        rasterize_cluster(&mut via_cluster, &verts, &tris, 7, false);

        let mut by_hand = VisBuffer::new(8, 8);
        for (i, t) in tris.iter().enumerate() {
            let tri = [verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]];
            rasterize_triangle(&mut by_hand, tri, pack_cluster_triangle(7, i as u32), false);
        }
        assert_eq!(via_cluster.pixels(), by_hand.pixels());
    }

    #[test]
    fn cluster_skips_out_of_range_indices_without_panic() {
        let verts = [sv(0.0, 0.0, 0.5), sv(4.0, 0.0, 0.5), sv(0.0, 4.0, 0.5)];
        // Second triple references a vertex that does not exist.
        let tris = [[0, 1, 2], [0, 1, 99]];
        let mut buffer = VisBuffer::new(4, 4);
        rasterize_cluster(&mut buffer, &verts, &tris, 1, false);
        // Triangle 0 still rasterized; the bad triple was skipped.
        assert_eq!(triangle_of(vis_payload(buffer.at(0, 0))), 0);
    }

    /// Column-major identity: world xyz passes straight to ndc, then to the
    /// viewport center for the origin.
    #[test]
    fn project_identity_maps_origin_to_viewport_center() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let v = project_vertex(&identity, [0.0, 0.0, 0.5], [200.0, 100.0]).unwrap();
        assert_eq!(v.pos, [100.0, 50.0]);
        assert_eq!(v.depth, 0.5);
    }

    /// `ndc_to_uv` flips y: the NDC top (`+1`) lands at the top pixel row
    /// (`y == 0`) and the NDC bottom (`-1`) at the last row.
    #[test]
    fn project_flips_y_for_screen_space() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let top = project_vertex(&identity, [0.0, 1.0, 0.0], [200.0, 100.0]).unwrap();
        let bottom = project_vertex(&identity, [0.0, -1.0, 0.0], [200.0, 100.0]).unwrap();
        assert_eq!(top.pos[1], 0.0);
        assert_eq!(bottom.pos[1], 100.0);
    }

    /// A homogeneous `w != 1` proves the perspective divide runs: raw clip.x is
    /// `2` but the divide by `w == 2` puts the vertex on the right screen edge.
    #[test]
    fn project_applies_perspective_divide() {
        let mut m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Constant homogeneous w of 2 regardless of world position.
        m[3][3] = 2.0;
        let v = project_vertex(&m, [2.0, 0.0, 0.0], [200.0, 100.0]).unwrap();
        // ndc.x = 2 / 2 = 1 -> u = 1 -> right edge; not off-screen at x = 4.
        assert_eq!(v.pos[0], 200.0);
    }

    /// A vertex with `w <= 0` is on or behind the camera plane; the divide is
    /// undefined, so the reference rejects it rather than projecting garbage.
    #[test]
    fn project_rejects_non_positive_w() {
        // Row 3 all zero -> clip.w == 0 for every input.
        let degenerate = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        assert!(project_vertex(&degenerate, [0.0, 0.0, 0.5], [8.0, 8.0]).is_none());
    }

    /// End-to-end: projecting three world vertices and rasterizing the cluster
    /// covers the full transform-plus-fill path the GPU twin is diffed against.
    #[test]
    fn project_then_rasterize_covers_full_path() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let viewport = [8.0, 8.0];
        // A triangle straddling the viewport center in NDC space.
        let world = [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.5, 0.5]];
        let verts: Vec<ScreenVertex> = world
            .iter()
            .map(|&w| project_vertex(&identity, w, viewport).unwrap())
            .collect();
        let mut buffer = VisBuffer::new(8, 8);
        rasterize_cluster(&mut buffer, &verts, &[[0, 1, 2]], 3, false);
        // The center pixel is inside the triangle and decodes to cluster 3 / tri 0.
        let center = buffer.at(4, 4);
        assert_ne!(center, 0, "center pixel must be covered by the projected triangle");
        assert_eq!(cluster_of(vis_payload(center)), 3);
        assert_eq!(triangle_of(vis_payload(center)), 0);
    }
}
