//! Maps visible-receiver footprints onto the shadow pages they touch.
//!
//! Residency needs to know *which* clipmap pages a frame actually uses so the
//! backend only streams and rasters shadow depth where visible geometry casts
//! or receives it. The geometry pass hands this layer, per visible cluster, its
//! light-space footprint — the axis-aligned bounds of the cluster projected onto
//! the plane perpendicular to the light — together with the shadow-texel size
//! the receiver needs. This layer selects the clip level for that texel size,
//! walks every page the footprint overlaps within the level's grid, and records
//! each one into a [`ShadowRequestBatch`] at the cluster's screen-importance
//! priority. Coalescing across the many clusters that share a page happens in
//! the batch, so a page pulled by any high-importance receiver streams in with
//! that urgency.
//!
//! Like the rest of the clipmap decision layer this is 2D and GPU-independent:
//! the caller supplies light-space bounds, so no trigonometry or GPU state
//! leaks in, and the page walk is deterministic for reproducible request sets.

use super::clipmap::ClipmapConfig;
use super::residency::ShadowRequestBatch;

/// Records every clip page a light-space receiver footprint overlaps.
///
/// `min` and `max` are the receiver's axis-aligned bounds in light space (any
/// component order is tolerated). `required_texel_size` selects the clip level
/// via [`ClipmapConfig::select_level`]; `priority` is the receiver's screen
/// importance, forwarded to [`ShadowRequestBatch::record`] so the batch keeps
/// the highest priority any receiver reported for a shared page.
///
/// Pages outside the selected level's grid are skipped rather than clamped onto
/// edge pages, so a footprint that only partly overlaps the covered region
/// marks just the pages it truly touches. Returns the number of distinct pages
/// recorded from this footprint (a page already present in the batch from an
/// earlier footprint is still counted here since it is overlapped again). An
/// empty or degenerate clipmap records nothing and returns zero.
pub fn mark_receiver_footprint(
    config: &ClipmapConfig,
    camera: [f32; 2],
    min: [f32; 2],
    max: [f32; 2],
    required_texel_size: f32,
    priority: f32,
    batch: &mut ShadowRequestBatch,
) -> usize {
    if config.level_count == 0 || config.resolution == 0 {
        return 0;
    }
    let level = config.select_level(required_texel_size);
    let ps = config.page_size(level);
    if ps <= 0.0 {
        return 0;
    }
    let origin = config.level_origin(camera, level);
    let (lo_x, hi_x) = ordered(min[0], max[0]);
    let (lo_y, hi_y) = ordered(min[1], max[1]);
    let last = i64::from(config.resolution) - 1;

    let x0 = page_index(lo_x, origin[0], ps);
    let x1 = page_index(hi_x, origin[0], ps);
    let y0 = page_index(lo_y, origin[1], ps);
    let y1 = page_index(hi_y, origin[1], ps);

    // Intersect the footprint's page span with the level grid `[0, last]`.
    let xs = x0.max(0);
    let xe = x1.min(last);
    let ys = y0.max(0);
    let ye = y1.min(last);
    if xs > xe || ys > ye {
        return 0;
    }

    let level16 = u16::from(level);
    let mut recorded = 0usize;
    let mut y = ys;
    while y <= ye {
        let mut x = xs;
        while x <= xe {
            batch.record(
                super::ShadowPageKey {
                    light: config.light,
                    level: level16,
                    x: x as u16,
                    y: y as u16,
                },
                priority,
            );
            recorded += 1;
            x += 1;
        }
        y += 1;
    }
    recorded
}

/// Orders a pair so the smaller component comes first.
#[must_use]
fn ordered(a: f32, b: f32) -> (f32, f32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Floors a light-space coordinate onto the level's page lattice.
///
/// The float-to-int cast saturates, so a footprint far outside the grid yields
/// an out-of-range index that the caller's clamp discards rather than wrapping.
#[must_use]
fn page_index(coord: f32, origin: f32, page_size: f32) -> i64 {
    ((coord - origin) / page_size).floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            light: 3,
            level_count: 4,
            resolution: 8,
            page_texel_dim: 128,
            level0_page_size: 4.0,
        }
    }

    #[test]
    fn point_footprint_marks_exactly_one_page() {
        let c = config();
        let camera = [0.0, 0.0];
        let mut batch = ShadowRequestBatch::new();
        let n = mark_receiver_footprint(&c, camera, [0.0, 0.0], [0.0, 0.0], 0.001, 1.0, &mut batch);
        assert_eq!(n, 1);
        assert_eq!(batch.len(), 1);
        // The single page agrees with page_of for the same position/level.
        let key = c.page_of(camera, [0.0, 0.0], 0).expect("in range");
        assert_eq!(batch.priority(key), Some(1.0));
    }

    #[test]
    fn footprint_spanning_two_pages_marks_both() {
        let c = config();
        let camera = [0.0, 0.0];
        let ps = c.page_size(0);
        let origin = c.level_origin(camera, 0);
        // A box straddling the boundary between two horizontally adjacent pages.
        let min = [origin[0] + ps * 0.5, origin[1] + ps * 0.5];
        let max = [origin[0] + ps * 1.5, origin[1] + ps * 0.5];
        let mut batch = ShadowRequestBatch::new();
        let n = mark_receiver_footprint(&c, camera, min, max, 0.001, 2.0, &mut batch);
        assert_eq!(n, 2);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn reversed_bounds_are_tolerated() {
        let c = config();
        let camera = [0.0, 0.0];
        let ps = c.page_size(0);
        let origin = c.level_origin(camera, 0);
        let a = [origin[0] + ps * 1.5, origin[1] + ps * 0.5];
        let b = [origin[0] + ps * 0.5, origin[1] + ps * 0.5];
        let mut batch = ShadowRequestBatch::new();
        // Passing max before min must yield the same two pages.
        let n = mark_receiver_footprint(&c, camera, a, b, 0.001, 1.0, &mut batch);
        assert_eq!(n, 2);
    }

    #[test]
    fn footprint_outside_grid_marks_nothing() {
        let c = config();
        let mut batch = ShadowRequestBatch::new();
        let n = mark_receiver_footprint(
            &c,
            [0.0, 0.0],
            [10_000.0, 10_000.0],
            [10_001.0, 10_001.0],
            0.001,
            1.0,
            &mut batch,
        );
        assert_eq!(n, 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn partial_overlap_clips_to_grid_pages() {
        let c = config();
        let camera = [0.0, 0.0];
        let origin = c.level_origin(camera, 0);
        let span = c.level_span(0);
        // Start one page before the grid and end one page inside it: only the
        // in-grid pages are marked, the outside column is clipped away.
        let ps = c.page_size(0);
        let min = [origin[0] - ps * 0.5, origin[1] + ps * 0.5];
        let max = [origin[0] + ps * 0.5, origin[1] + ps * 0.5];
        let mut batch = ShadowRequestBatch::new();
        let n = mark_receiver_footprint(&c, camera, min, max, 0.001, 1.0, &mut batch);
        // Column -1 is clipped, column 0 is kept.
        assert_eq!(n, 1);
        // The whole grid stays finite regardless of span sign.
        assert!(span > 0.0);
    }

    #[test]
    fn higher_priority_receiver_wins_shared_page() {
        let c = config();
        let camera = [0.0, 0.0];
        let mut batch = ShadowRequestBatch::new();
        mark_receiver_footprint(&c, camera, [0.0, 0.0], [0.0, 0.0], 0.001, 1.0, &mut batch);
        mark_receiver_footprint(&c, camera, [0.0, 0.0], [0.0, 0.0], 0.001, 5.0, &mut batch);
        let key = c.page_of(camera, [0.0, 0.0], 0).expect("in range");
        assert_eq!(batch.priority(key), Some(5.0));
    }

    #[test]
    fn empty_clipmap_marks_nothing() {
        let mut c = config();
        c.level_count = 0;
        let mut batch = ShadowRequestBatch::new();
        let n =
            mark_receiver_footprint(&c, [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], 1.0, 1.0, &mut batch);
        assert_eq!(n, 0);
        assert!(batch.is_empty());
    }
}
