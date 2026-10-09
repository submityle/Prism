//! Signed-distance-field (SDF) glyph atlas — the "any-scale sharp" tier **C**.
//!
//! Tier `B` ([`crate::vector`]) rasterises glyph outlines at the *exact* target
//! size every time: crisp, but it re-rasterises per size and has no GPU story.
//! Tier `C` bakes each glyph *once* into a single-channel distance field packed
//! into one atlas texture; a renderer then samples that atlas and recovers a
//! sharp edge at *any* scale with the standard `screenPxRange` coverage rule
//! (Valve/`msdfgen` lineage). The same math runs on the CPU reference
//! rasteriser here (ground truth) and, mirrored in WGSL, on the GPU backend —
//! upholding Loom's "CPU ↔ GPU pixel parity" rule.
//!
//! This is a true SDF (one channel); multi-channel MSDF (sharper corners) is a
//! future enhancement noted in `docs/prism_loom_text_rendering_design_zh.md`.
//! The field is built from [`crate::vector`] outline coverage via an exact
//! Euclidean distance transform (Felzenszwalb & Huttenlocher), so no extra
//! font dependency is pulled in. Gated behind the `vector` feature. Contains no
//! Unreal Engine source or derived code.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::vector;

/// Per-glyph placement inside an [`SdfAtlas`]. All geometric values are in
/// *em-relative* units (fraction of the em square) so one bake serves any size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGlyph {
    /// Atlas sub-rect (texels): inclusive-min, exclusive-max in `x`.
    pub px_min: (u32, u32),
    /// Atlas sub-rect (texels): exclusive-max.
    pub px_max: (u32, u32),
    /// Left/top bearing of the field box relative to the pen origin, in em units
    /// (y grows downward from the baseline, so the top is typically negative).
    pub bearing_em: (f32, f32),
    /// Field box size in em units (includes the SDF padding).
    pub size_em: (f32, f32),
    /// Monospace horizontal advance in em units.
    pub advance_em: f32,
}

/// A single-channel SDF atlas for printable ASCII plus the per-glyph layout.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfAtlas {
    /// Atlas width in texels.
    pub width: u32,
    /// Atlas height in texels.
    pub height: u32,
    /// Row-major single-channel distance field. Each texel encodes signed
    /// distance mapped to `0..=255`, where `128` is exactly on the edge, values
    /// `> 128` are inside the glyph and `< 128` outside. One texel ≈
    /// [`Self::px_range`] fraction of the distance window.
    pub data: Vec<u8>,
    /// The em size (device px) the field was baked at. Only affects field
    /// resolution, not final render size.
    pub bake_em: f32,
    /// Width of the signed-distance window in texels, i.e. the number of texels
    /// spanned as the encoded value goes `0 → 255`. Feeds `screenPxRange`.
    pub px_range: f32,
    glyphs: BTreeMap<char, SdfGlyph>,
}

impl SdfAtlas {
    /// Placement for `ch`, if it was baked (space and unsupported glyphs absent).
    #[must_use]
    pub fn glyph(&self, ch: char) -> Option<&SdfGlyph> {
        self.glyphs.get(&ch)
    }

    /// Reads the raw encoded distance at integer texel `(x, y)` (clamped).
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> u8 {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        self.data[(y * self.width + x) as usize]
    }

    /// Bilinearly samples the normalised signed distance (`0.0..=1.0`, `0.5` =
    /// edge) at continuous atlas texel coordinates.
    #[must_use]
    pub fn sample_distance(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.width.saturating_sub(1)) as f32);
        let y = y.clamp(0.0, (self.height.saturating_sub(1)) as f32);
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (x0, y0) = (x0 as u32, y0 as u32);
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));
        let d = |xx, yy| f32::from(self.texel(xx, yy)) / 255.0;
        let top = d(x0, y0) * (1.0 - fx) + d(x1, y0) * fx;
        let bot = d(x0, y1) * (1.0 - fx) + d(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    }
}

/// Converts an SDF sample plus the on-screen glyph scale into edge coverage,
/// using the standard `screenPxRange` anti-aliasing rule. `screen_px_range` is
/// `px_range * on_screen_glyph_px / baked_glyph_px`. This exact formula is
/// mirrored by the GPU WGSL path so the two backends agree per pixel.
#[must_use]
pub fn coverage(distance: f32, screen_px_range: f32) -> f32 {
    let r = screen_px_range.max(1.0);
    (r * (distance - 0.5) + 0.5).clamp(0.0, 1.0)
}

/// Bakes a single-channel SDF atlas for printable ASCII at `bake_em` device px.
///
/// `padding` texels are reserved around each glyph so the distance window never
/// clips; `px_range` is the texel span of the signed-distance window (a value
/// of `padding` is a good default). Returns `None` if the embedded vector face
/// is unavailable.
#[must_use]
pub fn bake_ascii(bake_em: f32, padding: u32, px_range: f32) -> Option<SdfAtlas> {
    if !vector::available() {
        return None;
    }
    let em = bake_em.max(4.0);
    let pad = padding.max(1);
    let metrics = vector::metrics(em)?;
    let advance_em = metrics.advance / em;

    // First pass: rasterise every glyph's coverage and compute its SDF cell.
    struct Cell {
        ch: char,
        w: u32,
        h: u32,
        bearing_em: (f32, f32),
        size_em: (f32, f32),
        field: Vec<u8>, // encoded SDF, row-major w*h
    }
    let mut cells: Vec<Cell> = Vec::new();
    for code in (crate::FIRST as u32)..=(crate::LAST as u32) {
        let ch = char::from_u32(code).unwrap_or('?');
        let Some(cov) = vector::glyph_coverage(ch, em) else {
            continue; // e.g. space: no ink, nothing to bake
        };
        let w = cov.width as u32 + 2 * pad;
        let h = cov.height as u32 + 2 * pad;
        // Place the coverage inside a padded cell so the field has room.
        let mut inside = vec![false; (w * h) as usize];
        for row in 0..cov.height {
            for col in 0..cov.width {
                if cov.data[row * cov.width + col] >= 0.5 {
                    let x = col as u32 + pad;
                    let y = row as u32 + pad;
                    inside[(y * w + x) as usize] = true;
                }
            }
        }
        let field = encode_sdf(&inside, w, h, px_range);
        cells.push(Cell {
            ch,
            w,
            h,
            bearing_em: (
                (cov.left - pad as f32) / em,
                (cov.top - pad as f32) / em,
            ),
            size_em: (w as f32 / em, h as f32 / em),
            field,
        });
    }

    // Second pass: shelf-pack the cells into one atlas.
    let max_w: u32 = cells.iter().map(|c| c.w).max().unwrap_or(1);
    // Aim for a roughly square atlas.
    let total: u32 = cells.iter().map(|c| c.w * c.h).sum();
    let atlas_w = (((total as f32).sqrt() * 1.2) as u32).max(max_w).max(16);
    let mut shelf_x = 0u32;
    let mut shelf_y = 0u32;
    let mut shelf_h = 0u32;
    let mut placed: Vec<(usize, u32, u32)> = Vec::with_capacity(cells.len());
    for (i, c) in cells.iter().enumerate() {
        if shelf_x + c.w > atlas_w {
            shelf_y += shelf_h;
            shelf_x = 0;
            shelf_h = 0;
        }
        placed.push((i, shelf_x, shelf_y));
        shelf_x += c.w;
        shelf_h = shelf_h.max(c.h);
    }
    let atlas_h = (shelf_y + shelf_h).max(1);

    let mut data = vec![0u8; (atlas_w * atlas_h) as usize];
    let mut glyphs = BTreeMap::new();
    for (i, ox, oy) in placed {
        let c = &cells[i];
        for row in 0..c.h {
            for col in 0..c.w {
                let v = c.field[(row * c.w + col) as usize];
                let ax = ox + col;
                let ay = oy + row;
                data[(ay * atlas_w + ax) as usize] = v;
            }
        }
        glyphs.insert(
            c.ch,
            SdfGlyph {
                px_min: (ox, oy),
                px_max: (ox + c.w, oy + c.h),
                bearing_em: c.bearing_em,
                size_em: c.size_em,
                advance_em,
            },
        );
    }

    Some(SdfAtlas {
        width: atlas_w,
        height: atlas_h,
        data,
        bake_em: em,
        px_range,
        glyphs,
    })
}

/// Builds the encoded signed distance field for an inside/outside mask using an
/// exact Euclidean distance transform, then maps distance to `0..=255` with the
/// edge at `128` and `px_range` texels spanning the full window.
fn encode_sdf(inside: &[bool], w: u32, h: u32, px_range: f32) -> Vec<u8> {
    let dist_to_out = edt(inside, w, h, false); // nearest OUTSIDE texel
    let dist_to_in = edt(inside, w, h, true); // nearest INSIDE texel
    let span = (px_range * 0.5).max(0.5);
    let mut out = vec![0u8; (w * h) as usize];
    for i in 0..(w * h) as usize {
        // Signed distance: positive inside (how deep), negative outside, 0 edge.
        let signed = if inside[i] {
            dist_to_out[i]
        } else {
            -dist_to_in[i]
        };
        let norm = (0.5 + 0.5 * (signed / span)).clamp(0.0, 1.0);
        out[i] = (norm * 255.0).round() as u8;
    }
    out
}

/// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher 2004).
/// Returns, for every texel, the distance to the nearest texel whose
/// `inside == target` flag. Distances for seed texels are `0`.
fn edt(inside: &[bool], w: u32, h: u32, target: bool) -> Vec<f32> {
    let (w, h) = (w as usize, h as usize);
    let inf = f32::INFINITY;
    // Seed grid: squared distance 0 at target texels, +inf elsewhere.
    let mut grid = vec![inf; w * h];
    for i in 0..w * h {
        if inside[i] == target {
            grid[i] = 0.0;
        }
    }
    // Transform columns, then rows (1-D parabola lower-envelope per Felzenszwalb).
    let mut col = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = grid[y * w + x];
        }
        let d = edt_1d(&col);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
    let mut row = vec![0.0f32; w];
    for y in 0..h {
        for x in 0..w {
            row[x] = grid[y * w + x];
        }
        let d = edt_1d(&row);
        for x in 0..w {
            grid[y * w + x] = d[x];
        }
    }
    grid.iter().map(|v| v.sqrt()).collect()
}

/// 1-D squared-distance transform over a function sampled at integer positions
/// (Felzenszwalb & Huttenlocher lower-envelope). `+inf` entries are treated as
/// "no seed here" and never win; a column/row with no finite entry stays `+inf`.
fn edt_1d(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    let mut d = vec![f32::INFINITY; n];
    if n == 0 {
        return d;
    }
    // First finite seed index; if none, the whole line is unreachable.
    let Some(q0) = (0..n).find(|&i| f[i].is_finite()) else {
        return d;
    };

    let mut v = vec![0usize; n]; // parabola apex indices in the lower envelope
    let mut z = vec![0.0f32; n + 1]; // breakpoints between consecutive parabolas
    let mut k = 0usize;
    v[0] = q0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in (q0 + 1)..n {
        if !f[q].is_finite() {
            continue;
        }
        // Pop parabolas that are no longer part of the lower envelope.
        let mut s;
        loop {
            let p = v[k];
            s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32))
                / (2.0 * q as f32 - 2.0 * p as f32);
            if s <= z[k] {
                // z[0] == -inf is a sentinel, so k never underflows below 0.
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }

    let mut k = 0usize;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let dx = q as f32 - p as f32;
        *dq = dx * dx + f[p];
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_point_source() {
        // A single seed in the middle: distances grow with Euclidean radius.
        let w = 5u32;
        let h = 5u32;
        let mut inside = vec![false; 25];
        inside[2 * 5 + 2] = true;
        let d = edt(&inside, w, h, true);
        assert!((d[2 * 5 + 2]).abs() < 1e-4); // seed
        assert!((d[2 * 5 + 3] - 1.0).abs() < 1e-4); // one right
        assert!((d[0] - (8.0f32).sqrt()).abs() < 1e-3); // corner (2,2)
    }

    #[test]
    fn atlas_bakes_ascii() {
        let atlas = bake_ascii(32.0, 4, 8.0).expect("vector face available");
        assert!(atlas.width > 0 && atlas.height > 0);
        assert_eq!(atlas.data.len(), (atlas.width * atlas.height) as usize);
        // A letter is present with a sane advance; space is absent (no ink).
        let a = atlas.glyph('A').expect("A baked");
        assert!(a.advance_em > 0.0);
        assert!(atlas.glyph(' ').is_none());
    }

    #[test]
    fn sdf_coverage_matches_outline() {
        // Render 'H' from the SDF atlas and from the direct outline at the bake
        // size (scale 1, so atlas texels map 1:1 to device pixels). The two
        // coverage masks should agree closely; SDF is an approximation so we
        // allow a small mean error.
        let pad = 6u32;
        let em = 48.0f32;
        let atlas = bake_ascii(em, pad, 12.0).expect("atlas");
        let g = *atlas.glyph('H').expect("H");
        let screen_px_range = atlas.px_range; // scale == 1

        let cov = vector::glyph_coverage('H', em).expect("H outline");

        let mut sum_abs = 0.0f32;
        let mut count = 0usize;
        for row in 0..cov.height {
            for col in 0..cov.width {
                // Outline pixel -> atlas texel (coverage sits at +pad in the cell).
                let u = (g.px_min.0 + pad) as f32 + col as f32 + 0.5;
                let v = (g.px_min.1 + pad) as f32 + row as f32 + 0.5;
                let d = atlas.sample_distance(u, v);
                let sdf_c = coverage(d, screen_px_range);
                let out_c = cov.data[row * cov.width + col];
                sum_abs += (sdf_c - out_c).abs();
                count += 1;
            }
        }
        let mean = sum_abs / count.max(1) as f32;
        // SDF is an approximation; the residual is edge anti-aliasing where the
        // field's soft ramp differs from the outline's own coverage AA. A gross
        // sign/placement error would push this far higher (~0.4), so this bound
        // still guards correctness while allowing honest reconstruction error.
        assert!(
            mean < 0.15,
            "SDF vs outline mean coverage error {mean:.3} too high"
        );
    }
}
