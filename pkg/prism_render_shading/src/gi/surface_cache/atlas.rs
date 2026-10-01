//! Surfel atlas addressing: deterministic tile allocation + octahedral storage.
//!
//! Each surfel owns a small square *tile* in a shared radiance atlas, and
//! within that tile it stores its directional outgoing radiance on an
//! octahedral map (reusing [`crate::gi::world_space::octahedral`]).  This
//! module is the pure addressing layer: it maps a surfel id to its tile, maps a
//! world-space direction to the texel inside that tile, and inverts a texel
//! back to the direction at its centre.  It stores no radiance itself — that
//! lives in [`super::integration`] — so it can be shared verbatim by the GPU
//! twin that owns the real atlas texture.
//!
//! # Conventions
//! * The atlas is a grid of `tiles_per_row` columns of square tiles, each
//!   `tile_resolution` texels on a side.  Surfel ids tile left-to-right then
//!   top-to-bottom: surfel `id` lives at tile column `id % tiles_per_row` and
//!   tile row `id / tiles_per_row`.  Ids at or beyond [`SurfelAtlas::capacity`]
//!   have no slot and every lookup returns `None`.
//! * Texel coordinates are the graphics convention (`x` right, `y` down) with
//!   texel centres at integer `+ 0.5`; a direction maps to the texel whose
//!   centre is nearest its octahedral UV, and the inverse decodes that centre.
//!   Round-tripping is therefore exact only up to the atlas quantisation set by
//!   `tile_resolution`.
//! * All constructor arguments are clamped to at least `1` so widths, heights,
//!   and capacities are always positive; every lookup is `None` or an in-range
//!   coordinate, never out of bounds.
//! * All helpers are deterministic pure functions (no RNG / IO / GPU / unsafe).

use bevy_math::{Vec2, Vec3};

use crate::gi::world_space::octahedral::{dir_to_oct, oct_to_dir};

/// A texel coordinate inside the atlas (global, not tile-local).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasTexel {
    /// Column (x) texel index.
    pub x: u32,
    /// Row (y) texel index.
    pub y: u32,
}

/// A tile coordinate (column, row) in units of whole tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCoord {
    /// Tile column.
    pub col: u32,
    /// Tile row.
    pub row: u32,
}

/// Deterministic surfel-atlas addressing descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfelAtlas {
    /// Number of tile columns across the atlas.
    pub tiles_per_row: u32,
    /// Number of tile rows down the atlas.
    pub tile_rows: u32,
    /// Side length (in texels) of each square tile's octahedral map.
    pub tile_resolution: u32,
}

impl SurfelAtlas {
    /// Construct an atlas descriptor, clamping every dimension to at least `1`.
    #[must_use]
    pub fn new(tiles_per_row: u32, tile_rows: u32, tile_resolution: u32) -> Self {
        Self {
            tiles_per_row: tiles_per_row.max(1),
            tile_rows: tile_rows.max(1),
            tile_resolution: tile_resolution.max(1),
        }
    }

    /// Maximum number of surfels the atlas can address (`cols * rows`).
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.tiles_per_row * self.tile_rows
    }

    /// Atlas width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.tiles_per_row * self.tile_resolution
    }

    /// Atlas height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.tile_rows * self.tile_resolution
    }

    /// Tile (column, row) for a surfel id, or `None` if the id is out of range.
    #[must_use]
    pub fn tile_coord(&self, surfel_id: u32) -> Option<TileCoord> {
        if surfel_id >= self.capacity() {
            return None;
        }
        Some(TileCoord {
            col: surfel_id % self.tiles_per_row,
            row: surfel_id / self.tiles_per_row,
        })
    }

    /// Inverse of [`tile_coord`](Self::tile_coord): linear surfel id for a tile.
    ///
    /// Returns `None` when the tile lies outside the atlas grid.
    #[must_use]
    pub fn surfel_id(&self, tile: TileCoord) -> Option<u32> {
        if tile.col >= self.tiles_per_row || tile.row >= self.tile_rows {
            return None;
        }
        Some(tile.row * self.tiles_per_row + tile.col)
    }

    /// Top-left (origin) texel of a surfel's tile, or `None` if out of range.
    #[must_use]
    pub fn tile_origin(&self, surfel_id: u32) -> Option<AtlasTexel> {
        let tile = self.tile_coord(surfel_id)?;
        Some(AtlasTexel {
            x: tile.col * self.tile_resolution,
            y: tile.row * self.tile_resolution,
        })
    }

    /// Tile-local texel for a direction: the nearest octahedral texel centre.
    ///
    /// Independent of the surfel id; the returned coordinates are in
    /// `[0, tile_resolution)` on each axis.
    #[must_use]
    pub fn local_texel(&self, dir: Vec3) -> (u32, u32) {
        let uv = dir_to_oct(dir);
        self.uv_to_local(uv)
    }

    /// Global atlas texel storing `dir`'s radiance for `surfel_id`.
    ///
    /// Combines [`tile_origin`](Self::tile_origin) with
    /// [`local_texel`](Self::local_texel); `None` when the id is out of range.
    #[must_use]
    pub fn dir_to_texel(&self, surfel_id: u32, dir: Vec3) -> Option<AtlasTexel> {
        let origin = self.tile_origin(surfel_id)?;
        let (lx, ly) = self.local_texel(dir);
        Some(AtlasTexel {
            x: origin.x + lx,
            y: origin.y + ly,
        })
    }

    /// Unit direction stored at a tile-local texel centre (inverse of
    /// [`local_texel`](Self::local_texel) up to atlas quantisation).
    #[must_use]
    pub fn local_texel_to_dir(&self, lx: u32, ly: u32) -> Vec3 {
        let res = self.tile_resolution as f32;
        let cx = (lx.min(self.tile_resolution - 1) as f32 + 0.5) / res;
        let cy = (ly.min(self.tile_resolution - 1) as f32 + 0.5) / res;
        oct_to_dir(Vec2::new(cx, cy))
    }

    /// Unit direction stored at a *global* atlas texel, decoding the tile-local
    /// offset from `surfel_id`'s tile.  `None` if the id is out of range or the
    /// texel falls outside the surfel's tile.
    #[must_use]
    pub fn texel_to_dir(&self, surfel_id: u32, texel: AtlasTexel) -> Option<Vec3> {
        let origin = self.tile_origin(surfel_id)?;
        if texel.x < origin.x || texel.y < origin.y {
            return None;
        }
        let lx = texel.x - origin.x;
        let ly = texel.y - origin.y;
        if lx >= self.tile_resolution || ly >= self.tile_resolution {
            return None;
        }
        Some(self.local_texel_to_dir(lx, ly))
    }

    /// Map an octahedral UV in `[0, 1]^2` to the nearest tile-local texel.
    #[must_use]
    fn uv_to_local(&self, uv: Vec2) -> (u32, u32) {
        let res = self.tile_resolution;
        let fx = (uv.x.clamp(0.0, 1.0) * res as f32).floor();
        let fy = (uv.y.clamp(0.0, 1.0) * res as f32).floor();
        let max = res - 1;
        let x = clamp_index(fx, max);
        let y = clamp_index(fy, max);
        (x, y)
    }
}

/// Clamp a (possibly non-finite) float index into `[0, max]` as a `u32`.
#[must_use]
fn clamp_index(value: f32, max: u32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let v = value as u32;
    v.min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_and_dimensions() {
        let atlas = SurfelAtlas::new(4, 3, 8);
        assert_eq!(atlas.capacity(), 12);
        assert_eq!(atlas.width(), 32);
        assert_eq!(atlas.height(), 24);
    }

    #[test]
    fn constructor_clamps_to_one() {
        let atlas = SurfelAtlas::new(0, 0, 0);
        assert_eq!(atlas.capacity(), 1);
        assert_eq!(atlas.tile_resolution, 1);
    }

    #[test]
    fn tile_coord_roundtrip() {
        let atlas = SurfelAtlas::new(4, 3, 8);
        for id in 0..atlas.capacity() {
            let tile = atlas.tile_coord(id).expect("in range");
            let back = atlas.surfel_id(tile).expect("valid tile");
            assert_eq!(back, id, "roundtrip failed for id {id}");
        }
    }

    #[test]
    fn out_of_range_ids_have_no_slot() {
        let atlas = SurfelAtlas::new(2, 2, 4);
        assert_eq!(atlas.capacity(), 4);
        assert!(atlas.tile_coord(4).is_none());
        assert!(atlas.tile_origin(99).is_none());
        assert!(atlas.dir_to_texel(4, Vec3::Z).is_none());
    }

    #[test]
    fn tile_origins_are_disjoint_and_in_bounds() {
        let atlas = SurfelAtlas::new(3, 2, 8);
        for id in 0..atlas.capacity() {
            let origin = atlas.tile_origin(id).unwrap();
            assert!(origin.x + atlas.tile_resolution <= atlas.width());
            assert!(origin.y + atlas.tile_resolution <= atlas.height());
        }
        // Distinct ids map to distinct origins.
        let a = atlas.tile_origin(0).unwrap();
        let b = atlas.tile_origin(1).unwrap();
        assert_ne!((a.x, a.y), (b.x, b.y));
    }

    #[test]
    fn direction_texel_lands_inside_tile() {
        let atlas = SurfelAtlas::new(4, 4, 16);
        let texel = atlas.dir_to_texel(5, Vec3::new(0.3, -0.6, 0.7)).unwrap();
        let origin = atlas.tile_origin(5).unwrap();
        assert!(texel.x >= origin.x && texel.x < origin.x + atlas.tile_resolution);
        assert!(texel.y >= origin.y && texel.y < origin.y + atlas.tile_resolution);
    }

    #[test]
    fn direction_roundtrip_within_quantisation() {
        // A coarse direction survives dir -> texel -> dir up to the texel size.
        let atlas = SurfelAtlas::new(2, 2, 64);
        for dir in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::Y,
            Vec3::new(0.4, 0.5, 0.76).normalize(),
            Vec3::new(-0.3, 0.2, -0.93).normalize(),
        ] {
            let texel = atlas.dir_to_texel(1, dir).unwrap();
            let back = atlas.texel_to_dir(1, texel).unwrap();
            // One texel spans ~2/res in octahedral UV, mapped onto the sphere;
            // allow a generous angular tolerance tied to the resolution.
            let err = (back - dir).length();
            assert!(err < 0.1, "dir {dir:?} back {back:?} err {err}");
            assert!((back.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn local_texel_to_dir_is_unit_length() {
        let atlas = SurfelAtlas::new(1, 1, 8);
        for ly in 0..atlas.tile_resolution {
            for lx in 0..atlas.tile_resolution {
                let dir = atlas.local_texel_to_dir(lx, ly);
                assert!((dir.length() - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn texel_to_dir_rejects_foreign_tiles() {
        let atlas = SurfelAtlas::new(2, 2, 8);
        // Texel belonging to surfel 0's tile is not inside surfel 3's tile.
        let origin0 = atlas.tile_origin(0).unwrap();
        assert!(atlas.texel_to_dir(3, origin0).is_none());
        assert!(atlas.texel_to_dir(0, origin0).is_some());
    }

    #[test]
    fn degenerate_direction_is_safe() {
        let atlas = SurfelAtlas::new(2, 2, 8);
        let texel = atlas.dir_to_texel(0, Vec3::ZERO).unwrap();
        // Zero vector maps to the +z pole (uv centre); texel stays in-tile.
        assert!(texel.x < atlas.tile_resolution && texel.y < atlas.tile_resolution);
    }

    #[test]
    fn determinism() {
        let atlas = SurfelAtlas::new(4, 4, 16);
        let d = Vec3::new(0.2, 0.3, 0.9).normalize();
        assert_eq!(atlas.dir_to_texel(7, d), atlas.dir_to_texel(7, d));
    }
}
