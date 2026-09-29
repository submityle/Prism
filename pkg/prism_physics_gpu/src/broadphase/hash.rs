//! Shared spatial-hash math for the broad phase.
//!
//! Both the `CPU` golden twin and the `GPU` `WGSL` kernel call the exact same
//! integer/float operations defined here (mirrored line-for-line in
//! `shaders/broadphase.wgsl`), so a particle always lands in the same cell and
//! the same hash bucket regardless of which path runs it. This bit-identical
//! bucketing is what makes the real-device parity test meaningful.
//!
//! Provenance: Teschner et al. 2003 three-prime spatial hash. No Unreal Engine
//! source or derived code.

use glam::Vec3;

/// First hashing prime (`x` axis).
const P1: u32 = 73_856_093;
/// Second hashing prime (`y` axis).
const P2: u32 = 19_349_663;
/// Third hashing prime (`z` axis).
const P3: u32 = 83_492_791;

/// Returns the integer grid cell a world-space point falls in.
///
/// Uses component-wise `floor(p / cell_size)` truncated to [`i32`], matching
/// the `WGSL` `vec3<i32>(floor(p / cell))` exactly. `cell_size` must be
/// strictly positive; the caller (config validation) guarantees this.
#[must_use]
pub fn cell_coord(position: Vec3, cell_size: f32) -> [i32; 3] {
    let scaled = position / cell_size;
    [
        floor_to_i32(scaled.x),
        floor_to_i32(scaled.y),
        floor_to_i32(scaled.z),
    ]
}

/// Floors `value` and truncates it to [`i32`] the same way `WGSL` does.
#[must_use]
fn floor_to_i32(value: f32) -> i32 {
    value.floor() as i32
}

/// Hashes an integer cell into `[0, table_size)`.
///
/// The three axes are reinterpreted as [`u32`] (two's-complement bit pattern,
/// matching `WGSL` `bitcast<u32>`), multiplied by distinct primes with
/// wrapping, exclusive-ored together, and reduced modulo `table_size`.
/// `table_size` must be non-zero, which config validation guarantees.
#[must_use]
pub fn hash_cell(cell: [i32; 3], table_size: u32) -> u32 {
    let ux = (cell[0] as u32).wrapping_mul(P1);
    let uy = (cell[1] as u32).wrapping_mul(P2);
    let uz = (cell[2] as u32).wrapping_mul(P3);
    (ux ^ uy ^ uz) % table_size
}
