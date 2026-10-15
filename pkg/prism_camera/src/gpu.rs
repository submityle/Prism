use prism_math::{DVec3, Mat4, Vec2};

pub const VIEW_SCHEMA_VERSION: u32 = 1;

/// CPU-side ABI record. The renderer may transmute/copy this into its backend
/// buffer after applying its own validated byte-layout checks.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(16))]
pub struct GpuViewRecord {
    pub clip_from_world: [[f32; 4]; 4],
    pub clip_from_world_unjittered: [[f32; 4]; 4],
    pub prev_clip_from_world: [[f32; 4]; 4],
    pub view_from_world: [[f32; 4]; 4],
    pub world_from_view: [[f32; 4]; 4],
    pub world_origin_hi: [f32; 4],
    pub world_origin_lo: [f32; 4],
    pub exposure: f32,
    pub schema_version: u32,
    pub flags: u32,
    pub view_index: u32,
    pub jitter: [f32; 2],
    pub viewport: [f32; 4],
    pub frustum: [[f32; 4]; 6],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn record_is_abi_aligned_and_flags_are_stable() {
        assert_eq!(size_of::<GpuViewRecord>() % 16, 0);
        assert_eq!(GpuViewRecord::flags(true, false, true, true), 0b1101);
    }

    #[test]
    fn origin_split_reconstructs_with_f32_parts() {
        let origin = DVec3::new(1_000_000.125, -2_000_000.25, 3.5);
        let (hi, lo) = GpuViewRecord::split_origin(origin);
        assert!(((hi[0] as f64 + lo[0] as f64) - origin.x).abs() < 0.01);
        assert!(((hi[1] as f64 + lo[1] as f64) - origin.y).abs() < 0.01);
    }
}

impl GpuViewRecord {
    pub fn split_origin(origin: DVec3) -> ([f32; 4], [f32; 4]) {
        fn split(v: f64) -> (f32, f32) {
            let hi = v as f32;
            let lo = (v - hi as f64) as f32;
            (hi, lo)
        }
        let (hx, lx) = split(origin.x);
        let (hy, ly) = split(origin.y);
        let (hz, lz) = split(origin.z);
        ([hx, hy, hz, 0.0], [lx, ly, lz, 0.0])
    }

    pub fn matrix(m: Mat4) -> [[f32; 4]; 4] {
        [
            [m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w],
            [m.y_axis.x, m.y_axis.y, m.y_axis.z, m.y_axis.w],
            [m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w],
            [m.w_axis.x, m.w_axis.y, m.w_axis.z, m.w_axis.w],
        ]
    }

    pub fn flags(history_valid: bool, teleport: bool, finite_far: bool, relative: bool) -> u32 {
        u32::from(history_valid)
            | (u32::from(teleport) << 1)
            | (u32::from(finite_far) << 2)
            | (u32::from(relative) << 3)
    }

    pub fn jitter(jitter: Vec2) -> [f32; 2] {
        [jitter.x, jitter.y]
    }
}
