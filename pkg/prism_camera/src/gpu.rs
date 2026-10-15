use prism_math::{DVec3, Mat4, Vec2};

pub const VIEW_SCHEMA_VERSION: u32 = 1;
pub const GPU_VIEW_RECORD_ALIGNMENT: usize = 16;
pub const GPU_VIEW_RECORD_SIZE: usize = 496;

pub const GPU_VIEW_OFFSET_CLIP_FROM_WORLD: usize = 0;
pub const GPU_VIEW_OFFSET_CLIP_FROM_WORLD_UNJITTERED: usize = 64;
pub const GPU_VIEW_OFFSET_PREV_CLIP_FROM_WORLD: usize = 128;
pub const GPU_VIEW_OFFSET_VIEW_FROM_WORLD: usize = 192;
pub const GPU_VIEW_OFFSET_WORLD_FROM_VIEW: usize = 256;
pub const GPU_VIEW_OFFSET_WORLD_ORIGIN_HI: usize = 320;
pub const GPU_VIEW_OFFSET_WORLD_ORIGIN_LO: usize = 336;
pub const GPU_VIEW_OFFSET_EXPOSURE: usize = 352;
pub const GPU_VIEW_OFFSET_SCHEMA_VERSION: usize = 356;
pub const GPU_VIEW_OFFSET_FLAGS: usize = 360;
pub const GPU_VIEW_OFFSET_VIEW_INDEX: usize = 364;
pub const GPU_VIEW_OFFSET_JITTER: usize = 368;
pub const GPU_VIEW_OFFSET_VIEWPORT: usize = 376;
pub const GPU_VIEW_OFFSET_FRUSTUM: usize = 392;

/// CPU-side ABI record. The renderer may transmute/copy this into its backend
/// buffer after applying its own validated byte-layout checks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    use core::mem::{align_of, size_of};

    #[test]
    fn record_has_stable_wgsl_layout() {
        assert_eq!(size_of::<GpuViewRecord>(), GPU_VIEW_RECORD_SIZE);
        assert_eq!(align_of::<GpuViewRecord>(), GPU_VIEW_RECORD_ALIGNMENT);
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, clip_from_world),
            GPU_VIEW_OFFSET_CLIP_FROM_WORLD
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, clip_from_world_unjittered),
            GPU_VIEW_OFFSET_CLIP_FROM_WORLD_UNJITTERED
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, prev_clip_from_world),
            GPU_VIEW_OFFSET_PREV_CLIP_FROM_WORLD
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, view_from_world),
            GPU_VIEW_OFFSET_VIEW_FROM_WORLD
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, world_from_view),
            GPU_VIEW_OFFSET_WORLD_FROM_VIEW
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, world_origin_hi),
            GPU_VIEW_OFFSET_WORLD_ORIGIN_HI
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, world_origin_lo),
            GPU_VIEW_OFFSET_WORLD_ORIGIN_LO
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, exposure),
            GPU_VIEW_OFFSET_EXPOSURE
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, schema_version),
            GPU_VIEW_OFFSET_SCHEMA_VERSION
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, flags),
            GPU_VIEW_OFFSET_FLAGS
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, view_index),
            GPU_VIEW_OFFSET_VIEW_INDEX
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, jitter),
            GPU_VIEW_OFFSET_JITTER
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, viewport),
            GPU_VIEW_OFFSET_VIEWPORT
        );
        assert_eq!(
            core::mem::offset_of!(GpuViewRecord, frustum),
            GPU_VIEW_OFFSET_FRUSTUM
        );
    }

    #[test]
    fn record_flags_are_stable() {
        assert_eq!(GpuViewRecord::flags(true, false, true, true), 0b1101);
    }

    #[test]
    fn origin_split_reconstructs_all_three_axes() {
        let origin = DVec3::new(1_000_000.125, -2_000_000.25, 3.5);
        let (hi, lo) = GpuViewRecord::split_origin(origin);
        assert!(((hi[0] as f64 + lo[0] as f64) - origin.x).abs() < 0.01);
        assert!(((hi[1] as f64 + lo[1] as f64) - origin.y).abs() < 0.01);
        assert!(((hi[2] as f64 + lo[2] as f64) - origin.z).abs() < 0.01);
    }

    #[test]
    fn matrix_and_bytes_use_column_major_order() {
        let record = GpuViewRecord {
            clip_from_world: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            ..Default::default()
        };
        let bytes = record.to_bytes_le();
        for (index, value) in (1_u32..=16).enumerate() {
            let start = index * 4;
            assert_eq!(&bytes[start..start + 4], &(value as f32).to_le_bytes());
        }
        assert!(bytes[488..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn bytes_keep_scalar_fields_at_declared_offsets() {
        let record = GpuViewRecord {
            exposure: 1.5,
            schema_version: 2,
            flags: 3,
            view_index: 4,
            jitter: [5.0, 6.0],
            viewport: [7.0, 8.0, 9.0, 10.0],
            ..Default::default()
        };
        let bytes = record.to_bytes_le();
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_EXPOSURE..][..4],
            &1.5_f32.to_le_bytes()
        );
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_SCHEMA_VERSION..][..4],
            &2_u32.to_le_bytes()
        );
        assert_eq!(&bytes[GPU_VIEW_OFFSET_FLAGS..][..4], &3_u32.to_le_bytes());
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_VIEW_INDEX..][..4],
            &4_u32.to_le_bytes()
        );
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_JITTER..][..4],
            &5.0_f32.to_le_bytes()
        );
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_JITTER + 4..][..4],
            &6.0_f32.to_le_bytes()
        );
        assert_eq!(
            &bytes[GPU_VIEW_OFFSET_VIEWPORT..][..4],
            &7.0_f32.to_le_bytes()
        );
        assert!(bytes[488..].iter().all(|byte| *byte == 0));
    }
}

impl GpuViewRecord {
    /// Serializes the record in the stable little-endian byte order consumed by
    /// the renderer. Matrices are emitted as four consecutive column vectors.
    pub fn to_bytes_le(self) -> [u8; GPU_VIEW_RECORD_SIZE] {
        let mut bytes = [0_u8; GPU_VIEW_RECORD_SIZE];
        let mut offset = 0;

        fn write_f32(bytes: &mut [u8], offset: &mut usize, value: f32) {
            bytes[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
            *offset += 4;
        }

        fn write_u32(bytes: &mut [u8], offset: &mut usize, value: u32) {
            bytes[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
            *offset += 4;
        }

        fn write_matrix(bytes: &mut [u8], offset: &mut usize, matrix: [[f32; 4]; 4]) {
            for column in matrix {
                for value in column {
                    write_f32(bytes, offset, value);
                }
            }
        }

        fn write_vec4(bytes: &mut [u8], offset: &mut usize, value: [f32; 4]) {
            for component in value {
                write_f32(bytes, offset, component);
            }
        }

        write_matrix(&mut bytes, &mut offset, self.clip_from_world);
        write_matrix(&mut bytes, &mut offset, self.clip_from_world_unjittered);
        write_matrix(&mut bytes, &mut offset, self.prev_clip_from_world);
        write_matrix(&mut bytes, &mut offset, self.view_from_world);
        write_matrix(&mut bytes, &mut offset, self.world_from_view);
        write_vec4(&mut bytes, &mut offset, self.world_origin_hi);
        write_vec4(&mut bytes, &mut offset, self.world_origin_lo);
        write_f32(&mut bytes, &mut offset, self.exposure);
        write_u32(&mut bytes, &mut offset, self.schema_version);
        write_u32(&mut bytes, &mut offset, self.flags);
        write_u32(&mut bytes, &mut offset, self.view_index);
        write_f32(&mut bytes, &mut offset, self.jitter[0]);
        write_f32(&mut bytes, &mut offset, self.jitter[1]);
        write_vec4(&mut bytes, &mut offset, self.viewport);
        for plane in self.frustum {
            write_vec4(&mut bytes, &mut offset, plane);
        }
        debug_assert_eq!(offset, GPU_VIEW_OFFSET_FRUSTUM + 6 * 16);
        bytes
    }

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
