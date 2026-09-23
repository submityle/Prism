use prism_render_architecture::abi::GenerationalHandle;

pub const VISIBILITY_BUFFER_ABI_VERSION: u32 = 1;
pub const INVALID_VISIBILITY_ID: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarycentricError {
    NonFinite,
    OutsideTriangle,
}

/// Stable pixel identity written by both standard meshes and virtual geometry.
///
/// Generations prevent delayed shading from resolving recycled scene/material
/// slots. `primitive_id` is local to the resolved geometry LOD or cluster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisibilityPixel {
    pub scene_index: u32,
    pub scene_generation: u32,
    pub primitive_id: u32,
    pub geometry_lod_or_cluster: u32,
    pub material_index: u32,
    pub material_generation: u32,
    pub barycentrics_unorm16: u32,
    pub coverage_and_flags: u32,
}

/// Physical render-target layout for one [`VisibilityPixel`].
///
/// The raster pass writes exactly two `Rgba32Uint` attachments. Keeping this
/// representation explicit prevents the CPU ABI and GPU attachment contract
/// from drifting apart as the resolve path evolves.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisibilityPixelTargets {
    pub ids: [u32; 4],
    pub metadata: [u32; 4],
}

impl VisibilityPixel {
    pub const INVALID: Self = Self {
        scene_index: INVALID_VISIBILITY_ID,
        scene_generation: 0,
        primitive_id: INVALID_VISIBILITY_ID,
        geometry_lod_or_cluster: INVALID_VISIBILITY_ID,
        material_index: INVALID_VISIBILITY_ID,
        material_generation: 0,
        barycentrics_unorm16: 0,
        coverage_and_flags: 0,
    };

    pub fn new(
        scene: GenerationalHandle,
        primitive_id: u32,
        geometry_lod_or_cluster: u32,
        material: GenerationalHandle,
        barycentrics: [f32; 3],
        coverage: u8,
    ) -> Result<Self, BarycentricError> {
        Ok(Self {
            scene_index: scene.index,
            scene_generation: scene.generation,
            primitive_id,
            geometry_lod_or_cluster,
            material_index: material.index,
            material_generation: material.generation,
            barycentrics_unorm16: encode_barycentrics(barycentrics)?,
            coverage_and_flags: u32::from(coverage),
        })
    }

    pub const fn is_valid(self) -> bool {
        self.scene_index != INVALID_VISIBILITY_ID
            && self.primitive_id != INVALID_VISIBILITY_ID
            && self.material_index != INVALID_VISIBILITY_ID
    }

    pub fn barycentrics(self) -> [f32; 3] {
        let x = (self.barycentrics_unorm16 & 0xffff) as f32 / 65535.0;
        let y = (self.barycentrics_unorm16 >> 16) as f32 / 65535.0;
        [x, y, (1.0 - x - y).max(0.0)]
    }

    pub const fn coverage(self) -> u8 {
        self.coverage_and_flags as u8
    }

    pub const fn targets(self) -> VisibilityPixelTargets {
        VisibilityPixelTargets {
            ids: [
                self.scene_index,
                self.scene_generation,
                self.primitive_id,
                self.geometry_lod_or_cluster,
            ],
            metadata: [
                self.material_index,
                self.material_generation,
                self.coverage_and_flags,
                self.barycentrics_unorm16,
            ],
        }
    }

    pub const fn from_targets(targets: VisibilityPixelTargets) -> Self {
        Self {
            scene_index: targets.ids[0],
            scene_generation: targets.ids[1],
            primitive_id: targets.ids[2],
            geometry_lod_or_cluster: targets.ids[3],
            material_index: targets.metadata[0],
            material_generation: targets.metadata[1],
            coverage_and_flags: targets.metadata[2],
            barycentrics_unorm16: targets.metadata[3],
        }
    }
}

pub fn encode_barycentrics(value: [f32; 3]) -> Result<u32, BarycentricError> {
    if !value.into_iter().all(f32::is_finite) {
        return Err(BarycentricError::NonFinite);
    }
    let epsilon = 1.0e-4;
    if value.into_iter().any(|component| component < -epsilon)
        || (value.into_iter().sum::<f32>() - 1.0).abs() > epsilon
    {
        return Err(BarycentricError::OutsideTriangle);
    }
    let x = (value[0].clamp(0.0, 1.0) * 65535.0).round() as u32;
    let y = (value[1].clamp(0.0, 1.0) * 65535.0).round() as u32;
    Ok(x | (y << 16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visibility_pixel_has_stable_32_byte_layout_and_invalid_sentinel() {
        assert_eq!(size_of::<VisibilityPixel>(), 32);
        assert_eq!(size_of::<VisibilityPixelTargets>(), 32);
        assert!(!VisibilityPixel::INVALID.is_valid());
        assert_eq!(VisibilityPixel::INVALID.coverage(), 0);
    }

    #[test]
    fn physical_targets_round_trip_the_cpu_abi() {
        let pixel = VisibilityPixel::new(
            GenerationalHandle {
                index: 7,
                generation: 3,
            },
            9,
            2,
            GenerationalHandle {
                index: 4,
                generation: 6,
            },
            [0.25, 0.5, 0.25],
            0b1011,
        )
        .unwrap();
        assert_eq!(VisibilityPixel::from_targets(pixel.targets()), pixel);
    }

    #[test]
    fn barycentrics_round_trip_and_reject_invalid_input() {
        let pixel = VisibilityPixel::new(
            GenerationalHandle {
                index: 7,
                generation: 3,
            },
            9,
            2,
            GenerationalHandle {
                index: 4,
                generation: 6,
            },
            [0.25, 0.5, 0.25],
            0b1011,
        )
        .unwrap();
        let decoded = pixel.barycentrics();
        assert!((decoded[0] - 0.25).abs() < 2.0 / 65535.0);
        assert!((decoded[1] - 0.5).abs() < 2.0 / 65535.0);
        assert!((decoded.into_iter().sum::<f32>() - 1.0).abs() < 2.0 / 65535.0);
        assert_eq!(pixel.coverage(), 0b1011);
        assert_eq!(encode_barycentrics([f32::NAN, 0.0, 1.0]), Err(BarycentricError::NonFinite));
        assert_eq!(
            encode_barycentrics([0.8, 0.8, -0.6]),
            Err(BarycentricError::OutsideTriangle)
        );
    }
}
