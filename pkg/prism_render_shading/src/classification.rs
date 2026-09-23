use alloc::vec::Vec;
use prism_render_material::{GpuMaterialHeader, MaterialRenderClass, MaterialShadingModel};

pub const MAX_SHADING_CLASSES: usize = 9;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum MaterialShadingClass {
    #[default]
    Principled,
    Unlit,
    Subsurface,
    ClearCoat,
    Cloth,
    Hair,
    Water,
    Npr,
    Custom,
}

impl MaterialShadingClass {
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationError {
    Inactive,
    NonSurfaceClass,
    UnsupportedShadingModel(u32),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadingWorkItem {
    pub pixel_index: u32,
    pub material_index: u32,
    pub shading_class: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadingWorkPlan {
    pub offsets: [u32; MAX_SHADING_CLASSES],
    pub counts: [u32; MAX_SHADING_CLASSES],
    pub work: Vec<ShadingWorkItem>,
}

impl ShadingWorkPlan {
    pub fn build(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
    ) -> Self {
        let mut classified: [Vec<ShadingWorkItem>; MAX_SHADING_CLASSES] = Default::default();
        for (pixel_index, pixel) in pixels {
            if !pixel.is_valid()
                || classified.iter().map(Vec::len).sum::<usize>() >= capacity as usize
            {
                continue;
            }
            let Some(header) = materials.get(pixel.material_index as usize) else {
                continue;
            };
            if header.generation != pixel.material_generation {
                continue;
            }
            let Ok(class) = classify_material_header(header) else {
                continue;
            };
            classified[class.index()].push(ShadingWorkItem {
                pixel_index,
                material_index: pixel.material_index,
                shading_class: class as u32,
                flags: header.feature_flags,
            });
        }
        let mut plan = Self::default();
        for (class, items) in classified.into_iter().enumerate() {
            plan.offsets[class] = plan.work.len() as u32;
            plan.counts[class] = items.len() as u32;
            plan.work.extend(items);
        }
        plan
    }
}

pub fn classify_material_header(
    header: &GpuMaterialHeader,
) -> Result<MaterialShadingClass, ClassificationError> {
    if header.active == 0 {
        return Err(ClassificationError::Inactive);
    }
    if !matches!(
        header.render_class,
        x if x == MaterialRenderClass::Opaque as u32
            || x == MaterialRenderClass::OpaqueTwoSided as u32
            || x == MaterialRenderClass::Masked as u32
            || x == MaterialRenderClass::MaskedTwoSided as u32
            || x == MaterialRenderClass::Hair as u32
            || x == MaterialRenderClass::Water as u32
            || x == MaterialRenderClass::NprOpaque as u32
            || x == MaterialRenderClass::CustomOpaque as u32
    ) {
        return Err(ClassificationError::NonSurfaceClass);
    }
    match header.shading_model {
        x if x == MaterialShadingModel::Principled as u32 => Ok(MaterialShadingClass::Principled),
        x if x == MaterialShadingModel::Unlit as u32 => Ok(MaterialShadingClass::Unlit),
        x if x == MaterialShadingModel::Subsurface as u32 => Ok(MaterialShadingClass::Subsurface),
        x if x == MaterialShadingModel::ClearCoat as u32 => Ok(MaterialShadingClass::ClearCoat),
        x if x == MaterialShadingModel::Cloth as u32 => Ok(MaterialShadingClass::Cloth),
        x if x == MaterialShadingModel::Hair as u32 => Ok(MaterialShadingClass::Hair),
        x if x == MaterialShadingModel::Water as u32 => Ok(MaterialShadingClass::Water),
        x if x == MaterialShadingModel::Npr as u32 => Ok(MaterialShadingClass::Npr),
        x if x == MaterialShadingModel::Custom as u32 => Ok(MaterialShadingClass::Custom),
        other => Err(ClassificationError::UnsupportedShadingModel(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_material::fallback_material_header;

    #[test]
    fn classifies_every_supported_model_without_per_asset_permutations() {
        for (model, class) in [
            (
                MaterialShadingModel::Principled,
                MaterialShadingClass::Principled,
            ),
            (MaterialShadingModel::Unlit, MaterialShadingClass::Unlit),
            (
                MaterialShadingModel::Subsurface,
                MaterialShadingClass::Subsurface,
            ),
            (
                MaterialShadingModel::ClearCoat,
                MaterialShadingClass::ClearCoat,
            ),
            (MaterialShadingModel::Cloth, MaterialShadingClass::Cloth),
            (MaterialShadingModel::Hair, MaterialShadingClass::Hair),
            (MaterialShadingModel::Water, MaterialShadingClass::Water),
            (MaterialShadingModel::Npr, MaterialShadingClass::Npr),
            (MaterialShadingModel::Custom, MaterialShadingClass::Custom),
        ] {
            let mut header = fallback_material_header(1);
            header.shading_model = model as u32;
            assert_eq!(classify_material_header(&header), Ok(class));
        }
    }

    #[test]
    fn hair_and_water_surface_render_classes_are_classifiable() {
        for (model, render_class, expected) in [
            (
                MaterialShadingModel::Hair,
                MaterialRenderClass::Hair,
                MaterialShadingClass::Hair,
            ),
            (
                MaterialShadingModel::Water,
                MaterialRenderClass::Water,
                MaterialShadingClass::Water,
            ),
        ] {
            let mut header = fallback_material_header(1);
            header.shading_model = model as u32;
            header.render_class = render_class as u32;
            assert_eq!(classify_material_header(&header), Ok(expected));
        }
    }

    #[test]
    fn work_plan_is_class_contiguous_and_generation_safe() {
        let mut pbr = fallback_material_header(1);
        pbr.generation = 2;
        let mut npr = fallback_material_header(1);
        npr.generation = 4;
        npr.shading_model = MaterialShadingModel::Npr as u32;
        npr.render_class = MaterialRenderClass::NprOpaque as u32;
        let pixel = |material: GenerationalHandle| {
            super::super::VisibilityPixel::new(
                GenerationalHandle {
                    index: 1,
                    generation: 1,
                },
                0,
                0,
                material,
                [1.0, 0.0, 0.0],
                1,
            )
            .unwrap()
        };
        let plan = ShadingWorkPlan::build(
            [
                (
                    9,
                    pixel(GenerationalHandle {
                        index: 1,
                        generation: 4,
                    }),
                ),
                (
                    3,
                    pixel(GenerationalHandle {
                        index: 0,
                        generation: 2,
                    }),
                ),
                (
                    7,
                    pixel(GenerationalHandle {
                        index: 1,
                        generation: 3,
                    }),
                ),
            ],
            &[pbr, npr],
            8,
        );

        assert_eq!(plan.counts[MaterialShadingClass::Principled.index()], 1);
        assert_eq!(plan.counts[MaterialShadingClass::Npr.index()], 1);
        assert_eq!(plan.work.len(), 2);
        assert_eq!(plan.work[0].pixel_index, 3);
        assert_eq!(plan.work[1].pixel_index, 9);
    }
}
