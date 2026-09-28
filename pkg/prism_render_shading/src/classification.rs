use alloc::vec::Vec;
use prism_render_material::{ClosureKind, GpuMaterialHeader, Illumination, MaterialRenderClass};

pub const MAX_SHADING_CLASSES: usize = 9;

/// The runtime shading bucket a pixel is binned into for the deferred/compute
/// shading pass. This is a *derived* projection of the orthogonal material axes
/// (`illumination` + `closure_mask` + `render_class`), not a stored field: the
/// material ABI no longer carries a single `shading_model`. Keeping the 9-way
/// enum lets the shading pass stay wavefront-coherent while the material side
/// stays fully orthogonal.
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
    UnsupportedIllumination(u32),
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

/// True when a closure bit is present in the packed mask.
fn has_closure(mask: u32, kind: ClosureKind) -> bool {
    mask & (1 << kind as u32) != 0
}

/// Project a `Lit` surface's closure graph onto a coherent shading bucket. The
/// dominant physically-based closure wins; a plain metal/dielectric surface is
/// `Principled`. This mirrors the old per-model buckets without forcing the
/// material author to pick one exclusive model.
fn lit_class_from_closures(closure_mask: u32) -> MaterialShadingClass {
    if has_closure(closure_mask, ClosureKind::Hair) {
        MaterialShadingClass::Hair
    } else if has_closure(closure_mask, ClosureKind::Subsurface) {
        MaterialShadingClass::Subsurface
    } else if has_closure(closure_mask, ClosureKind::ClearCoat) {
        MaterialShadingClass::ClearCoat
    } else if has_closure(closure_mask, ClosureKind::Sheen) {
        MaterialShadingClass::Cloth
    } else {
        MaterialShadingClass::Principled
    }
}

pub fn classify_material_header(
    header: &GpuMaterialHeader,
) -> Result<MaterialShadingClass, ClassificationError> {
    if header.active == 0 {
        return Err(ClassificationError::Inactive);
    }
    // Surface-domain blend families only. `Npr*`/`Custom*` no longer exist as
    // render classes; style lives on the illumination axis instead.
    if !matches!(
        header.render_class,
        x if x == MaterialRenderClass::Opaque as u32
            || x == MaterialRenderClass::OpaqueTwoSided as u32
            || x == MaterialRenderClass::Masked as u32
            || x == MaterialRenderClass::MaskedTwoSided as u32
            || x == MaterialRenderClass::Hair as u32
            || x == MaterialRenderClass::Water as u32
    ) {
        return Err(ClassificationError::NonSurfaceClass);
    }

    // Dedicated subsystem render classes bind their bucket directly, regardless
    // of the illumination axis (a stylized water shader is still water work).
    if header.render_class == MaterialRenderClass::Water as u32 {
        return Ok(MaterialShadingClass::Water);
    }
    if header.render_class == MaterialRenderClass::Hair as u32 {
        return Ok(MaterialShadingClass::Hair);
    }

    match Illumination::from_u32(header.illumination) {
        Some(Illumination::Unlit) => Ok(MaterialShadingClass::Unlit),
        Some(Illumination::Stylized) => Ok(MaterialShadingClass::Npr),
        Some(Illumination::Custom) => Ok(MaterialShadingClass::Custom),
        Some(Illumination::Lit) => Ok(lit_class_from_closures(header.closure_mask)),
        None => Err(ClassificationError::UnsupportedIllumination(
            header.illumination,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_material::fallback_material_header;

    /// Set the illumination axis on a header, mirroring how the material
    /// registry emits it.
    fn with_illumination(mut header: GpuMaterialHeader, illum: Illumination) -> GpuMaterialHeader {
        header.illumination = illum as u32;
        header
    }

    /// Set a single closure bit (the "dominant" physically-based closure).
    fn with_closure(mut header: GpuMaterialHeader, kind: ClosureKind) -> GpuMaterialHeader {
        header.closure_mask = 1 << kind as u32;
        header
    }

    #[test]
    fn illumination_axis_maps_to_style_buckets_without_per_asset_permutations() {
        let base = fallback_material_header(1);
        assert_eq!(
            classify_material_header(&with_illumination(base, Illumination::Unlit)),
            Ok(MaterialShadingClass::Unlit)
        );
        assert_eq!(
            classify_material_header(&with_illumination(base, Illumination::Stylized)),
            Ok(MaterialShadingClass::Npr)
        );
        assert_eq!(
            classify_material_header(&with_illumination(base, Illumination::Custom)),
            Ok(MaterialShadingClass::Custom)
        );
    }

    #[test]
    fn lit_surfaces_derive_their_bucket_from_the_dominant_closure() {
        let base = with_illumination(fallback_material_header(1), Illumination::Lit);
        for (kind, expected) in [
            (ClosureKind::Diffuse, MaterialShadingClass::Principled),
            (ClosureKind::Conductor, MaterialShadingClass::Principled),
            (ClosureKind::Subsurface, MaterialShadingClass::Subsurface),
            (ClosureKind::ClearCoat, MaterialShadingClass::ClearCoat),
            (ClosureKind::Sheen, MaterialShadingClass::Cloth),
            (ClosureKind::Hair, MaterialShadingClass::Hair),
        ] {
            assert_eq!(
                classify_material_header(&with_closure(base, kind)),
                Ok(expected),
                "closure {kind:?} should classify as {expected:?}"
            );
        }
    }

    #[test]
    fn hair_and_water_render_classes_bind_their_bucket_directly() {
        for (render_class, expected) in [
            (MaterialRenderClass::Hair, MaterialShadingClass::Hair),
            (MaterialRenderClass::Water, MaterialShadingClass::Water),
        ] {
            let mut header = fallback_material_header(1);
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
        npr.illumination = Illumination::Stylized as u32;
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
