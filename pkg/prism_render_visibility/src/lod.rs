use alloc::vec::Vec;
use prism_render_architecture::gpu_scene::GeometryHandle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryLod {
    pub level: u16,
    pub screen_error: f32,
    pub resident: bool,
    pub fallback: bool,
}

/// The element and vertex windows consumed by one concrete mesh draw.
///
/// `first_element..first_element + element_count` addresses the mesh's index
/// buffer for indexed draws, or its vertex buffer for non-indexed draws.
/// `base_vertex` is the signed indexed-draw vertex offset. `vertex_count` is
/// the number of vertices in the LOD's vertex window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryLodDrawWindow {
    pub first_element: u32,
    pub element_count: u32,
    pub base_vertex: i32,
    pub vertex_count: u32,
}

/// A fully specified LOD suitable for driving an actual mesh draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryLodDefinition {
    pub level: u16,
    pub screen_error: f32,
    pub resident: bool,
    pub fallback: bool,
    pub draw_window: GeometryLodDrawWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryLodValidationError {
    EmptyElementWindow,
    EmptyVertexWindow,
    ElementWindowOverflow,
    ElementWindowOutOfBounds,
    VertexWindowOutOfBounds,
    IndexedVertexOutOfBounds,
    InvalidScreenError,
    DuplicateLevel(u16),
    AllocationFailed,
}

impl GeometryLodDrawWindow {
    /// Validates the draw window against the backing mesh buffer lengths.
    ///
    /// For indexed draws, the caller must additionally ensure every selected
    /// index plus `base_vertex` addresses a vertex in the supplied vertex
    /// window; that requires inspecting the actual index data.
    pub fn validate(
        self,
        total_elements: u32,
        total_vertices: u32,
    ) -> Result<(), GeometryLodValidationError> {
        if self.element_count == 0 {
            return Err(GeometryLodValidationError::EmptyElementWindow);
        }
        if self.vertex_count == 0 {
            return Err(GeometryLodValidationError::EmptyVertexWindow);
        }
        let element_end = self
            .first_element
            .checked_add(self.element_count)
            .ok_or(GeometryLodValidationError::ElementWindowOverflow)?;
        if element_end > total_elements {
            return Err(GeometryLodValidationError::ElementWindowOutOfBounds);
        }
        // `base_vertex` is a signed draw offset, not the beginning of a
        // contiguous vertex range. Its effective bounds depend on the actual
        // selected index values and are intentionally left to mesh validation.
        if self.vertex_count > total_vertices {
            return Err(GeometryLodValidationError::VertexWindowOutOfBounds);
        }
        Ok(())
    }

    /// Validates an indexed draw against its actual index buffer contents.
    pub fn validate_indexed(
        self,
        indices: &[u32],
        total_vertices: u32,
    ) -> Result<(), GeometryLodValidationError> {
        let total_elements = u32::try_from(indices.len())
            .map_err(|_| GeometryLodValidationError::ElementWindowOutOfBounds)?;
        self.validate(total_elements, total_vertices)?;
        let end = self.first_element + self.element_count;
        for &index in &indices[self.first_element as usize..end as usize] {
            let addressed_vertex = i64::from(index) + i64::from(self.base_vertex);
            if !(0..i64::from(self.vertex_count)).contains(&addressed_vertex) {
                return Err(GeometryLodValidationError::IndexedVertexOutOfBounds);
            }
        }
        Ok(())
    }
}

impl GeometryLodDefinition {
    pub fn validate(
        self,
        total_elements: u32,
        total_vertices: u32,
    ) -> Result<(), GeometryLodValidationError> {
        if !self.screen_error.is_finite() || self.screen_error < 0.0 {
            return Err(GeometryLodValidationError::InvalidScreenError);
        }
        self.draw_window.validate(total_elements, total_vertices)
    }
}

/// Validates that a complete set of drawable LOD definitions has unique levels.
pub fn validate_geometry_lod_levels(
    lods: &[GeometryLodDefinition],
    total_elements: u32,
    total_vertices: u32,
) -> Result<(), GeometryLodValidationError> {
    let mut levels = Vec::new();
    levels
        .try_reserve_exact(lods.len())
        .map_err(|_| GeometryLodValidationError::AllocationFailed)?;
    for lod in lods {
        lod.validate(total_elements, total_vertices)?;
        levels.push(lod.level);
    }
    levels.sort_unstable();
    if let Some(pair) = levels.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(GeometryLodValidationError::DuplicateLevel(pair[0]));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryLodChain {
    pub geometry: GeometryHandle,
    pub lods: Vec<GeometryLod>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSelection {
    pub level: u16,
    pub used_fallback: bool,
}

impl GeometryLodChain {
    pub fn select(
        &self,
        projected_radius: f32,
        lod_scale: f32,
        previous: Option<u16>,
    ) -> Option<LodSelection> {
        let target = 1.0 / (projected_radius.max(1.0e-4) * lod_scale.max(1.0e-4));
        let mut selected = self
            .lods
            .iter()
            .filter(|lod| lod.resident && lod.screen_error <= target)
            .max_by(|a, b| a.screen_error.total_cmp(&b.screen_error));
        if let Some(previous) = previous
            && let Some(previous_lod) = self
                .lods
                .iter()
                .find(|lod| lod.level == previous && lod.resident)
            && (previous_lod.screen_error - target).abs() < target * 0.15
        {
            selected = Some(previous_lod);
        }
        selected
            .or_else(|| {
                self.lods
                    .iter()
                    .filter(|lod| lod.resident && lod.fallback)
                    .min_by_key(|lod| lod.level)
            })
            .map(|lod| LodSelection {
                level: lod.level,
                used_fallback: lod.fallback,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;

    fn chain() -> GeometryLodChain {
        GeometryLodChain {
            geometry: GenerationalHandle {
                index: 1,
                generation: 1,
            },
            lods: vec![
                GeometryLod {
                    level: 0,
                    screen_error: 0.1,
                    resident: true,
                    fallback: true,
                },
                GeometryLod {
                    level: 1,
                    screen_error: 0.5,
                    resident: true,
                    fallback: false,
                },
                GeometryLod {
                    level: 2,
                    screen_error: 1.0,
                    resident: true,
                    fallback: false,
                },
            ],
        }
    }

    #[test]
    fn selects_coarsest_lod_within_error_budget() {
        assert_eq!(chain().select(1.0, 1.0, None).unwrap().level, 2);
        assert_eq!(chain().select(4.0, 1.0, None).unwrap().level, 0);
    }

    #[test]
    fn preserves_previous_lod_inside_hysteresis_band() {
        assert_eq!(chain().select(1.9, 1.0, Some(1)).unwrap().level, 1);
    }

    fn definition(level: u16, window: GeometryLodDrawWindow) -> GeometryLodDefinition {
        GeometryLodDefinition {
            level,
            screen_error: 0.1,
            resident: true,
            fallback: level == 0,
            draw_window: window,
        }
    }

    fn valid_window() -> GeometryLodDrawWindow {
        GeometryLodDrawWindow {
            first_element: 8,
            element_count: 12,
            base_vertex: -2,
            vertex_count: 16,
        }
    }

    #[test]
    fn validates_unique_lod_levels_and_mesh_window_bounds() {
        let lods = [definition(0, valid_window()), definition(1, valid_window())];
        assert_eq!(validate_geometry_lod_levels(&lods, 20, 16), Ok(()));
        assert_eq!(
            validate_geometry_lod_levels(&lods, 19, 16),
            Err(GeometryLodValidationError::ElementWindowOutOfBounds)
        );

        let duplicate = [definition(3, valid_window()), definition(3, valid_window())];
        assert_eq!(
            validate_geometry_lod_levels(&duplicate, 20, 16),
            Err(GeometryLodValidationError::DuplicateLevel(3))
        );
    }

    #[test]
    fn rejects_empty_and_overflowing_windows() {
        let mut window = valid_window();
        window.element_count = 0;
        assert_eq!(
            window.validate(20, 16),
            Err(GeometryLodValidationError::EmptyElementWindow)
        );

        window = valid_window();
        window.first_element = u32::MAX - 2;
        window.element_count = 4;
        assert_eq!(
            window.validate(u32::MAX, 16),
            Err(GeometryLodValidationError::ElementWindowOverflow)
        );

        window = valid_window();
        window.vertex_count = 17;
        assert_eq!(
            window.validate(20, 16),
            Err(GeometryLodValidationError::VertexWindowOutOfBounds)
        );
    }

    #[test]
    fn rejects_invalid_screen_error() {
        let mut lod = definition(0, valid_window());
        lod.screen_error = f32::NAN;
        assert_eq!(
            lod.validate(20, 16),
            Err(GeometryLodValidationError::InvalidScreenError)
        );
        lod.screen_error = -0.1;
        assert_eq!(
            lod.validate(20, 16),
            Err(GeometryLodValidationError::InvalidScreenError)
        );
    }

    #[test]
    fn indexed_validation_applies_signed_base_vertex_to_selected_elements() {
        let window = valid_window();
        let indices = [
            0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        ];
        assert_eq!(window.validate_indexed(&indices, 16), Ok(()));

        let invalid = [
            0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        ];
        assert_eq!(
            window.validate_indexed(&invalid, 16),
            Err(GeometryLodValidationError::IndexedVertexOutOfBounds)
        );
    }
}
