//! Stable geometry metadata shared by visibility and draw consumers.

use crate::gpu_scene::GeometryHandle;

/// Version of the initial geometry table ABI.
pub const GEOMETRY_ABI_VERSION: u32 = 1;

/// Logical primitive encoding. Physical buffers remain backend-owned.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum GeometryPrimitiveKind {
    #[default]
    Indexed,
    NonIndexed,
}

/// Draw metadata for one resident LOD.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryLodRecord {
    pub level: u32,
    pub primitive_kind: GeometryPrimitiveKind,
    pub element_count: u32,
    pub first_element: u32,
    pub base_vertex: i32,
    pub vertex_count: u32,
    pub screen_error: f32,
    pub resident: bool,
    pub fallback: bool,
}

/// Stable record addressed by a generational geometry handle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryRecord {
    pub handle: GeometryHandle,
    pub revision: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub primitive_start: u32,
    pub primitive_count: u32,
    pub lods: Vec<GeometryLodRecord>,
}

impl GeometryRecord {
    /// Selects the requested resident LOD, falling back toward finer levels
    /// and finally to the explicitly marked resident fallback.
    pub fn resolve_lod(&self, requested: u32) -> Option<&GeometryLodRecord> {
        self.lods
            .iter()
            .filter(|lod| lod.resident && lod.level <= requested)
            .max_by_key(|lod| lod.level)
            .or_else(|| self.lods.iter().find(|lod| lod.resident && lod.fallback))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_to_resident_ancestor_then_fallback() {
        let record = GeometryRecord {
            lods: vec![
                GeometryLodRecord {
                    level: 0,
                    resident: true,
                    fallback: true,
                    ..Default::default()
                },
                GeometryLodRecord {
                    level: 1,
                    resident: false,
                    ..Default::default()
                },
                GeometryLodRecord {
                    level: 2,
                    resident: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(record.resolve_lod(1).unwrap().level, 0);
        assert_eq!(record.resolve_lod(2).unwrap().level, 2);
    }
}
