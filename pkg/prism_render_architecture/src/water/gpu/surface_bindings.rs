//! The water-surface raster draw's `GPU` resource contract: the `@group(0)`
//! binding layout, the per-view uniform byte layout, and the indexed grid
//! draw-call sizing the scene crate's raster node builds against
//! `water_surface_raster.wesl`.
//!
//! [`super::surface_pass`] fixes the *pass* state (blend, depth, target, and the
//! stable `WESL` entry-point names). This module fixes the *resource* side of
//! the same draw: which bindings sit at which `@group(0)` slot, which shader
//! stage each is visible to, how the per-view uniform block is laid out in
//! bytes, and how many vertices/indices one surface patch draws. Together they
//! are the full `CPU`-testable contract the scene draw node is a dumb executor
//! of — exactly as [`super::pipeline::PlannedDispatch`] is for the compute side.
//!
//! Like the rest of [`super`], this is pure integer bookkeeping: no `GPU`
//! handles, no floats, no wall clock. The byte offsets and binding slots mirror
//! `water_surface_raster.wesl` field-for-field, so a mismatch between the host
//! layout and the device struct is caught by the unit tests here rather than by
//! a silent misread on the device. The indexed grid draw mirrors the surface
//! mesh every shipping ocean rasterizes (`UE5` Single Layer Water, `Crest`,
//! `WaveWorks`): a regular `verts_x` by `verts_z` lattice of displaced vertices,
//! two triangles per quad.

/// Byte stride of one surface-vertex storage record: an `array<vec4<f32>>`
/// element (`base_positions`, `surface_uvs`, `displacement`, `normal_foam`),
/// 16 bytes at the `std430` stride. All four per-vertex storage arrays share
/// this stride.
pub const SURFACE_VERTEX_RECORD_STRIDE: u32 = 16;

/// The `@group(0)` binding slots the water-surface raster draw declares, in
/// shader-declaration order. The discriminants are the literal `@binding(n)`
/// indices in `water_surface_raster.wesl`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurfaceBinding {
    /// `@binding(0)`: the per-view uniform block ([`WaterSurfaceView`] layout).
    View,
    /// `@binding(1)`: undisplaced base grid positions (`xyz`; `w = 1`).
    BasePositions,
    /// `@binding(2)`: per-vertex texture coordinates (`xy`).
    SurfaceUvs,
    /// `@binding(3)`: solver displacement per vertex (`xyz` world offset).
    Displacement,
    /// `@binding(4)`: surface normal (`xyz`) and foam coverage (`w`).
    NormalFoam,
    /// `@binding(5)`: already-rendered opaque scene color (refraction source).
    SceneColor,
    /// `@binding(6)`: the sampler the refraction lookup uses.
    SceneSampler,
}

impl SurfaceBinding {
    /// Every binding in shader-declaration order.
    pub const ALL: [SurfaceBinding; 7] = [
        SurfaceBinding::View,
        SurfaceBinding::BasePositions,
        SurfaceBinding::SurfaceUvs,
        SurfaceBinding::Displacement,
        SurfaceBinding::NormalFoam,
        SurfaceBinding::SceneColor,
        SurfaceBinding::SceneSampler,
    ];

    /// The literal `@binding(n)` slot index this binding occupies in `@group(0)`.
    #[must_use]
    pub fn index(self) -> u32 {
        match self {
            SurfaceBinding::View => 0,
            SurfaceBinding::BasePositions => 1,
            SurfaceBinding::SurfaceUvs => 2,
            SurfaceBinding::Displacement => 3,
            SurfaceBinding::NormalFoam => 4,
            SurfaceBinding::SceneColor => 5,
            SurfaceBinding::SceneSampler => 6,
        }
    }

    /// The resource kind the backend must bind at this slot.
    #[must_use]
    pub fn kind(self) -> SurfaceBindingKind {
        match self {
            SurfaceBinding::View => SurfaceBindingKind::Uniform,
            SurfaceBinding::BasePositions
            | SurfaceBinding::SurfaceUvs
            | SurfaceBinding::Displacement
            | SurfaceBinding::NormalFoam => SurfaceBindingKind::StorageRead,
            SurfaceBinding::SceneColor => SurfaceBindingKind::SampledTexture,
            SurfaceBinding::SceneSampler => SurfaceBindingKind::Sampler,
        }
    }

    /// Whether the vertex stage reads this binding.
    ///
    /// The displaced-mesh geometry stage reads the uniform (for the
    /// clip transform) and all four per-vertex storage arrays; the refraction
    /// texture and its sampler are fragment-only.
    #[must_use]
    pub fn visible_in_vertex(self) -> bool {
        matches!(
            self,
            SurfaceBinding::View
                | SurfaceBinding::BasePositions
                | SurfaceBinding::SurfaceUvs
                | SurfaceBinding::Displacement
                | SurfaceBinding::NormalFoam
        )
    }

    /// Whether the fragment stage reads this binding.
    ///
    /// The lighting-response stage reads the uniform (light, water params) and
    /// the scene-color texture/sampler (refraction); the position/uv arrays are
    /// consumed by the vertex stage and arrive interpolated.
    #[must_use]
    pub fn visible_in_fragment(self) -> bool {
        matches!(
            self,
            SurfaceBinding::View
                | SurfaceBinding::NormalFoam
                | SurfaceBinding::SceneColor
                | SurfaceBinding::SceneSampler
        )
    }
}

/// The resource kind of a surface-draw binding, mapped to the backend's
/// bind-group-layout entry types.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurfaceBindingKind {
    /// A `var<uniform>` block.
    Uniform,
    /// A `var<storage, read>` array.
    StorageRead,
    /// A sampled `texture_2d<f32>`.
    SampledTexture,
    /// A filtering `sampler`.
    Sampler,
}

/// Byte layout of the `WaterSurfaceView` uniform block, mirroring the `WESL`
/// `struct` in `water_surface_raster.wesl` field-for-field at the `std140`
/// stride every field already satisfies (a `mat4x4<f32>` followed by five
/// 16-byte-aligned `vec4<f32>` rows).
pub mod view {
    /// Offset of `clip_from_world: mat4x4<f32>`.
    pub const CLIP_FROM_WORLD_OFFSET: u32 = 0;
    /// Offset of `world_camera_position: vec4<f32>` (`xyz`; `w` = refraction
    /// screen offset).
    pub const WORLD_CAMERA_POSITION_OFFSET: u32 = 64;
    /// Offset of `surface_params: vec4<f32>` (roughness, reflectance,
    /// thickness, foam-whiten).
    pub const SURFACE_PARAMS_OFFSET: u32 = 80;
    /// Offset of `water_color: vec4<f32>` (`rgb` albedo, `a` min alpha).
    pub const WATER_COLOR_OFFSET: u32 = 96;
    /// Offset of `style_params: vec4<f32>` (ramp steps, foam threshold, tint,
    /// hybrid bias).
    pub const STYLE_PARAMS_OFFSET: u32 = 112;
    /// Offset of `viewport: vec4<f32>` (`xy` framebuffer pixel size).
    pub const VIEWPORT_OFFSET: u32 = 128;
    /// Total size of the uniform block in bytes (`64 + 5 * 16`), already a
    /// multiple of 16 so it needs no tail padding.
    pub const SIZE: u32 = 144;
}

/// A regular displaced surface patch: a `verts_x` by `verts_z` lattice of
/// vertices, two triangles per interior quad. The solved per-vertex fields
/// (`base_positions`/`surface_uvs`/`displacement`/`normal_foam`) hold one
/// record per lattice vertex.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SurfaceGrid {
    /// Vertex count along the `x` axis.
    pub verts_x: u32,
    /// Vertex count along the `z` axis.
    pub verts_z: u32,
}

impl SurfaceGrid {
    /// Total lattice vertices (`verts_x * verts_z`), the length each per-vertex
    /// storage array must hold. Saturating so a pathological grid can never
    /// wrap the allocation size.
    #[must_use]
    pub fn vertex_count(self) -> u32 {
        self.verts_x.saturating_mul(self.verts_z)
    }

    /// Interior quads (`(verts_x - 1) * (verts_z - 1)`), zero when either axis
    /// has fewer than two vertices (a degenerate patch draws nothing).
    #[must_use]
    pub fn quad_count(self) -> u32 {
        let qx = self.verts_x.saturating_sub(1);
        let qz = self.verts_z.saturating_sub(1);
        qx.saturating_mul(qz)
    }

    /// Index count for the triangle-list draw (`quads * 6`, two triangles per
    /// quad). Saturating.
    #[must_use]
    pub fn index_count(self) -> u32 {
        self.quad_count().saturating_mul(6)
    }

    /// Byte size of one per-vertex storage array
    /// (`vertex_count * SURFACE_VERTEX_RECORD_STRIDE`). Saturating.
    #[must_use]
    pub fn vertex_array_bytes(self) -> u32 {
        self.vertex_count()
            .saturating_mul(SURFACE_VERTEX_RECORD_STRIDE)
    }
}

/// The indexed draw call for one surface patch: the index count the raster
/// node issues and the single instance it draws.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SurfaceDrawCall {
    /// Number of indices in the triangle-list draw.
    pub index_count: u32,
    /// Number of lattice vertices the storage arrays hold.
    pub vertex_count: u32,
    /// Instances drawn (one patch per draw; instancing is a later slice).
    pub instance_count: u32,
}

impl SurfaceDrawCall {
    /// `true` when the patch has no interior quads and so records nothing — the
    /// raster node must skip it rather than issue a zero-index draw.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.index_count == 0
    }
}

/// Plan the indexed draw call for one surface grid.
///
/// Deterministic: identical grid in, identical draw call out, so the draw is
/// stable across frames. A grid thinner than two vertices on either axis yields
/// an empty draw ([`SurfaceDrawCall::is_empty`]).
#[must_use]
pub fn plan_surface_draw_call(grid: SurfaceGrid) -> SurfaceDrawCall {
    SurfaceDrawCall {
        index_count: grid.index_count(),
        vertex_count: grid.vertex_count(),
        instance_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_indices_are_the_shader_declaration_order() {
        for (slot, binding) in SurfaceBinding::ALL.iter().enumerate() {
            assert_eq!(binding.index(), u32::try_from(slot).unwrap());
        }
    }

    #[test]
    fn binding_indices_are_unique_and_contiguous() {
        let mut seen = [false; 7];
        for binding in SurfaceBinding::ALL {
            let i = binding.index() as usize;
            assert!(i < seen.len());
            assert!(!seen[i], "binding slot {i} declared twice");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "binding slots must be contiguous");
    }

    #[test]
    fn binding_kinds_match_the_shader() {
        assert_eq!(SurfaceBinding::View.kind(), SurfaceBindingKind::Uniform);
        assert_eq!(
            SurfaceBinding::BasePositions.kind(),
            SurfaceBindingKind::StorageRead
        );
        assert_eq!(
            SurfaceBinding::SurfaceUvs.kind(),
            SurfaceBindingKind::StorageRead
        );
        assert_eq!(
            SurfaceBinding::Displacement.kind(),
            SurfaceBindingKind::StorageRead
        );
        assert_eq!(
            SurfaceBinding::NormalFoam.kind(),
            SurfaceBindingKind::StorageRead
        );
        assert_eq!(
            SurfaceBinding::SceneColor.kind(),
            SurfaceBindingKind::SampledTexture
        );
        assert_eq!(
            SurfaceBinding::SceneSampler.kind(),
            SurfaceBindingKind::Sampler
        );
    }

    #[test]
    fn vertex_stage_reads_the_uniform_and_all_four_vertex_arrays() {
        for binding in SurfaceBinding::ALL {
            let expected = matches!(
                binding,
                SurfaceBinding::View
                    | SurfaceBinding::BasePositions
                    | SurfaceBinding::SurfaceUvs
                    | SurfaceBinding::Displacement
                    | SurfaceBinding::NormalFoam
            );
            assert_eq!(binding.visible_in_vertex(), expected, "{binding:?}");
        }
    }

    #[test]
    fn fragment_stage_reads_the_uniform_normal_foam_and_refraction() {
        assert!(SurfaceBinding::View.visible_in_fragment());
        assert!(SurfaceBinding::NormalFoam.visible_in_fragment());
        assert!(SurfaceBinding::SceneColor.visible_in_fragment());
        assert!(SurfaceBinding::SceneSampler.visible_in_fragment());
        // The position/uv arrays are consumed in the vertex stage only.
        assert!(!SurfaceBinding::BasePositions.visible_in_fragment());
        assert!(!SurfaceBinding::SurfaceUvs.visible_in_fragment());
        assert!(!SurfaceBinding::Displacement.visible_in_fragment());
    }

    #[test]
    fn every_binding_is_visible_to_at_least_one_stage() {
        for binding in SurfaceBinding::ALL {
            assert!(
                binding.visible_in_vertex() || binding.visible_in_fragment(),
                "{binding:?} is bound but read by no stage"
            );
        }
    }

    #[test]
    fn uniform_offsets_are_monotonic_sixteen_byte_aligned_and_packed() {
        let offsets = [
            view::CLIP_FROM_WORLD_OFFSET,
            view::WORLD_CAMERA_POSITION_OFFSET,
            view::SURFACE_PARAMS_OFFSET,
            view::WATER_COLOR_OFFSET,
            view::STYLE_PARAMS_OFFSET,
            view::VIEWPORT_OFFSET,
        ];
        for pair in offsets.windows(2) {
            assert!(pair[1] > pair[0], "uniform fields must be ordered");
        }
        for offset in offsets {
            assert_eq!(offset % 16, 0, "every field is 16-byte aligned");
        }
        // The mat4x4 spans 64 bytes; every later vec4 row is exactly 16 apart.
        assert_eq!(view::WORLD_CAMERA_POSITION_OFFSET, 64);
        for pair in offsets[1..].windows(2) {
            assert_eq!(pair[1] - pair[0], 16);
        }
        assert_eq!(view::VIEWPORT_OFFSET + 16, view::SIZE);
        assert_eq!(view::SIZE % 16, 0, "no tail padding needed");
    }

    #[test]
    fn grid_draw_call_counts_two_triangles_per_quad() {
        let grid = SurfaceGrid {
            verts_x: 4,
            verts_z: 3,
        };
        assert_eq!(grid.vertex_count(), 12);
        assert_eq!(grid.quad_count(), 3 * 2);
        assert_eq!(grid.index_count(), 3 * 2 * 6);

        let draw = plan_surface_draw_call(grid);
        assert_eq!(draw.vertex_count, 12);
        assert_eq!(draw.index_count, 36);
        assert_eq!(draw.instance_count, 1);
        assert!(!draw.is_empty());
    }

    #[test]
    fn degenerate_grids_draw_nothing() {
        for grid in [
            SurfaceGrid {
                verts_x: 1,
                verts_z: 8,
            },
            SurfaceGrid {
                verts_x: 8,
                verts_z: 1,
            },
            SurfaceGrid {
                verts_x: 0,
                verts_z: 0,
            },
            SurfaceGrid {
                verts_x: 1,
                verts_z: 1,
            },
        ] {
            let draw = plan_surface_draw_call(grid);
            assert_eq!(draw.index_count, 0, "{grid:?}");
            assert!(draw.is_empty(), "{grid:?}");
        }
    }

    #[test]
    fn draw_call_is_deterministic() {
        let grid = SurfaceGrid {
            verts_x: 129,
            verts_z: 129,
        };
        assert_eq!(plan_surface_draw_call(grid), plan_surface_draw_call(grid));
    }

    #[test]
    fn pathological_grid_saturates_without_wrapping() {
        let grid = SurfaceGrid {
            verts_x: u32::MAX,
            verts_z: u32::MAX,
        };
        // Saturating arithmetic: counts clamp at u32::MAX rather than wrapping
        // to a small value that would under-allocate the device buffers.
        assert_eq!(grid.vertex_count(), u32::MAX);
        assert_eq!(grid.index_count(), u32::MAX);
        assert_eq!(grid.vertex_array_bytes(), u32::MAX);
    }

    #[test]
    fn vertex_array_bytes_tracks_the_record_stride() {
        let grid = SurfaceGrid {
            verts_x: 8,
            verts_z: 8,
        };
        assert_eq!(
            grid.vertex_array_bytes(),
            grid.vertex_count() * SURFACE_VERTEX_RECORD_STRIDE
        );
        assert_eq!(SURFACE_VERTEX_RECORD_STRIDE, 16);
    }
}
