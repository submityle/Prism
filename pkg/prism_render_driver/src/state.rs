//! Fixed-function pipeline state: primitive assembly, depth/stencil, and
//! multisampling.

use crate::compare::CompareFunction;
use crate::format::TextureFormat;

/// How vertices are assembled into primitives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PrimitiveTopology {
    /// Each vertex is a point.
    PointList,
    /// Each pair of vertices is a line.
    LineList,
    /// Connected line segments.
    LineStrip,
    /// Each triple of vertices is a triangle.
    #[default]
    TriangleList,
    /// Connected triangles sharing edges.
    TriangleStrip,
}

impl PrimitiveTopology {
    /// Whether the topology forms strips, which require an index-format hint
    /// for primitive-restart handling.
    #[must_use]
    pub const fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }
}

/// Which winding order is considered front-facing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum FrontFace {
    /// Counter-clockwise winding is the front face.
    #[default]
    Ccw,
    /// Clockwise winding is the front face.
    Cw,
}

/// Which face, if any, is culled.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Face {
    /// Cull front-facing triangles.
    Front,
    /// Cull back-facing triangles.
    Back,
}

/// How polygons are rasterized.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PolygonMode {
    /// Fill the interior (the normal mode).
    #[default]
    Fill,
    /// Draw only edges.
    Line,
    /// Draw only vertices.
    Point,
}

/// The index width for indexed draws.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum IndexFormat {
    /// 16-bit indices.
    Uint16,
    /// 32-bit indices.
    #[default]
    Uint32,
}

impl IndexFormat {
    /// The byte size of one index.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Primitive assembly and rasterization state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PrimitiveState {
    /// How vertices form primitives.
    pub topology: PrimitiveTopology,
    /// The index format for strip restart, required for strip topologies.
    pub strip_index_format: Option<IndexFormat>,
    /// The front-facing winding order.
    pub front_face: FrontFace,
    /// Which face to cull, if any.
    pub cull_mode: Option<Face>,
    /// How polygons are filled.
    pub polygon_mode: PolygonMode,
    /// Whether fragments beyond the depth range are clamped instead of clipped.
    pub unclipped_depth: bool,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
        }
    }
}

impl PrimitiveState {
    /// Whether this state is internally consistent: strip topologies must
    /// declare a `strip_index_format`, and list topologies must not.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        if self.topology.is_strip() {
            self.strip_index_format.is_some()
        } else {
            self.strip_index_format.is_none()
        }
    }
}

/// The operation applied to the stored stencil value for one test outcome.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StencilOperation {
    /// Keep the current value.
    #[default]
    Keep,
    /// Set the value to zero.
    Zero,
    /// Replace with the reference value.
    Replace,
    /// Bitwise invert the value.
    Invert,
    /// Increment, clamping at the maximum.
    IncrementClamp,
    /// Decrement, clamping at zero.
    DecrementClamp,
    /// Increment, wrapping to zero on overflow.
    IncrementWrap,
    /// Decrement, wrapping to the maximum on underflow.
    DecrementWrap,
}

/// The stencil test and operations for one face.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StencilFaceState {
    /// The comparison between reference and stored stencil values.
    pub compare: CompareFunction,
    /// What to do when the stencil test fails.
    pub fail_op: StencilOperation,
    /// What to do when the stencil test passes but the depth test fails.
    pub depth_fail_op: StencilOperation,
    /// What to do when both tests pass.
    pub pass_op: StencilOperation,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

impl StencilFaceState {
    /// A face state that never touches the stencil buffer.
    pub const IGNORE: Self = Self {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

/// Full stencil state across both faces plus read/write masks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct StencilState {
    /// The stencil test for front-facing primitives.
    pub front: StencilFaceState,
    /// The stencil test for back-facing primitives.
    pub back: StencilFaceState,
    /// Which bits are read during the stencil test.
    pub read_mask: u32,
    /// Which bits may be written by stencil operations.
    pub write_mask: u32,
}

impl StencilState {
    /// Whether stencil testing is effectively enabled (either face does
    /// something other than always-keep).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.front != StencilFaceState::IGNORE || self.back != StencilFaceState::IGNORE
    }
}

/// Depth-bias (polygon offset) parameters.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DepthBiasState {
    /// A constant depth offset in device units.
    pub constant: i32,
    /// A slope-scaled depth offset.
    pub slope_scale: f32,
    /// The maximum (or minimum) bias clamp.
    pub clamp: f32,
}

/// Depth/stencil attachment test and write state.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthStencilState {
    /// The depth/stencil attachment format.
    pub format: TextureFormat,
    /// Whether depth writes are enabled.
    pub depth_write_enabled: bool,
    /// The depth comparison predicate.
    pub depth_compare: CompareFunction,
    /// The stencil state.
    pub stencil: StencilState,
    /// Depth-bias parameters.
    pub bias: DepthBiasState,
}

impl DepthStencilState {
    /// Standard reverse-Z-friendly depth state: writes enabled, `GreaterEqual`
    /// comparison. Pass the attachment `format`.
    #[must_use]
    pub fn reverse_z(format: TextureFormat) -> Self {
        Self {
            format,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }
    }

    /// Read-only depth test with the given comparison (no depth writes).
    #[must_use]
    pub fn read_only(format: TextureFormat, depth_compare: CompareFunction) -> Self {
        Self {
            format,
            depth_write_enabled: false,
            depth_compare,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }
    }
}

/// Multisample (MSAA) resolve state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MultisampleState {
    /// The sample count (1 means no multisampling).
    pub count: u32,
    /// The sample coverage mask.
    pub mask: u64,
    /// Whether alpha-to-coverage is enabled.
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

impl MultisampleState {
    /// A multisample state at `count` samples with a full coverage mask.
    #[must_use]
    pub fn with_count(count: u32) -> Self {
        Self {
            count,
            ..Self::default()
        }
    }
}
