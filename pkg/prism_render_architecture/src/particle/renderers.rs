//! Particle renderer matrix, phase routing, and per-renderer geometry contracts
//! (design §15, with sort routing per §12 and shading per §16).
//!
//! One emitter may drive several renderers, each issuing an indirect draw into a
//! render phase (`Transparent3d` / `AlphaMask3d` / `Opaque3d`). Every renderer
//! can pair with any shading model (design §16); `Light` is the sole exception,
//! contributing to the clustered light list rather than emitting shaded
//! geometry. This module owns the renderer-kind taxonomy, the phase/sort routing
//! contract, and the deterministic `CPU` reference logic for the geometry a
//! `GPU` build pass would generate (`Sprite` billboards and flipbooks, `Ribbon`
//! strip segmentation, `Beam` chains, and `Light` count budgeting).
//!
//! The routing mirrors production `VFX` stacks (Unreal `Niagara` renderers and
//! Unity `VFX Graph` outputs) at the algorithm level: additive/premultiplied
//! blends are order-independent and never sort, straight alpha blends route
//! through the scene's shared `OIT` path (or a standalone view-depth sort as a
//! fallback), and opaque/alpha-masked draws are depth-tested without a sort.
//! Only `sqrt` (through [`Vec3`]) and ordinary arithmetic are used, so this
//! `CPU` reference stays bit-reproducible against a future `GPU` kernel.

use alloc::vec::Vec;

use super::sort_cull::{self, SortDecision};
use super::{EmberShadingModel, SortStrategy, Vec3, EPS_LEN_SQ};

/// The render primitive an emitter's renderer emits (design §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RendererKind {
    /// Camera/velocity/axis-aligned billboards; flipbook; soft particles;
    /// `SDF` shapes.
    Sprite,
    /// One mesh instance per particle (instanced, full shading closures).
    Mesh,
    /// `RibbonId`-linked strips generated on the `GPU`.
    Ribbon,
    /// Two/multi-point beams (lightning, lasers).
    Beam,
    /// Particle-driven lights routed into the clustered light list.
    Light,
    /// Ground decals routed into the deferred/forward decal path.
    Decal,
    /// Grid-fluid density volume ray-marched (design §10, §20).
    Volume,
}

/// The render phase a draw is routed into (design §12, §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderPhase {
    /// Opaque geometry with depth write.
    Opaque3d,
    /// Alpha-tested cutout geometry.
    AlphaMask3d,
    /// Order-dependent translucency.
    Transparent3d,
}

/// How a shading model responds to light *and* whether it operates on a surface
/// or a participating medium (design §15, §16, §20).
///
/// This expresses the per-renderer shading constraint from the design matrix:
/// surface renderers (`Sprite` / `Mesh` / `Ribbon` / `Beam` / `Decal`) shade a
/// surfel, `Volume` runs the volumetric closure (six-way lighting, phase
/// function), and `Light` shades nothing because it emits no geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadingDomain {
    /// Surface shading (a surfel with normal/roughness where the model needs it).
    Surface,
    /// Volumetric shading (ray-marched density, six-way lighting; design §20).
    Volumetric,
    /// No shading: the renderer emits no shaded geometry (`Light`).
    None,
}

/// How a renderer's fragments composite into the frame, which drives both the
/// [`RenderPhase`] and whether a depth sort is required (design §12, §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParticleBlend {
    /// Opaque: depth-tested, order-independent, depth-writing.
    Opaque,
    /// Alpha-tested cutout: depth-tested, order-independent.
    AlphaMask,
    /// Additive: commutative, order-independent.
    Additive,
    /// Premultiplied alpha over an additive-safe pipeline: order-independent.
    Premultiplied,
    /// Straight alpha blending: order-dependent, needs `OIT` or a depth sort.
    AlphaBlend,
}

impl ParticleBlend {
    /// The [`RenderPhase`] this blend routes into.
    ///
    /// Opaque draws go to [`RenderPhase::Opaque3d`], alpha-masked cutouts to
    /// [`RenderPhase::AlphaMask3d`], and every translucent blend (additive,
    /// premultiplied, straight alpha) to [`RenderPhase::Transparent3d`].
    #[must_use]
    pub const fn phase(self) -> RenderPhase {
        match self {
            ParticleBlend::Opaque => RenderPhase::Opaque3d,
            ParticleBlend::AlphaMask => RenderPhase::AlphaMask3d,
            ParticleBlend::Additive | ParticleBlend::Premultiplied | ParticleBlend::AlphaBlend => {
                RenderPhase::Transparent3d
            }
        }
    }

    /// Whether correct compositing requires an explicit depth sort (or shared
    /// `OIT`): only straight alpha blending is order-dependent (design §12).
    #[must_use]
    pub const fn needs_sort(self) -> bool {
        matches!(self, ParticleBlend::AlphaBlend)
    }

    /// Maps to the sort-cull blend axis so the §12 strategy matrix is reused
    /// rather than duplicated. Alpha-masked cutouts sort like opaque geometry
    /// (they do not sort at all).
    #[must_use]
    const fn sort_blend(self) -> sort_cull::BlendMode {
        match self {
            ParticleBlend::Opaque | ParticleBlend::AlphaMask => sort_cull::BlendMode::Opaque,
            ParticleBlend::Additive => sort_cull::BlendMode::Additive,
            ParticleBlend::Premultiplied => sort_cull::BlendMode::Premultiplied,
            ParticleBlend::AlphaBlend => sort_cull::BlendMode::AlphaBlend,
        }
    }
}

impl RendererKind {
    /// Whether this renderer draws geometry at all. `Light` contributes to the
    /// clustered light list rather than emitting a draw, so it has no phase.
    #[must_use]
    pub const fn draws_geometry(self) -> bool {
        !matches!(self, RendererKind::Light)
    }

    /// The shading domain this renderer operates in (design §15, §16, §20).
    ///
    /// Surface renderers shade a surfel, `Volume` runs the volumetric closure,
    /// and `Light` shades nothing.
    #[must_use]
    pub const fn shading_domain(self) -> ShadingDomain {
        match self {
            RendererKind::Sprite
            | RendererKind::Mesh
            | RendererKind::Ribbon
            | RendererKind::Beam
            | RendererKind::Decal => ShadingDomain::Surface,
            RendererKind::Volume => ShadingDomain::Volumetric,
            RendererKind::Light => ShadingDomain::None,
        }
    }
}

/// Whether a renderer can shade with a given model (design §15, §16).
///
/// All four shading responses are equal citizens, so every geometry-emitting
/// renderer is compatible with every [`EmberShadingModel`] (the `_shading`
/// argument is accepted to make that invariant explicit at call sites and to
/// leave room for future per-model restrictions). `Light` emits no shaded
/// geometry, so it is compatible with no shading model.
#[must_use]
pub const fn is_compatible(kind: RendererKind, _shading: EmberShadingModel) -> bool {
    kind.draws_geometry()
}

/// The resolved routing for one renderer draw: its render phase and its sort
/// strategy (design §12, §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RenderRouting {
    /// The phase the indirect draw is submitted into.
    pub phase: RenderPhase,
    /// The sort strategy applied before compositing.
    pub sort: SortStrategy,
}

/// Resolves the [`RenderPhase`] for a `(kind, blend)` pair (design §15).
///
/// Returns `None` for `Light`, which emits no geometry and therefore has no
/// phase; every other renderer routes purely by its [`ParticleBlend`].
#[must_use]
pub const fn resolve_phase(kind: RendererKind, blend: ParticleBlend) -> Option<RenderPhase> {
    if kind.draws_geometry() {
        Some(blend.phase())
    } else {
        None
    }
}

/// Inputs to the full phase + sort routing decision (design §12, §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutingInput {
    /// The renderer emitting the draw.
    pub kind: RendererKind,
    /// The emitter's shading model (four equal citizens; design §16).
    pub shading: EmberShadingModel,
    /// How the renderer's fragments composite.
    pub blend: ParticleBlend,
    /// Live particle count after culling (drives the standalone sort choice).
    pub visible_count: u32,
    /// At or above this count a standalone sort uses radix; below it, bitonic.
    pub radix_min_count: u32,
    /// Prefer the scene's shared `OIT` path over a standalone per-emitter sort.
    pub prefer_shared_oit: bool,
}

/// Resolves the render phase and sort strategy for a renderer draw
/// (design §12, §15).
///
/// Returns `None` for `Light` (no geometry). For every other renderer the phase
/// comes from the [`ParticleBlend`] and the sort strategy is delegated to the
/// shared §12 matrix in [`sort_cull::choose_sort_strategy`]: additive,
/// premultiplied, opaque, and alpha-masked draws never sort; straight alpha
/// blends route through [`SortStrategy::SharedOit`] when the caller prefers the
/// shared path, otherwise a standalone [`SortStrategy::ViewDepthRadix`] (large
/// counts) or [`SortStrategy::ViewDepthBitonic`] (small counts).
///
/// The `shading` model does not change routing (every model shares the geometry
/// and sorting services; design §16), but an incompatible pairing — a shading
/// model on a `Light` renderer — is rejected by returning `None`.
#[must_use]
pub fn resolve_routing(input: RoutingInput) -> Option<RenderRouting> {
    if !is_compatible(input.kind, input.shading) {
        return None;
    }
    let phase = input.blend.phase();
    let sort = sort_cull::choose_sort_strategy(SortDecision {
        blend: input.blend.sort_blend(),
        particle_count: input.visible_count,
        radix_min_count: input.radix_min_count,
        prefer_shared_oit: input.prefer_shared_oit,
    });
    Some(RenderRouting { phase, sort })
}

// ---------------------------------------------------------------------------
// Sprite / billboard alignment and flipbook (design §15).
// ---------------------------------------------------------------------------

/// How a `Sprite` renderer orients its quad (design §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BillboardAlign {
    /// Fully camera-facing: the quad normal points at the camera.
    CameraFacing,
    /// The quad's local up follows the particle velocity, facing the camera
    /// around that axis (spark streaks, speed lines).
    VelocityAligned,
    /// The quad's local up is locked to a fixed world axis (grass, beams).
    FixedAxis,
}

/// An orthonormal billboard basis in world space: `right`, `up`, and the
/// `forward` normal the quad faces along.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BillboardBasis {
    /// Quad tangent (local +X).
    pub right: Vec3,
    /// Quad bitangent (local +Y).
    pub up: Vec3,
    /// Quad normal (faces the camera).
    pub forward: Vec3,
}

/// The camera frame a billboard is built against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFrame {
    /// Unit-ish vector from the particle toward the camera.
    pub to_camera: Vec3,
    /// The camera's world up, used to complete a camera-facing basis.
    pub up: Vec3,
}

/// Returns any unit vector perpendicular to `v` (deterministic), or `+X` when
/// `v` is (numerically) zero. Used as a fallback when a cross product collapses.
#[must_use]
fn any_perpendicular(v: Vec3) -> Vec3 {
    // Cross with the world axis least aligned with `v` to avoid a near-zero
    // cross product; compare squared components with no transcendentals.
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    let reference = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let perp = v.cross(reference).normalize_or_zero();
    if perp.length_squared() > EPS_LEN_SQ {
        perp
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// Builds a camera-facing orthonormal basis whose normal points at the camera.
fn camera_facing_basis(cam: CameraFrame) -> BillboardBasis {
    let mut forward = cam.to_camera.normalize_or_zero();
    if forward.length_squared() <= EPS_LEN_SQ {
        forward = Vec3::new(0.0, 0.0, 1.0);
    }
    let mut right = cam.up.cross(forward).normalize_or_zero();
    if right.length_squared() <= EPS_LEN_SQ {
        right = any_perpendicular(forward);
    }
    let up = forward.cross(right);
    BillboardBasis { right, up, forward }
}

/// Builds a basis whose local up is locked to `axis` while still facing the
/// camera around that axis (shared by velocity-aligned and fixed-axis sprites).
fn axis_locked_basis(axis: Vec3, cam: CameraFrame, fallback_axis: Vec3) -> BillboardBasis {
    let mut up = axis.normalize_or_zero();
    if up.length_squared() <= EPS_LEN_SQ {
        up = fallback_axis.normalize_or_zero();
        if up.length_squared() <= EPS_LEN_SQ {
            up = Vec3::new(0.0, 1.0, 0.0);
        }
    }
    let mut view = cam.to_camera.normalize_or_zero();
    if view.length_squared() <= EPS_LEN_SQ {
        view = Vec3::new(0.0, 0.0, 1.0);
    }
    let mut right = up.cross(view).normalize_or_zero();
    if right.length_squared() <= EPS_LEN_SQ {
        right = any_perpendicular(up);
    }
    let forward = right.cross(up);
    BillboardBasis { right, up, forward }
}

/// Builds the orthonormal billboard basis for a sprite (design §15).
///
/// `fixed_axis` is only consulted for [`BillboardAlign::FixedAxis`] and
/// `velocity` only for [`BillboardAlign::VelocityAligned`]. Every path is
/// degenerate-safe: a zero view vector falls back to `+Z`, a zero velocity or
/// fixed axis falls back to the camera up, and a collapsed cross product falls
/// back to a deterministic perpendicular, so the result is always orthonormal
/// and never `NaN`.
#[must_use]
pub fn billboard_basis(
    align: BillboardAlign,
    fixed_axis: Vec3,
    velocity: Vec3,
    cam: CameraFrame,
) -> BillboardBasis {
    match align {
        BillboardAlign::CameraFacing => camera_facing_basis(cam),
        BillboardAlign::VelocityAligned => axis_locked_basis(velocity, cam, cam.up),
        BillboardAlign::FixedAxis => axis_locked_basis(fixed_axis, cam, cam.up),
    }
}

/// How a flipbook animation behaves once the last frame is reached.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FlipbookWrap {
    /// Hold the final frame.
    Clamp,
    /// Wrap back to the first frame.
    Loop,
}

/// A `Sprite` flipbook: a `frames`-cell sub-image sheet advanced at `fps`
/// (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flipbook {
    /// Number of frames in the sheet. Zero is treated as a single static cell.
    pub frames: u32,
    /// Playback rate in frames per second (clamped to be non-negative).
    pub fps: f32,
    /// Behaviour past the last frame.
    pub wrap: FlipbookWrap,
}

impl Flipbook {
    /// Frame index for a particle of age `age_seconds` (design §15).
    ///
    /// Negative ages clamp to `0`. With `frames == 0` the index is always `0`.
    /// [`FlipbookWrap::Clamp`] holds the last frame; [`FlipbookWrap::Loop`]
    /// wraps modulo `frames`. The result is always in `0..frames.max(1)`.
    #[must_use]
    pub fn frame_for_age(self, age_seconds: f32) -> u32 {
        if self.frames == 0 {
            return 0;
        }
        let age = age_seconds.max(0.0);
        let fps = self.fps.max(0.0);
        // `age >= 0` and `fps >= 0`, so the cast floors toward zero safely.
        let raw = (age * fps) as u32;
        self.wrap_index(raw)
    }

    /// Frame index for a normalized life fraction `life` in `0..=1`.
    ///
    /// `life` is clamped to `0..=1`; `0` maps to the first frame and values at
    /// or above `1` map to the last frame (clamp) or the first frame (loop).
    #[must_use]
    pub fn frame_for_normalized(self, life: f32) -> u32 {
        if self.frames == 0 {
            return 0;
        }
        let t = life.clamp(0.0, 1.0);
        let raw = (t * self.frames as f32) as u32;
        self.wrap_index(raw)
    }

    /// Applies the wrap mode to a raw (unbounded) frame counter.
    #[must_use]
    fn wrap_index(self, raw: u32) -> u32 {
        match self.wrap {
            FlipbookWrap::Clamp => raw.min(self.frames - 1),
            FlipbookWrap::Loop => raw % self.frames,
        }
    }
}

/// A `Sprite` renderer's `CPU`-verifiable configuration (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteConfig {
    /// Quad orientation policy.
    pub align: BillboardAlign,
    /// The locked axis for [`BillboardAlign::FixedAxis`] (ignored otherwise).
    pub fixed_axis: Vec3,
    /// Optional flipbook animation.
    pub flipbook: Option<Flipbook>,
    /// Whether the sprite fades against scene depth (needs the depth prepass).
    pub soft_particles: bool,
}

impl SpriteConfig {
    /// Whether this sprite requires the shared depth prepass (soft particles
    /// depth-fade against scene geometry; design §12).
    #[must_use]
    pub const fn needs_depth_prepass(self) -> bool {
        self.soft_particles
    }
}

// ---------------------------------------------------------------------------
// Size-over-life LUT sampling (design §15 ribbon width, §6.2 LUTs).
// ---------------------------------------------------------------------------

/// Nearest-neighbour index into a size-over-life `LUT` for a normalized life
/// fraction `life` (design §6.2, §15).
///
/// `life` is clamped to `0..=1` and mapped across `0..lut_len`; an empty `LUT`
/// (`lut_len == 0`) yields `0`. The index is always in `0..lut_len.max(1)`.
#[must_use]
pub fn size_lut_index(life: f32, lut_len: u32) -> u32 {
    if lut_len == 0 {
        return 0;
    }
    let t = life.clamp(0.0, 1.0);
    let scaled = t * (lut_len - 1) as f32;
    ((scaled + 0.5) as u32).min(lut_len - 1)
}

/// Samples a size-over-life `LUT` at normalized life `life` (design §15).
///
/// An empty `LUT` returns the neutral width `1.0`; otherwise the nearest cell
/// selected by [`size_lut_index`] is returned.
#[must_use]
pub fn sample_size_over_life(lut: &[f32], life: f32) -> f32 {
    if lut.is_empty() {
        return 1.0;
    }
    // `lut.len()` cast is bounded by the slice length; the index is in range.
    lut[size_lut_index(life, lut.len() as u32) as usize]
}

// ---------------------------------------------------------------------------
// Ribbon / trail segmentation (design §15).
// ---------------------------------------------------------------------------

/// The `GPU` tessellation quality of a ribbon span (design §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessellationLod {
    /// Highest quality: most subdivisions per span.
    High,
    /// Balanced quality.
    Medium,
    /// Lowest quality: a single quad per span (distant/`LOD`-reduced ribbons).
    Low,
}

impl TessellationLod {
    /// Number of quad subdivisions the geometry pass emits per ribbon span.
    #[must_use]
    pub const fn subdivisions(self) -> u32 {
        match self {
            TessellationLod::High => 4,
            TessellationLod::Medium => 2,
            TessellationLod::Low => 1,
        }
    }

    /// Picks a tessellation tier from a screen-coverage fraction in `0..=1`:
    /// large ribbons tessellate finely, distant/small ones collapse to `Low`.
    #[must_use]
    pub fn for_coverage(coverage: f32) -> Self {
        let c = coverage.clamp(0.0, 1.0);
        if c >= 0.25 {
            TessellationLod::High
        } else if c >= 0.05 {
            TessellationLod::Medium
        } else {
            TessellationLod::Low
        }
    }
}

/// One sample feeding ribbon segmentation: a point on some ribbon at some age
/// (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonSample {
    /// The ribbon this sample belongs to; samples sharing an id form one strip.
    pub ribbon_id: u32,
    /// The sample's age in seconds; ordering and break detection use it.
    pub age: f32,
    /// Normalized life fraction in `0..=1`, used for size-over-life width.
    pub life: f32,
    /// World-space position of the sample.
    pub position: Vec3,
}

/// One vertex of the segmented ribbon topology (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonVertex {
    /// Index of the originating [`RibbonSample`] in the input slice.
    pub source_index: u32,
    /// The ribbon id this vertex belongs to.
    pub ribbon_id: u32,
    /// Previous vertex along the strip (index into the topology), if any.
    pub prev: Option<u32>,
    /// Next vertex along the strip (index into the topology), if any.
    pub next: Option<u32>,
    /// Cumulative arc length from the chain head, for `UV` parameterization.
    pub arc_length: f32,
    /// Strip width sampled from the size-over-life `LUT` at this vertex.
    pub width: f32,
}

/// A contiguous ribbon strip: a run of vertices with no age break (design §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RibbonChain {
    /// Index of the head vertex in [`RibbonTopology::vertices`].
    pub head: u32,
    /// Number of vertices in this chain (always `>= 1`).
    pub count: u32,
}

/// The segmented ribbon topology produced by [`segment_ribbons`] (design §15).
///
/// Vertices are grouped into chains; within a chain the `prev`/`next` links form
/// a doubly linked strip and `arc_length` grows monotonically from the head.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RibbonTopology {
    /// All strip vertices, grouped chain by chain in order.
    pub vertices: Vec<RibbonVertex>,
    /// The chains (contiguous strips) discovered during segmentation.
    pub chains: Vec<RibbonChain>,
}

/// Segments ribbon samples into contiguous strips (design §15, build pass).
///
/// Samples are grouped by `ribbon_id` and ordered by `age` (ties broken by
/// input index for determinism). Consecutive samples whose age gap exceeds
/// `break_age_gap` start a new chain (break detection). Each vertex records its
/// `prev`/`next` neighbours, its cumulative `arc_length` from the chain head
/// (for `UV` parameterization), and its `width` sampled from `size_lut` at the
/// sample's normalized life.
///
/// The routine is deterministic and safe for empty input (empty topology),
/// single-point ribbons (a one-vertex chain with no neighbours), and any
/// ordering of the input.
#[must_use]
pub fn segment_ribbons(
    samples: &[RibbonSample],
    break_age_gap: f32,
    size_lut: &[f32],
) -> RibbonTopology {
    let mut topology = RibbonTopology::default();
    if samples.is_empty() {
        return topology;
    }

    // Deterministic total order: by ribbon id, then age (total order over the
    // float bit pattern), then original index to break exact ties.
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by(|&a, &b| {
        let sa = &samples[a];
        let sb = &samples[b];
        sa.ribbon_id
            .cmp(&sb.ribbon_id)
            .then_with(|| sa.age.total_cmp(&sb.age))
            .then_with(|| a.cmp(&b))
    });

    let mut chain_head: usize = 0;
    let mut chain_len: u32 = 0;

    for pos in 0..order.len() {
        let src = order[pos];
        let sample = &samples[src];

        // Decide whether this sample extends the current chain or starts a new
        // one: a new ribbon id or an age gap over the threshold breaks the strip.
        let starts_new_chain = if chain_len == 0 {
            true
        } else {
            let prev_sample = &samples[order[pos - 1]];
            let same_ribbon = prev_sample.ribbon_id == sample.ribbon_id;
            let age_gap = sample.age - prev_sample.age;
            !same_ribbon || age_gap > break_age_gap
        };

        if starts_new_chain && chain_len > 0 {
            topology.chains.push(RibbonChain {
                head: chain_head as u32,
                count: chain_len,
            });
            chain_len = 0;
        }

        let vertex_index = topology.vertices.len() as u32;
        if starts_new_chain {
            chain_head = topology.vertices.len();
        }

        let (prev, arc_length) = if starts_new_chain {
            (None, 0.0)
        } else {
            let prev_index = vertex_index - 1;
            let prev_vertex = &topology.vertices[prev_index as usize];
            let prev_sample = &samples[prev_vertex.source_index as usize];
            let span = sample.position.distance(prev_sample.position);
            (Some(prev_index), prev_vertex.arc_length + span)
        };

        // Link the previous vertex forward to this one within the same chain.
        if let Some(prev_index) = prev {
            topology.vertices[prev_index as usize].next = Some(vertex_index);
        }

        topology.vertices.push(RibbonVertex {
            source_index: src as u32,
            ribbon_id: sample.ribbon_id,
            prev,
            next: None,
            arc_length,
            width: sample_size_over_life(size_lut, sample.life),
        });
        chain_len += 1;
    }

    if chain_len > 0 {
        topology.chains.push(RibbonChain {
            head: chain_head as u32,
            count: chain_len,
        });
    }

    topology
}

// ---------------------------------------------------------------------------
// Beam topology and light budgeting (design §15).
// ---------------------------------------------------------------------------

/// One segment of a beam: an index pair into the control-point slice and its
/// world-space length (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeamSegment {
    /// Index of the segment's start control point.
    pub start: u32,
    /// Index of the segment's end control point.
    pub end: u32,
    /// World-space length of the segment.
    pub length: f32,
}

/// Builds the segment chain for a beam from its control points (design §15).
///
/// A two-point beam yields one segment; an `n`-point chain yields `n - 1`
/// segments (lightning/laser arcs). Fewer than two points yields no segments.
#[must_use]
pub fn beam_segments(points: &[Vec3]) -> Vec<BeamSegment> {
    let mut segments = Vec::new();
    if points.len() < 2 {
        return segments;
    }
    for i in 0..points.len() - 1 {
        segments.push(BeamSegment {
            start: i as u32,
            end: (i + 1) as u32,
            length: points[i].distance(points[i + 1]),
        });
    }
    segments
}

/// Resolves how many particle-driven lights are actually submitted (design §15).
///
/// The requested count is first divided by the `LOD` divisor (clamped to at
/// least `1`, so a divisor of `0` behaves like `1`), then clamped to the
/// clustered-light budget `max_lights`. This is the "count clamp + `LOD`"
/// behaviour of the `Light` renderer: over-budget requests are truncated rather
/// than overflowing the clustered light list.
#[must_use]
pub fn resolve_light_count(requested: u32, max_lights: u32, lod_divisor: u32) -> u32 {
    let divisor = lod_divisor.max(1);
    (requested / divisor).min(max_lights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::ShadingBasis;

    // --- taxonomy -------------------------------------------------------

    #[test]
    fn light_renderer_draws_no_geometry() {
        assert!(!RendererKind::Light.draws_geometry());
        assert!(RendererKind::Sprite.draws_geometry());
        assert!(RendererKind::Volume.draws_geometry());
    }

    #[test]
    fn renderer_kinds_are_distinct() {
        assert_ne!(RendererKind::Sprite, RendererKind::Mesh);
        assert_ne!(RendererKind::Ribbon, RendererKind::Beam);
    }

    #[test]
    fn shading_domain_per_renderer() {
        assert_eq!(
            RendererKind::Sprite.shading_domain(),
            ShadingDomain::Surface
        );
        assert_eq!(RendererKind::Mesh.shading_domain(), ShadingDomain::Surface);
        assert_eq!(RendererKind::Decal.shading_domain(), ShadingDomain::Surface);
        assert_eq!(
            RendererKind::Volume.shading_domain(),
            ShadingDomain::Volumetric
        );
        assert_eq!(RendererKind::Light.shading_domain(), ShadingDomain::None);
    }

    // --- compatibility --------------------------------------------------

    #[test]
    fn every_geometry_renderer_pairs_with_every_shading_model() {
        let renderers = [
            RendererKind::Sprite,
            RendererKind::Mesh,
            RendererKind::Ribbon,
            RendererKind::Beam,
            RendererKind::Decal,
            RendererKind::Volume,
        ];
        let models = [
            EmberShadingModel::Unlit,
            EmberShadingModel::Pbr,
            EmberShadingModel::Npr,
            EmberShadingModel::Custom(7),
            EmberShadingModel::Hybrid {
                base: ShadingBasis::Pbr,
                overlay: ShadingBasis::Npr,
                weight: 0.5,
            },
        ];
        for kind in renderers {
            for model in models {
                assert!(is_compatible(kind, model), "{kind:?} + {model:?}");
            }
        }
    }

    #[test]
    fn light_is_incompatible_with_all_shading_models() {
        assert!(!is_compatible(
            RendererKind::Light,
            EmberShadingModel::Unlit
        ));
        assert!(!is_compatible(RendererKind::Light, EmberShadingModel::Pbr));
        assert!(!is_compatible(
            RendererKind::Light,
            EmberShadingModel::Custom(1)
        ));
    }

    // --- phase routing --------------------------------------------------

    #[test]
    fn blend_maps_to_expected_phase() {
        assert_eq!(ParticleBlend::Opaque.phase(), RenderPhase::Opaque3d);
        assert_eq!(ParticleBlend::AlphaMask.phase(), RenderPhase::AlphaMask3d);
        assert_eq!(ParticleBlend::Additive.phase(), RenderPhase::Transparent3d);
        assert_eq!(
            ParticleBlend::Premultiplied.phase(),
            RenderPhase::Transparent3d
        );
        assert_eq!(
            ParticleBlend::AlphaBlend.phase(),
            RenderPhase::Transparent3d
        );
    }

    #[test]
    fn resolve_phase_is_none_for_light() {
        assert_eq!(
            resolve_phase(RendererKind::Light, ParticleBlend::Additive),
            None
        );
        assert_eq!(
            resolve_phase(RendererKind::Sprite, ParticleBlend::Opaque),
            Some(RenderPhase::Opaque3d)
        );
    }

    fn routing(kind: RendererKind, blend: ParticleBlend, prefer_shared_oit: bool) -> RoutingInput {
        RoutingInput {
            kind,
            shading: EmberShadingModel::Pbr,
            blend,
            visible_count: 4096,
            radix_min_count: 2048,
            prefer_shared_oit,
        }
    }

    #[test]
    fn additive_never_sorts() {
        let r = resolve_routing(routing(RendererKind::Sprite, ParticleBlend::Additive, true))
            .expect("sprite routes");
        assert_eq!(r.phase, RenderPhase::Transparent3d);
        assert_eq!(r.sort, SortStrategy::None);
    }

    #[test]
    fn opaque_and_mask_never_sort() {
        let opaque = resolve_routing(routing(RendererKind::Mesh, ParticleBlend::Opaque, false))
            .expect("mesh routes");
        assert_eq!(opaque.phase, RenderPhase::Opaque3d);
        assert_eq!(opaque.sort, SortStrategy::None);

        let mask = resolve_routing(routing(
            RendererKind::Sprite,
            ParticleBlend::AlphaMask,
            true,
        ))
        .expect("sprite routes");
        assert_eq!(mask.phase, RenderPhase::AlphaMask3d);
        assert_eq!(mask.sort, SortStrategy::None);
    }

    #[test]
    fn alpha_blend_prefers_shared_oit() {
        let r = resolve_routing(routing(
            RendererKind::Sprite,
            ParticleBlend::AlphaBlend,
            true,
        ))
        .expect("sprite routes");
        assert_eq!(r.phase, RenderPhase::Transparent3d);
        assert_eq!(r.sort, SortStrategy::SharedOit);
    }

    #[test]
    fn alpha_blend_falls_back_to_standalone_sort() {
        let big = resolve_routing(routing(
            RendererKind::Sprite,
            ParticleBlend::AlphaBlend,
            false,
        ))
        .expect("sprite routes");
        assert_eq!(big.sort, SortStrategy::ViewDepthRadix);

        let small = resolve_routing(RoutingInput {
            visible_count: 64,
            ..routing(RendererKind::Sprite, ParticleBlend::AlphaBlend, false)
        })
        .expect("sprite routes");
        assert_eq!(small.sort, SortStrategy::ViewDepthBitonic);
    }

    #[test]
    fn light_renderer_has_no_routing() {
        assert_eq!(
            resolve_routing(routing(RendererKind::Light, ParticleBlend::Additive, true)),
            None
        );
    }

    // --- billboard basis ------------------------------------------------

    fn is_orthonormal(basis: BillboardBasis) -> bool {
        let eps = 1e-4;
        let unit = |v: Vec3| (v.length() - 1.0).abs() < eps;
        let ortho = |a: Vec3, b: Vec3| a.dot(b).abs() < eps;
        unit(basis.right)
            && unit(basis.up)
            && unit(basis.forward)
            && ortho(basis.right, basis.up)
            && ortho(basis.right, basis.forward)
            && ortho(basis.up, basis.forward)
    }

    #[test]
    fn camera_facing_basis_is_orthonormal_and_faces_camera() {
        let cam = CameraFrame {
            to_camera: Vec3::new(0.0, 0.0, 1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        };
        let basis = billboard_basis(BillboardAlign::CameraFacing, Vec3::ZERO, Vec3::ZERO, cam);
        assert!(is_orthonormal(basis));
        assert!(basis.forward.dot(cam.to_camera) > 0.0);
    }

    #[test]
    fn velocity_aligned_basis_follows_velocity() {
        let cam = CameraFrame {
            to_camera: Vec3::new(0.0, 0.0, 1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        };
        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let basis = billboard_basis(BillboardAlign::VelocityAligned, Vec3::ZERO, velocity, cam);
        assert!(is_orthonormal(basis));
        // Local up is parallel to the (normalized) velocity direction.
        assert!(basis.up.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.999);
    }

    #[test]
    fn fixed_axis_basis_locks_up_axis() {
        let cam = CameraFrame {
            to_camera: Vec3::new(0.0, 0.0, 1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        };
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let basis = billboard_basis(BillboardAlign::FixedAxis, axis, Vec3::ZERO, cam);
        assert!(is_orthonormal(basis));
        assert!(basis.up.dot(Vec3::new(0.0, 0.0, 1.0)).abs() > 0.999);
    }

    #[test]
    fn billboard_basis_is_safe_for_degenerate_inputs() {
        let cam = CameraFrame {
            to_camera: Vec3::ZERO,
            up: Vec3::ZERO,
        };
        for align in [
            BillboardAlign::CameraFacing,
            BillboardAlign::VelocityAligned,
            BillboardAlign::FixedAxis,
        ] {
            let basis = billboard_basis(align, Vec3::ZERO, Vec3::ZERO, cam);
            assert!(is_orthonormal(basis), "{align:?} produced a bad basis");
        }
    }

    // --- flipbook -------------------------------------------------------

    #[test]
    fn flipbook_clamp_holds_last_frame() {
        let fb = Flipbook {
            frames: 4,
            fps: 10.0,
            wrap: FlipbookWrap::Clamp,
        };
        assert_eq!(fb.frame_for_age(0.0), 0);
        assert_eq!(fb.frame_for_age(0.25), 2);
        assert_eq!(fb.frame_for_age(100.0), 3);
        // Negative age clamps to the first frame.
        assert_eq!(fb.frame_for_age(-5.0), 0);
    }

    #[test]
    fn flipbook_loop_wraps() {
        let fb = Flipbook {
            frames: 4,
            fps: 10.0,
            wrap: FlipbookWrap::Loop,
        };
        // 0.5s * 10fps = frame 5 -> 5 % 4 = 1.
        assert_eq!(fb.frame_for_age(0.5), 1);
        assert_eq!(fb.frame_for_age(0.8), 0);
    }

    #[test]
    fn flipbook_zero_frames_is_safe() {
        let fb = Flipbook {
            frames: 0,
            fps: 30.0,
            wrap: FlipbookWrap::Loop,
        };
        assert_eq!(fb.frame_for_age(3.0), 0);
        assert_eq!(fb.frame_for_normalized(0.9), 0);
    }

    #[test]
    fn flipbook_normalized_bounds() {
        let fb = Flipbook {
            frames: 8,
            fps: 24.0,
            wrap: FlipbookWrap::Clamp,
        };
        assert_eq!(fb.frame_for_normalized(0.0), 0);
        assert_eq!(fb.frame_for_normalized(1.0), 7);
        assert_eq!(fb.frame_for_normalized(2.0), 7);

        let looped = Flipbook {
            wrap: FlipbookWrap::Loop,
            ..fb
        };
        // t == 1.0 -> raw 8 -> 8 % 8 = 0.
        assert_eq!(looped.frame_for_normalized(1.0), 0);
    }

    #[test]
    fn flipbook_is_deterministic() {
        let fb = Flipbook {
            frames: 6,
            fps: 12.0,
            wrap: FlipbookWrap::Loop,
        };
        assert_eq!(fb.frame_for_age(0.37), fb.frame_for_age(0.37));
    }

    // --- size LUT -------------------------------------------------------

    #[test]
    fn size_lut_index_clamps_and_maps() {
        assert_eq!(size_lut_index(0.0, 4), 0);
        assert_eq!(size_lut_index(1.0, 4), 3);
        assert_eq!(size_lut_index(-1.0, 4), 0);
        assert_eq!(size_lut_index(5.0, 4), 3);
        assert_eq!(size_lut_index(0.5, 1), 0);
        assert_eq!(size_lut_index(0.5, 0), 0);
    }

    #[test]
    fn sample_size_over_life_handles_empty_lut() {
        assert_eq!(sample_size_over_life(&[], 0.5), 1.0);
        let lut = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(sample_size_over_life(&lut, 0.0), 1.0);
        assert_eq!(sample_size_over_life(&lut, 1.0), 4.0);
    }

    // --- ribbon segmentation --------------------------------------------

    fn sample(id: u32, age: f32, x: f32) -> RibbonSample {
        RibbonSample {
            ribbon_id: id,
            age,
            life: 0.0,
            position: Vec3::new(x, 0.0, 0.0),
        }
    }

    #[test]
    fn empty_ribbon_input_is_empty_topology() {
        let topo = segment_ribbons(&[], 1.0, &[]);
        assert!(topo.vertices.is_empty());
        assert!(topo.chains.is_empty());
    }

    #[test]
    fn single_point_ribbon_is_one_vertex_chain() {
        let samples = alloc::vec![sample(0, 0.0, 0.0)];
        let topo = segment_ribbons(&samples, 1.0, &[]);
        assert_eq!(topo.vertices.len(), 1);
        assert_eq!(topo.chains.len(), 1);
        assert_eq!(topo.chains[0].count, 1);
        assert_eq!(topo.vertices[0].prev, None);
        assert_eq!(topo.vertices[0].next, None);
        assert_eq!(topo.vertices[0].arc_length, 0.0);
    }

    #[test]
    fn contiguous_ribbon_links_and_arc_length() {
        let samples = alloc::vec![
            sample(0, 0.0, 0.0),
            sample(0, 0.1, 1.0),
            sample(0, 0.2, 3.0),
        ];
        let topo = segment_ribbons(&samples, 1.0, &[]);
        assert_eq!(topo.chains.len(), 1);
        assert_eq!(topo.chains[0].count, 3);
        assert_eq!(topo.vertices[0].prev, None);
        assert_eq!(topo.vertices[0].next, Some(1));
        assert_eq!(topo.vertices[1].prev, Some(0));
        assert_eq!(topo.vertices[1].next, Some(2));
        assert_eq!(topo.vertices[2].next, None);
        // Arc length accumulates distances along the strip.
        assert_eq!(topo.vertices[0].arc_length, 0.0);
        assert_eq!(topo.vertices[1].arc_length, 1.0);
        assert_eq!(topo.vertices[2].arc_length, 3.0);
    }

    #[test]
    fn age_gap_breaks_ribbon_into_two_chains() {
        let samples = alloc::vec![
            sample(0, 0.0, 0.0),
            sample(0, 0.1, 1.0),
            // Large age gap -> break.
            sample(0, 5.0, 2.0),
            sample(0, 5.1, 3.0),
        ];
        let topo = segment_ribbons(&samples, 1.0, &[]);
        assert_eq!(topo.chains.len(), 2);
        assert_eq!(topo.chains[0].count, 2);
        assert_eq!(topo.chains[1].count, 2);
        // The break resets links and arc length.
        let head_of_second = topo.chains[1].head as usize;
        assert_eq!(topo.vertices[head_of_second].prev, None);
        assert_eq!(topo.vertices[head_of_second].arc_length, 0.0);
    }

    #[test]
    fn distinct_ribbon_ids_form_distinct_chains() {
        let samples = alloc::vec![
            sample(1, 0.0, 0.0),
            sample(2, 0.05, 1.0),
            sample(1, 0.1, 2.0),
        ];
        let topo = segment_ribbons(&samples, 1.0, &[]);
        // Ribbon 1 (two samples) then ribbon 2 (one sample).
        assert_eq!(topo.chains.len(), 2);
        assert_eq!(topo.vertices[0].ribbon_id, 1);
        assert_eq!(topo.chains[0].count, 2);
        assert_eq!(topo.chains[1].count, 1);
    }

    #[test]
    fn ribbon_width_samples_size_lut() {
        let lut = [2.0_f32, 4.0];
        let samples = alloc::vec![
            RibbonSample {
                ribbon_id: 0,
                age: 0.0,
                life: 0.0,
                position: Vec3::ZERO,
            },
            RibbonSample {
                ribbon_id: 0,
                age: 0.1,
                life: 1.0,
                position: Vec3::new(1.0, 0.0, 0.0),
            },
        ];
        let topo = segment_ribbons(&samples, 1.0, &lut);
        assert_eq!(topo.vertices[0].width, 2.0);
        assert_eq!(topo.vertices[1].width, 4.0);
    }

    #[test]
    fn ribbon_segmentation_is_deterministic_and_order_independent() {
        let ordered = alloc::vec![
            sample(0, 0.0, 0.0),
            sample(0, 0.1, 1.0),
            sample(0, 0.2, 2.0),
        ];
        let shuffled = alloc::vec![
            sample(0, 0.2, 2.0),
            sample(0, 0.0, 0.0),
            sample(0, 0.1, 1.0),
        ];
        let a = segment_ribbons(&ordered, 1.0, &[]);
        let b = segment_ribbons(&shuffled, 1.0, &[]);
        // Same chain structure and arc lengths regardless of input order.
        assert_eq!(a.chains, b.chains);
        assert_eq!(a.vertices.len(), b.vertices.len());
        for (va, vb) in a.vertices.iter().zip(b.vertices.iter()) {
            assert_eq!(va.arc_length, vb.arc_length);
            assert_eq!(va.prev, vb.prev);
            assert_eq!(va.next, vb.next);
        }
    }

    // --- tessellation ---------------------------------------------------

    #[test]
    fn tessellation_subdivisions_and_coverage() {
        assert_eq!(TessellationLod::High.subdivisions(), 4);
        assert_eq!(TessellationLod::Medium.subdivisions(), 2);
        assert_eq!(TessellationLod::Low.subdivisions(), 1);
        assert_eq!(TessellationLod::for_coverage(0.5), TessellationLod::High);
        assert_eq!(TessellationLod::for_coverage(0.1), TessellationLod::Medium);
        assert_eq!(TessellationLod::for_coverage(0.0), TessellationLod::Low);
        assert_eq!(TessellationLod::for_coverage(2.0), TessellationLod::High);
    }

    // --- beam -----------------------------------------------------------

    #[test]
    fn beam_two_point_is_single_segment() {
        let points = alloc::vec![Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0)];
        let segs = beam_segments(&points);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 1);
        assert_eq!(segs[0].length, 5.0);
    }

    #[test]
    fn beam_multi_point_chain() {
        let points = alloc::vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let segs = beam_segments(&points);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].start, 1);
        assert_eq!(segs[1].end, 2);
    }

    #[test]
    fn beam_degenerate_inputs_are_empty() {
        assert!(beam_segments(&[]).is_empty());
        assert!(beam_segments(&[Vec3::ZERO]).is_empty());
    }

    // --- light budgeting ------------------------------------------------

    #[test]
    fn light_count_clamps_to_budget() {
        assert_eq!(resolve_light_count(100, 16, 1), 16);
        assert_eq!(resolve_light_count(8, 16, 1), 8);
    }

    #[test]
    fn light_count_applies_lod_divisor() {
        assert_eq!(resolve_light_count(100, 64, 4), 25);
        // A zero divisor behaves like 1 (no division by zero).
        assert_eq!(resolve_light_count(10, 64, 0), 10);
    }

    #[test]
    fn light_count_lod_then_clamp() {
        // 100 / 2 = 50, then clamped to the budget of 16.
        assert_eq!(resolve_light_count(100, 16, 2), 16);
    }

    // --- sprite config --------------------------------------------------

    #[test]
    fn soft_particles_need_depth_prepass() {
        let soft = SpriteConfig {
            align: BillboardAlign::CameraFacing,
            fixed_axis: Vec3::new(0.0, 1.0, 0.0),
            flipbook: None,
            soft_particles: true,
        };
        assert!(soft.needs_depth_prepass());
        let hard = SpriteConfig {
            soft_particles: false,
            ..soft
        };
        assert!(!hard.needs_depth_prepass());
    }
}
