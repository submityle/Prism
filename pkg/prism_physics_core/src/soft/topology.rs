//! Deterministic topology-level tearing for deformable triangle meshes.
//!
//! The legacy damage pass removes stretch constraints. [`TearableMesh`] goes
//! one step further: a broken edge becomes a seam, incident triangle fans are
//! split into disconnected sectors, and duplicated vertices create a real
//! renderable crack boundary. The implementation keeps a stable root identity
//! for every duplicated vertex, so seam state remains valid after subsequent
//! front propagation.
//!
//! The mesh owns the state needed by a CPU simulation bridge (positions,
//! velocities and inverse masses). Every mutation is deterministic: candidates
//! are ordered by root edge, then by triangle/vertex index, and all per-frame
//! work is bounded by the configured front and vertex limits. The mesh also
//! exposes immutable render and collision snapshots; consumers can rebuild
//! their local GPU buffers and acceleration structures from one versioned
//! snapshot without reimplementing the tear decision.

mod index;
pub mod stress;

pub use stress::{MembraneMaterial, PrincipalMembraneStress};

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

/// An undirected edge identified by its two stable root vertex ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyEdge {
    /// Lower endpoint id.
    pub a: u32,
    /// Higher endpoint id.
    pub b: u32,
}

impl TopologyEdge {
    /// Creates a canonical undirected edge. Equal endpoints are accepted but
    /// are ignored by mesh construction and seam propagation.
    #[must_use]
    pub const fn new(a: u32, b: u32) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

/// A topology-level tearing event for effects, audio and render bridges.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyTearEvent {
    /// Event category.
    pub kind: TopologyTearEventKind,
    /// World-space event position.
    pub position: Vec3,
    /// Stable seam edge for an edge break, or the selected triangle normal for
    /// a puncture.
    pub normal: Vec3,
    /// Root edge involved in an edge break, when applicable.
    pub edge: Option<TopologyEdge>,
    /// Newly allocated vertex id for a vertex split, when applicable.
    pub vertex: Option<u32>,
    /// Revision before this topology edit.
    pub previous_topology_version: u64,
    /// Revision after this topology edit.
    pub topology_version: u64,
    /// Physical vertex count after this edit.
    pub vertex_count: u32,
    /// Triangle count after this edit.
    pub triangle_count: u32,
}

/// Kinds emitted by [`TearableMesh`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TopologyTearEventKind {
    /// A structural edge crossed its damage threshold.
    EdgeBreak,
    /// A triangle fan was split and a duplicate vertex was allocated.
    VertexSplit,
    /// A point query selected a triangle and opened its strongest edge.
    Puncture,
}

/// A physical edge in the post-split index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyBoundaryEdge {
    /// Current physical vertex slots.
    pub vertices: [u32; 2],
    /// Stable pre-split identity of this edge.
    pub root: TopologyEdge,
    /// Whether this edge belongs to an opened seam.
    pub seam: bool,
}

/// A render vertex emitted by TearableMesh::render_snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyRenderVertex {
    /// Current world-space position.
    pub position: Vec3,
    /// Area-weighted normalized triangle normal.
    pub normal: Vec3,
    /// Stable source/root vertex id for UV/material lookup.
    pub root: u32,
}

/// Versioned render bridge data for a tearable mesh.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyRenderSnapshot {
    /// Revision represented by every array in this snapshot.
    pub topology_version: u64,
    /// Current vertex stream.
    pub vertices: Vec<TopologyRenderVertex>,
    /// Current triangle index buffer.
    pub triangles: Vec<[u32; 3]>,
    /// All current boundary edges.
    pub boundary_edges: Vec<TopologyBoundaryEdge>,
    /// Boundary edges whose root identity is an opened seam.
    pub seam_edges: Vec<TopologyBoundaryEdge>,
}

/// Axis-aligned bounds for one post-tear triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyTriangleBounds {
    /// Triangle index in TopologyCollisionUpdate::triangles.
    pub triangle: u32,
    /// Minimum corner of the current bounds.
    pub min: Vec3,
    /// Maximum corner of the current bounds.
    pub max: Vec3,
}

/// Versioned collision/CCD refresh data emitted after a topology edit.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyCollisionUpdate {
    /// Revision represented by all fields.
    pub topology_version: u64,
    /// Current triangle index buffer.
    pub triangles: Vec<[u32; 3]>,
    /// Complete replacement set for triangle broad-phase proxies.
    pub triangle_bounds: Vec<TopologyTriangleBounds>,
    /// Current physical boundary and seam edges.
    pub boundary_edges: Vec<TopologyBoundaryEdge>,
    /// Same-triangle vertex pairs excluded from self-collision.
    pub self_collision_exclusions: Vec<[u32; 2]>,
    /// Cache invalidation token for continuous collision detection.
    pub ccd_revision: u64,
}

/// The complete replacement payload produced by one topology edit.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyTearUpdate {
    /// The revision represented by every field in this update.
    pub topology_version: u64,
    /// Events generated by the edit, all stamped with the same revision.
    pub events: Vec<TopologyTearEvent>,
    /// Complete render replacement data.
    pub render: TopologyRenderSnapshot,
    /// Complete collision and CCD replacement data.
    pub collision: TopologyCollisionUpdate,
}

/// Why a versioned topology payload could not be consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyUpdateError {
    /// Render, collision and CCD payloads do not describe the same revision.
    RevisionMismatch,
    /// The consumer received a duplicate or older payload.
    StaleRevision,
    /// The consumer received a revision with a missing predecessor.
    RevisionGap,
    /// The replacement arrays are not mutually consistent.
    InconsistentPayload,
}

/// Version-aware atomic consumer for render, collision and CCD topology data.
///
/// A consumer owns the last accepted complete replacement. Applying an update
/// validates every array before mutating any field, so a failed or stale update
/// cannot leave render and collision state at different revisions.
#[derive(Clone, Debug, PartialEq)]
pub struct TopologyUpdateConsumer {
    topology_version: u64,
    render: TopologyRenderSnapshot,
    collision: TopologyCollisionUpdate,
    last_events: Vec<TopologyTearEvent>,
    previous_positions: Vec<Vec3>,
    swept_triangle_bounds: Vec<TopologyTriangleBounds>,
}

impl TopologyUpdateConsumer {
    /// Creates a consumer from an authoritative current mesh without emitting
    /// a synthetic event or requiring an update at revision zero.
    #[must_use]
    pub fn from_mesh(mesh: &TearableMesh) -> Self {
        Self {
            topology_version: mesh.topology_version,
            render: mesh.render_snapshot(),
            collision: mesh.collision_update(),
            last_events: Vec::new(),
            previous_positions: mesh.positions.clone(),
            swept_triangle_bounds: mesh.collision_update().triangle_bounds,
        }
    }

    /// Creates a consumer from the first complete replacement payload.
    pub fn new(update: TopologyTearUpdate) -> Result<Self, TopologyUpdateError> {
        validate_update(&update, None)?;
        let previous_positions = update.render.vertices.iter().map(|v| v.position).collect();
        let swept_triangle_bounds = update.collision.triangle_bounds.clone();
        Ok(Self {
            topology_version: update.topology_version,
            render: update.render,
            collision: update.collision,
            last_events: update.events,
            previous_positions,
            swept_triangle_bounds,
        })
    }

    /// Applies one complete next revision atomically.
    pub fn apply(&mut self, update: TopologyTearUpdate) -> Result<(), TopologyUpdateError> {
        validate_update(&update, Some(self.topology_version))?;
        // Tear updates append duplicates. Existing physical identities must
        // not silently be rebound to other particles by a producer.
        if update.render.vertices.len() < self.render.vertices.len()
            || self
                .render
                .vertices
                .iter()
                .zip(&update.render.vertices)
                .any(|(old, new)| old.root != new.root)
        {
            return Err(TopologyUpdateError::InconsistentPayload);
        }
        let roots: BTreeMap<_, _> = self
            .render
            .vertices
            .iter()
            .enumerate()
            .rev()
            .map(|(index, vertex)| (vertex.root, index))
            .collect();
        let previous_positions: Vec<_> = update
            .render
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| {
                self.render
                    .vertices
                    .get(index)
                    .or_else(|| {
                        roots
                            .get(&vertex.root)
                            .map(|&old| &self.render.vertices[old])
                    })
                    .map_or(vertex.position, |previous| previous.position)
            })
            .collect();
        let swept_triangle_bounds = update
            .collision
            .triangle_bounds
            .iter()
            .zip(&update.render.triangles)
            .map(|(bounds, triangle)| {
                let mut swept = *bounds;
                for &vertex in triangle {
                    let previous = previous_positions[vertex as usize];
                    swept.min = swept.min.min(previous);
                    swept.max = swept.max.max(previous);
                }
                swept
            })
            .collect();
        self.topology_version = update.topology_version;
        self.render = update.render;
        self.collision = update.collision;
        self.last_events = update.events;
        self.previous_positions = previous_positions;
        self.swept_triangle_bounds = swept_triangle_bounds;
        Ok(())
    }

    /// Returns the last accepted topology revision.
    #[must_use]
    pub const fn topology_version(&self) -> u64 {
        self.topology_version
    }

    /// Returns the current render replacement owned by this consumer.
    #[must_use]
    pub const fn render(&self) -> &TopologyRenderSnapshot {
        &self.render
    }

    /// Returns the current collision/CCD replacement owned by this consumer.
    #[must_use]
    pub const fn collision(&self) -> &TopologyCollisionUpdate {
        &self.collision
    }

    /// Returns events from the most recently accepted update.
    #[must_use]
    pub fn last_events(&self) -> &[TopologyTearEvent] {
        &self.last_events
    }

    /// CCD start positions remapped onto the current post-split vertex stream.
    /// New duplicates inherit the lowest previous physical slot of their root.
    #[must_use]
    pub fn previous_positions(&self) -> &[Vec3] {
        &self.previous_positions
    }

    /// Conservative bounds covering both accepted endpoints of each triangle.
    #[must_use]
    pub fn swept_triangle_bounds(&self) -> &[TopologyTriangleBounds] {
        &self.swept_triangle_bounds
    }
}

fn validate_update(
    update: &TopologyTearUpdate,
    current: Option<u64>,
) -> Result<(), TopologyUpdateError> {
    let version = update.topology_version;
    if update.render.topology_version != version
        || update.collision.topology_version != version
        || update.collision.ccd_revision != version
    {
        return Err(TopologyUpdateError::RevisionMismatch);
    }
    if let Some(current) = current {
        if version <= current {
            return Err(TopologyUpdateError::StaleRevision);
        }
        if version != current.saturating_add(1) {
            return Err(TopologyUpdateError::RevisionGap);
        }
    }
    if update.events.iter().any(|event| {
        event.topology_version != version
            || event.previous_topology_version != version.saturating_sub(1)
            || event.vertex_count == 0
            || event.triangle_count == 0
            || event.vertex_count as usize != update.render.vertices.len()
            || event.triangle_count as usize != update.render.triangles.len()
            || !event.position.is_finite()
            || !event.normal.is_finite()
            || event
                .vertex
                .is_some_and(|v| v as usize >= update.render.vertices.len())
    }) {
        return Err(TopologyUpdateError::InconsistentPayload);
    }
    let render = &update.render;
    if render.vertices.is_empty()
        || render.triangles.is_empty()
        || render.vertices.iter().enumerate().any(|(index, vertex)| {
            !vertex.position.is_finite()
                || !vertex.normal.is_finite()
                || vertex.root as usize > index
                || render.vertices[vertex.root as usize].root != vertex.root
        })
        || render.triangles != update.collision.triangles
        || render.boundary_edges != update.collision.boundary_edges
        || render.seam_edges
            != render
                .boundary_edges
                .iter()
                .copied()
                .filter(|edge| edge.seam)
                .collect::<Vec<_>>()
        || render
            .boundary_edges
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || render.boundary_edges.iter().any(|edge| {
            let [a, b] = edge.vertices;
            a == b
                || a as usize >= render.vertices.len()
                || b as usize >= render.vertices.len()
                || edge.root
                    != TopologyEdge::new(
                        render.vertices[a as usize].root,
                        render.vertices[b as usize].root,
                    )
        })
        || render.triangles.iter().any(|triangle| {
            triangle
                .iter()
                .any(|&vertex| vertex as usize >= render.vertices.len())
                || triangle[0] == triangle[1]
                || triangle[1] == triangle[2]
                || triangle[0] == triangle[2]
        })
    {
        return Err(TopologyUpdateError::InconsistentPayload);
    }
    let collision = &update.collision;
    if collision.triangle_bounds.len() != collision.triangles.len()
        || collision
            .triangle_bounds
            .iter()
            .enumerate()
            .any(|(index, bounds)| {
                bounds.triangle != index as u32
                    || !bounds.min.is_finite()
                    || !bounds.max.is_finite()
                    || !bounds.min.cmple(bounds.max).all()
                    || {
                        let [a, b, c] = collision.triangles[index]
                            .map(|vertex| render.vertices[vertex as usize].position);
                        bounds.min != a.min(b).min(c) || bounds.max != a.max(b).max(c)
                    }
            })
        || collision
            .self_collision_exclusions
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || collision
            .self_collision_exclusions
            .iter()
            .any(|[a, b]| a >= b)
    {
        return Err(TopologyUpdateError::InconsistentPayload);
    }
    let expected_exclusions: BTreeSet<[u32; 2]> = render
        .triangles
        .iter()
        .flat_map(|&[a, b, c]| {
            [
                [a.min(b), a.max(b)],
                [b.min(c), b.max(c)],
                [a.min(c), a.max(c)],
            ]
        })
        .collect();
    if collision.self_collision_exclusions != expected_exclusions.into_iter().collect::<Vec<_>>() {
        return Err(TopologyUpdateError::InconsistentPayload);
    }
    Ok(())
}

/// Configuration for bounded, deterministic topology tearing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyTearingConfig {
    /// Maximum principal tensile stress required to initiate a crack.
    pub break_stress: Real,
    /// Orthotropic membrane constitutive law.
    pub material: MembraneMaterial,
    /// Propagation threshold / initiation threshold, clamped to [0, 1].
    pub propagation_ratio: Real,
    /// Maximum root edges inspected for spontaneous initiation per update.
    /// A rotating cursor eventually visits the whole mesh without a frame spike.
    pub max_initiation_edges: usize,
    /// Maximum number of root edges opened in one update.
    pub max_front_steps: usize,
    /// Maximum number of physical vertices allowed after splitting.
    pub max_split_vertices: usize,
    /// Non-negative directional threshold gain. A value of `0` is isotropic;
    /// positive values make principal tension along this material axis tougher.
    pub anisotropy_gain: Real,
}

impl Default for TopologyTearingConfig {
    fn default() -> Self {
        Self {
            break_stress: 0.5,
            material: MembraneMaterial::default(),
            propagation_ratio: 0.8,
            max_initiation_edges: 256,
            max_front_steps: 8,
            max_split_vertices: 1_000_000,
            anisotropy_gain: 0.0,
        }
    }
}

impl TopologyTearingConfig {
    /// Returns a finite, safe configuration. Invalid thresholds disable
    /// automatic tearing rather than making every edge fail.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            break_stress: if self.break_stress.is_finite() && self.break_stress >= 0.0 {
                self.break_stress
            } else {
                Real::INFINITY
            },
            material: self.material,
            propagation_ratio: if self.propagation_ratio.is_finite() {
                self.propagation_ratio.clamp(0.0, 1.0)
            } else {
                1.0
            },
            max_initiation_edges: self.max_initiation_edges,
            max_front_steps: self.max_front_steps,
            max_split_vertices: self.max_split_vertices.max(1),
            anisotropy_gain: if self.anisotropy_gain.is_finite() {
                self.anisotropy_gain.max(0.0)
            } else {
                0.0
            },
        }
    }
}

/// Errors returned while constructing a tearable mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyMeshError {
    /// Position and inverse-mass arrays have different lengths.
    AttributeLengthMismatch,
    /// A triangle references a vertex outside the supplied arrays.
    TriangleOutOfRange,
    /// A triangle contains a repeated vertex.
    DegenerateTriangle,
    /// No triangles were supplied.
    EmptyTopology,
    /// A position, inverse mass, or triangle coordinate is not finite.
    NonFiniteAttribute,
    /// An inverse mass is negative and therefore cannot represent a particle.
    NegativeInverseMass,
}

/// A triangle mesh with stable root identities and real seam topology.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TearableMesh {
    /// Rest positions, including duplicated vertices.
    pub rest_positions: Vec<Vec3>,
    /// Current positions, including duplicated vertices.
    pub positions: Vec<Vec3>,
    /// Current velocities, including duplicated vertices.
    pub velocities: Vec<Vec3>,
    /// Inverse masses, aligned with [`positions`](Self::positions).
    pub inverse_masses: Vec<Real>,
    /// Triangle index buffer after all splits.
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Stable root id for each physical vertex slot.
    pub(crate) vertex_roots: Vec<u32>,
    /// Root edges that have become open seams.
    pub(crate) seams: BTreeSet<TopologyEdge>,
    /// Monotonically increasing topology revision.
    pub topology_version: u64,
    vertex_triangles: Vec<Vec<usize>>,
    root_edges: Vec<index::RootEdge>,
    root_neighbors: Vec<Vec<usize>>,
    front_roots: BTreeSet<u32>,
    seam_degrees: Vec<usize>,
    initiation_cursor: usize,
}

impl TearableMesh {
    /// Builds a tearable mesh from rest/current positions and triangle indices.
    #[must_use]
    pub fn new(
        rest_positions: Vec<Vec3>,
        positions: Vec<Vec3>,
        inverse_masses: Vec<Real>,
        triangles: Vec<[u32; 3]>,
    ) -> Result<Self, TopologyMeshError> {
        if rest_positions.len() != positions.len() || positions.len() != inverse_masses.len() {
            return Err(TopologyMeshError::AttributeLengthMismatch);
        }
        if triangles.is_empty() {
            return Err(TopologyMeshError::EmptyTopology);
        }
        if rest_positions.iter().any(|p| !p.is_finite())
            || positions.iter().any(|p| !p.is_finite())
            || inverse_masses.iter().any(|&w| !w.is_finite())
        {
            return Err(TopologyMeshError::NonFiniteAttribute);
        }
        if inverse_masses.iter().any(|&w| w < 0.0) {
            return Err(TopologyMeshError::NegativeInverseMass);
        }
        let vertex_count = positions.len() as u32;
        for [a, b, c] in &triangles {
            if *a >= vertex_count || *b >= vertex_count || *c >= vertex_count {
                return Err(TopologyMeshError::TriangleOutOfRange);
            }
            if a == b || b == c || a == c {
                return Err(TopologyMeshError::DegenerateTriangle);
            }
        }
        let velocities = vec![Vec3::ZERO; positions.len()];
        let vertex_roots = (0..vertex_count).collect();
        let mut mesh = Self {
            rest_positions,
            positions,
            velocities,
            inverse_masses,
            triangles,
            vertex_roots,
            seams: BTreeSet::new(),
            topology_version: 0,
            vertex_triangles: Vec::new(),
            root_edges: Vec::new(),
            root_neighbors: Vec::new(),
            front_roots: BTreeSet::new(),
            seam_degrees: vec![0; vertex_count as usize],
            initiation_cursor: 0,
        };
        mesh.build_index();
        Ok(mesh)
    }

    /// Returns the number of physical vertex slots.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns the number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Returns the number of currently open root seams.
    #[must_use]
    pub fn seam_count(&self) -> usize {
        self.seams.len()
    }

    /// Read-only triangle stream. Topology mutation is confined to the tear
    /// operations so its local adjacency index cannot become stale.
    #[must_use]
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Stable pre-split identity of each physical vertex.
    #[must_use]
    pub fn vertex_roots(&self) -> &[u32] {
        &self.vertex_roots
    }

    /// Computes area-weighted membrane stress from only this vertex's one-ring.
    #[must_use]
    pub fn vertex_stress(
        &self,
        vertex: u32,
        material: MembraneMaterial,
    ) -> Option<PrincipalMembraneStress> {
        let incident = self.vertex_triangles.get(vertex as usize)?;
        stress::principal_membrane_stress(incident.iter().filter_map(|&index| {
            let triangle = self.triangles[index];
            stress::triangle_membrane_stress(
                triangle.map(|v| self.rest_positions[v as usize]),
                triangle.map(|v| self.positions[v as usize]),
                material,
            )
        }))
    }

    /// Returns the root edge for a physical edge.
    #[must_use]
    pub fn root_edge(&self, a: u32, b: u32) -> TopologyEdge {
        TopologyEdge::new(self.vertex_roots[a as usize], self.vertex_roots[b as usize])
    }

    /// Returns active physical edges in deterministic triangle traversal order.
    #[must_use]
    pub fn active_edges(&self) -> Vec<(u32, u32)> {
        let mut edges = BTreeSet::new();
        for [a, b, c] in &self.triangles {
            for (u, v) in [(*a, *b), (*b, *c), (*c, *a)] {
                if !self.seams.contains(&self.root_edge(u, v)) {
                    edges.insert(TopologyEdge::new(u, v));
                }
            }
        }
        edges.into_iter().map(|e| (e.a, e.b)).collect()
    }

    /// Returns all current physical boundary edges in deterministic order.
    #[must_use]
    pub fn boundary_edges(&self) -> Vec<TopologyBoundaryEdge> {
        let mut occurrences: BTreeMap<TopologyEdge, Vec<[u32; 2]>> = BTreeMap::new();
        for [a, b, c] in &self.triangles {
            for pair in [[*a, *b], [*b, *c], [*c, *a]] {
                let root = self.root_edge(pair[0], pair[1]);
                occurrences.entry(root).or_default().push(pair);
            }
        }
        let mut result = Vec::new();
        for (root, mut physical_edges) in occurrences {
            physical_edges.sort_unstable();
            physical_edges.dedup();
            let seam = self.seams.contains(&root);
            if seam || physical_edges.len() == 1 {
                for vertices in physical_edges {
                    result.push(TopologyBoundaryEdge {
                        vertices,
                        root,
                        seam,
                    });
                }
            }
        }
        result.sort_unstable();
        result
    }

    /// Builds the complete render-side replacement stream for this revision.
    #[must_use]
    pub fn render_snapshot(&self) -> TopologyRenderSnapshot {
        let mut normals = vec![Vec3::ZERO; self.vertex_count()];
        for [a, b, c] in &self.triangles {
            let pa = self.positions[*a as usize];
            let pb = self.positions[*b as usize];
            let pc = self.positions[*c as usize];
            let face = (pb - pa).cross(pc - pa);
            if face.length_squared() > 1.0e-20 {
                normals[*a as usize] += face;
                normals[*b as usize] += face;
                normals[*c as usize] += face;
            }
        }
        let vertices = self
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| TopologyRenderVertex {
                position: *position,
                normal: normals[index].try_normalize().unwrap_or(Vec3::Y),
                root: self.vertex_roots[index],
            })
            .collect();
        let boundary_edges = self.boundary_edges();
        let seam_edges = boundary_edges
            .iter()
            .copied()
            .filter(|edge| edge.seam)
            .collect();
        TopologyRenderSnapshot {
            topology_version: self.topology_version,
            vertices,
            triangles: self.triangles.clone(),
            boundary_edges,
            seam_edges,
        }
    }

    /// Builds the complete collision/CCD replacement set for this revision.
    #[must_use]
    pub fn collision_update(&self) -> TopologyCollisionUpdate {
        let triangle_bounds = self
            .triangles
            .iter()
            .enumerate()
            .map(|(triangle, [a, b, c])| {
                let pa = self.positions[*a as usize];
                let pb = self.positions[*b as usize];
                let pc = self.positions[*c as usize];
                TopologyTriangleBounds {
                    triangle: triangle as u32,
                    min: pa.min(pb).min(pc),
                    max: pa.max(pb).max(pc),
                }
            })
            .collect();
        let mut exclusions = BTreeSet::new();
        for [a, b, c] in &self.triangles {
            for pair in [[*a, *b], [*b, *c], [*a, *c]] {
                exclusions.insert(if pair[0] <= pair[1] {
                    pair
                } else {
                    [pair[1], pair[0]]
                });
            }
        }
        TopologyCollisionUpdate {
            topology_version: self.topology_version,
            triangles: self.triangles.clone(),
            triangle_bounds,
            boundary_edges: self.boundary_edges(),
            self_collision_exclusions: exclusions.into_iter().collect(),
            ccd_revision: self.topology_version,
        }
    }

    /// Builds one atomic render/collision update with an event list.
    #[must_use]
    pub fn tear_update(&self, events: Vec<TopologyTearEvent>) -> TopologyTearUpdate {
        TopologyTearUpdate {
            topology_version: self.topology_version,
            events,
            render: self.render_snapshot(),
            collision: self.collision_update(),
        }
    }

    /// Advances damage and opens up to `max_front_steps` candidate root edges.
    /// The returned events are suitable for a render/effects bridge.
    pub fn advance(&mut self, config: TopologyTearingConfig) -> Vec<TopologyTearEvent> {
        let config = config.sanitized();
        if config.max_front_steps == 0
            || self.vertex_count() >= config.max_split_vertices
            || !config.material.is_valid()
            || !config.break_stress.is_finite()
        {
            return Vec::new();
        }
        let previous_topology_version = self.topology_version;
        let mut candidates = Vec::new();
        let (edge_indices, propagating) = self.candidate_edges(config.max_initiation_edges);
        let mut vertex_stresses = BTreeMap::new();
        for edge_index in edge_indices {
            let indexed = &self.root_edges[edge_index];
            let root = indexed.edge;
            if indexed.triangles.len() != 2 || self.seams.contains(&root) {
                continue;
            }
            let Some((a, b)) = self.physical_endpoints(edge_index) else {
                continue;
            };
            let mut best: Option<(Real, Vec3)> = None;
            for vertex in [a, b] {
                let state = *vertex_stresses
                    .entry(vertex)
                    .or_insert_with(|| self.vertex_stress(vertex, config.material));
                let Some(state) = state else { continue };
                let gain = state.direction.dot(state.warp).powi(2) * config.anisotropy_gain;
                let threshold = config.break_stress
                    * (1.0 + gain)
                    * if propagating {
                        config.propagation_ratio
                    } else {
                        1.0
                    };
                // No compression failure, even for a zero fracture threshold.
                if state.maximum <= threshold.max(1.0e-6) {
                    continue;
                }
                let tangent = (self.positions[b as usize] - self.positions[a as usize])
                    .try_normalize()
                    .unwrap_or(Vec3::ZERO);
                // Crack edges follow the direction perpendicular to tension.
                // Alignment participates in candidate ordering, not the damage
                // threshold, so an oblique triangulation cannot suppress tears.
                let alignment = 1.0 - tangent.dot(state.direction).powi(2);
                let score = (state.maximum - threshold) * alignment.max(1.0e-4);
                if best.is_none_or(|(old, _)| score > old) {
                    best = Some((score, state.direction));
                }
            }
            if let Some((score, normal)) = best {
                candidates.push((root, a, b, score, normal));
            }
        }
        candidates.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3).then_with(|| lhs.0.cmp(&rhs.0)));
        let mut events = Vec::new();
        let mut affected = BTreeSet::new();
        let mut proposed_seams = self.seams.clone();
        let mut accepted_edges = 0usize;
        for (root, a, b, _, normal) in candidates {
            if accepted_edges >= config.max_front_steps {
                break;
            }
            proposed_seams.insert(root);
            let mut proposed_affected = affected.clone();
            proposed_affected.extend([a, b]);
            let fits = self
                .required_split_vertices(&proposed_seams, &proposed_affected)
                .and_then(|required| self.vertex_count().checked_add(required))
                .is_some_and(|count| count <= config.max_split_vertices);
            if !fits {
                proposed_seams.remove(&root);
                continue;
            }
            self.open_seam(root);
            affected = proposed_affected;
            accepted_edges += 1;
            let position = (self.positions[a as usize] + self.positions[b as usize]) * 0.5;
            events.push(TopologyTearEvent {
                kind: TopologyTearEventKind::EdgeBreak,
                position,
                normal,
                edge: Some(root),
                vertex: None,
                previous_topology_version: 0,
                topology_version: 0,
                vertex_count: 0,
                triangle_count: 0,
            });
        }
        if !events.is_empty() {
            events.extend(self.split_sectors(config.max_split_vertices, affected));
            self.topology_version = self.topology_version.wrapping_add(1);
            self.finalize_events(&mut events, previous_topology_version);
        }
        events
    }

    /// Opens the strongest edge of the nearest triangle to a point, then
    /// performs the same deterministic sector split as [`advance`](Self::advance).
    pub fn puncture(
        &mut self,
        point: Vec3,
        direction: Vec3,
        radius: Real,
        config: TopologyTearingConfig,
    ) -> Option<Vec<TopologyTearEvent>> {
        let config = config.sanitized();
        if !point.is_finite()
            || !direction.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || config.max_front_steps == 0
            || self.vertex_count() >= config.max_split_vertices
        {
            return None;
        }
        let direction = direction.try_normalize().unwrap_or(Vec3::ZERO);
        let mut best: Option<(Real, usize, TopologyEdge)> = None;
        for (index, [a, b, c]) in self.triangles.iter().copied().enumerate() {
            let closest = prism_physics_geometry::closest_point_on_triangle(
                point,
                self.positions[a as usize],
                self.positions[b as usize],
                self.positions[c as usize],
            );
            let distance = closest.distance(point);
            if distance > radius {
                continue;
            }
            let mut longest = TopologyEdge::new(a, b);
            for edge in [TopologyEdge::new(b, c), TopologyEdge::new(c, a)] {
                if self.physical_edge_length(edge) > self.physical_edge_length(longest) {
                    longest = edge;
                }
            }
            let mut interior = [
                TopologyEdge::new(a, b),
                TopologyEdge::new(b, c),
                TopologyEdge::new(c, a),
            ]
            .into_iter()
            .filter(|&edge| self.edge_incident_triangle_count(edge) == 2)
            .filter(|&edge| !self.seams.contains(&self.root_edge(edge.a, edge.b)))
            .max_by(|lhs, rhs| {
                self.physical_edge_length(*lhs)
                    .partial_cmp(&self.physical_edge_length(*rhs))
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let Some(interior_edge) = interior.take().or_else(|| {
                (self.edge_incident_triangle_count(longest) == 2
                    && !self.seams.contains(&self.root_edge(longest.a, longest.b)))
                .then_some(longest)
            }) else {
                continue;
            };
            if best.is_none_or(|current| (distance, index, interior_edge) < current) {
                best = Some((distance, index, interior_edge));
            }
        }
        let (_, triangle_index, edge) = best?;
        let previous_topology_version = self.topology_version;
        let [a, b, c] = self.triangles[triangle_index];
        let normal = (self.positions[b as usize] - self.positions[a as usize])
            .cross(self.positions[c as usize] - self.positions[a as usize])
            .try_normalize()
            .unwrap_or_else(|| direction.try_normalize().unwrap_or(Vec3::Y));
        let root_edge = self.root_edge(edge.a, edge.b);
        let mut proposed_seams = self.seams.clone();
        proposed_seams.insert(root_edge);
        let affected = BTreeSet::from([edge.a, edge.b]);
        let fits = self
            .required_split_vertices(&proposed_seams, &affected)
            .and_then(|required| self.vertex_count().checked_add(required))
            .is_some_and(|count| count <= config.max_split_vertices);
        if !fits {
            return None;
        }
        if !self.open_seam(root_edge) {
            return None;
        }
        let midpoint = (self.positions[edge.a as usize] + self.positions[edge.b as usize]) * 0.5;
        let mut events = vec![TopologyTearEvent {
            kind: TopologyTearEventKind::Puncture,
            position: midpoint,
            normal,
            edge: Some(root_edge),
            vertex: None,
            previous_topology_version: 0,
            topology_version: 0,
            vertex_count: 0,
            triangle_count: 0,
        }];
        events.extend(self.split_sectors(config.max_split_vertices, affected));
        self.topology_version = self.topology_version.wrapping_add(1);
        self.finalize_events(&mut events, previous_topology_version);
        Some(events)
    }

    fn finalize_events(&self, events: &mut [TopologyTearEvent], previous_topology_version: u64) {
        for event in events {
            event.previous_topology_version = previous_topology_version;
            event.topology_version = self.topology_version;
            event.vertex_count = self.vertex_count() as u32;
            event.triangle_count = self.triangle_count() as u32;
        }
    }

    fn physical_edge_length(&self, edge: TopologyEdge) -> Real {
        (self.rest_positions[edge.a as usize] - self.rest_positions[edge.b as usize]).length()
    }

    fn edge_incident_triangle_count(&self, edge: TopologyEdge) -> usize {
        self.vertex_triangles[edge.a as usize]
            .iter()
            .filter(|&&index| self.triangles[index].contains(&edge.b))
            .count()
    }

    fn split_sectors(
        &mut self,
        max_vertices: usize,
        affected: BTreeSet<u32>,
    ) -> Vec<TopologyTearEvent> {
        let mut events = Vec::new();
        for vertex in affected.into_iter().map(|v| v as usize) {
            if self.positions.len() >= max_vertices {
                break;
            }
            let components = self.sector_components(vertex, &self.seams);
            if components.len() < 2 {
                continue;
            }
            if self.positions.len() + components.len() - 1 > max_vertices {
                continue;
            }
            let component_areas: Vec<Real> = components
                .iter()
                .map(|component| {
                    component
                        .iter()
                        .map(|&triangle_index| self.triangle_area(self.triangles[triangle_index]))
                        .sum::<Real>()
                })
                .collect();
            let total_area: Real = component_areas.iter().sum();
            if total_area <= 1.0e-12 {
                continue;
            }
            let original_inverse_mass = self.inverse_masses[vertex];
            let mut components = components.into_iter();
            self.vertex_triangles[vertex] = components.next().expect("at least two components");
            let first_share = component_areas[0] / total_area;
            self.inverse_masses[vertex] = if original_inverse_mass == 0.0 {
                0.0
            } else {
                original_inverse_mass / first_share
            };
            for (component_index, component) in components.enumerate() {
                let component_index = component_index + 1;
                if self.positions.len() >= max_vertices {
                    break;
                }
                let duplicate = self.positions.len() as u32;
                self.rest_positions.push(self.rest_positions[vertex]);
                self.positions.push(self.positions[vertex]);
                self.velocities.push(self.velocities[vertex]);
                let share = component_areas[component_index] / total_area;
                self.inverse_masses.push(if original_inverse_mass == 0.0 {
                    0.0
                } else {
                    original_inverse_mass / share
                });
                self.vertex_roots.push(self.vertex_roots[vertex]);
                self.vertex_triangles.push(component.clone());
                for triangle_index in component {
                    for index in &mut self.triangles[triangle_index] {
                        if *index == vertex as u32 {
                            *index = duplicate;
                        }
                    }
                }
                events.push(TopologyTearEvent {
                    kind: TopologyTearEventKind::VertexSplit,
                    position: self.positions[duplicate as usize],
                    normal: Vec3::ZERO,
                    edge: None,
                    vertex: Some(duplicate),
                    previous_topology_version: 0,
                    topology_version: 0,
                    vertex_count: 0,
                    triangle_count: 0,
                });
            }
        }
        events
    }

    fn sector_components(&self, vertex: usize, seams: &BTreeSet<TopologyEdge>) -> Vec<Vec<usize>> {
        let incident = self.vertex_triangles[vertex].clone();
        if incident.len() < 2 {
            return vec![incident];
        }
        let mut adjacency: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut neighbors = BTreeMap::new();
        for &triangle in &incident {
            for other in self.triangles[triangle] {
                if other == vertex as u32 || seams.contains(&self.root_edge(vertex as u32, other)) {
                    continue;
                }
                if let Some(&previous) = neighbors.get(&other) {
                    adjacency.entry(previous).or_default().push(triangle);
                    adjacency.entry(triangle).or_default().push(previous);
                } else {
                    neighbors.insert(other, triangle);
                }
            }
        }
        let mut components = Vec::new();
        let mut visited = BTreeSet::new();
        for &start in &incident {
            if !visited.insert(start) {
                continue;
            }
            let mut stack = vec![start];
            let mut component = Vec::new();
            while let Some(current) = stack.pop() {
                component.push(current);
                if let Some(neighbors) = adjacency.get(&current) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components
    }

    fn required_split_vertices(
        &self,
        seams: &BTreeSet<TopologyEdge>,
        affected: &BTreeSet<u32>,
    ) -> Option<usize> {
        let mut required = 0usize;
        for &vertex in affected {
            let components = self.sector_components(vertex as usize, seams);
            if components.len() < 2 {
                continue;
            }
            let total_area: Real = components
                .iter()
                .flat_map(|component| component.iter())
                .map(|&triangle| self.triangle_area(self.triangles[triangle]))
                .sum();
            if !total_area.is_finite() || total_area <= 1.0e-12 {
                return None;
            }
            required = required.checked_add(components.len() - 1)?;
        }
        Some(required)
    }

    fn triangle_area(&self, triangle: [u32; 3]) -> Real {
        let [a, b, c] = triangle.map(|index| self.rest_positions[index as usize]);
        0.5 * (b - a).cross(c - a).length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn square() -> TearableMesh {
        TearableMesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0; 4],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .unwrap()
    }

    #[test]
    fn strain_break_creates_real_boundary_and_duplicate() {
        let mut mesh = square();
        let events = mesh.advance(TopologyTearingConfig {
            break_stress: 0.5,
            max_front_steps: 1,
            ..Default::default()
        });
        assert!(events
            .iter()
            .any(|event| event.kind == TopologyTearEventKind::EdgeBreak));
        assert!(events
            .iter()
            .any(|event| event.kind == TopologyTearEventKind::VertexSplit));
        assert!(mesh.vertex_count() > 4);
        assert_eq!(mesh.seam_count(), 1);
        assert_eq!(mesh.topology_version, 1);
        assert!(events
            .iter()
            .all(|event| event.previous_topology_version == 0
                && event.topology_version == 1
                && event.vertex_count == mesh.vertex_count() as u32
                && event.triangle_count == mesh.triangle_count() as u32));
        let update = mesh.tear_update(events);
        assert_eq!(update.topology_version, 1);
        assert_eq!(update.render.topology_version, 1);
        assert_eq!(update.render.vertices.len(), mesh.vertex_count());
        assert_eq!(update.render.triangles.len(), mesh.triangle_count());
        assert!(!update.render.seam_edges.is_empty());
        assert_eq!(
            update.collision.triangle_bounds.len(),
            mesh.triangle_count()
        );
        assert_eq!(update.collision.ccd_revision, 1);
        assert!(update
            .collision
            .self_collision_exclusions
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn front_budget_is_hard_and_deterministic() {
        let mut mesh = square();
        let config = TopologyTearingConfig {
            break_stress: 0.0,
            max_front_steps: 1,
            ..Default::default()
        };
        let first = mesh.advance(config);
        assert_eq!(
            first
                .iter()
                .filter(|event| event.kind == TopologyTearEventKind::EdgeBreak)
                .count(),
            1
        );
        assert_eq!(mesh.seam_count(), 1);
    }

    #[test]
    fn insufficient_split_capacity_rejects_edge_atomically() {
        let mut mesh = square();
        let before = mesh.clone();
        let events = mesh.advance(TopologyTearingConfig {
            break_stress: 0.0,
            max_front_steps: 1,
            max_split_vertices: 5,
            ..Default::default()
        });
        assert!(events.is_empty());
        assert_eq!(mesh, before);

        let events = mesh.puncture(
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::Z,
            1.0,
            TopologyTearingConfig {
                max_split_vertices: 5,
                ..Default::default()
            },
        );
        assert!(events.is_none());
        assert_eq!(mesh, before);
    }

    #[test]
    fn puncture_emits_normal_and_changes_topology() {
        let mut mesh = square();
        let before = mesh.vertex_count();
        let events = mesh
            .puncture(Vec3::new(0.5, 0.5, 0.0), Vec3::Z, 1.0, Default::default())
            .unwrap();
        assert!(events
            .iter()
            .any(|event| event.kind == TopologyTearEventKind::Puncture));
        assert!(events
            .iter()
            .any(|event| event.kind == TopologyTearEventKind::VertexSplit));
        assert!(mesh.vertex_count() > before);
        assert_eq!(
            events
                .iter()
                .find(|event| event.kind == TopologyTearEventKind::Puncture)
                .unwrap()
                .normal,
            Vec3::Z
        );
    }

    #[test]
    fn invalid_mesh_is_rejected() {
        assert_eq!(
            TearableMesh::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![]),
            Err(TopologyMeshError::EmptyTopology)
        );
    }
}
