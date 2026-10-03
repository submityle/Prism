//! Acoustic geometry primitives for the Encoder Input Format (EIF) scene.
//!
//! The EIF describes a scene's geometry declaratively: planar faces, axis
//! aligned boxes, and references to triangle-mesh assets, each carrying a
//! reference into the scene's material table. This module provides those
//! primitives together with the derived quantities a renderer needs to fold
//! them into the shared bounding-volume hierarchy and reflection models:
//! axis-aligned bounds, face normals, and face areas.
//!
//! Geometry here is pure data plus closed-form geometry; it performs no
//! ray tracing itself. It is the asset-time description that maps onto the
//! engine's shared BVH (section 31) and section 14 acoustic materials, exactly
//! as section 49.1 requires ("EIF geometry maps to the renderer's shared BVH
//! and section 14 acoustic materials", not a second acoustic truth).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The primitives (planar polygon, axis-aligned box, mesh reference) and the
//! Newell normal / shoelace area used here are standard computational geometry.
//!
//! # Relationship
//!
//! References [`crate::eif::material::AcousticMaterial`] by table index via
//! [`MaterialRef`]. Aggregated by [`crate::eif::scene::EifScene`]. The derived
//! [`Aabb`] bounds are the hook the importer ([`crate::eif::import`]) uses to
//! place each primitive into the engine's shared BVH.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;

use prism_audio_core::math::Sample;

/// A reference into the owning [`EifScene`](crate::eif::scene::EifScene)
/// material table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialRef(pub usize);

impl MaterialRef {
    /// The underlying table index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// An opaque handle to an external triangle-mesh asset (resolved at load time
/// against the engine's asset system, not stored inline in the EIF).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshRef(pub u64);

impl MeshRef {
    /// The underlying asset id.
    #[inline]
    #[must_use]
    pub const fn id(self) -> u64 {
        self.0
    }
}

/// An axis-aligned bounding box in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    /// The component-wise minimum corner.
    pub min: Vec3,
    /// The component-wise maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box from two corners, normalising so `min <= max` component
    /// wise regardless of the order the corners are passed.
    #[must_use]
    pub fn from_corners(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// An empty (inverted) box that absorbs points via [`Aabb::expand`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(Sample::INFINITY),
            max: Vec3::splat(Sample::NEG_INFINITY),
        }
    }

    /// Whether the box is empty (any `min` component exceeds its `max`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Grows the box to include `point`.
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Returns the union of two boxes.
    #[must_use]
    pub fn merge(&self, other: &Aabb) -> Aabb {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The geometric centre of the box (meaningless for an empty box).
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Whether `point` lies inside or on the box (always `false` when empty).
    #[must_use]
    pub fn contains(&self, point: Vec3) -> bool {
        if self.is_empty() {
            return false;
        }
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

/// A planar polygon face described by an ordered ring of world-space vertices.
///
/// The winding order defines the outward normal via the right-hand rule
/// (see [`AcousticFace::normal`]). A face needs at least three vertices to have
/// a well-defined normal and area.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticFace {
    /// The ordered polygon vertices in world space.
    pub vertices: Vec<Vec3>,
    /// The surface material.
    pub material: MaterialRef,
}

impl AcousticFace {
    /// Builds a face from a vertex ring and a material reference.
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, material: MaterialRef) -> Self {
        Self { vertices, material }
    }

    /// The outward unit normal computed by Newell's method, which is robust for
    /// non-planar or near-degenerate rings. Returns [`Vec3::ZERO`] for a
    /// degenerate face (fewer than three vertices or zero area).
    #[must_use]
    pub fn normal(&self) -> Vec3 {
        if self.vertices.len() < 3 {
            return Vec3::ZERO;
        }
        let mut n = Vec3::ZERO;
        let count = self.vertices.len();
        for i in 0..count {
            let current = self.vertices[i];
            let next = self.vertices[(i + 1) % count];
            n.x += (current.y - next.y) * (current.z + next.z);
            n.y += (current.z - next.z) * (current.x + next.x);
            n.z += (current.x - next.x) * (current.y + next.y);
        }
        n.normalize_or_zero()
    }

    /// The polygon area via the Newell vector magnitude (half the length of the
    /// unnormalised Newell normal). Zero for a degenerate face.
    #[must_use]
    pub fn area(&self) -> Sample {
        if self.vertices.len() < 3 {
            return 0.0;
        }
        let mut n = Vec3::ZERO;
        let count = self.vertices.len();
        for i in 0..count {
            let current = self.vertices[i];
            let next = self.vertices[(i + 1) % count];
            n.x += (current.y - next.y) * (current.z + next.z);
            n.y += (current.z - next.z) * (current.x + next.x);
            n.z += (current.x - next.x) * (current.y + next.y);
        }
        n.length() * 0.5
    }

    /// The axis-aligned bounds enclosing the face vertices.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        let mut bounds = Aabb::empty();
        for &v in &self.vertices {
            bounds.expand(v);
        }
        bounds
    }
}

/// An axis-aligned box primitive (a six-sided room shell or a solid obstacle).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticBox {
    /// The box extent.
    pub bounds: Aabb,
    /// The surface material applied to all six faces.
    pub material: MaterialRef,
}

impl AcousticBox {
    /// Builds a box primitive from an extent and a material.
    #[must_use]
    pub fn new(bounds: Aabb, material: MaterialRef) -> Self {
        Self { bounds, material }
    }
}

/// A reference to an external triangle mesh with a single applied material.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticMesh {
    /// The mesh asset handle.
    pub mesh: MeshRef,
    /// Pre-resolved world bounds of the mesh (filled by the asset loader).
    pub bounds: Aabb,
    /// The surface material.
    pub material: MaterialRef,
}

impl AcousticMesh {
    /// Builds a mesh reference primitive.
    #[must_use]
    pub fn new(mesh: MeshRef, bounds: Aabb, material: MaterialRef) -> Self {
        Self {
            mesh,
            bounds,
            material,
        }
    }
}

/// One geometry primitive of an EIF scene.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GeometryPrimitive {
    /// A planar polygon face.
    Face(AcousticFace),
    /// An axis-aligned box.
    Box(AcousticBox),
    /// A reference to an external triangle mesh.
    Mesh(AcousticMesh),
}

impl GeometryPrimitive {
    /// The material reference carried by this primitive.
    #[must_use]
    pub fn material(&self) -> MaterialRef {
        match self {
            GeometryPrimitive::Face(f) => f.material,
            GeometryPrimitive::Box(b) => b.material,
            GeometryPrimitive::Mesh(m) => m.material,
        }
    }

    /// The axis-aligned world bounds of this primitive, the hook used to place
    /// it into the engine's shared BVH.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        match self {
            GeometryPrimitive::Face(f) => f.bounds(),
            GeometryPrimitive::Box(b) => b.bounds,
            GeometryPrimitive::Mesh(m) => m.bounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn unit_square() -> AcousticFace {
        // A one-metre square in the z = 0 plane, wound so the normal points +z.
        AcousticFace::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            MaterialRef(0),
        )
    }

    #[test]
    fn face_normal_and_area() {
        let f = unit_square();
        let n = f.normal();
        assert!(close(n.x, 0.0) && close(n.y, 0.0) && close(n.z, 1.0));
        assert!(close(f.area(), 1.0));
    }

    #[test]
    fn degenerate_face_is_safe() {
        let f = AcousticFace::new(vec![Vec3::ZERO, Vec3::X], MaterialRef(2));
        assert_eq!(f.normal(), Vec3::ZERO);
        assert!(close(f.area(), 0.0));
    }

    #[test]
    fn face_bounds_enclose_vertices() {
        let b = unit_square().bounds();
        assert!(close(b.min.x, 0.0) && close(b.max.x, 1.0));
        assert!(close(b.min.y, 0.0) && close(b.max.y, 1.0));
    }

    #[test]
    fn aabb_from_corners_normalises() {
        let b = Aabb::from_corners(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 5.0));
        assert_eq!(b.min, Vec3::new(-1.0, 0.0, 3.0));
        assert_eq!(b.max, Vec3::new(1.0, 2.0, 5.0));
        assert!(b.contains(Vec3::new(0.0, 1.0, 4.0)));
        assert!(!b.contains(Vec3::new(2.0, 1.0, 4.0)));
    }

    #[test]
    fn aabb_merge_and_expand() {
        let mut b = Aabb::empty();
        assert!(b.is_empty());
        b.expand(Vec3::ZERO);
        b.expand(Vec3::splat(2.0));
        assert_eq!(b.center(), Vec3::splat(1.0));
        let other = Aabb::from_corners(Vec3::splat(-1.0), Vec3::splat(0.5));
        let m = b.merge(&other);
        assert_eq!(m.min, Vec3::splat(-1.0));
        assert_eq!(m.max, Vec3::splat(2.0));
    }

    #[test]
    fn primitive_bounds_and_material() {
        let b = AcousticBox::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::splat(3.0)),
            MaterialRef(5),
        );
        let p = GeometryPrimitive::Box(b);
        assert_eq!(p.material(), MaterialRef(5));
        assert_eq!(p.bounds().max, Vec3::splat(3.0));
    }
}
