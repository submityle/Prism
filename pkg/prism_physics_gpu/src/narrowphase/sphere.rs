//! Sphere-sphere contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! [`sphere_sphere_contact`] is the single source of truth for the collision
//! test and manifold construction; the `CPU` twin and the device kernel both run
//! this exact arithmetic so their contacts agree to within the floating-point
//! tolerance the parity test allows (only the square root and the reciprocal
//! differ in their low bits).
//!
//! # Geometry
//!
//! Two spheres with centres `pa`, `pb` and radii `ra`, `rb` penetrate when the
//! distance between centres is strictly less than `ra + rb`. The contact normal
//! is the unit vector from `a` to `b`, the penetration depth is
//! `(ra + rb) - dist`, and the contact point sits on the plane midway through
//! the overlap, `pa + normal * (ra - depth / 2)`.
//!
//! # Degenerate centres
//!
//! When the centres coincide (or are closer than [`COINCIDENT_EPS2`]) the
//! direction is undefined, so the test falls back to the `+x` axis and a full
//! `ra + rb` depth. This is deterministic and identical on both paths, so a
//! stacked-particle pair never produces a `NaN` normal.
//!
//! Provenance: textbook sphere-sphere collision; no Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::contact::Contact;

/// Squared-distance threshold below which two centres are treated as
/// coincident. Kept identical to the `WGSL` constant so both paths pick the
/// fallback normal on the same pairs.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// Tests whether the two spheres penetrate and, if so, builds their contact.
///
/// Returns [`Some`] with the manifold for the pair `(a, b)` when the spheres
/// overlap, or [`None`] when they are separated or exactly touching. The
/// arithmetic mirrors `narrowphase_contacts.wgsl` operation for operation.
#[must_use]
pub(crate) fn sphere_sphere_contact(
    a: u32,
    b: u32,
    pa: Vec3,
    ra: f32,
    pb: Vec3,
    rb: f32,
) -> Option<Contact> {
    let delta = pb - pa;
    let dist2 = delta.dot(delta);
    let sum_r = ra + rb;
    // Strict overlap: exactly touching spheres share only a boundary point and
    // carry no penetration, so they are not a contact.
    if dist2 >= sum_r * sum_r {
        return None;
    }

    let (normal, dist) = if dist2 <= COINCIDENT_EPS2 {
        // Coincident centres: choose a stable axis rather than dividing by zero.
        (Vec3::X, 0.0)
    } else {
        let dist = dist2.sqrt();
        (delta / dist, dist)
    };

    let depth = sum_r - dist;
    let point = pa + normal * (ra - depth * 0.5);
    Some(Contact::new(a, b, normal, depth, point))
}
