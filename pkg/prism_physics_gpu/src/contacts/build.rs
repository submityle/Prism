//! Builds contact constraints from a narrow-phase contact list.
//!
//! [`contact_constraints`] turns the dense, one-slot-per-candidate output of
//! the narrow phase (a [`Contact`] wherever a pair penetrates, [`None`]
//! otherwise) into the flat list of [`ContactConstraint`]s the solver consumes.
//! Only the [`Some`] slots become constraints; empty slots are dropped, so the
//! constraint list is compact regardless of how sparse the contact list is.
//!
//! The rest separation of each constraint is recomputed here from the two
//! particle radii (`rest = r_a + r_b`) rather than trusting a value carried on
//! the contact, keeping the radii the single source of truth for how far apart
//! the solver drives the pair.
//!
//! Provenance: trivial list transform; no Unreal Engine source or derived code.

use crate::broadphase::Particle;
use crate::narrowphase::Contact;

use super::constraint::ContactConstraint;

/// Builds one [`ContactConstraint`] per populated slot of `contacts`.
///
/// `particles` supplies the radii used to compute each contact's rest
/// separation; `compliance` is applied uniformly to every constraint (`0` for
/// a perfectly rigid contact). Empty ([`None`]) slots produce no constraint.
#[must_use]
pub fn contact_constraints(
    particles: &[Particle],
    contacts: &[Option<Contact>],
    compliance: f32,
) -> Vec<ContactConstraint> {
    contacts
        .iter()
        .filter_map(|slot| slot.as_ref())
        .map(|contact| {
            let rest = particles[contact.a as usize].radius + particles[contact.b as usize].radius;
            ContactConstraint::new(contact.a, contact.b, rest, compliance)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn particle(x: f32, r: f32) -> Particle {
        Particle::new(Vec3::new(x, 0.0, 0.0), r)
    }

    #[test]
    fn only_populated_slots_become_constraints() {
        let particles = vec![particle(0.0, 0.5), particle(0.8, 0.5), particle(5.0, 1.0)];
        let contacts = vec![
            Some(Contact::new(0, 1, Vec3::X, 0.2, Vec3::new(0.4, 0.0, 0.0))),
            None,
        ];
        let cons = contact_constraints(&particles, &contacts, 0.0);
        assert_eq!(cons.len(), 1);
        assert_eq!(cons[0].a, 0);
        assert_eq!(cons[0].b, 1);
        // rest = r_a + r_b = 0.5 + 0.5.
        assert!((cons[0].rest - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rest_is_the_sum_of_radii_not_the_contact() {
        let particles = vec![particle(0.0, 0.25), particle(1.0, 0.75)];
        let contacts = vec![Some(Contact::new(0, 1, Vec3::X, 0.1, Vec3::ZERO))];
        let cons = contact_constraints(&particles, &contacts, 0.5);
        assert!((cons[0].rest - 1.0).abs() < 1e-6);
        assert!((cons[0].compliance - 0.5).abs() < 1e-6);
    }

    #[test]
    fn an_all_empty_list_yields_no_constraints() {
        let particles = vec![particle(0.0, 0.5)];
        let contacts = vec![None, None, None];
        assert!(contact_constraints(&particles, &contacts, 0.0).is_empty());
    }
}
