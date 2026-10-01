//! Teleport / reset handling for persistent `GPU` cloth pieces.
//!
//! A persistent piece keeps its particle position and velocity pools resident on
//! the device so the solver evolves them in place across frames (see
//! [`ClothPieceGpuBuffers::write_dynamic`](super::bind_groups::ClothPieceGpuBuffers::write_dynamic)).
//! That continuity is exactly wrong when a garment's reference frame jumps
//! discontinuously — a character teleports, a cutscene cuts, an actor respawns —
//! because the resident state would stretch the cloth across the whole jump
//! before the solver could pull it back, a frame of catastrophic artefacts UE's
//! `Chaos` cloth avoids with an explicit teleport path.
//!
//! An author signals the jump by bumping the garment's teleport *generation*
//! (see [`ClothGarment::request_teleport`](super::garment::ClothGarment::request_teleport))
//! and choosing a [`ClothTeleportMode`]. The prepare stage compares the garment's
//! generation against the one the resident piece last applied and, on a change,
//! restreams the authored state the mode asks for instead of letting the stale
//! resident pools ride through the jump.

/// How a persistent cloth piece reacts when its reference frame jumps.
///
/// The default every ordinary frame is [`Continuous`](Self::Continuous): the
/// solver keeps evolving the resident state in place. The two teleporting modes
/// are latched by an author on the frame of a discontinuity and differ only on
/// whether the garment keeps the motion it had accumulated before the jump.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ClothTeleportMode {
    /// No discontinuity: the solver keeps evolving the resident position and
    /// velocity pools in place. This is the default for a garment that is not
    /// teleporting this generation.
    #[default]
    Continuous,
    /// The reference frame jumped but the cloth should keep its motion: the
    /// resident positions are snapped to the authored positions (the new pose)
    /// while the resident velocities ride through unchanged, so the garment keeps
    /// billowing across the cut instead of snapping rigid.
    Teleport,
    /// The reference frame jumped and the cloth should settle: both the resident
    /// positions and velocities are restreamed from the authored state, so the
    /// garment lands on its authored rest pose carrying no inherited motion.
    TeleportAndReset,
}

/// The resident pools a [`ClothTeleportMode`] restreams from the authored upload
/// when the prepare stage applies a teleport. Derived purely from the mode so the
/// prepare stage and its unit tests share one source of truth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TeleportRestream {
    /// Snap the resident positions back to the authored positions.
    pub(crate) positions: bool,
    /// Restream the authored velocities, dropping the evolved resident motion.
    pub(crate) velocities: bool,
}

impl ClothTeleportMode {
    /// The resident pools this mode restreams from the authored upload.
    ///
    /// [`Continuous`](Self::Continuous) restreams nothing (the prepare stage never
    /// even applies it); the two teleporting modes both snap positions and differ
    /// only on whether the inherited velocity survives the jump.
    #[must_use]
    pub(crate) const fn restream(self) -> TeleportRestream {
        match self {
            Self::Continuous => TeleportRestream {
                positions: false,
                velocities: false,
            },
            Self::Teleport => TeleportRestream {
                positions: true,
                velocities: false,
            },
            Self::TeleportAndReset => TeleportRestream {
                positions: true,
                velocities: true,
            },
        }
    }

    /// Whether this mode asks the prepare stage to restream any resident pool.
    /// [`Continuous`](Self::Continuous) is the only inert mode, so a generation
    /// bump latched with it collapses to a no-op apply.
    #[must_use]
    pub(crate) const fn restreams_any(self) -> bool {
        !matches!(self, Self::Continuous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_restreams_nothing() {
        let r = ClothTeleportMode::Continuous.restream();
        assert!(!r.positions, "continuous must not snap positions");
        assert!(!r.velocities, "continuous must not drop velocities");
        assert!(!ClothTeleportMode::Continuous.restreams_any());
    }

    #[test]
    fn teleport_snaps_positions_but_keeps_velocity() {
        let r = ClothTeleportMode::Teleport.restream();
        assert!(r.positions, "teleport must snap positions to the new pose");
        assert!(
            !r.velocities,
            "teleport must keep the inherited motion so the garment keeps billowing",
        );
        assert!(ClothTeleportMode::Teleport.restreams_any());
    }

    #[test]
    fn teleport_and_reset_restreams_both_pools() {
        let r = ClothTeleportMode::TeleportAndReset.restream();
        assert!(r.positions, "reset must snap positions to the authored pose");
        assert!(
            r.velocities,
            "reset must drop inherited motion so the garment settles",
        );
        assert!(ClothTeleportMode::TeleportAndReset.restreams_any());
    }

    #[test]
    fn continuous_is_the_default() {
        assert_eq!(ClothTeleportMode::default(), ClothTeleportMode::Continuous);
    }
}
