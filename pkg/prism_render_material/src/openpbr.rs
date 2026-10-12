//! `OpenPBR` super-set hard constraints (§4.5): the pure-`OpenPBR` conformance
//! gate and the lossy `project_to_openpbr` downgrade exporter.
//!
//! `Prism`'s über-BSDF is *defined* as a truth-bearing superset of `OpenPBR`
//! Surface: `Prism über = OpenPBR Surface ∪ { FACE } ∪ { open-vocabulary
//! lobes }` (design doc §4.4/§4.5). For that claim to be checkable rather than
//! aspirational, §4.5 fixes three hard constraints, two of which live here:
//!
//! 1. **Conformance (§4.5 constraint 1)** — a surface whose [`LobeMask`] uses
//!    only the `OpenPBR` specification-core region is a *pure `OpenPBR`* material,
//!    and Prism must leave it untouched. [`openpbr_conformance`] asserts the
//!    projection is the byte-level identity on such masks, so "turn off the
//!    extensions and `Prism` *is* `OpenPBR`" is a machine-checked fact.
//! 2. **Lossy downgrade projection (§4.5 constraint 3)** —
//!    [`project_to_openpbr`] turns a Prism über that carries private/extension
//!    lobes into a legal pure-OpenPBR parameter block for export to an external
//!    `OpenPBR` or `MaterialX` consumer. The projection is one-way (superset →
//!    baseline), never claims visual equivalence, and — crucially — *reports
//!    every dropped lobe explicitly* so a lossy export can never masquerade as
//!    lossless.
//!
//! Constraint 2 (bit-region isolation) is enforced structurally by
//! [`LobeMask::KHR_REGION`]/[`LobeMask::EXTENSION_REGION`] and their
//! schema-parity gate; this module consumes that split rather than restating
//! it.
//!
//! Both operations are pure and allocation-bounded by the fixed lobe count, so
//! they are safe to run in authoring/export tooling and in CI without touching
//! the runtime ABI.

use alloc::vec::Vec;

use crate::surface_layout::SURFACE_LOBES;
use crate::{LobeMask, SurfaceParameterBlock};

/// One extension lobe that [`project_to_openpbr`] removed from a Prism über.
///
/// Every field is identity metadata from the schema-generated lobe table
/// (§17.5), so the record survives namespace promotion (`X_studio → EXT → KHR`
/// keeps the same `uid` and slot) and can be surfaced verbatim in an export
/// report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DroppedLobe {
    /// Schema lobe id, e.g. `"face"`.
    pub id: &'static str,
    /// Globally unique capability id, stable across promotion (§5.4).
    pub uid: &'static str,
    /// Governance namespace that placed the lobe outside the core region
    /// (`"PRISM"` / `"EXT"` / `"X_STUDIO"`).
    pub namespace: &'static str,
    /// The single dropped [`LobeMask`] bit.
    pub bit: LobeMask,
    /// The canonical `OpenPBR` lobe whose response the exporter approximates this
    /// lobe's effect with, or `None` when there is no faithful canonical
    /// equivalent and the effect is dropped outright.
    ///
    /// The projection never fabricates an approximation it cannot justify: a
    /// `None` here is an honest "this effect does not survive export".
    pub approximated_by: Option<&'static str>,
}

/// The pure-OpenPBR block produced by [`project_to_openpbr`], together with the
/// explicit record of everything the downgrade lost.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenPbrProjection {
    /// The projected block. Its [`LobeMask`] is confined to
    /// [`LobeMask::KHR_REGION`], so it is legal pure-OpenPBR.
    pub block: SurfaceParameterBlock,
    /// Named extension lobes that were dropped, in canonical pack order.
    pub dropped: Vec<DroppedLobe>,
    /// Undefined/non-lobe bits that were present on the input mask and pruned.
    ///
    /// A well-formed mask never sets these, so in practice this is empty;
    /// carrying it separately keeps the projection total (it never panics on a
    /// corrupt mask) without inventing fake lobe identities.
    pub unknown_bits: LobeMask,
}

impl OpenPbrProjection {
    /// Whether the downgrade preserved every input lobe, i.e. the input was
    /// already pure `OpenPBR`.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.dropped.is_empty() && self.unknown_bits.bits() == 0
    }

    /// Globally unique ids of every dropped named lobe, in pack order.
    pub fn dropped_uids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.dropped.iter().map(|lobe| lobe.uid)
    }
}

/// The verdict of the §4.5 constraint-1 conformance gate for one surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenPbrConformance {
    /// The surface is pure `OpenPBR` and projection left it byte-identical:
    /// `Prism` reproduces the `OpenPBR` baseline exactly (gate passes).
    Conformant,
    /// The surface carries extension lobes, so `OpenPBR` conformance is not
    /// applicable to it as-is; project it first, then re-check the projected
    /// block. Not a failure.
    NotApplicable,
    /// A pure-OpenPBR surface changed under projection. This can only happen if
    /// the region masks and the lobe table disagree — an internal invariant
    /// break that must fail CI.
    Breached,
}

/// §4.5 constraint 3: lossily project a Prism über down to a legal pure-OpenPBR
/// block, reporting every dropped lobe.
///
/// The projection keeps the always-present core and every lobe whose bit lives
/// in [`LobeMask::KHR_REGION`], drops the rest, and never mutates a kept lobe's
/// stored value. For an input that is already [`LobeMask::is_pure_openpbr`] the
/// result is the byte-level identity, which is exactly what
/// [`openpbr_conformance`] asserts.
#[must_use]
pub fn project_to_openpbr(block: &SurfaceParameterBlock) -> OpenPbrProjection {
    let kept = block.lobe_mask.canonical_region();
    let mut dropped = Vec::new();
    let mut named = LobeMask::default();
    for lobe in SURFACE_LOBES {
        let bit = LobeMask(1 << lobe.registry_slot);
        if !block.lobe_mask.contains(bit) {
            continue;
        }
        named = named.union(bit);
        if kept.contains(bit) {
            continue;
        }
        dropped.push(DroppedLobe {
            id: lobe.id,
            uid: lobe.uid,
            namespace: lobe.namespace,
            bit,
            approximated_by: canonical_equivalent(lobe.id),
        });
    }
    // Any set bit that is not one of the named lobes is corrupt; keep it out of
    // the projection but record it so the export report stays honest.
    let unknown_bits = LobeMask(block.lobe_mask.bits() & !named.bits());

    let mut projected = *block;
    projected.lobe_mask = kept;
    OpenPbrProjection {
        block: projected,
        dropped,
        unknown_bits,
    }
}

/// §4.5 constraint 1: check that projection reproduces a pure-OpenPBR surface
/// byte-for-byte.
///
/// Returns [`OpenPbrConformance::NotApplicable`] for a surface that carries
/// extension lobes (downgrade it first), [`OpenPbrConformance::Conformant`] when
/// a pure-OpenPBR surface is the fixed point of the projection, and
/// [`OpenPbrConformance::Breached`] when a pure-OpenPBR surface was mutated —
/// the CI failure that would mean the bit-region split no longer isolates the
/// specification core.
#[must_use]
pub fn openpbr_conformance(block: &SurfaceParameterBlock) -> OpenPbrConformance {
    if !block.lobe_mask.is_pure_openpbr() {
        return OpenPbrConformance::NotApplicable;
    }
    let projection = project_to_openpbr(block);
    if projection.is_lossless() && projection.block.pack() == block.pack() {
        OpenPbrConformance::Conformant
    } else {
        OpenPbrConformance::Breached
    }
}

/// The nearest canonical `OpenPBR` lobe an extension lobe's effect folds into, or
/// `None` when no canonical lobe is a faithful carrier.
///
/// This is intentionally conservative and total: it never invents a mapping.
/// The stylized SDF face-shadow lobe (`PRISM`) has no physical `OpenPBR`
/// counterpart, so its effect is dropped outright and an exporter must warn the
/// author; a future extension lobe that genuinely refines a canonical lobe
/// (e.g. a cross-vendor extension of coat) would return that lobe's id here.
fn canonical_equivalent(lobe_id: &str) -> Option<&'static str> {
    // The extension→canonical carrier table. It is empty today: the only
    // extension lobe is FACE (stylized SDF terminator softening) and `OpenPBR`
    // models no such lobe, so it is dropped rather than folded. Adding a
    // faithful mapping is a one-line, reviewable change here.
    CORE_CARRIERS
        .iter()
        .find(|(id, _)| *id == lobe_id)
        .map(|(_, canonical)| *canonical)
}

/// Extension lobe id → the canonical `OpenPBR` lobe that faithfully carries its
/// effect. Empty until an extension genuinely refines a canonical lobe.
const CORE_CARRIERS: [(&str, &str); 0] = [];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuSurfaceParameters;

    fn authored() -> GpuSurfaceParameters {
        GpuSurfaceParameters {
            base_color: [0.1, 0.2, 0.3, 1.0],
            emissive: [2.0, 0.0, 0.0, 1.0],
            metallic: 0.25,
            perceptual_roughness: 0.4,
            reflectance: 0.55,
            ambient_occlusion: 0.9,
            normal_scale: 0.8,
            alpha_cutoff: 0.37,
            transmission: 0.6,
            thickness: 1.5,
            clearcoat: 0.7,
            clearcoat_roughness: 0.15,
            anisotropy: 0.3,
            anisotropy_rotation: 0.2,
            sheen: 0.45,
            subsurface: 0.65,
            index_of_refraction: 1.33,
            dispersion: 0.05,
            face_softness: 0.25,
            _pad_face0: 0.0,
            _pad_face1: 0.0,
            _pad_face2: 0.0,
        }
    }

    fn block(mask: LobeMask) -> SurfaceParameterBlock {
        SurfaceParameterBlock::from_full(&authored(), mask)
    }

    #[test]
    fn pure_openpbr_projects_to_a_byte_identical_fixed_point() {
        // Every subset of the specification-core region must be its own
        // projection, with nothing reported as lost.
        for raw in 0..=LobeMask::KHR_REGION.bits() {
            let mask = LobeMask(raw & LobeMask::KHR_REGION.bits());
            let input = block(mask);
            let projection = project_to_openpbr(&input);
            assert!(projection.is_lossless(), "mask {mask:?} reported loss");
            assert!(projection.unknown_bits.bits() == 0);
            assert_eq!(projection.block.lobe_mask, mask);
            assert_eq!(projection.block.pack(), input.pack());
            assert_eq!(openpbr_conformance(&input), OpenPbrConformance::Conformant);
        }
    }

    #[test]
    fn extension_lobes_are_dropped_and_flagged() {
        let input = block(LobeMask::FACE);
        let projection = project_to_openpbr(&input);
        assert!(!projection.is_lossless());
        assert_eq!(projection.dropped.len(), 1);
        let dropped = projection.dropped[0];
        assert_eq!(dropped.id, "face");
        assert_eq!(dropped.uid, "studio.prism.face_shadow");
        assert_eq!(dropped.namespace, "PRISM");
        assert_eq!(dropped.bit, LobeMask::FACE);
        // No canonical OpenPBR lobe carries stylized SDF face shading, so the
        // projection reports an outright drop rather than a fake mapping.
        assert_eq!(dropped.approximated_by, None);
        // The projected block is legal pure OpenPBR with the core preserved.
        assert!(projection.block.lobe_mask.is_pure_openpbr());
        assert_eq!(projection.block.lobe_mask, LobeMask::default());
    }

    #[test]
    fn projection_preserves_every_canonical_lobe_verbatim() {
        let input = block(LobeMask::all());
        let projection = project_to_openpbr(&input);
        // Only FACE is an extension lobe today.
        assert_eq!(projection.dropped.len(), 1);
        assert_eq!(projection.block.lobe_mask, LobeMask::KHR_REGION);
        // Kept lobes are value-identical after projection.
        assert_eq!(projection.block.emission, input.emission);
        assert_eq!(projection.block.transmission, input.transmission);
        assert_eq!(projection.block.clearcoat, input.clearcoat);
        // The projection packs as core + six canonical lobes.
        assert_eq!(
            projection.block.packed_len_words(),
            crate::SURFACE_CORE_WORDS + 6 * crate::SURFACE_LOBE_WORDS
        );
    }

    #[test]
    fn projection_is_idempotent_and_the_second_pass_is_lossless() {
        let once = project_to_openpbr(&block(LobeMask::all()));
        let twice = project_to_openpbr(&once.block);
        assert!(twice.is_lossless());
        assert_eq!(twice.block, once.block);
        assert_eq!(twice.block.pack(), once.block.pack());
        assert_eq!(
            openpbr_conformance(&once.block),
            OpenPbrConformance::Conformant
        );
        // The input as-is is not conformant-applicable (it carries extensions).
        assert_eq!(
            openpbr_conformance(&block(LobeMask::all())),
            OpenPbrConformance::NotApplicable
        );
    }

    #[test]
    fn dropped_report_is_in_canonical_pack_order() {
        // A mask carrying several extension lobes reports them low-bit first,
        // long before the next bit is defined. Only FACE exists today, so we
        // also prove undefined high bits never fabricate a named drop.
        let mut input = block(LobeMask::FACE);
        input.lobe_mask = LobeMask(LobeMask::FACE.bits() | (1 << 31));
        let projection = project_to_openpbr(&input);
        assert_eq!(projection.dropped.len(), 1);
        assert_eq!(projection.dropped[0].uid, "studio.prism.face_shadow");
        assert_eq!(projection.unknown_bits, LobeMask(1 << 31));
        assert!(!projection.is_lossless());
    }

    #[test]
    fn empty_mask_is_the_lossless_fixed_point() {
        let input = block(LobeMask::default());
        let projection = project_to_openpbr(&input);
        assert!(projection.is_lossless());
        assert_eq!(projection.block, input);
        assert_eq!(
            projection.block.packed_len_words(),
            crate::SURFACE_CORE_WORDS
        );
    }
}
