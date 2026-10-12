//! The built-in PCM catalogue: the `OpenPBR` specification core registered as
//! `KHR` items plus Prism's own `PRISM` lobe and channel extensions (§5.1).
//!
//! # Dogfooding (§5.1)
//!
//! The specification's own lobes are **not** special-cased: `base`/`coat`/…
//! are ordinary `kind = Lobe` capabilities in namespace `KHR`. There is exactly
//! one registry and one manifest format; the only difference between a core
//! lobe and a studio lobe is its namespace, stability and bit region. This is
//! the Vulkan `EXT → KHR` promotion model and the USD "built-in schemas are
//! just schemas" model applied to the whole material pipeline.
//!
//! # Slots are pinned to the schema
//!
//! The `KHR` lobe slots below are the same immutable `registry_slot`s declared
//! by `prism_material_schema`'s `surface.toml` (and re-exported as
//! [`LobeMask`](https://docs.rs/prism_render_material) bits). A parity gate in
//! `prism_render_material` asserts the two never drift, so this catalogue
//! cannot become a second source of truth.

use crate::manifest::ExtensionManifest;
use crate::namespace::{Kind, Namespace, Stability};
use crate::registry::{ExtensionRegistry, RegistryError};
use crate::{Capability, CapabilitySet};

/// The `OpenPBR` specification namespace vendor string.
pub const VENDOR_KHRONOS: &str = "khronos";
/// Prism's own vendor string.
pub const VENDOR_PRISM: &str = "prism";

/// `uid` of the emission lobe (`KHR`).
pub const UID_LOBE_EMISSION: &str = "org.khronos.openpbr.emission";
/// `uid` of the clearcoat (coat) lobe (`KHR`).
pub const UID_LOBE_CLEARCOAT: &str = "org.khronos.openpbr.coat";
/// `uid` of the specular-anisotropy lobe (`KHR`).
pub const UID_LOBE_ANISOTROPY: &str = "org.khronos.openpbr.specular_anisotropy";
/// `uid` of the sheen (fuzz) lobe (`KHR`).
pub const UID_LOBE_SHEEN: &str = "org.khronos.openpbr.fuzz";
/// `uid` of the subsurface lobe (`KHR`).
pub const UID_LOBE_SUBSURFACE: &str = "org.khronos.openpbr.subsurface";
/// `uid` of the transmission lobe (`KHR`).
pub const UID_LOBE_TRANSMISSION: &str = "org.khronos.openpbr.transmission";
/// `uid` of the thin-film lobe (`KHR`).
pub const UID_LOBE_THIN_FILM: &str = "org.khronos.openpbr.thin_film";
/// `uid` of the stylized SDF face-shadow lobe (`PRISM`).
///
/// The spelling is the schema's: `studio.prism.face_shadow`. Identity is
/// `uid`-based and immutable once a material has been serialized, so it keeps
/// its original spelling even though the human-facing `id` became `npr_face`
/// (`PRISM_lobe_npr_face`) when the lobe was renamed — see
/// `prism_material_schema/schema/surface.toml`.
pub const UID_LOBE_FACE: &str = "studio.prism.face_shadow";
/// `uid` of the wetness dynamic channel (`PRISM`).
pub const UID_CHANNEL_WETNESS: &str = "org.prism.channel.wetness";
/// `uid` of the snow-cover dynamic channel (`PRISM`).
pub const UID_CHANNEL_SNOW: &str = "org.prism.channel.snow";
/// `uid` of the damage dynamic channel (`PRISM`).
pub const UID_CHANNEL_DAMAGE: &str = "org.prism.channel.damage";
/// `uid` of the dissolve dynamic channel (`PRISM`).
pub const UID_CHANNEL_DISSOLVE: &str = "org.prism.channel.dissolve";

/// The specification-core lobe slots, in schema order.
///
/// These are the *same* numbers as the `LobeMask` bits of the same name; the
/// parity gate in `prism_render_material` enforces it.
pub const KHR_LOBE_SLOTS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

/// The `KHR` specification-core lobes, view-dependence flags included.
///
/// `view_dependent` drives §14.3: a view-independent extension (emission,
/// subsurface) may be cached in the object-space shading cache, a
/// view-dependent one (coat, both specular lobes, transmission, thin film) may
/// not.
const KHR_LOBES: [(&str, u32, bool); 7] = [
    (UID_LOBE_EMISSION, 0, false),
    (UID_LOBE_CLEARCOAT, 1, true),
    (UID_LOBE_ANISOTROPY, 2, true),
    (UID_LOBE_SHEEN, 3, true),
    (UID_LOBE_SUBSURFACE, 4, false),
    (UID_LOBE_TRANSMISSION, 5, true),
    (UID_LOBE_THIN_FILM, 6, true),
];

/// The Prism-private extensions, with the stability and capability each needs.
///
/// A missing capability is *not* a silent fold-in: `snow` without virtual
/// texturing is gracefully ignored (the material renders dry), `wetness`
/// without the object-space cache is ignored (the coat renders un-wetted), and
/// neither ever errors because neither is `required`.
///
/// The stability column mirrors the schema: the `npr_face` lobe is
/// `experimental` (its SDF-atlas provider and direction field are still
/// landing, §5.5), while the dynamic channels are `candidate` (layout frozen,
/// response settling). `tests/pcm_parity.rs` in `prism_render_material` asserts
/// the two catalogues agree row for row, so this column cannot drift.
const PRISM_EXTENSIONS: [(&str, u32, Kind, Stability, CapabilitySet, bool); 5] = [
    (
        UID_LOBE_FACE,
        16,
        Kind::Lobe,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_CHANNEL_WETNESS,
        17,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::of(Capability::ObjectSpaceCache),
        false,
    ),
    (
        UID_CHANNEL_SNOW,
        18,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::of(Capability::VirtualTexture),
        false,
    ),
    (
        UID_CHANNEL_DAMAGE,
        19,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_CHANNEL_DISSOLVE,
        20,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
];

/// Build a registry containing the full built-in catalogue.
///
/// The function is allocation-free on the success path and returns a value (not
/// a static) so it stays usable from `no_std` builds without a lazy cell; the
/// catalogue is small and fixed, and §5.2 keeps it off the hot path anyway.
///
/// # Panics
///
/// Panics only if the compiled-in catalogue violates the registry's own
/// invariants, which is a build-time programming error caught by the crate's
/// tests.
pub fn builtin_registry() -> ExtensionRegistry {
    let mut registry = ExtensionRegistry::new();
    for (uid, slot, view_dependent) in KHR_LOBES {
        let manifest = ExtensionManifest::new(uid, slot, Namespace::Khr, Kind::Lobe)
            .with_vendor(VENDOR_KHRONOS)
            .with_stability(Stability::Stable)
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in KHR lobe catalogue is valid");
    }
    for (uid, slot, kind, stability, requires_caps, view_dependent) in PRISM_EXTENSIONS {
        let manifest = ExtensionManifest::new(uid, slot, Namespace::Prism, kind)
            .with_vendor(VENDOR_PRISM)
            .with_stability(stability)
            .with_requires_caps(requires_caps)
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in Prism extension catalogue is valid");
    }
    registry
}

/// Register a brand-new Prism extension into `registry`, allocating its slot.
///
/// # Errors
///
/// Propagates [`RegistryError`] — including
/// [`RegistryError::PoolExhausted`], the explicit failure that replaces the old
/// "silently fold it into an existing bit" behaviour.
pub fn register_prism_extension(
    registry: &mut ExtensionRegistry,
    uid: &'static str,
    kind: Kind,
) -> Result<ExtensionManifest, RegistryError> {
    registry.register_new(uid, Namespace::Prism, kind)
}

/// The capability set a fully-featured Prism target offers.
pub fn full_capabilities() -> CapabilitySet {
    CapabilitySet::full()
}
