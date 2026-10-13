//! Load-time capability negotiation (§5.3).
//!
//! Borrowed from the Vulkan feature/limit model: an extension declares
//! `requires_caps`, the target platform broadcasts what it offers, and the
//! handshake runs once at bake/load time. A missing **required** capability is
//! a hard error (no runtime surprise); a missing **used-only** capability is a
//! graceful ignore — the material is projected back to the variant without that
//! extension, exactly like a glTF importer dropping an unknown extension.

use crate::{Capability, CapabilitySet, ExtensionRegistry, ExtensionSet};
use alloc::{string::String, vec::Vec};
use core::fmt;

/// One requested extension the target cannot honour.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsupportedExtension {
    /// Its registry slot.
    pub slot: u32,
    /// Its `uid`, or `None` when this build does not register the slot at all.
    pub uid: Option<String>,
    /// The capabilities it needs that are missing.
    pub missing: CapabilitySet,
}

impl fmt::Display for UnsupportedExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.uid {
            Some(uid) => write!(f, "slot {} ({uid})", self.slot),
            None => write!(f, "slot {} (unregistered)", self.slot),
        }
    }
}

/// A required extension cannot be satisfied on this target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiationError {
    /// Every extension that could not be honoured.
    pub unsupported: Vec<UnsupportedExtension>,
    /// The capabilities the target actually offers.
    pub available: CapabilitySet,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "material requires extensions this target cannot provide: "
        )?;
        for (index, extension) in self.unsupported.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{extension}")?;
            if !extension.missing.is_empty() {
                write!(f, " missing ")?;
                for (capability_index, capability) in extension.missing.iter().enumerate() {
                    if capability_index > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "{capability}")?;
                }
            }
        }
        Ok(())
    }
}

impl core::error::Error for NegotiationError {}

/// The outcome of a successful negotiation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Negotiation {
    extensions: ExtensionSet,
    ignored: Vec<UnsupportedExtension>,
}

impl Negotiation {
    /// The digest after graceful projection: `required` is always preserved,
    /// unsatisfiable `used`-only extensions are dropped.
    pub const fn extensions(&self) -> ExtensionSet {
        self.extensions
    }

    /// The `used`-only extensions dropped by graceful ignore.
    pub fn ignored(&self) -> &[UnsupportedExtension] {
        &self.ignored
    }

    /// Whether anything was dropped.
    pub fn is_lossless(&self) -> bool {
        self.ignored.is_empty()
    }
}

/// Why `requested`'s extension needs cannot be met by `available`.
fn unmet(
    registry: &ExtensionRegistry,
    slot: u32,
    available: CapabilitySet,
) -> Option<UnsupportedExtension> {
    let Some(manifest) = registry.manifest_at(slot) else {
        // The build does not even register this slot: the extension is absent.
        return Some(UnsupportedExtension {
            slot,
            uid: None,
            missing: CapabilitySet::EMPTY,
        });
    };
    let missing = manifest
        .requires_caps()
        .intersection(CapabilitySet::from_bits(!available.bits()));
    if manifest.requires_caps().is_subset_of(available) {
        debug_assert!(missing.is_empty());
        None
    } else {
        debug_assert!(!missing.is_empty());
        Some(UnsupportedExtension {
            slot,
            uid: Some(String::from(manifest.uid())),
            missing,
        })
    }
}

/// Negotiate a material's requested extension digest against a target's
/// capabilities (§5.3).
///
/// # Errors
///
/// Returns [`NegotiationError`] listing every **required** extension whose
/// capabilities the target lacks. A `used`-only extension that cannot be
/// honoured is reported through [`Negotiation::ignored`] and removed from the
/// resulting digest, so nothing is lost without a record.
pub fn negotiate(
    registry: &ExtensionRegistry,
    requested: ExtensionSet,
    available: CapabilitySet,
) -> Result<Negotiation, NegotiationError> {
    let mut unsupported = Vec::new();
    for slot in requested.iter_required() {
        if let Some(extension) = unmet(registry, slot, available) {
            unsupported.push(extension);
        }
    }
    if !unsupported.is_empty() {
        return Err(NegotiationError {
            unsupported,
            available,
        });
    }

    let mut ignored = Vec::new();
    let mut keep: u64 = 0;
    for slot in requested.iter_used() {
        match unmet(registry, slot, available) {
            None => keep |= 1u64 << slot,
            Some(extension) => ignored.push(extension),
        }
    }
    // Required slots are always kept (they were validated above).
    keep |= requested.required();

    Ok(Negotiation {
        extensions: requested.intersect_used(keep),
        ignored,
    })
}

/// Convenience: the capability set implied by a target that offers `caps`.
pub fn capabilities(caps: impl IntoIterator<Item = Capability>) -> CapabilitySet {
    caps.into_iter().collect()
}
