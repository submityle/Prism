//! Driver-local module interning. Strings occur only at this cold boundary.

use super::{
    ArtifactDigest, PermutationError, PermutationIndex, PermutationSpace, ShaderPackageId,
};
use alloc::{collections::BTreeMap, string::String};
use core::sync::atomic::{AtomicU64, Ordering};

/// Numeric identity of code, entry point and binding ABI on one driver.
/// Never serialize this handle into material artifacts or persisted PSO keys.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModuleId(pub u64);

impl ModuleId {
    /// Reserved for the renderer's explicitly documented stand-in provider.
    pub const SPINE: Self = Self(0);
}

static NEXT_MODULE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModuleIdentity {
    pub package: ShaderPackageId,
    pub digest: ArtifactDigest,
    pub entry_point: String,
    pub binding_layout_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct InternedModule {
    pub identity: ModuleIdentity,
    pub permutations: PermutationSpace,
    pub full_lobe_fallback: Option<PermutationIndex>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleInternError {
    Permutation(PermutationError),
    DeclarationChanged,
    ReservedIdentityConflict,
    Exhausted,
}

#[derive(Default)]
pub struct ModuleInterner {
    by_identity: BTreeMap<ModuleIdentity, ModuleId>,
    modules: BTreeMap<ModuleId, InternedModule>,
}

impl ModuleInterner {
    fn validate_declaration(
        permutations: &PermutationSpace,
        full_lobe_fallback: Option<PermutationIndex>,
    ) -> Result<(), ModuleInternError> {
        permutations
            .checked_total()
            .map_err(ModuleInternError::Permutation)?;
        if let Some(index) = full_lobe_fallback {
            permutations
                .check_index(index.get())
                .map_err(ModuleInternError::Permutation)?;
        }
        Ok(())
    }

    pub fn intern(
        &mut self,
        identity: ModuleIdentity,
        permutations: PermutationSpace,
        full_lobe_fallback: Option<PermutationIndex>,
    ) -> Result<ModuleId, ModuleInternError> {
        Self::validate_declaration(&permutations, full_lobe_fallback)?;
        if let Some(id) = self.by_identity.get(&identity) {
            let existing = &self.modules[id];
            if existing.permutations != permutations
                || existing.full_lobe_fallback != full_lobe_fallback
            {
                return Err(ModuleInternError::DeclarationChanged);
            }
            return Ok(*id);
        }
        let raw = NEXT_MODULE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| ModuleInternError::Exhausted)?;
        let id = ModuleId(raw);
        self.by_identity.insert(identity.clone(), id);
        self.modules.insert(
            id,
            InternedModule {
                identity,
                permutations,
                full_lobe_fallback,
            },
        );
        Ok(id)
    }

    /// Binds a reserved driver-local module id to a declaration.
    ///
    /// Reserved ids are not allocated from the ordinary process-wide counter,
    /// but they still participate in the same identity/declaration table. This
    /// prevents the spine from bypassing permutation validation or allowing a
    /// later ordinary intern to silently claim the same declaration.
    pub fn reserve(
        &mut self,
        id: ModuleId,
        identity: ModuleIdentity,
        permutations: PermutationSpace,
        full_lobe_fallback: Option<PermutationIndex>,
    ) -> Result<ModuleId, ModuleInternError> {
        Self::validate_declaration(&permutations, full_lobe_fallback)?;
        if let Some(existing_id) = self.by_identity.get(&identity).copied() {
            if existing_id != id {
                return Err(ModuleInternError::ReservedIdentityConflict);
            }
        }
        if let Some(existing) = self.modules.get(&id) {
            if existing.identity != identity
                || existing.permutations != permutations
                || existing.full_lobe_fallback != full_lobe_fallback
            {
                return Err(ModuleInternError::DeclarationChanged);
            }
            return Ok(id);
        }
        self.by_identity.insert(identity.clone(), id);
        self.modules.insert(
            id,
            InternedModule {
                identity,
                permutations,
                full_lobe_fallback,
            },
        );
        Ok(id)
    }

    pub fn get(&self, id: ModuleId) -> Option<&InternedModule> {
        self.modules.get(&id)
    }
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    fn identity() -> ModuleIdentity {
        ModuleIdentity {
            package: ShaderPackageId::new("module-test"),
            digest: ArtifactDigest([1; 32]),
            entry_point: "main".into(),
            binding_layout_digest: [2; 32],
        }
    }

    #[test]
    fn module_interning_binds_content_entry_and_layout() {
        let mut modules = ModuleInterner::default();
        let space = PermutationSpace::try_new([2, 3]).unwrap();
        let fallback = space.check_index(5).unwrap();
        let first = modules
            .intern(identity(), space.clone(), Some(fallback))
            .unwrap();
        assert_eq!(
            modules.intern(identity(), space.clone(), Some(fallback)),
            Ok(first)
        );
        assert_eq!(
            modules.intern(identity(), space.clone(), None),
            Err(ModuleInternError::DeclarationChanged)
        );
        for field in 0..3 {
            let mut changed = identity();
            match field {
                0 => changed.digest.0[0] += 1,
                1 => changed.entry_point.push('2'),
                _ => changed.binding_layout_digest[0] += 1,
            }
            assert_ne!(
                modules
                    .intern(changed, space.clone(), Some(fallback))
                    .unwrap(),
                first
            );
        }
        let mut other_driver = ModuleInterner::default();
        assert_ne!(
            other_driver
                .intern(identity(), space, Some(fallback))
                .unwrap(),
            first
        );
        assert_ne!(first, ModuleId::SPINE);
    }

    #[test]
    fn cold_package_names_are_interned() {
        let first = ShaderPackageId::new("same-package");
        let second = ShaderPackageId::new(String::from("same-package"));
        assert_eq!(first, second);
        assert!(core::ptr::eq(first.as_str(), second.as_str()));
        assert_ne!(first, ShaderPackageId::new("different-package"));
        assert_eq!(size_of::<ShaderPackageId>(), size_of::<usize>());
    }

    #[test]
    fn reserved_module_uses_the_same_declaration_validation() {
        let mut modules = ModuleInterner::default();
        let identity = identity();
        let space = PermutationSpace::try_new([2]).unwrap();
        let fallback = space.check_index(1).unwrap();
        assert_eq!(
            modules.reserve(
                ModuleId::SPINE,
                identity.clone(),
                space.clone(),
                Some(fallback),
            ),
            Ok(ModuleId::SPINE)
        );
        assert_eq!(
            modules.intern(identity.clone(), space.clone(), Some(fallback)),
            Ok(ModuleId::SPINE)
        );
        assert_eq!(
            modules.reserve(ModuleId::SPINE, identity, space, None),
            Err(ModuleInternError::DeclarationChanged)
        );
    }
}
