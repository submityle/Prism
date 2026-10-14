//! Derived views only: accesses remain the sole declaration authority.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantResourceUse {
    pub feature: FeatureId,
    pub variant: VariantId,
    pub pass: PassKey,
    pub usage: ResourceUse,
}

impl PassTemplate {
    pub fn reads(&self) -> impl Iterator<Item = &ResourceUse> {
        self.accesses.iter().filter(|usage| !usage.access.writes())
    }
    pub fn writes(&self) -> impl Iterator<Item = &ResourceUse> {
        self.accesses.iter().filter(|usage| usage.access.writes())
    }
}

impl FeatureDefinition {
    /// Includes rejected and inactive variants, preserving their identities.
    pub fn accesses(&self) -> impl Iterator<Item = VariantResourceUse> + '_ {
        self.variants.iter().flat_map(move |variant| {
            variant.passes.iter().flat_map(move |pass| {
                pass.accesses.iter().map(move |&usage| VariantResourceUse {
                    feature: self.id,
                    variant: variant.id,
                    pass: pass.id,
                    usage,
                })
            })
        })
    }
    pub fn reads(&self) -> impl Iterator<Item = VariantResourceUse> + '_ {
        self.accesses()
            .filter(|access| !access.usage.access.writes())
    }
    pub fn writes(&self) -> impl Iterator<Item = VariantResourceUse> + '_ {
        self.accesses()
            .filter(|access| access.usage.access.writes())
    }
}
