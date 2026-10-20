//! Explicit importer/cook recipe identity.

use crate::executor::ArtifactSpec;
use crate::hash::HashError;
use crate::job::{BuildJob, JobId, Processor};
use crate::profile::{EffectiveProfile, ProfileError, ProfileReader};
use crate::source::SourceSnapshot;
use crate::types::{ArtifactKey, AssetId};

/// Stable processor identity used by recipes, registries, and artifact keys.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcessorIdentity {
    /// Stable protocol namespace for the processor.
    pub id: u128,
    /// Implementation/schema version included in cache identity.
    pub version: u32,
}

impl ProcessorIdentity {
    /// Creates a stable processor identity.
    pub const fn new(id: u128, version: u32) -> Self {
        Self { id, version }
    }
}

/// Processor identity that can be checked against an [`ImportRecipe`].
pub trait ImportProcessor: Processor {
    /// Stable importer/processor protocol identity.
    const IMPORTER_ID: u128;
    /// Importer implementation/schema version.
    const IMPORTER_VERSION: u32;

    /// Returns the identity represented by the associated constants.
    fn identity() -> ProcessorIdentity {
        ProcessorIdentity::new(Self::IMPORTER_ID, Self::IMPORTER_VERSION)
    }
}

/// Failure while binding a recipe to a concrete importer processor.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportRecipeError {
    /// Recipe identity does not describe the processor that will run.
    ProcessorIdentityMismatch {
        /// Importer identity declared by the recipe.
        recipe_id: u128,
        /// Importer version declared by the recipe.
        recipe_version: u32,
        /// Importer identity declared by the concrete processor.
        processor_id: u128,
        /// Importer version declared by the concrete processor.
        processor_version: u32,
    },
    /// Registry selection failed before a graph node could be created.
    ImporterSelection(crate::import::ImporterSelectionError),
    /// The selected importer profile dependencies could not be projected.
    Profile(ProfileError),
    /// The frozen output contract cannot be encoded.
    Hash(HashError),
}

impl std::fmt::Display for ImportRecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProcessorIdentityMismatch {
                recipe_id,
                recipe_version,
                processor_id,
                processor_version,
            } => write!(
                f,
                "import recipe identifies {:032x} v{}, but processor is {:032x} v{}",
                recipe_id, recipe_version, processor_id, processor_version
            ),
            Self::ImporterSelection(error) => error.fmt(f),
            Self::Profile(error) => error.fmt(f),
            Self::Hash(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ImportRecipeError {}

impl From<crate::import::ImporterSelectionError> for ImportRecipeError {
    fn from(value: crate::import::ImporterSelectionError) -> Self {
        Self::ImporterSelection(value)
    }
}

impl From<ProfileError> for ImportRecipeError {
    fn from(value: ProfileError) -> Self {
        Self::Profile(value)
    }
}

impl From<HashError> for ImportRecipeError {
    fn from(value: HashError) -> Self {
        Self::Hash(value)
    }
}

/// The minimum executable import node: a source-dependent job and its output contract.
pub struct ImportPlan {
    /// Job to add to the build graph.
    pub job: BuildJob,
    /// Artifact contract to pass to [`crate::BuildExecutor`].
    pub spec: ArtifactSpec,
}

/// All semantic inputs needed to derive an imported artifact key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportRecipe {
    /// Stable importer/processor protocol identity.
    pub importer_id: u128,
    /// Importer implementation/schema version.
    pub importer_version: u32,
    /// Canonically encoded import settings.
    pub canonical_settings: Vec<u8>,
    /// Explicit platform/profile identity.
    pub profile_hash: [u8; 32],
    /// Explicit toolchain fingerprint.
    pub toolchain_fingerprint: Vec<u8>,
    /// Explicit declared environment fingerprint.
    pub environment_fingerprint: Vec<u8>,
}

impl ImportRecipe {
    /// Configures an owned processor through a recording profile reader.
    ///
    /// All successful reads, including absent optional fields and fields used
    /// to choose a branch, enter both the artifact key and the input snapshot.
    /// Unread fields do not invalidate this node. The factory runs during graph
    /// planning, before any DDC lookup; it must capture every required value in
    /// the returned processor rather than reading ambient state at execution.
    /// The processor is owned and cannot retain the borrowed reader.
    ///
    /// The snapshot also records settings, identity, output contract, routing,
    /// toolchain and environment, so callers need not maintain a parallel
    /// dependency list. Planning failure leaves the snapshot unchanged.
    pub fn plan_with_profile<P, F>(
        &self,
        asset: AssetId,
        job_id: JobId,
        source_job: JobId,
        profile: &EffectiveProfile,
        inputs: &mut crate::BuildInputSnapshot,
        configure: F,
        kind: u128,
        schema: u32,
        alignment: u32,
    ) -> Result<ImportPlan, ImportRecipeError>
    where
        P: ImportProcessor + 'static,
        F: FnOnce(&mut ProfileReader<'_>) -> Result<P, ProfileError>,
    {
        let mut reader = profile.reader();
        let processor = configure(&mut reader)?;
        let projection = reader.finish();
        let mut recipe = self.clone();
        recipe.profile_hash = projection.hash();
        let plan = recipe.plan(
            asset, job_id, source_job, processor, kind, schema, alignment,
        )?;
        // Profile changes are explained separately from processor settings.
        // This fingerprint includes all remaining semantic contract fields.
        let mut contract = plan.spec.clone();
        contract.profile_hash = [0; 32];
        let mut settings = asset.0.to_le_bytes().to_vec();
        settings.extend_from_slice(&contract.artifact_key(1, &[])?.0);
        inputs.set_settings(job_id, settings);
        inputs.set_profile(job_id, projection);
        Ok(plan)
    }

    /// Binds a concrete importer to this recipe and creates its graph node.
    ///
    /// The identity check makes the processor implementation part of the
    /// recipe contract instead of relying on callers to keep two fields in
    /// sync manually.
    pub fn plan<P>(
        &self,
        asset: AssetId,
        job_id: JobId,
        source_job: JobId,
        processor: P,
        kind: u128,
        schema: u32,
        alignment: u32,
    ) -> Result<ImportPlan, ImportRecipeError>
    where
        P: ImportProcessor + 'static,
    {
        if self.importer_id != P::IMPORTER_ID || self.importer_version != P::IMPORTER_VERSION {
            return Err(ImportRecipeError::ProcessorIdentityMismatch {
                recipe_id: self.importer_id,
                recipe_version: self.importer_version,
                processor_id: P::IMPORTER_ID,
                processor_version: P::IMPORTER_VERSION,
            });
        }
        Ok(ImportPlan {
            job: BuildJob::new(job_id, vec![source_job], processor),
            spec: self.artifact_spec(asset, kind, schema, alignment),
        })
    }

    /// Plans an import by selecting a registered plugin from extension/MIME metadata.
    pub fn plan_from_registry(
        &self,
        registry: &crate::import::ImporterRegistry,
        asset: AssetId,
        job_id: JobId,
        source_job: JobId,
        selector: &crate::import::ImportSelector,
        kind: u128,
        schema: u32,
        alignment: u32,
    ) -> Result<ImportPlan, ImportRecipeError> {
        let importer = registry.select(selector)?;
        if self.importer_id != importer.identity().id
            || self.importer_version != importer.identity().version
        {
            return Err(ImportRecipeError::ProcessorIdentityMismatch {
                recipe_id: self.importer_id,
                recipe_version: self.importer_version,
                processor_id: importer.identity().id,
                processor_version: importer.identity().version,
            });
        }
        Ok(ImportPlan {
            job: BuildJob::new(job_id, vec![source_job], importer),
            spec: self.artifact_spec(asset, kind, schema, alignment),
        })
    }

    /// Returns a recipe whose profile hash covers exactly the selected fields.
    pub fn with_profile_dependencies<I, S>(
        mut self,
        profile: &EffectiveProfile,
        fields: I,
    ) -> Result<Self, ProfileError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.profile_hash = profile.project(fields)?.hash();
        Ok(self)
    }

    /// Returns a recipe whose profile hash covers a registered importer's fields.
    pub fn with_importer_profile(
        self,
        profile: &EffectiveProfile,
        importer: &crate::import::ImporterDescriptor,
    ) -> Result<Self, ProfileError> {
        self.with_profile_dependencies(profile, importer.profile_fields().iter())
    }

    /// Derives a recipe key using a neutral output contract.
    ///
    /// Use [`Self::artifact_key_for_output`] when the concrete artifact kind,
    /// schema, or alignment is known.
    pub fn artifact_key(
        &self,
        protocol_version: u32,
        source: &SourceSnapshot,
        imported_inputs: &[ArtifactKey],
    ) -> Result<ArtifactKey, HashError> {
        let mut ordered_inputs = Vec::with_capacity(imported_inputs.len() + 1);
        ordered_inputs.push(source.artifact_key());
        ordered_inputs.extend_from_slice(imported_inputs);
        self.artifact_spec(AssetId(0), 0, 1, 1)
            .artifact_key(protocol_version, &ordered_inputs)
    }

    /// Derives the artifact key using the concrete output contract.
    pub fn artifact_key_for_output(
        &self,
        protocol_version: u32,
        source: &SourceSnapshot,
        imported_inputs: &[ArtifactKey],
        kind: u128,
        schema: u32,
        alignment: u32,
    ) -> Result<ArtifactKey, HashError> {
        let mut ordered_inputs = Vec::with_capacity(imported_inputs.len() + 1);
        ordered_inputs.push(source.artifact_key());
        ordered_inputs.extend_from_slice(imported_inputs);
        self.artifact_spec(AssetId(0), kind, schema, alignment)
            .artifact_key(protocol_version, &ordered_inputs)
    }

    /// Creates an output spec for a concrete imported artifact.
    #[must_use]
    pub fn artifact_spec(
        &self,
        asset: AssetId,
        kind: u128,
        schema: u32,
        alignment: u32,
    ) -> ArtifactSpec {
        ArtifactSpec {
            asset,
            processor_id: self.importer_id,
            processor_version: self.importer_version,
            canonical_settings: self.canonical_settings.clone(),
            profile_hash: self.profile_hash,
            toolchain_fingerprint: self.toolchain_fingerprint.clone(),
            environment_fingerprint: self.environment_fingerprint.clone(),
            kind,
            schema,
            alignment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestProcessor;

    impl Processor for TestProcessor {
        fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, crate::ProcessorError> {
            Ok(inputs[0].to_vec())
        }
    }

    impl ImportProcessor for TestProcessor {
        const IMPORTER_ID: u128 = 0x54455354;
        const IMPORTER_VERSION: u32 = 7;
    }

    fn recipe() -> ImportRecipe {
        ImportRecipe {
            importer_id: 0x494d504f52544552,
            importer_version: 3,
            canonical_settings: b"{\"srgb\":true}".to_vec(),
            profile_hash: [7; 32],
            toolchain_fingerprint: b"toolchain-v1".to_vec(),
            environment_fingerprint: b"declared-platform".to_vec(),
        }
    }

    #[test]
    fn recipe_key_is_deterministic_and_all_declared_inputs_matter() {
        let source = SourceSnapshot::new("textures/hero.png", b"source".to_vec()).unwrap();
        let first = recipe().artifact_key(1, &source, &[]).unwrap();
        assert_eq!(first, recipe().artifact_key(1, &source, &[]).unwrap());

        let mut changed = recipe();
        changed.canonical_settings = b"{\"srgb\":false}".to_vec();
        assert_ne!(first, changed.artifact_key(1, &source, &[]).unwrap());
        assert_ne!(
            first,
            recipe()
                .artifact_key(1, &source, &[ArtifactKey([9; 32])])
                .unwrap()
        );
    }

    #[test]
    fn output_contract_is_part_of_the_recipe_key() {
        let source = SourceSnapshot::new("textures/hero.png", b"source".to_vec()).unwrap();
        let first = recipe()
            .artifact_key_for_output(1, &source, &[], 9, 1, 4)
            .unwrap();
        let changed_schema = recipe()
            .artifact_key_for_output(1, &source, &[], 9, 2, 4)
            .unwrap();
        let changed_alignment = recipe()
            .artifact_key_for_output(1, &source, &[], 9, 1, 8)
            .unwrap();
        assert_ne!(first, changed_schema);
        assert_ne!(first, changed_alignment);
    }

    #[test]
    fn plan_binds_recipe_identity_and_declares_source_dependency() {
        let recipe = ImportRecipe {
            importer_id: TestProcessor::IMPORTER_ID,
            importer_version: TestProcessor::IMPORTER_VERSION,
            canonical_settings: b"settings".to_vec(),
            profile_hash: [1; 32],
            toolchain_fingerprint: b"toolchain".to_vec(),
            environment_fingerprint: b"environment".to_vec(),
        };
        let plan = recipe
            .plan(AssetId(2), JobId(11), JobId(3), TestProcessor, 9, 1, 4)
            .unwrap();
        assert_eq!(plan.job.id(), JobId(11));
        assert_eq!(plan.job.dependencies(), &[JobId(3)]);
        assert_eq!(plan.spec.asset, AssetId(2));
        assert_eq!(plan.spec.processor_id, TestProcessor::IMPORTER_ID);
        assert_eq!(plan.spec.processor_version, TestProcessor::IMPORTER_VERSION);
    }

    #[test]
    fn plan_rejects_a_recipe_for_a_different_processor() {
        let recipe = ImportRecipe {
            importer_id: 0xBAD,
            importer_version: 1,
            canonical_settings: Vec::new(),
            profile_hash: [0; 32],
            toolchain_fingerprint: Vec::new(),
            environment_fingerprint: Vec::new(),
        };
        assert!(matches!(
            recipe.plan(AssetId(2), JobId(11), JobId(3), TestProcessor, 9, 1, 4),
            Err(ImportRecipeError::ProcessorIdentityMismatch {
                recipe_id: 0xBAD,
                recipe_version: 1,
                processor_id: TestProcessor::IMPORTER_ID,
                processor_version: TestProcessor::IMPORTER_VERSION,
            })
        ));
    }

    #[test]
    fn registry_plan_binds_selected_processor_and_profile_dependencies() {
        let registry = crate::import::ImporterRegistry::with_builtin().unwrap();
        let schema = crate::profile::ProfileSchema::new(
            "test.profile",
            1,
            [(
                "texture.semantic",
                crate::profile::ProfileField::new(
                    crate::profile::ProfileFieldType::String {
                        min_bytes: 1,
                        max_bytes: 16,
                        allowed_values: None,
                    },
                    crate::profile::ProfileMergePolicy::ScalarOverride,
                ),
            )],
        )
        .unwrap();
        let mut layer =
            crate::profile::ProfileLayerValues::new(crate::profile::ProfileLayer::EngineBaseline);
        layer
            .set(
                "texture.semantic",
                crate::profile::ProfileValue::String("color".into()),
            )
            .unwrap();
        let profile = schema.resolve(&[layer]).unwrap();
        let mut recipe = recipe();
        recipe.importer_id = crate::import::CanonicalPngProcessor::IMPORTER_ID;
        recipe.importer_version = crate::import::CanonicalPngProcessor::IMPORTER_VERSION;
        let recipe = recipe
            .with_profile_dependencies(&profile, ["texture.semantic"])
            .unwrap();
        let plan = recipe
            .plan_from_registry(
                &registry,
                AssetId(9),
                JobId(10),
                JobId(11),
                &crate::import::ImportSelector::for_path("textures/hero.PNG", None),
                crate::import::IMAGE_ARTIFACT_KIND,
                2,
                4,
            )
            .unwrap();
        assert_eq!(plan.job.dependencies(), &[JobId(11)]);
        assert_eq!(
            plan.spec.profile_hash,
            profile.project(["texture.semantic"]).unwrap().hash()
        );
    }
}
