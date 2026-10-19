//! Executable backends for recipes that are independent of the frame host.
//!
//! This module owns the first reusable direct-light recipe backends. They
//! consume the existing deterministic `MegaLights`/RIS primitives and return
//! a numerically testable result. The frame host may later bind the same work
//! record to a GPU pass; this layer deliberately does not own GPU resources or
//! graph scheduling.

use super::contract::{RecipeCost, RecipeId};
use super::culling::ClusterBounds;
use super::stochastic::{select_tile_lights, MegaLight, StochasticTile};

/// A dense visibility table produced by a virtual-shadow implementation.
///
/// Entry `light_index` is the visibility multiplier in `[0, 1]`. The table is
/// intentionally separate from RIS selection: shadow representation changes
/// the evaluated contribution, while the light-selection distribution remains
/// the same deterministic MegaLights distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSlotTable<'a> {
    pub visibility: &'a [f32],
}

impl<'a> ShadowSlotTable<'a> {
    #[must_use]
    pub const fn new(visibility: &'a [f32]) -> Self {
        Self { visibility }
    }

    #[must_use]
    pub fn for_light(self, light_index: u32) -> f32 {
        self.visibility
            .get(light_index as usize)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    }
}

/// One work item shared by the direct RIS backends and their deterministic
/// twin. `contributions` contains the unshadowed RGB contribution for each
/// light at the current cluster/surface.
#[derive(Clone, Copy, Debug)]
pub struct DirectRisWork<'a> {
    pub cluster: ClusterBounds,
    pub lights: &'a [MegaLight],
    pub contributions: &'a [[f32; 3]],
    pub shadow_slots: Option<ShadowSlotTable<'a>>,
    pub sample_budget: u32,
    pub seed: u64,
    pub cluster_id: u32,
}

/// Output of one direct RIS recipe execution.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectRisOutput {
    pub value: [f32; 3],
    pub selected: StochasticTile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecipeBackendError {
    ContributionCountMismatch,
    ShadowSlotCountMismatch,
    ShadowSlotsRequired,
    ShadowSlotsForbidden,
    UnsupportedRecipe,
}

/// The concrete backend registrations available to the architecture planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecipeBackendRegistration {
    pub recipe: RecipeId,
    pub cost: RecipeCost,
    pub deterministic_twin: bool,
}

const REGISTRATIONS: [RecipeBackendRegistration; 2] = [
    RecipeBackendRegistration {
        recipe: RecipeId::DirectStochasticRt,
        cost: RecipeId::DirectStochasticRt.cost(),
        deterministic_twin: true,
    },
    RecipeBackendRegistration {
        recipe: RecipeId::DirectStochasticVsm,
        cost: RecipeId::DirectStochasticVsm.cost(),
        deterministic_twin: true,
    },
];

/// Returns the concrete backend registration for a recipe, if one exists.
#[must_use]
pub const fn registration(recipe: RecipeId) -> Option<RecipeBackendRegistration> {
    match recipe {
        RecipeId::DirectStochasticRt => Some(REGISTRATIONS[0]),
        RecipeId::DirectStochasticVsm => Some(REGISTRATIONS[1]),
        _ => None,
    }
}

/// Whether a recipe has a registered executable backend and deterministic twin.
#[must_use]
pub const fn is_registered(recipe: RecipeId) -> bool {
    matches!(
        recipe,
        RecipeId::DirectStochasticRt | RecipeId::DirectStochasticVsm
    )
}

/// Executes the registered direct RIS backend.
pub fn execute(
    recipe: RecipeId,
    work: DirectRisWork<'_>,
) -> Result<DirectRisOutput, RecipeBackendError> {
    match recipe {
        RecipeId::DirectStochasticRt => execute_ris(work, None),
        RecipeId::DirectStochasticVsm => {
            let Some(slots) = work.shadow_slots else {
                return Err(RecipeBackendError::ShadowSlotsRequired);
            };
            execute_ris(work, Some(slots))
        }
        _ => Err(RecipeBackendError::UnsupportedRecipe),
    }
}

/// Deterministic CPU twin used by replay and backend parity tests.
///
/// This is the same recipe contract as [`execute`], exposed explicitly so a
/// backend test can compare the registered execution path with the stable
/// architecture-side twin without depending on a render device.
#[must_use]
pub fn deterministic_twin(
    recipe: RecipeId,
    work: DirectRisWork<'_>,
) -> Result<DirectRisOutput, RecipeBackendError> {
    execute(recipe, work)
}

fn execute_ris(
    work: DirectRisWork<'_>,
    shadow_slots: Option<ShadowSlotTable<'_>>,
) -> Result<DirectRisOutput, RecipeBackendError> {
    if work.contributions.len() != work.lights.len() {
        return Err(RecipeBackendError::ContributionCountMismatch);
    }
    if let Some(slots) = shadow_slots {
        if slots.visibility.len() != work.lights.len() {
            return Err(RecipeBackendError::ShadowSlotCountMismatch);
        }
    } else if work.shadow_slots.is_some() {
        return Err(RecipeBackendError::ShadowSlotsForbidden);
    }

    let selected = select_tile_lights(
        work.cluster,
        work.lights,
        work.sample_budget,
        work.seed,
        work.cluster_id,
    );
    let mut value = [0.0; 3];
    for sample in &selected.samples {
        let contribution = work.contributions[sample.light_index as usize];
        let visibility = shadow_slots
            .map(|slots| slots.for_light(sample.light_index))
            .unwrap_or(1.0);
        for channel in 0..3 {
            value[channel] += contribution[channel] * visibility * sample.weight;
        }
    }
    Ok(DirectRisOutput { value, selected })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::{culling::LightVolume, LightHandle};

    fn work<'a>(
        lights: &'a [MegaLight],
        contributions: &'a [[f32; 3]],
        shadow_slots: Option<ShadowSlotTable<'a>>,
        sample_budget: u32,
    ) -> DirectRisWork<'a> {
        DirectRisWork {
            cluster: ClusterBounds {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            lights,
            contributions,
            shadow_slots,
            sample_budget,
            seed: 17,
            cluster_id: 3,
        }
    }

    fn lights() -> [MegaLight; 2] {
        [
            MegaLight {
                volume: LightVolume {
                    light: LightHandle(0),
                    center: [0.0, 0.0, 0.0],
                    radius: 2.0,
                },
                power: 2.0,
            },
            MegaLight {
                volume: LightVolume {
                    light: LightHandle(1),
                    center: [0.5, 0.0, 0.0],
                    radius: 2.0,
                },
                power: 1.0,
            },
        ]
    }

    #[test]
    fn registrations_cover_the_first_direct_ris_pair() {
        assert!(is_registered(RecipeId::DirectStochasticRt));
        assert!(is_registered(RecipeId::DirectStochasticVsm));
        assert!(!is_registered(RecipeId::DirectDeterministicCsm));
        assert!(
            registration(RecipeId::DirectStochasticRt)
                .expect("registered RT recipe")
                .deterministic_twin
        );
    }

    #[test]
    fn exact_budget_matches_the_full_many_light_sum() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let output = execute(
            RecipeId::DirectStochasticRt,
            work(&lights, &contributions, None, 2),
        )
        .expect("RIS execution");
        assert_eq!(output.value, [5.0, 7.0, 9.0]);
        assert!(output.selected.exact);
    }

    #[test]
    fn shadow_slots_modulate_the_same_ris_selection() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let visibility = [0.5, 0.25];
        let output = execute(
            RecipeId::DirectStochasticVsm,
            work(
                &lights,
                &contributions,
                Some(ShadowSlotTable::new(&visibility)),
                2,
            ),
        )
        .expect("VSM execution");
        assert_eq!(output.value, [1.5, 2.25, 3.0]);
    }

    #[test]
    fn backend_and_deterministic_twin_are_bitwise_stable() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let first = execute(
            RecipeId::DirectStochasticRt,
            work(&lights, &contributions, None, 1),
        )
        .expect("RIS execution");
        let second = deterministic_twin(
            RecipeId::DirectStochasticRt,
            work(&lights, &contributions, None, 1),
        )
        .expect("RIS twin");
        assert_eq!(first, second);
        assert_eq!(
            first,
            execute(
                RecipeId::DirectStochasticRt,
                work(&lights, &contributions, None, 1),
            )
            .expect("repeat")
        );
    }

    #[test]
    fn malformed_shadow_table_is_rejected() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let visibility = [1.0];
        let error = execute(
            RecipeId::DirectStochasticVsm,
            work(
                &lights,
                &contributions,
                Some(ShadowSlotTable::new(&visibility)),
                2,
            ),
        )
        .expect_err("short shadow table must fail");
        assert_eq!(error, RecipeBackendError::ShadowSlotCountMismatch);
    }
}
