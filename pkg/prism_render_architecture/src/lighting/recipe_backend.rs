//! Executable backends for recipes that are independent of the frame host.
//!
//! This module owns the first reusable direct-light recipe backends. They
//! consume the existing deterministic `MegaLights`/RIS primitives and return
//! a numerically testable result. The frame host may later bind the same work
//! record to a GPU pass; this layer deliberately does not own GPU resources or
//! graph scheduling.

use super::contract::{RecipeCost, RecipeId};
use super::culling::ClusterBounds;
use super::restir_gi::{stream_initial as stream_initial_gi, GiCandidate, GiReservoir};
use super::stochastic::{select_tile_lights, MegaLight, StochasticTile};
use super::ReservoirBudget;
use crate::particle::reservoir_sample::Rng;

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

/// Output of one initial WorldReSTIR-GI reservoir execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRestirOutput {
    /// The finalized path-sample reservoir, retained for temporal/spatial reuse.
    pub reservoir: GiReservoir,
    /// Scene-linear indirect outgoing radiance represented by this reservoir.
    pub radiance: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecipeBackendError {
    ContributionCountMismatch,
    ShadowSlotCountMismatch,
    ShadowSlotsRequired,
    ShadowSlotsForbidden,
    InvalidGiCandidate { index: usize },
    UnsupportedRecipe,
}

/// The concrete backend registrations available to the architecture planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecipeBackendRegistration {
    pub recipe: RecipeId,
    pub cost: RecipeCost,
    pub deterministic_twin: bool,
}

const REGISTRATIONS: [RecipeBackendRegistration; 4] = [
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
    RecipeBackendRegistration {
        recipe: RecipeId::SpecializedDeterministic,
        cost: RecipeId::SpecializedDeterministic.cost(),
        deterministic_twin: true,
    },
    RecipeBackendRegistration {
        recipe: RecipeId::WorldRestir,
        cost: RecipeId::WorldRestir.cost(),
        deterministic_twin: true,
    },
];

/// Returns every registered recipe/provider pair.
///
/// The planner and diagnostics must consume this table instead of keeping a
/// second list of executable recipes. A registration is only executable when
/// it also advertises the deterministic twin required by the architecture
/// contract.
#[must_use]
pub const fn registrations() -> &'static [RecipeBackendRegistration] {
    &REGISTRATIONS
}

/// Returns the concrete backend registration for a recipe, if one exists.
#[must_use]
pub const fn registration(recipe: RecipeId) -> Option<RecipeBackendRegistration> {
    match recipe {
        RecipeId::DirectStochasticRt => Some(REGISTRATIONS[0]),
        RecipeId::DirectStochasticVsm => Some(REGISTRATIONS[1]),
        RecipeId::SpecializedDeterministic => Some(REGISTRATIONS[2]),
        RecipeId::WorldRestir => Some(REGISTRATIONS[3]),
        _ => None,
    }
}

/// Whether a recipe has a registered executable backend and deterministic twin.
#[must_use]
pub const fn is_registered(recipe: RecipeId) -> bool {
    match registration(recipe) {
        Some(registration) => registration.deterministic_twin,
        None => false,
    }
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
        RecipeId::SpecializedDeterministic => {
            let exact_work = DirectRisWork {
                sample_budget: u32::MAX,
                ..work
            };
            execute_ris(exact_work, work.shadow_slots)
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

/// Executes the initial WorldReSTIR-GI candidate stream on the CPU.
/// The registered provider supplies the deterministic reference used for
/// parity and replay. Temporal/spatial reuse remains a subsequent operation.
pub fn execute_world_restir(
    recipe: RecipeId,
    candidates: &[GiCandidate],
    budget: ReservoirBudget,
    seed: u32,
) -> Result<WorldRestirOutput, RecipeBackendError> {
    if recipe != RecipeId::WorldRestir {
        return Err(RecipeBackendError::UnsupportedRecipe);
    }
    let candidate_count = candidates
        .len()
        .min(usize::from(budget.initial_candidates));
    for (index, candidate) in candidates[..candidate_count].iter().enumerate() {
        let sample = candidate.sample;
        let finite = candidate.target_pdf.is_finite()
            && candidate.source_pdf.is_finite()
            && sample.sample_point.iter().all(|value| value.is_finite())
            && sample.sample_normal.iter().all(|value| value.is_finite())
            && sample.radiance.iter().all(|value| value.is_finite());
        let nonnegative = candidate.target_pdf >= 0.0
            && candidate.source_pdf >= 0.0
            && sample.radiance.iter().all(|value| *value >= 0.0);
        if !finite || !nonnegative {
            return Err(RecipeBackendError::InvalidGiCandidate { index });
        }
    }
    let mut rng = Rng::new(seed);
    let mut reservoir = stream_initial_gi(candidates, budget, &mut rng);
    reservoir.finalize();
    let radiance = if reservoir.is_empty() {
        [0.0; 3]
    } else {
        [
            reservoir.sample.radiance[0] * reservoir.w,
            reservoir.sample.radiance[1] * reservoir.w,
            reservoir.sample.radiance[2] * reservoir.w,
        ]
    };
    Ok(WorldRestirOutput {
        reservoir,
        radiance,
    })
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
    fn registrations_cover_executable_direct_recipe_backends() {
        assert_eq!(registrations().len(), 4);
        assert!(is_registered(RecipeId::DirectStochasticRt));
        assert!(is_registered(RecipeId::DirectStochasticVsm));
        assert!(!is_registered(RecipeId::DirectDeterministicCsm));
        assert!(is_registered(RecipeId::SpecializedDeterministic));
        assert!(is_registered(RecipeId::WorldRestir));
        assert!(
            registration(RecipeId::DirectStochasticRt)
                .expect("registered RT recipe")
                .deterministic_twin
        );
        for registration in registrations() {
            assert_eq!(registration.cost, registration.recipe.cost());
            assert!(registration.deterministic_twin);
            assert!(RecipeId::ALL.contains(&registration.recipe));
        }
    }

    #[test]
    fn registration_table_is_the_single_recipe_execution_whitelist() {
        let mut registered_from_all = Vec::new();
        for recipe in RecipeId::ALL {
            let registration = registration(recipe);
            assert_eq!(recipe.has_execution_backend(), registration.is_some());
            if let Some(registration) = registration {
                assert_eq!(registration.recipe, recipe);
                assert_eq!(registration.cost, recipe.cost());
                assert!(registration.deterministic_twin);
                registered_from_all.push(recipe);
            }
        }

        assert_eq!(
            registered_from_all,
            vec![
                RecipeId::DirectStochasticRt,
                RecipeId::DirectStochasticVsm,
                RecipeId::SpecializedDeterministic,
                RecipeId::WorldRestir,
            ]
        );
        assert_eq!(
            registrations()
                .iter()
                .map(|entry| entry.recipe)
                .collect::<Vec<_>>(),
            registered_from_all
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
    fn specialized_deterministic_backend_always_evaluates_every_overlapping_light() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut work = work(&lights, &contributions, None, 0);
        work.cluster_id = 99;
        let output = execute(RecipeId::SpecializedDeterministic, work)
            .expect("the exact direct backend is executable");
        assert_eq!(output.value, [5.0, 7.0, 9.0]);
        assert!(output.selected.exact);
        assert_eq!(output.selected.overlap_count, 2);
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

    #[test]
    fn unregistered_recipe_cannot_enter_the_execution_function() {
        let lights = lights();
        let contributions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let error = execute(
            RecipeId::WorldRestir,
            work(&lights, &contributions, None, 1),
        )
        .expect_err("an unregistered recipe must not have a pseudo execution path");
        assert_eq!(error, RecipeBackendError::UnsupportedRecipe);
    }

    #[test]
    fn world_restir_registration_executes_gi_reservoir_and_is_deterministic() {
        let candidates = [
            GiCandidate {
                sample: super::super::restir_gi::GiSample {
                    sample_point: [0.0, 0.0, 1.0],
                    sample_normal: [0.0, 0.0, -1.0],
                    radiance: [2.0, 1.0, 0.5],
                },
                target_pdf: 1.0,
                source_pdf: 1.0,
            },
            GiCandidate {
                sample: super::super::restir_gi::GiSample {
                    sample_point: [1.0, 0.0, 1.0],
                    sample_normal: [0.0, 0.0, -1.0],
                    radiance: [4.0, 2.0, 1.0],
                },
                target_pdf: 2.0,
                source_pdf: 1.0,
            },
        ];
        let budget = ReservoirBudget {
            initial_candidates: 2,
            ..ReservoirBudget::default()
        };
        let first = execute_world_restir(RecipeId::WorldRestir, &candidates, budget, 19)
            .expect("registered WorldRestir provider");
        let second = execute_world_restir(RecipeId::WorldRestir, &candidates, budget, 19)
            .expect("deterministic twin");
        assert_eq!(first, second);
        assert_eq!(first.reservoir.m, 2);
        assert!((first.reservoir.w - 0.75).abs() < 1.0e-6);
        assert_eq!(
            first.radiance,
            [
                first.reservoir.sample.radiance[0] * first.reservoir.w,
                first.reservoir.sample.radiance[1] * first.reservoir.w,
                first.reservoir.sample.radiance[2] * first.reservoir.w,
            ]
        );
    }

    #[test]
    fn world_restir_rejects_non_finite_or_negative_candidates_before_resampling() {
        let candidate = GiCandidate {
            sample: super::super::restir_gi::GiSample {
                sample_point: [0.0, 0.0, 1.0],
                sample_normal: [0.0, 0.0, -1.0],
                radiance: [f32::NAN, 1.0, 1.0],
            },
            target_pdf: 1.0,
            source_pdf: 1.0,
        };
        let error = execute_world_restir(
            RecipeId::WorldRestir,
            &[candidate],
            ReservoirBudget {
                initial_candidates: 1,
                ..ReservoirBudget::default()
            },
            0,
        )
        .expect_err("non-finite radiance must not poison the reservoir");
        assert_eq!(error, RecipeBackendError::InvalidGiCandidate { index: 0 });
    }

    #[test]
    fn world_restir_does_not_inspect_candidates_outside_the_budget() {
        let invalid_unused = GiCandidate {
            sample: super::super::restir_gi::GiSample {
                sample_point: [0.0, 0.0, 1.0],
                sample_normal: [0.0, 0.0, 1.0],
                radiance: [-1.0, 0.0, 0.0],
            },
            target_pdf: 1.0,
            source_pdf: 1.0,
        };
        let output = execute_world_restir(
            RecipeId::WorldRestir,
            &[invalid_unused],
            ReservoirBudget::default(),
            0,
        )
        .expect("unconsumed candidate is outside this execution");
        assert!(output.reservoir.is_empty());
        assert_eq!(output.radiance, [0.0; 3]);
    }
}
