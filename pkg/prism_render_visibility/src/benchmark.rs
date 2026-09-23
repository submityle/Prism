use crate::{cull_view, VisibilityDiagnostics, VisibilityInput};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VisibilityBenchmarkResult {
    pub iterations: u32,
    pub input_instances: u64,
    pub output_instances: u64,
    pub elapsed_seconds: f64,
}

impl VisibilityBenchmarkResult {
    pub fn instances_per_second(self) -> f64 {
        if self.elapsed_seconds <= 0.0 {
            0.0
        } else {
            self.input_instances as f64 / self.elapsed_seconds
        }
    }
}

#[cfg(feature = "std")]
pub fn benchmark_cpu_reference(
    view: &crate::GpuViewRecord,
    input: VisibilityInput<'_>,
    iterations: u32,
) -> (VisibilityBenchmarkResult, VisibilityDiagnostics) {
    use std::time::Instant;

    let started = Instant::now();
    let mut diagnostics = VisibilityDiagnostics::default();
    let mut output_instances = 0_u64;
    for _ in 0..iterations {
        let (work, iteration) = cull_view(
            view,
            VisibilityInput {
                scene: input.scene,
                handles: input.handles,
                geometry: input.geometry,
                materials: input.materials,
                previous_lods: input.previous_lods,
                occluded: input.occluded,
                capacity: input.capacity,
                previous_history_epoch: input.previous_history_epoch,
            },
        );
        output_instances += work.len() as u64;
        diagnostics = iteration;
    }
    (
        VisibilityBenchmarkResult {
            iterations,
            input_instances: input.handles.len() as u64 * iterations as u64,
            output_instances,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        },
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_is_guarded_against_zero_duration() {
        assert_eq!(
            VisibilityBenchmarkResult::default().instances_per_second(),
            0.0
        );
        assert_eq!(
            VisibilityBenchmarkResult {
                input_instances: 200,
                elapsed_seconds: 2.0,
                ..Default::default()
            }
            .instances_per_second(),
            100.0
        );
    }
}
