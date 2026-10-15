#![allow(missing_docs)]

#[path = "../tests/support/response_composition_workload.rs"]
mod workload;

use std::{env, hint::black_box, time::Instant};

use prism_render_material::{BlendDomain, MaterialRenderClass};
use prism_render_shading::{
    blend_responses, evaluate_response_pair, AlphaMergeRule, Auxiliary, AuxiliaryMergeRule,
    AuxiliaryMergeRules, AuxiliaryValue, CompositionOptions, LightingContext, ResponseOutput,
    SharedSurfaceAlpha,
};
use workload::{build_plan, continuous_item_count, plan_digest, standard_workload, WorkloadLedger};

const DEFAULT_SAMPLES: usize = 31;

fn percentile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    let index = ((sorted.len() - 1) * numerator + denominator / 2) / denominator;
    sorted[index]
}

fn timed_samples(samples: usize, mut operation: impl FnMut()) -> Vec<u128> {
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        operation();
        timings.push(start.elapsed().as_nanos());
    }
    timings.sort_unstable();
    timings
}

fn main() {
    let samples = env::args()
        .skip_while(|arg| arg != "--samples")
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SAMPLES);
    assert!(samples >= 3, "--samples must be at least 3");

    let workload = standard_workload();
    let ledger = WorkloadLedger::for_workload(&workload);
    let warm_plan = build_plan(&workload);
    let expected_digest = plan_digest(&warm_plan);
    let continuous_count = continuous_item_count(&warm_plan);

    let plan_timings = timed_samples(samples, || {
        let plan = build_plan(black_box(&workload));
        assert_eq!(plan_digest(&plan), expected_digest);
        black_box(plan);
    });

    let mut left_auxiliary = Auxiliary::default();
    left_auxiliary
        .fields
        .insert("reactive_mask".into(), AuxiliaryValue::Scalar(0.2));
    let mut right_auxiliary = Auxiliary::default();
    right_auxiliary
        .fields
        .insert("reactive_mask".into(), AuxiliaryValue::Scalar(0.8));
    let left = ResponseOutput {
        direct: [0.4, 0.3, 0.2],
        indirect_diffuse: [0.2, 0.3, 0.4],
        reflection: [0.8, 0.7, 0.6],
        transmission: [0.1, 0.2, 0.3],
        emission: [0.05, 0.04, 0.03],
        alpha: 0.75,
        auxiliary: left_auxiliary,
    };
    let right = ResponseOutput {
        direct: [0.1, 0.2, 0.3],
        indirect_diffuse: [0.5, 0.4, 0.3],
        reflection: [0.2, 0.3, 0.4],
        transmission: [0.4, 0.3, 0.2],
        emission: [0.01, 0.02, 0.03],
        alpha: 0.65,
        auxiliary: right_auxiliary,
    };
    let lighting = LightingContext {
        direct: &[],
        indirect_diffuse_irradiance: None,
        reflection_radiance: None,
        transmission_radiance: None,
        shared_terms: &(),
    };
    let (left, right) = evaluate_response_pair(&lighting, |_| left.clone(), |_| right.clone())
        .expect("fixed response outputs must validate");
    let mut auxiliary_rules = AuxiliaryMergeRules::default();
    auxiliary_rules
        .fields
        .insert("reactive_mask".into(), AuxiliaryMergeRule::Linear);
    let masks: Vec<f32> = warm_plan
        .work
        .iter()
        .filter(|item| item.right_shading_class != u32::MAX)
        .map(|item| (item.response_meta & u16::MAX as u32) as f32 / u16::MAX as f32)
        .collect();
    let composition_timings = timed_samples(samples, || {
        let mut checksum = 0.0_f32;
        for mask in &masks {
            let output = blend_responses(
                black_box(&left),
                black_box(&right),
                CompositionOptions {
                    mask: *mask,
                    domain: BlendDomain::DirectOnly,
                    shared_alpha: SharedSurfaceAlpha::new(0.7, MaterialRenderClass::Opaque)
                        .unwrap(),
                    alpha_rule: AlphaMergeRule::Shared,
                    auxiliary_rules: &auxiliary_rules,
                },
            )
            .unwrap();
            checksum += output.direct[0];
        }
        black_box(checksum);
    });

    println!("{{");
    println!("  \"mode\": \"cpu\",");
    println!("  \"gpu_adapter\": null,");
    println!("  \"workload\": \"{}\",", workload.name);
    println!("  \"width\": {},", workload.width);
    println!("  \"height\": {},", workload.height);
    println!("  \"workload_pixels\": {},", ledger.workload_pixels);
    println!("  \"continuous_items\": {},", continuous_count);
    println!("  \"bucket_count\": {},", ledger.bucket_count);
    println!("  \"auxiliary_jobs\": {},", ledger.auxiliary_jobs);
    println!(
        "  \"plan_build_ns\": {{\"p50\": {}, \"p95\": {}, \"p99\": {}}},",
        percentile(&plan_timings, 50, 100),
        percentile(&plan_timings, 95, 100),
        percentile(&plan_timings, 99, 100)
    );
    println!(
        "  \"cpu_composition_ns\": {{\"p50\": {}, \"p95\": {}, \"p99\": {}}},",
        percentile(&composition_timings, 50, 100),
        percentile(&composition_timings, 95, 100),
        percentile(&composition_timings, 99, 100)
    );
    println!(
        "  \"composition_payload_bytes\": {},",
        ledger.composition_payload_bytes
    );
    println!(
        "  \"classification_payload_bytes\": {},",
        ledger.classification_payload_bytes
    );
    println!(
        "  \"known_gpu_payload_bytes\": {},",
        ledger.known_gpu_payload_bytes
    );
    println!(
        "  \"cpu_input_payload_bytes\": {},",
        ledger.cpu_input_payload_bytes
    );
    println!(
        "  \"work_item_payload_bytes\": {}",
        ledger.work_item_payload_bytes
    );
    println!("}}");
}
