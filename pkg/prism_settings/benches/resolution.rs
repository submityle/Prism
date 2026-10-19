//! Release-mode resolver benchmark at the design's 10k-definition/9-layer scale.

use std::{hint::black_box, time::Instant};

use prism_settings::{
    Domain, DomainMask, LayerKind, LayerSource, PrecedenceProfile, RegistryBuilder, Resolver,
    SettingDef, SettingKey, TrustLevel, Value,
};

const DEFINITIONS: usize = 10_000;
const SAMPLES: usize = 1_000;

fn main() {
    let mut builder = RegistryBuilder::new();
    builder
        .namespace("render", "prism_settings_bench")
        .expect("unique benchmark namespace");
    for index in 0..DEFINITIONS {
        let key = format!("render.setting_{index:05}");
        let mut def = SettingDef::new(key, "render", Value::Int(index as i64));
        def.domains = DomainMask::RUNTIME;
        def.precedence = PrecedenceProfile::UserChoiceWins;
        builder.register(def);
    }
    let registry = builder.finish().expect("valid benchmark registry");
    let resolver = Resolver::new(&registry);
    let layers = [
        LayerKind::Project,
        LayerKind::Platform,
        LayerKind::Recommended,
        LayerKind::Remote,
        LayerKind::User,
        LayerKind::Device,
        LayerKind::Session,
        LayerKind::Runtime,
        LayerKind::Mandatory,
    ];
    let sources = layers
        .into_iter()
        .enumerate()
        .map(|(layer_index, layer)| {
            let trust = if layer == LayerKind::Mandatory {
                TrustLevel::Trusted
            } else if layer == LayerKind::Project {
                TrustLevel::Project
            } else {
                TrustLevel::User
            };
            let mut source = LayerSource::new(layer, trust, format!("bench-{layer_index}"));
            for index in 0..DEFINITIONS {
                source.set(
                    SettingKey::from(format!("render.setting_{index:05}")),
                    Value::Int((index + layer_index) as i64),
                );
            }
            source
        })
        .collect::<Vec<_>>();

    for generation in 0..100 {
        black_box(
            resolver
                .resolve(Domain::Runtime, &sources, generation)
                .expect("benchmark inputs resolve"),
        );
    }

    let mut elapsed_ns = Vec::with_capacity(SAMPLES);
    for generation in 100..100 + SAMPLES as u64 {
        let start = Instant::now();
        let snapshot = resolver
            .resolve(Domain::Runtime, &sources, generation)
            .expect("benchmark inputs resolve");
        black_box(snapshot);
        elapsed_ns.push(start.elapsed().as_nanos());
    }
    elapsed_ns.sort_unstable();

    let percentile = |percent: usize| {
        let index = (elapsed_ns.len() - 1) * percent / 100;
        elapsed_ns[index]
    };
    println!(
        "resolve: definitions={DEFINITIONS}, source_layers={}, samples={SAMPLES}, p50={}us, p95={}us, p99={}us, budget=5000us",
        layers.len(),
        percentile(50) / 1_000,
        percentile(95) / 1_000,
        percentile(99) / 1_000,
    );
}
