// Precipitation-classification twin: classifies a cell's precipitation phase
// and intensity on-device, mirroring the CPU golden `classify_precip` (in
// `prism_render_architecture::volumetric::weather`).
//
// The weather map's precip (B) channel, the cloud kind and the air temperature
// together decide whether a cell rains, snows or stays dry, and how hard
// (design section 9). The classification is a pure threshold cascade:
//   1. precipitation < PRECIP_TRIGGER  -> dry (None, intensity 0)
//   2. else phase = Snow when temperature <= FREEZING_POINT_C, else Rain
//   3. intensity = saturate(precipitation * gain * max(coverage, 0)), where
//      gain = CUMULONIMBUS_PRECIP_GAIN for the deep-convective storm cloud
//      (kind ordinal 3) and DEFAULT_PRECIP_GAIN otherwise.
// The freezing edge itself is Snow, deterministically. The CPU golden owns that
// logic; this kernel is the on-device twin that runs one thread per query.
//
// The result packs the phase ordinal (0 = None, 1 = Rain, 2 = Snow) and the
// intensity per query. It uses only the portable core-WGSL subset - comparisons
// and clamp - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard threshold-cascade precipitation classification; no
// Unreal Engine source or derived code.

const PRECIP_TRIGGER: f32 = 0.5;
const FREEZING_POINT_C: f32 = 0.0;
const CUMULONIMBUS_PRECIP_GAIN: f32 = 1.0;
const DEFAULT_PRECIP_GAIN: f32 = 0.5;
// CloudKind ordinal for the only precipitation-capable kind (Cumulonimbus).
const CUMULONIMBUS: u32 = 3u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One classification query: the weather-map precip and coverage channels, the
// air temperature (Celsius) and the CloudKind ordinal (Cumulus 0, Stratus 1,
// Cirrus 2, Cumulonimbus 3). Matches `GpuQuery` in `src/classify_precip.rs`.
struct Query {
    precipitation: f32,
    coverage: f32,
    temperature: f32,
    kind: u32,
}

// One result: the phase ordinal plus the intensity. Matches `GpuResult` in
// `src/classify_precip.rs`.
struct Res {
    kind: u32,
    intensity: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// The classification, mirroring `classify_precip`.
fn classify_precip(precipitation: f32, coverage: f32, temperature: f32, kind: u32) -> Res {
    if (precipitation < PRECIP_TRIGGER) {
        return Res(0u, 0.0);
    }
    let gain = select(DEFAULT_PRECIP_GAIN, CUMULONIMBUS_PRECIP_GAIN, kind == CUMULONIMBUS);
    let intensity = saturate_s(precipitation * gain * max(coverage, 0.0));
    // Snow at or below freezing, Rain above it.
    let phase = select(1u, 2u, temperature <= FREEZING_POINT_C);
    return Res(phase, intensity);
}

@compute @workgroup_size(64)
fn classify_precip_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = classify_precip(q.precipitation, q.coverage, q.temperature, q.kind);
}
