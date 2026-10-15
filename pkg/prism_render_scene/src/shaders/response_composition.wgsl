// Five consecutive RGB channels and alpha. No vec3 padding in the 64-byte ABI.
struct ResponseOutput { words: array<f32, 16> }
struct CompositionControl {
    mask: f32,
    channels: u32,
    shared_alpha: f32,
    alpha_rule: u32,
}
struct AuxiliaryMerge {
    left: vec4<u32>,
    right: vec4<u32>,
    shared_value: vec4<u32>,
    kind: u32,
    rule: u32,
    mask: f32,
    reserved: u32,
}
@group(0) @binding(0) var<storage, read_write> left: array<ResponseOutput>;
@group(0) @binding(1) var<storage, read> right: array<ResponseOutput>;
@group(0) @binding(2) var<storage, read> controls: array<CompositionControl>;
@group(0) @binding(3) var<storage, read> water_coverage: array<u32>;

fn linear(a: f32, b: f32, mask: f32) -> f32 {
    if mask == 0.0 { return a; }
    if mask == 1.0 { return b; }
    return (1.0 - mask) * a + mask * b;
}

@compute @workgroup_size(64)
fn compose_responses(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&controls) || i >= arrayLength(&left) || i >= arrayLength(&right)
        || i >= arrayLength(&water_coverage) { return; }
    let c = controls[i];
    // Neutral controls plus zero coverage are the ordinary single-arm path.
    // Skipping here keeps the resident compositor from rewriting untouched
    // pixels while making the water producer's coverage sidecar an actual
    // input to the composition contract.
    if c.channels == 0u && water_coverage[i] == 0u { return; }
    var out = left[i];
    let b = right[i];
    for (var channel = 0u; channel < 5u; channel++) {
        if (c.channels & (1u << channel)) != 0u {
            for (var component = 0u; component < 3u; component++) {
                let word = 3u * channel + component;
                out.words[word] = linear(out.words[word], b.words[word], c.mask);
            }
        }
    }
    switch c.alpha_rule {
        case 0u: { out.words[15] = c.shared_alpha; }
        case 1u: {}
        case 2u: { out.words[15] = b.words[15]; }
        case 3u: { out.words[15] = min(out.words[15], b.words[15]); }
        case 4u: { out.words[15] = max(out.words[15], b.words[15]); }
        default: { return; }
    }
    left[i] = out;
}

// Auxiliary has its own entry-point-specific auto layout. Keep it at binding 4
// so the radiance entry can reserve binding 3 for the water coverage sidecar
// without creating a cross-entry binding collision.
@group(0) @binding(4) var<storage, read_write> auxiliary: array<AuxiliaryMerge>;

@compute @workgroup_size(64)
fn compose_auxiliary(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&auxiliary) { return; }
    let job = auxiliary[i];
    var out = job.left;
    switch job.rule {
        case 0u: { out = job.shared_value; }
        case 1u, 3u: {}
        case 2u: { out = job.right; }
        case 7u: { out.x = job.left.x | job.right.x; }
        default: {
            let count = select(1u, 3u, job.kind == 1u);
            for (var j = 0u; j < count; j++) {
                if job.kind <= 1u {
                    let a = bitcast<f32>(job.left[j]);
                    let b = bitcast<f32>(job.right[j]);
                    var result = linear(a, b, job.mask);
                    if job.rule == 5u { result = min(a, b); }
                    if job.rule == 6u { result = max(a, b); }
                    out[j] = bitcast<u32>(result);
                } else {
                    if job.rule == 5u { out[j] = min(job.left[j], job.right[j]); }
                    if job.rule == 6u { out[j] = max(job.left[j], job.right[j]); }
                }
            }
        }
    }
    auxiliary[i].left = out;
}
