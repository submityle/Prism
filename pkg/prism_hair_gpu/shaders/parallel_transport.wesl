// Parallel-transport material-frame twin: builds on-device the same
// torsion-free (Bishop) frame chain the CPU golden
// `parallel_transport_frames` computes, one thread per strand.
//
// A material frame is transported along a strand poly-line so it never twists:
// the first segment's frame rotates its local +z (d3) onto the first edge, and
// each later frame is the previous frame rotated by the minimal rotation
// carrying the previous edge direction onto the current one
// (`Quat::from_min_rotation`, a pure half-vector cross/dot construction with no
// trigonometric call). This frame chain seeds Cosserat rest Darboux vectors,
// ribbon/card expansion and anisotropic hair shading, so it must match the
// reference term for term.
//
// The recurrence is strictly serial along a strand (frame[i] depends on
// frame[i-1]), so this is NOT a per-edge embarrassingly-parallel dispatch like
// the strand-tangent twin: one thread owns a whole strand and walks its edge
// list in order, exactly mirroring the scalar reference loop. Independent
// strands are the parallel axis, matching the per-strand Cosserat solver twin.
//
// Every reference branch is reproduced: a strand of fewer than two points emits
// nothing; a degenerate (zero-length) edge reuses the running frame without
// advancing `prev_dir`; the first non-degenerate edge seeds from the frame's
// transported +z axis while later edges chain from the previous edge direction;
// `from_min_rotation` falls back to a 180-degree rotation about an orthogonal
// axis when the two directions are near-antiparallel, and to identity when
// either input is degenerate. The host supplies finite inputs, so the kernel
// omits the reference `is_finite` guards (core WGSL has no `isFinite`) but keeps
// the `EPS_LEN_SQ` zero-length thresholds verbatim.
//
// Quaternions travel through the buffers as `vec4<f32>` with lane order
// (w, x, y, z); points travel as `vec4<f32>` with the position in `xyz`.
//
// It uses only the portable core-WGSL subset (dot / cross / sqrt / min / max),
// no transcendental and no optional feature, so the twin runs unmodified on
// Metal, Vulkan and DX12. It is not bit-exact with the scalar reference: a GPU
// may fuse a multiply-add the reference leaves separate, and the per-step
// re-normalize compounds that over the chain, so the parity test asserts a
// per-component tolerance rather than exact equality.
//
// Provenance: standard rotation-minimizing / parallel-transport frame (Bishop
// frame) construction; no Unreal Engine source or derived code.

const EPS_LEN_SQ: f32 = 1.0e-24;
const ANTIPARALLEL_DOT: f32 = -0.999999;

struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened control points, position in xyz (w unused).
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
// Per strand: (point_offset, point_count, frame_offset, pad).
@group(0) @binding(2) var<storage, read> strands: array<vec4<u32>>;
// Per strand initial quaternion (w, x, y, z).
@group(0) @binding(3) var<storage, read> initials: array<vec4<f32>>;
// Output frames (w, x, y, z), frame_offset + edge index.
@group(0) @binding(4) var<storage, read_write> frames_out: array<vec4<f32>>;

const QUAT_IDENTITY: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);

// Hamilton product a * b, both (w, x, y, z); matches the reference `Quat::mul`.
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let aw = a.x;
    let ax = a.y;
    let ay = a.z;
    let az = a.w;
    let bw = b.x;
    let bx = b.y;
    let by = b.z;
    let bz = b.w;
    return vec4<f32>(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    );
}

fn qconj(a: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x, -a.y, -a.z, -a.w);
}

// Unit quaternion along `q`, falling back to identity below EPS_LEN_SQ.
fn qnormalize(q: vec4<f32>) -> vec4<f32> {
    let n2 = dot(q, q);
    if n2 > EPS_LEN_SQ {
        return q * (1.0 / sqrt(n2));
    }
    return QUAT_IDENTITY;
}

// Rotates vector `v` by unit quaternion `q` via q * (0,v) * conj(q).
fn qrotate_vec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let p = vec4<f32>(0.0, v.x, v.y, v.z);
    let r = qmul(qmul(q, p), qconj(q));
    return vec3<f32>(r.y, r.z, r.w);
}

// Unit vector along `v`, or zero below EPS_LEN_SQ.
fn vnormalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > EPS_LEN_SQ {
        return v * (1.0 / sqrt(len2));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Unit vector orthogonal to unit `n`, matching the reference `orthonormal`.
fn orthonormal(n: vec3<f32>) -> vec3<f32> {
    var base = vec3<f32>(1.0, 0.0, 0.0);
    if abs(n.x) >= 0.9 {
        base = vec3<f32>(0.0, 1.0, 0.0);
    }
    return vnormalize_or_zero(cross(n, base));
}

// Minimal rotation taking unit-ish `a` onto unit-ish `b` via the half-vector
// identity, mirroring the reference `Quat::from_min_rotation` branch for branch.
fn from_min_rotation(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    let na = vnormalize_or_zero(a);
    let nb = vnormalize_or_zero(b);
    if dot(na, na) < 0.5 || dot(nb, nb) < 0.5 {
        return QUAT_IDENTITY;
    }
    let d = dot(na, nb);
    if d < ANTIPARALLEL_DOT {
        let axis = orthonormal(na);
        return vec4<f32>(0.0, axis.x, axis.y, axis.z);
    }
    let h = vnormalize_or_zero(na + nb);
    if dot(h, h) < 0.5 {
        let axis = orthonormal(na);
        return vec4<f32>(0.0, axis.x, axis.y, axis.z);
    }
    let w = dot(na, h);
    let vv = cross(na, h);
    return qnormalize(vec4<f32>(w, vv.x, vv.y, vv.z));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }
    let sm = strands[s];
    let p_off = sm.x;
    let p_cnt = sm.y;
    let f_off = sm.z;
    if p_cnt < 2u {
        return;
    }

    var frame = qnormalize(initials[s]);
    var prev_dir = vec3<f32>(0.0, 0.0, 0.0);
    var have_prev = false;
    var i = 0u;
    loop {
        if i + 1u >= p_cnt {
            break;
        }
        let a = points[p_off + i].xyz;
        let b = points[p_off + i + 1u].xyz;
        let dir = vnormalize_or_zero(b - a);
        if dot(dir, dir) < 0.5 {
            // Degenerate edge: keep the running frame unchanged.
            frames_out[f_off + i] = frame;
            i = i + 1u;
            continue;
        }
        var rot: vec4<f32>;
        if have_prev {
            rot = from_min_rotation(prev_dir, dir);
        } else {
            let d3 = qrotate_vec(frame, vec3<f32>(0.0, 0.0, 1.0));
            rot = from_min_rotation(d3, dir);
        }
        frame = qnormalize(qmul(rot, frame));
        frames_out[f_off + i] = frame;
        prev_dir = dir;
        have_prev = true;
        i = i + 1u;
    }
}
