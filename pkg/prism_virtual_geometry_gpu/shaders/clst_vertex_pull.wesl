struct Params {
    vertex_count: u32,
    index_count: u32,
    attribute_offset: u32,
    attribute_format: u32,
    attribute_stride: u32,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> payload: array<u32>;

fn byte_at(offset: u32) -> u32 {
    let word = payload[offset / 4u];
    let shift = (offset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn u16_at(offset: u32) -> u32 {
    return byte_at(offset) | (byte_at(offset + 1u) << 8u);
}

fn u32_at(offset: u32) -> u32 {
    return byte_at(offset)
        | (byte_at(offset + 1u) << 8u)
        | (byte_at(offset + 2u) << 16u)
        | (byte_at(offset + 3u) << 24u);
}

fn f32_at(offset: u32) -> f32 {
    return bitcast<f32>(u32_at(offset));
}

fn position_at(index: u32) -> vec3<f32> {
    let min_v = vec3<f32>(f32_at(10u), f32_at(14u), f32_at(18u));
    let max_v = vec3<f32>(f32_at(22u), f32_at(26u), f32_at(30u));
    let q_offset = 34u + index * 6u;
    let q = vec3<f32>(f32(u16_at(q_offset)), f32(u16_at(q_offset + 2u)), f32(u16_at(q_offset + 4u))) / 65535.0;
    return min_v + (max_v - min_v) * q;
}

fn attr_at(index: u32) -> vec4<f32> {
    let offset = params.attribute_offset + index * params.attribute_stride;
    if (params.attribute_format == 0u) {
        return vec4<f32>(f32_at(offset), 0.0, 0.0, 1.0);
    }
    if (params.attribute_format == 1u) {
        return vec4<f32>(f32_at(offset), f32_at(offset + 4u), 0.0, 1.0);
    }
    if (params.attribute_format == 2u) {
        return vec4<f32>(f32_at(offset), f32_at(offset + 4u), f32_at(offset + 8u), 1.0);
    }
    if (params.attribute_format == 3u) {
        return vec4<f32>(f32_at(offset), f32_at(offset + 4u), f32_at(offset + 8u), f32_at(offset + 12u));
    }
    if (params.attribute_format == 4u) {
        return vec4<f32>(unpack2x16float(u32_at(offset)), 0.0, 1.0);
    }
    if (params.attribute_format == 5u) {
        return vec4<f32>(unpack2x16float(u32_at(offset)), unpack2x16float(u32_at(offset + 4u)));
    }
    if (params.attribute_format == 6u) {
        return vec4<f32>(f32(u32_at(offset)), 0.0, 0.0, 1.0);
    }
    if (params.attribute_format == 7u) {
        return vec4<f32>(f32(u32_at(offset)), f32(u32_at(offset + 4u)), 0.0, 1.0);
    }
    if (params.attribute_format == 8u) {
        return vec4<f32>(f32(u32_at(offset)), f32(u32_at(offset + 4u)), f32(u32_at(offset + 8u)), f32(u32_at(offset + 12u)));
    }
    if (params.attribute_format == 9u) {
        return vec4<f32>(f32(bitcast<i32>(u32_at(offset))), 0.0, 0.0, 1.0);
    }
    if (params.attribute_format == 10u) {
        return vec4<f32>(
            f32(byte_at(offset)) / 255.0,
            f32(byte_at(offset + 1u)) / 255.0,
            f32(byte_at(offset + 2u)) / 255.0,
            f32(byte_at(offset + 3u)) / 255.0,
        );
    }
    if (params.attribute_format == 11u) {
        return vec4<f32>(
            snorm8(byte_at(offset)),
            snorm8(byte_at(offset + 1u)),
            snorm8(byte_at(offset + 2u)),
            snorm8(byte_at(offset + 3u)),
        );
    }
    return vec4<f32>(
        f32_at(offset),
        f32_at(offset + 4u),
        f32_at(offset + 8u),
        f32_at(offset + 12u),
    );
}

fn snorm8(value: u32) -> f32 {
    let signed = select(i32(value), i32(value) - 256, value >= 128u);
    return clamp(f32(signed) / 127.0, -1.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    let local_index_offset = 34u + params.vertex_count * 6u + vertex_index * 2u;
    let local_index = u16_at(local_index_offset);
    var output: VsOut;
    output.position = vec4<f32>(position_at(local_index), 1.0);
    output.color = attr_at(local_index);
    return output;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    return input.color;
}
