@group(0) @binding(0)
var<storage, read_write> indirect: array<atomic<u32>, 4>;

@compute @workgroup_size(1)
fn placement_indirect_count_reset_main() {
    atomicStore(&indirect[1], 0u);
}
