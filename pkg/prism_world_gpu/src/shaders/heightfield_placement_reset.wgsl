// Clears only transient placement state. indirect[0] is the immutable vertex
// count and must survive every request; indirect[1] is the generated instance
// count. Words 2 and 3 are immutable zero draw arguments initialized by host.
// One invocation is sufficient and requires only core u32 atomics.
struct Status {
    overflow_count: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> indirect: array<atomic<u32>, 4>;
@group(0) @binding(1) var<storage, read_write> status: Status;

@compute @workgroup_size(1)
fn reset_main() {
    atomicStore(&indirect[1], 0u);
    atomicStore(&status.overflow_count, 0u);
}
