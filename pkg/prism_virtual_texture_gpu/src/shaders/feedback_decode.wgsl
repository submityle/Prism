// Sampler-feedback min-mip grid decode — per-cell map stage.
//
// One invocation handles one base-level page cell of a streamable texture's
// dense min-mip feedback grid (row-major, `pages_x * pages_y` cells, one byte
// each expanded to one word on the host). For a cell that was sampled it emits
// the collapsed `TexturePageKey` as the same three compare words the CPU golden
// `GpuPageTable::compare_words` produces, plus the clamped desired mip; for a
// cell no view sampled (`NOT_REQUESTED`) it emits the `REQ_NONE` sentinel so the
// host can skip it during dedup.
//
// The map is pure integer work: a desired mip `d` coarser than `base_mip`
// collapses a base cell `(x, y)` onto the coarser page `(x >> (d - base_mip),
// y >> (d - base_mip))`, exactly as the golden `decode_feedback`. Dedup, key
// ordering, and the `resident_mip` lookup stay host-side (a BTreeMap is serial
// and ordered, and residency is a host callback), so this kernel reproduces the
// golden's per-cell arithmetic bit-for-bit and the host folds the identical
// first-writer-wins map over the row-major cell order.
//
// Integer-only: no floating point anywhere, so device output equals the golden
// output bit-for-bit (exact equality, not tolerance).

// Min-mip sentinel: a base cell no view sampled this frame.
const NOT_REQUESTED: u32 = 0xFFu;
// Output sentinel in the desired-mip slot for a not-requested cell.
const REQ_NONE: u32 = 0xFFFFFFFFu;
// Words emitted per cell: three compare words plus the desired mip.
const OUT_WORDS: u32 = 4u;

struct Params {
    // Source texture identifier shared with `TexturePageKey::texture`.
    texture: u32,
    // Array layer / cube face this grid describes.
    layer: u32,
    // Finest streamable mip (the grid's own level).
    base_mip: u32,
    // Coarsest streamable mip a request may address (inclusive).
    max_mip: u32,
    // Base-level page-grid width in cells.
    pages_x: u32,
    // Base-level page-grid height in cells.
    pages_y: u32,
    // Total cells = pages_x * pages_y.
    cell_count: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Min-mip grid, one cell value per word, row-major.
@group(0) @binding(1) var<storage, read> grid: array<u32>;
// Per-cell output: `OUT_WORDS` words each — `(w0, w1, w2, desired)`.
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cell_count) {
        return;
    }
    let obase = i * OUT_WORDS;

    let cell = grid[i];
    if (cell == NOT_REQUESTED) {
        out[obase] = 0u;
        out[obase + 1u] = 0u;
        out[obase + 2u] = 0u;
        out[obase + 3u] = REQ_NONE;
        return;
    }

    // Clamp the requested absolute mip into the streamable range
    // `base_mip..=max_mip`, matching `u8::clamp` on the golden path.
    var desired = cell;
    if (desired < params.base_mip) {
        desired = params.base_mip;
    }
    if (desired > params.max_mip) {
        desired = params.max_mip;
    }

    // Collapse the base cell onto the desired mip's coarser page grid.
    let shift = desired - params.base_mip;
    let x = i % params.pages_x;
    let y = i / params.pages_x;
    let px = x >> shift;
    let py = y >> shift;

    // Pack identically to the golden `GpuPageTable::compare_words`:
    //   w0 = texture
    //   w1 = (mip << 24) | (layer << 8)
    //   w2 = (x << 16) | y
    out[obase] = params.texture;
    out[obase + 1u] = (desired << 24u) | (params.layer << 8u);
    out[obase + 2u] = (px << 16u) | py;
    out[obase + 3u] = desired;
}
