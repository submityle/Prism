# Bevy Benchmarks

This is a crate with a collection of benchmarks for Bevy.

## Running benchmarks

Benchmarks can be run through Cargo:

```sh
# Run all benchmarks. (This will take a while!)
cargo bench -p benches

# Just compile the benchmarks, do not run them.
cargo bench -p benches --no-run

# Run the benchmarks for a specific crate. (See `Cargo.toml` for a complete list of crates
# tracked.)
cargo bench -p benches --bench ecs

# Filter which benchmarks are run based on the name. This will only run benchmarks whose name
# contains "name_fragment".
cargo bench -p benches -- name_fragment

# List all available benchmarks.
cargo bench -p benches -- --list

# Save a baseline to be compared against later.
cargo bench -p benches -- --save-baseline before

# Compare the current benchmarks against a baseline to find performance gains and regressions.
cargo bench -p benches -- --baseline before
```

## The GPU-driven culling benchmark

`prism_gpu_cull` measures the M2 GPU-driven submission against a naive
per-object one: the same scene drawn by one compute cull plus one indirect draw,
and by one `DrawIndexed` per object. It reports the CPU cost of assembling a
frame, the cost of assembling and submitting it, and the end-to-end frame time,
for 100, 1000 and 10000 objects.

It needs a graphics adapter. Without one it prints a skip notice and does
nothing; set `PRISM_REQUIRE_GPU=1` to make a missing adapter fail the run instead
of skipping it (useful on a GPU machine, where a silent skip would look like a
green result).

```sh
PRISM_REQUIRE_GPU=1 cargo bench -p benches --bench prism_gpu_cull
```

## Criterion

Bevy's benchmarks use [Criterion](https://crates.io/crates/criterion). If you want to learn more about using Criterion for comparing performance against a baseline or generating detailed reports, you can read the [Criterion.rs documentation](https://bheisler.github.io/criterion.rs/book/criterion_rs.html).

You can generate nicely formatted comparison tables of named benchmark runs (baselines) using the [critcmp](https://github.com/BurntSushi/critcmp) tool.
