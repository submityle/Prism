// Deterministic spatial coverage, no frame number or compact index dependence.
// This is ordered dither, not blue noise or a temporal antialiasing solution.
fn world_density_survives(coverage: f32, pixel: vec2<u32>, phase: u32) -> bool {
    let bayer = array<u32, 16>(
        0u, 8u, 2u, 10u,
        12u, 4u, 14u, 6u,
        3u, 11u, 1u, 9u,
        15u, 7u, 13u, 5u,
    );
    let x = (pixel.x + (phase & 3u)) & 3u;
    let y = (pixel.y + ((phase >> 2u) & 3u)) & 3u;
    return coverage > (f32(bayer[y * 4u + x]) + 0.5) / 16.0;
}
