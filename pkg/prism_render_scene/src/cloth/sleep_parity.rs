//! 休眠动能归约内核 `sleep_max_kinetic`（`cloth_sleep.wesl`）的**逐位** CPU
//! 转写，对齐架构层黄金 [`max_kinetic_indicator`]。
//!
//! `sleep_gpu_tests` 在**有** wgpu 适配器时才把归约真机派发；无头/沙盒环境里
//! 它整组 `skipping` 直接通过，于是**证明不了**内核的算术逐位复刻了 CPU 黄金
//! （设计 §9：不造假 parity）。本模块补上这条与设备无关的闭环：把内核的归约
//! 逐条算术独立转写成 CPU 版本，按 host 上传路径打包速度缓冲（`vec4` 的 `.w`
//! 承载 inverse mass 以跳过被钉粒子），再对多种批次断言结果与
//! [`max_kinetic_indicator`] **逐位（`to_bits`）一致**。
//!
//! ## 逐位一致的关键约定
//! * 每粒子的平方速度按 `x*x + y*y + z*z` 从左到右累加，与黄金
//!   `Vec3::length_squared == dot(self, self)` 的结合序**逐词一致**。
//! * 内核把运行最大值当作 `f32` 的 `u32` 位型存进原子，用整数 `atomicMax`
//!   折叠；所有贡献值均为非负（平方和或 `0.0` 地板），而非负 IEEE-754 浮点在
//!   `bitcast` 下保序，故整数 `atomicMax` 选出的最大值与黄金的浮点 `>` 归约
//!   逐位相同。host 以 `bitcast(0.0)` 播种、`f32::from_bits` 回读，CPU 与 GPU
//!   共用同一套算术与同一枚种子。本转写照此用「位型取大」折叠而非浮点 `max`，
//!   以**精确**镜像设备路径。
//! * 被钉粒子（`is_pinned == inverse_mass <= 0.0`）在黄金里被 `continue` 跳过，
//!   在内核里走 `inverse_mass > 0.0` 之外的分支保留 `0.0` 地板——`0.0` 永不
//!   夺魁（除非整批皆被钉或为空，此时两侧同为 `0.0`）。
//! * 定义域取有限速度（帧速度恒有限）：`NaN` 的平方和会让黄金的 `>` 判否、却
//!   可能在 `bitcast` 下夺得整数 `atomicMax`，属设备级非确定，不在本 parity
//!   范畴，故样例一律使用有限分量。

#![cfg(test)]

use prism_render_architecture::cloth::sleep::max_kinetic_indicator;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// `cloth_sleep.wesl` 里 `@workgroup_size(64)` 的工作组宽度；转写按整组折叠以
/// 忠实镜像设备上的树形归约结构（结果与扁平归约逐位相同，但保留结构更贴近源）。
const CLOTH_SLEEP_WORKGROUP: usize = 64;

/// 把一组粒子按 host 上传布局打包成速度缓冲：`xyz` 为速度，`.w` 为 inverse
/// mass（仅用于跳过被钉粒子），与 `sleep_gpu_tests::upload_velocities` 同款。
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, p.inverse_mass])
        .collect()
}

/// 单条 lane 的取值：越界或被钉（`inverse_mass <= 0.0`）保留 `0.0` 地板，否则
/// 回传 `x*x + y*y + z*z`（逐词镜像黄金 `length_squared`）。
fn lane_value(velocities: &[[f32; 4]], index: usize, particle_count: usize) -> f32 {
    if index >= particle_count || index >= velocities.len() {
        return 0.0;
    }
    let sample = velocities[index];
    let inverse_mass = sample[3];
    if inverse_mass > 0.0 {
        sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]
    } else {
        0.0
    }
}

/// 两个非负 `f32` 的「位型取大」折叠：把双方当作 `u32` 位型比较，回传较大者
/// 对应的浮点。精确镜像内核的整数 `atomicMax`。
fn fold_bits_max(accumulator: f32, value: f32) -> f32 {
    if value.to_bits() > accumulator.to_bits() {
        value
    } else {
        accumulator
    }
}

/// `sleep_max_kinetic` 的 CPU 转写：按工作组宽度分块做组内树形取大，再把每组
/// 偏旁折进全局原子（种子 `bitcast(0.0)`），全程用位型取大。
fn wesl_max_kinetic(velocities: &[[f32; 4]], particle_count: usize) -> f32 {
    // 全局原子的种子：host 以 `bitcast(0.0)` 播种。
    let mut global = f32::from_bits(0.0_f32.to_bits());
    let mut index = 0;
    while index < particle_count {
        // 组内树形归约的草稿：一组 64 个槽位，越界填 `0.0` 地板。
        let mut scratch = [0.0_f32; CLOTH_SLEEP_WORKGROUP];
        for (lane, slot) in scratch.iter_mut().enumerate() {
            *slot = lane_value(velocities, index + lane, particle_count);
        }
        // 树形折叠：把上半折进下半，直至 lane 0 持有本组最大值。
        let mut stride = CLOTH_SLEEP_WORKGROUP / 2;
        while stride != 0 {
            for lane in 0..stride {
                scratch[lane] = fold_bits_max(scratch[lane], scratch[lane + stride]);
            }
            stride /= 2;
        }
        // 单 lane 把本组偏旁折进全局原子。
        global = fold_bits_max(global, scratch[0]);
        index += CLOTH_SLEEP_WORKGROUP;
    }
    global
}

/// 对一组粒子断言：WESL 转写的归约结果与黄金 `max_kinetic_indicator` **逐位**
/// （`to_bits`）一致。
fn assert_bit_exact(particles: &[ClothParticle]) {
    let golden = max_kinetic_indicator(particles);
    let velocities = upload_velocities(particles);
    let twin = wesl_max_kinetic(&velocities, particles.len());
    assert_eq!(
        twin.to_bits(),
        golden.to_bits(),
        "sleep kinetic indicator diverged: twin={twin} ({:#010x}) golden={golden} ({:#010x})",
        twin.to_bits(),
        golden.to_bits()
    );
}

/// 构造一枚速度为 `velocity`、inverse mass 为 `inverse_mass` 的粒子（归约从不
/// 读位置，位置固定在原点）。
fn particle(velocity: Vec3, inverse_mass: f32) -> ClothParticle {
    let mut p = ClothParticle::new(Vec3::ZERO, inverse_mass);
    p.velocity = velocity;
    p
}

/// 自由粒子（单位 inverse mass）。
fn free(velocity: Vec3) -> ClothParticle {
    particle(velocity, 1.0)
}

/// 被钉粒子（零 inverse mass），无论多快都被归约跳过。
fn pinned(velocity: Vec3) -> ClothParticle {
    particle(velocity, 0.0)
}

#[test]
fn empty_batch_matches_golden_bit_for_bit() {
    assert_bit_exact(&[]);
}

#[test]
fn single_free_particle_matches_golden_bit_for_bit() {
    assert_bit_exact(&[free(Vec3::new(0.5, -1.25, 2.0))]);
}

#[test]
fn zero_velocity_batch_matches_golden_bit_for_bit() {
    assert_bit_exact(&[free(Vec3::ZERO), free(Vec3::ZERO), free(Vec3::ZERO)]);
}

#[test]
fn all_pinned_matches_golden_bit_for_bit() {
    assert_bit_exact(&[
        pinned(Vec3::new(10.0, 0.0, 0.0)),
        pinned(Vec3::new(0.0, 12.0, 0.0)),
        pinned(Vec3::new(0.0, 0.0, 7.0)),
    ]);
}

#[test]
fn pinned_fast_particle_is_excluded_bit_for_bit() {
    // 被钉的高速粒子不得夺魁；最大值应取自较慢的自由粒子。
    assert_bit_exact(&[
        pinned(Vec3::new(100.0, 0.0, 0.0)),
        free(Vec3::new(0.0, 3.0, 0.0)),
        free(Vec3::new(1.0, 1.0, 1.0)),
    ]);
}

#[test]
fn negative_inverse_mass_is_pinned_bit_for_bit() {
    // 负 inverse mass 也算被钉（`<= 0.0`），应被跳过。
    assert_bit_exact(&[
        particle(Vec3::new(50.0, 50.0, 50.0), -2.0),
        free(Vec3::new(2.0, 0.0, 0.0)),
    ]);
}

#[test]
fn max_is_order_independent_bit_for_bit() {
    // 最大值在批次中部，验证树形归约跨槽位选出同一峰值。
    assert_bit_exact(&[
        free(Vec3::new(0.1, 0.2, 0.3)),
        free(Vec3::new(4.0, 4.0, 4.0)),
        free(Vec3::new(1.0, 0.0, 0.0)),
        free(Vec3::new(0.0, 2.0, 0.0)),
    ]);
}

#[test]
fn dense_batch_across_workgroups_matches_golden_bit_for_bit() {
    // 超过一个 64 宽工作组，峰值落在第二组，验证跨组原子折叠。
    let mut particles = Vec::new();
    for i in 0..200u32 {
        let f = i as f32;
        particles.push(free(Vec3::new(f * 0.01, f * 0.02, f * 0.015)));
    }
    // 在尾部塞一个明确的峰值。
    particles.push(free(Vec3::new(9.0, 9.0, 9.0)));
    assert_bit_exact(&particles);
}

#[test]
fn mixed_pinned_and_free_across_workgroups_bit_for_bit() {
    let mut particles = Vec::new();
    for i in 0..150u32 {
        let f = i as f32;
        if i % 3 == 0 {
            // 被钉但高速：不得影响结果。
            particles.push(pinned(Vec3::new(f, f, f)));
        } else {
            particles.push(free(Vec3::new(f * 0.03, f * 0.01, f * 0.02)));
        }
    }
    assert_bit_exact(&particles);
}

#[test]
fn tie_values_match_golden_bit_for_bit() {
    // 两个相等峰值：浮点 `>` 与位型取大都应回传同一位型。
    assert_bit_exact(&[
        free(Vec3::new(3.0, 0.0, 0.0)),
        free(Vec3::new(0.0, 3.0, 0.0)),
        free(Vec3::new(0.0, 0.0, 3.0)),
    ]);
}

#[test]
fn jittered_batch_matches_golden_bit_for_bit() {
    // 非规整分量，确保平方和的尾数位在两侧逐位吻合。
    let mut particles = Vec::new();
    let seeds = [0.137_f32, 1.919, 2.718, 0.577, 3.141, 1.414, 0.618, 2.236];
    for (i, s) in seeds.iter().cycle().take(130).enumerate() {
        let j = i as f32;
        particles.push(free(Vec3::new(s * 0.7 + j * 0.001, s * -0.3, s * 1.1 - j * 0.002)));
    }
    assert_bit_exact(&particles);
}
