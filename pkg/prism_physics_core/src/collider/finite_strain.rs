//! 从给定的变形梯度 `F` 推导连续介质力学中的有限应变度量。
//!
//! 本模块是一个纯函数式、零耦合的分析原语：输入一个 `3x3` 的变形梯度
//! `F`（例如由 `nonaffine_displacement::deformation_gradient` 或
//! `tet_fem_basis::deformation_gradient` 产生），输出一组标准应变张量与不
//! 变量。它只做张量代数（乘、加、行列式），不涉及任何求解器状态或时间推进，
//! 因此可被 DEM / FEM / MPM 等任意上游复用而不引入耦合。
//!
//! 提供的度量：
//! - 体积比 `J = det(F)`（雅可比，物理上要求 `J > 0`，否则材料发生翻转）。
//! - 右 Cauchy-Green 张量 `C = Fᵀ F`。
//! - 左 Cauchy-Green 张量 `b = F Fᵀ`。
//! - Green-Lagrange 有限应变 `E = (C - I) / 2`（对大转动精确，刚体运动下为零）。
//! - 小应变（线性化）`ε = (F + Fᵀ) / 2 - I`（仅在小转动下有效，大转动会引入
//!   虚假应变——本模块的测试显式演示了这一经典差异）。
//! - `C` 的三个主不变量 `I1 = tr C`、`I2 = ½(I1² - tr(C²))`、`I3 = det C = J²`。
//! - 等容（isochoric）分解：`F̄ = J^(-1/3) F`、`C̄ = J^(-2/3) C`，用于分离剪切
//!   与体积响应（近不可压本构的标准做法）。

/// 行主序存储的 `3x3` 矩阵（`m[row][col]`）。
type Mat3 = [[f32; 3]; 3];

/// 单位矩阵。
fn identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// 转置。
fn transpose(m: &Mat3) -> Mat3 {
    let mut t = [[0.0_f32; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

/// 矩阵乘法 `a * b`。
fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0_f32;
            for (k, arow_k) in a[i].iter().enumerate() {
                acc += arow_k * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    c
}

/// 行列式（代数余子式展开）。
fn det(m: &Mat3) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 迹（对角线之和）。
fn trace(m: &Mat3) -> f32 {
    m[0][0] + m[1][1] + m[2][2]
}

/// 标量缩放。
fn scale(m: &Mat3, s: f32) -> Mat3 {
    let mut r = [[0.0_f32; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            r[i][j] = v * s;
        }
    }
    r
}

/// 逐元素相加。
fn add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    r
}

/// 逐元素相减 `a - b`。
fn sub(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][j] - b[i][j];
        }
    }
    r
}

/// 判断矩阵的所有分量是否有限。
fn all_finite(m: &Mat3) -> bool {
    m.iter().all(|row| row.iter().all(|v| v.is_finite()))
}

/// 由变形梯度 `F` 推导出的一组有限应变度量。
///
/// 使用 [`FiniteStrain::from_deformation_gradient`] 构造。所有张量以行主序
/// `[[f32; 3]; 3]` 返回；`C`、`b`、`E`、`ε` 均为对称张量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteStrain {
    f: Mat3,
    j: f32,
    c: Mat3,
    b: Mat3,
    e: Mat3,
    eps: Mat3,
    i1: f32,
    i2: f32,
    i3: f32,
}

impl FiniteStrain {
    /// 从变形梯度 `F` 构造全部有限应变度量。
    ///
    /// 当任意分量非有限，或体积比 `J = det(F) ≤ 0`（材料翻转/退化，物理上不
    /// 合法）时返回 `None`。
    #[must_use]
    pub fn from_deformation_gradient(f: Mat3) -> Option<Self> {
        if !all_finite(&f) {
            return None;
        }
        let j = det(&f);
        if !j.is_finite() || j <= 0.0 {
            return None;
        }

        let ft = transpose(&f);
        let c = mul(&ft, &f); // 右 Cauchy-Green: C = Fᵀ F
        let b = mul(&f, &ft); // 左 Cauchy-Green: b = F Fᵀ
        let id = identity();

        // Green-Lagrange 有限应变 E = (C - I) / 2。
        let e = scale(&sub(&c, &id), 0.5);

        // 小应变 ε = (F + Fᵀ) / 2 - I = sym(H)，其中位移梯度 H = F - I。
        let eps = sub(&scale(&add(&f, &ft), 0.5), &id);

        // C 的主不变量。
        let i1 = trace(&c);
        let c_sq = mul(&c, &c);
        let i2 = 0.5 * (i1 * i1 - trace(&c_sq));
        let i3 = det(&c);

        Some(Self {
            f,
            j,
            c,
            b,
            e,
            eps,
            i1,
            i2,
            i3,
        })
    }

    /// 输入的变形梯度 `F`。
    #[must_use]
    pub fn deformation_gradient(&self) -> Mat3 {
        self.f
    }

    /// 体积比（雅可比）`J = det(F)`。`J > 1` 膨胀，`J < 1` 压缩。
    #[must_use]
    pub fn jacobian(&self) -> f32 {
        self.j
    }

    /// 右 Cauchy-Green 变形张量 `C = Fᵀ F`（对称正定）。
    #[must_use]
    pub fn right_cauchy_green(&self) -> Mat3 {
        self.c
    }

    /// 左 Cauchy-Green 变形张量 `b = F Fᵀ`（对称正定）。
    #[must_use]
    pub fn left_cauchy_green(&self) -> Mat3 {
        self.b
    }

    /// Green-Lagrange 有限应变张量 `E = (C - I) / 2`。
    ///
    /// 对大转动精确：任意刚体运动（纯平移 + 纯转动）下为零。
    #[must_use]
    pub fn green_lagrange_strain(&self) -> Mat3 {
        self.e
    }

    /// 小应变（线性化）张量 `ε = (F + Fᵀ) / 2 - I`。
    ///
    /// 仅在小转动下有效；有限转动会使其产生非零的虚假应变。
    #[must_use]
    pub fn small_strain(&self) -> Mat3 {
        self.eps
    }

    /// 小应变的体积分量 `tr(ε)`，小变形下 `≈ J - 1`。
    #[must_use]
    pub fn volumetric_small_strain(&self) -> f32 {
        trace(&self.eps)
    }

    /// Green-Lagrange 应变的迹 `tr(E)`。
    #[must_use]
    pub fn green_lagrange_trace(&self) -> f32 {
        trace(&self.e)
    }

    /// `C` 的第一主不变量 `I1 = tr(C)`。
    #[must_use]
    pub fn first_invariant(&self) -> f32 {
        self.i1
    }

    /// `C` 的第二主不变量 `I2 = ½(I1² - tr(C²))`。
    #[must_use]
    pub fn second_invariant(&self) -> f32 {
        self.i2
    }

    /// `C` 的第三主不变量 `I3 = det(C) = J²`。
    #[must_use]
    pub fn third_invariant(&self) -> f32 {
        self.i3
    }

    /// 等容变形梯度 `F̄ = J^(-1/3) F`，满足 `det(F̄) = 1`。
    #[must_use]
    pub fn isochoric_deformation_gradient(&self) -> Mat3 {
        // J^(-1/3) 在 f64 下用立方根求得，再降回 f32（规避 f32 cbrt/powf）。
        let s = (1.0 / f64::from(self.j).cbrt()) as f32;
        scale(&self.f, s)
    }

    /// 等容右 Cauchy-Green 张量 `C̄ = J^(-2/3) C`，满足 `det(C̄) = 1`。
    #[must_use]
    pub fn isochoric_right_cauchy_green(&self) -> Mat3 {
        let inv_cbrt = 1.0 / f64::from(self.j).cbrt();
        let s = (inv_cbrt * inv_cbrt) as f32;
        scale(&self.c, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn mat_close(a: &Mat3, b: &Mat3) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(ra, rb)| ra.iter().zip(rb.iter()).all(|(x, y)| (x - y).abs() <= EPS))
    }

    fn is_symmetric(m: &Mat3) -> bool {
        (m[0][1] - m[1][0]).abs() <= EPS
            && (m[0][2] - m[2][0]).abs() <= EPS
            && (m[1][2] - m[2][1]).abs() <= EPS
    }

    #[test]
    fn identity_gradient_is_undeformed() {
        let fs = FiniteStrain::from_deformation_gradient(identity()).unwrap();
        assert!((fs.jacobian() - 1.0).abs() <= EPS);
        assert!(mat_close(&fs.right_cauchy_green(), &identity()));
        assert!(mat_close(&fs.left_cauchy_green(), &identity()));
        assert!(mat_close(&fs.green_lagrange_strain(), &[[0.0; 3]; 3]));
        assert!(mat_close(&fs.small_strain(), &[[0.0; 3]; 3]));
        assert!((fs.first_invariant() - 3.0).abs() <= EPS);
        assert!((fs.second_invariant() - 3.0).abs() <= EPS);
        assert!((fs.third_invariant() - 1.0).abs() <= EPS);
        assert!((fs.volumetric_small_strain()).abs() <= EPS);
        assert!((fs.green_lagrange_trace()).abs() <= EPS);
    }

    #[test]
    fn uniaxial_stretch() {
        // F = diag(2, 1, 1): 沿 x 拉伸两倍。
        let f = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        assert!((fs.jacobian() - 2.0).abs() <= EPS);
        // C = diag(4,1,1)。
        assert!((fs.right_cauchy_green()[0][0] - 4.0).abs() <= EPS);
        // E = diag(1.5, 0, 0)。
        assert!((fs.green_lagrange_strain()[0][0] - 1.5).abs() <= EPS);
        // ε = diag(1, 0, 0)，体积小应变 = 1。
        assert!((fs.small_strain()[0][0] - 1.0).abs() <= EPS);
        assert!((fs.volumetric_small_strain() - 1.0).abs() <= EPS);
        // I1 = 6, I3 = det C = 4 = J²。
        assert!((fs.first_invariant() - 6.0).abs() <= EPS);
        assert!((fs.third_invariant() - 4.0).abs() <= EPS);
        assert!((fs.third_invariant() - fs.jacobian() * fs.jacobian()).abs() <= EPS);
    }

    #[test]
    fn simple_shear_distinguishes_finite_from_small_strain() {
        // 简单剪切 F = [[1, γ, 0], [0, 1, 0], [0, 0, 1]]，γ = 0.5，J = 1（等容）。
        let g = 0.5_f32;
        let f = [[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        assert!((fs.jacobian() - 1.0).abs() <= EPS);
        // 小应变 ε 对角线为零（纯剪切无线性体积变化）。
        assert!(fs.volumetric_small_strain().abs() <= EPS);
        assert!((fs.small_strain()[0][1] - g / 2.0).abs() <= EPS);
        // 但 Green-Lagrange E 含二次对角项 γ²/2 = 0.125——有限应变的几何非线性。
        assert!((fs.green_lagrange_strain()[1][1] - g * g / 2.0).abs() <= EPS);
        assert!((fs.green_lagrange_strain()[0][1] - g / 2.0).abs() <= EPS);
        assert!(is_symmetric(&fs.right_cauchy_green()));
        assert!(is_symmetric(&fs.green_lagrange_strain()));
    }

    #[test]
    fn rigid_rotation_has_zero_green_lagrange_but_nonzero_small_strain() {
        // 绕 z 轴 90° 旋转 F = [[0,-1,0],[1,0,0],[0,0,1]]，det = 1。
        let f = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        assert!((fs.jacobian() - 1.0).abs() <= EPS);
        // C = I，Green-Lagrange 应变为零（对大转动精确）。
        assert!(mat_close(&fs.right_cauchy_green(), &identity()));
        assert!(mat_close(&fs.green_lagrange_strain(), &[[0.0; 3]; 3]));
        // 小应变在有限转动下产生虚假应变（ε11 = ε22 = -1）——线性化的经典缺陷。
        assert!((fs.small_strain()[0][0] + 1.0).abs() <= EPS);
        assert!((fs.small_strain()[1][1] + 1.0).abs() <= EPS);
    }

    #[test]
    fn isochoric_split_has_unit_jacobian() {
        // 一般非等容变形 F = diag(2, 3, 4)，J = 24。
        let f = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        assert!((fs.jacobian() - 24.0).abs() <= 1e-3);
        // det(F̄) = 1。
        let fbar = fs.isochoric_deformation_gradient();
        assert!((det(&fbar) - 1.0).abs() <= 1e-3);
        // det(C̄) = 1。
        let cbar = fs.isochoric_right_cauchy_green();
        assert!((det(&cbar) - 1.0).abs() <= 1e-2);
    }

    #[test]
    fn isochoric_deformation_is_unchanged_when_already_volume_preserving() {
        // F = diag(2, 0.5, 1)，J = 1：等容分解应为恒等。
        let f = [[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        assert!((fs.jacobian() - 1.0).abs() <= EPS);
        assert!(mat_close(&fs.isochoric_deformation_gradient(), &f));
        assert!(mat_close(
            &fs.isochoric_right_cauchy_green(),
            &fs.right_cauchy_green()
        ));
    }

    #[test]
    fn rejects_non_finite_and_inverted_gradients() {
        let nan = [[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(FiniteStrain::from_deformation_gradient(nan).is_none());
        // 翻转：J = -1 < 0。
        let inverted = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(FiniteStrain::from_deformation_gradient(inverted).is_none());
        // 退化：J = 0。
        let degenerate = [[0.0; 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(FiniteStrain::from_deformation_gradient(degenerate).is_none());
    }

    #[test]
    fn invariants_match_direct_formula() {
        // 对一般对称变形验证 I2 = ½(I1² - tr(C²)) 与特征乘积一致性。
        let f = [[1.2, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.1]];
        let fs = FiniteStrain::from_deformation_gradient(f).unwrap();
        // I3 = det C = J²。
        assert!((fs.third_invariant() - fs.jacobian() * fs.jacobian()).abs() <= 1e-4);
        // C 对称。
        assert!(is_symmetric(&fs.right_cauchy_green()));
    }
}
