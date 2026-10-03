//! Full swizzle coverage for the `f32` vector types.
//!
//! Every 2-, 3-, and 4-component permutation (with repeats) of a vector's
//! lanes is exposed as a method named after the selected lanes, matching the
//! glam-shaped API. For example `v.zyx()` reverses a [`Vec3`], and
//! `v.xxxx()` broadcasts the `x` lane into a [`Vec4`]. 2-lane results are
//! [`Vec2`], 4-lane results are [`Vec4`], and 3-lane results are [`Vec3`]
//! (or [`Vec3A`] when swizzling a [`Vec3A`]).

use crate::vec::{Vec2, Vec3, Vec3A, Vec4};

impl Vec2 {
    /// Swizzle `xx`.
    #[inline]
    #[must_use]
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Swizzle `xy`.
    #[inline]
    #[must_use]
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Swizzle `yx`.
    #[inline]
    #[must_use]
    pub fn yx(self) -> Vec2 {
        Vec2::new(self.y, self.x)
    }

    /// Swizzle `yy`.
    #[inline]
    #[must_use]
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Swizzle `xxx`.
    #[inline]
    #[must_use]
    pub fn xxx(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.x)
    }

    /// Swizzle `xxy`.
    #[inline]
    #[must_use]
    pub fn xxy(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.y)
    }

    /// Swizzle `xyx`.
    #[inline]
    #[must_use]
    pub fn xyx(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.x)
    }

    /// Swizzle `xyy`.
    #[inline]
    #[must_use]
    pub fn xyy(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.y)
    }

    /// Swizzle `yxx`.
    #[inline]
    #[must_use]
    pub fn yxx(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.x)
    }

    /// Swizzle `yxy`.
    #[inline]
    #[must_use]
    pub fn yxy(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.y)
    }

    /// Swizzle `yyx`.
    #[inline]
    #[must_use]
    pub fn yyx(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.x)
    }

    /// Swizzle `yyy`.
    #[inline]
    #[must_use]
    pub fn yyy(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.y)
    }

    /// Swizzle `xxxx`.
    #[inline]
    #[must_use]
    pub fn xxxx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.x)
    }

    /// Swizzle `xxxy`.
    #[inline]
    #[must_use]
    pub fn xxxy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.y)
    }

    /// Swizzle `xxyx`.
    #[inline]
    #[must_use]
    pub fn xxyx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.x)
    }

    /// Swizzle `xxyy`.
    #[inline]
    #[must_use]
    pub fn xxyy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.y)
    }

    /// Swizzle `xyxx`.
    #[inline]
    #[must_use]
    pub fn xyxx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.x)
    }

    /// Swizzle `xyxy`.
    #[inline]
    #[must_use]
    pub fn xyxy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.y)
    }

    /// Swizzle `xyyx`.
    #[inline]
    #[must_use]
    pub fn xyyx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.x)
    }

    /// Swizzle `xyyy`.
    #[inline]
    #[must_use]
    pub fn xyyy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.y)
    }

    /// Swizzle `yxxx`.
    #[inline]
    #[must_use]
    pub fn yxxx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.x)
    }

    /// Swizzle `yxxy`.
    #[inline]
    #[must_use]
    pub fn yxxy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.y)
    }

    /// Swizzle `yxyx`.
    #[inline]
    #[must_use]
    pub fn yxyx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.x)
    }

    /// Swizzle `yxyy`.
    #[inline]
    #[must_use]
    pub fn yxyy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.y)
    }

    /// Swizzle `yyxx`.
    #[inline]
    #[must_use]
    pub fn yyxx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.x)
    }

    /// Swizzle `yyxy`.
    #[inline]
    #[must_use]
    pub fn yyxy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.y)
    }

    /// Swizzle `yyyx`.
    #[inline]
    #[must_use]
    pub fn yyyx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.x)
    }

    /// Swizzle `yyyy`.
    #[inline]
    #[must_use]
    pub fn yyyy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.y)
    }
}

impl Vec3 {
    /// Swizzle `xx`.
    #[inline]
    #[must_use]
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Swizzle `xy`.
    #[inline]
    #[must_use]
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Swizzle `xz`.
    #[inline]
    #[must_use]
    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Swizzle `yx`.
    #[inline]
    #[must_use]
    pub fn yx(self) -> Vec2 {
        Vec2::new(self.y, self.x)
    }

    /// Swizzle `yy`.
    #[inline]
    #[must_use]
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Swizzle `yz`.
    #[inline]
    #[must_use]
    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Swizzle `zx`.
    #[inline]
    #[must_use]
    pub fn zx(self) -> Vec2 {
        Vec2::new(self.z, self.x)
    }

    /// Swizzle `zy`.
    #[inline]
    #[must_use]
    pub fn zy(self) -> Vec2 {
        Vec2::new(self.z, self.y)
    }

    /// Swizzle `zz`.
    #[inline]
    #[must_use]
    pub fn zz(self) -> Vec2 {
        Vec2::new(self.z, self.z)
    }

    /// Swizzle `xxx`.
    #[inline]
    #[must_use]
    pub fn xxx(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.x)
    }

    /// Swizzle `xxy`.
    #[inline]
    #[must_use]
    pub fn xxy(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.y)
    }

    /// Swizzle `xxz`.
    #[inline]
    #[must_use]
    pub fn xxz(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.z)
    }

    /// Swizzle `xyx`.
    #[inline]
    #[must_use]
    pub fn xyx(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.x)
    }

    /// Swizzle `xyy`.
    #[inline]
    #[must_use]
    pub fn xyy(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.y)
    }

    /// Swizzle `xyz`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Swizzle `xzx`.
    #[inline]
    #[must_use]
    pub fn xzx(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.x)
    }

    /// Swizzle `xzy`.
    #[inline]
    #[must_use]
    pub fn xzy(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.y)
    }

    /// Swizzle `xzz`.
    #[inline]
    #[must_use]
    pub fn xzz(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.z)
    }

    /// Swizzle `yxx`.
    #[inline]
    #[must_use]
    pub fn yxx(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.x)
    }

    /// Swizzle `yxy`.
    #[inline]
    #[must_use]
    pub fn yxy(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.y)
    }

    /// Swizzle `yxz`.
    #[inline]
    #[must_use]
    pub fn yxz(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.z)
    }

    /// Swizzle `yyx`.
    #[inline]
    #[must_use]
    pub fn yyx(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.x)
    }

    /// Swizzle `yyy`.
    #[inline]
    #[must_use]
    pub fn yyy(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.y)
    }

    /// Swizzle `yyz`.
    #[inline]
    #[must_use]
    pub fn yyz(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.z)
    }

    /// Swizzle `yzx`.
    #[inline]
    #[must_use]
    pub fn yzx(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.x)
    }

    /// Swizzle `yzy`.
    #[inline]
    #[must_use]
    pub fn yzy(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.y)
    }

    /// Swizzle `yzz`.
    #[inline]
    #[must_use]
    pub fn yzz(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.z)
    }

    /// Swizzle `zxx`.
    #[inline]
    #[must_use]
    pub fn zxx(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.x)
    }

    /// Swizzle `zxy`.
    #[inline]
    #[must_use]
    pub fn zxy(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.y)
    }

    /// Swizzle `zxz`.
    #[inline]
    #[must_use]
    pub fn zxz(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.z)
    }

    /// Swizzle `zyx`.
    #[inline]
    #[must_use]
    pub fn zyx(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.x)
    }

    /// Swizzle `zyy`.
    #[inline]
    #[must_use]
    pub fn zyy(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.y)
    }

    /// Swizzle `zyz`.
    #[inline]
    #[must_use]
    pub fn zyz(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.z)
    }

    /// Swizzle `zzx`.
    #[inline]
    #[must_use]
    pub fn zzx(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.x)
    }

    /// Swizzle `zzy`.
    #[inline]
    #[must_use]
    pub fn zzy(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.y)
    }

    /// Swizzle `zzz`.
    #[inline]
    #[must_use]
    pub fn zzz(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.z)
    }

    /// Swizzle `xxxx`.
    #[inline]
    #[must_use]
    pub fn xxxx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.x)
    }

    /// Swizzle `xxxy`.
    #[inline]
    #[must_use]
    pub fn xxxy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.y)
    }

    /// Swizzle `xxxz`.
    #[inline]
    #[must_use]
    pub fn xxxz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.z)
    }

    /// Swizzle `xxyx`.
    #[inline]
    #[must_use]
    pub fn xxyx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.x)
    }

    /// Swizzle `xxyy`.
    #[inline]
    #[must_use]
    pub fn xxyy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.y)
    }

    /// Swizzle `xxyz`.
    #[inline]
    #[must_use]
    pub fn xxyz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.z)
    }

    /// Swizzle `xxzx`.
    #[inline]
    #[must_use]
    pub fn xxzx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.x)
    }

    /// Swizzle `xxzy`.
    #[inline]
    #[must_use]
    pub fn xxzy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.y)
    }

    /// Swizzle `xxzz`.
    #[inline]
    #[must_use]
    pub fn xxzz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.z)
    }

    /// Swizzle `xyxx`.
    #[inline]
    #[must_use]
    pub fn xyxx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.x)
    }

    /// Swizzle `xyxy`.
    #[inline]
    #[must_use]
    pub fn xyxy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.y)
    }

    /// Swizzle `xyxz`.
    #[inline]
    #[must_use]
    pub fn xyxz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.z)
    }

    /// Swizzle `xyyx`.
    #[inline]
    #[must_use]
    pub fn xyyx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.x)
    }

    /// Swizzle `xyyy`.
    #[inline]
    #[must_use]
    pub fn xyyy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.y)
    }

    /// Swizzle `xyyz`.
    #[inline]
    #[must_use]
    pub fn xyyz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.z)
    }

    /// Swizzle `xyzx`.
    #[inline]
    #[must_use]
    pub fn xyzx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.x)
    }

    /// Swizzle `xyzy`.
    #[inline]
    #[must_use]
    pub fn xyzy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.y)
    }

    /// Swizzle `xyzz`.
    #[inline]
    #[must_use]
    pub fn xyzz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.z)
    }

    /// Swizzle `xzxx`.
    #[inline]
    #[must_use]
    pub fn xzxx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.x)
    }

    /// Swizzle `xzxy`.
    #[inline]
    #[must_use]
    pub fn xzxy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.y)
    }

    /// Swizzle `xzxz`.
    #[inline]
    #[must_use]
    pub fn xzxz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.z)
    }

    /// Swizzle `xzyx`.
    #[inline]
    #[must_use]
    pub fn xzyx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.x)
    }

    /// Swizzle `xzyy`.
    #[inline]
    #[must_use]
    pub fn xzyy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.y)
    }

    /// Swizzle `xzyz`.
    #[inline]
    #[must_use]
    pub fn xzyz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.z)
    }

    /// Swizzle `xzzx`.
    #[inline]
    #[must_use]
    pub fn xzzx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.x)
    }

    /// Swizzle `xzzy`.
    #[inline]
    #[must_use]
    pub fn xzzy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.y)
    }

    /// Swizzle `xzzz`.
    #[inline]
    #[must_use]
    pub fn xzzz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.z)
    }

    /// Swizzle `yxxx`.
    #[inline]
    #[must_use]
    pub fn yxxx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.x)
    }

    /// Swizzle `yxxy`.
    #[inline]
    #[must_use]
    pub fn yxxy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.y)
    }

    /// Swizzle `yxxz`.
    #[inline]
    #[must_use]
    pub fn yxxz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.z)
    }

    /// Swizzle `yxyx`.
    #[inline]
    #[must_use]
    pub fn yxyx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.x)
    }

    /// Swizzle `yxyy`.
    #[inline]
    #[must_use]
    pub fn yxyy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.y)
    }

    /// Swizzle `yxyz`.
    #[inline]
    #[must_use]
    pub fn yxyz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.z)
    }

    /// Swizzle `yxzx`.
    #[inline]
    #[must_use]
    pub fn yxzx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.x)
    }

    /// Swizzle `yxzy`.
    #[inline]
    #[must_use]
    pub fn yxzy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.y)
    }

    /// Swizzle `yxzz`.
    #[inline]
    #[must_use]
    pub fn yxzz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.z)
    }

    /// Swizzle `yyxx`.
    #[inline]
    #[must_use]
    pub fn yyxx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.x)
    }

    /// Swizzle `yyxy`.
    #[inline]
    #[must_use]
    pub fn yyxy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.y)
    }

    /// Swizzle `yyxz`.
    #[inline]
    #[must_use]
    pub fn yyxz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.z)
    }

    /// Swizzle `yyyx`.
    #[inline]
    #[must_use]
    pub fn yyyx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.x)
    }

    /// Swizzle `yyyy`.
    #[inline]
    #[must_use]
    pub fn yyyy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.y)
    }

    /// Swizzle `yyyz`.
    #[inline]
    #[must_use]
    pub fn yyyz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.z)
    }

    /// Swizzle `yyzx`.
    #[inline]
    #[must_use]
    pub fn yyzx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.x)
    }

    /// Swizzle `yyzy`.
    #[inline]
    #[must_use]
    pub fn yyzy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.y)
    }

    /// Swizzle `yyzz`.
    #[inline]
    #[must_use]
    pub fn yyzz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.z)
    }

    /// Swizzle `yzxx`.
    #[inline]
    #[must_use]
    pub fn yzxx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.x)
    }

    /// Swizzle `yzxy`.
    #[inline]
    #[must_use]
    pub fn yzxy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.y)
    }

    /// Swizzle `yzxz`.
    #[inline]
    #[must_use]
    pub fn yzxz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.z)
    }

    /// Swizzle `yzyx`.
    #[inline]
    #[must_use]
    pub fn yzyx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.x)
    }

    /// Swizzle `yzyy`.
    #[inline]
    #[must_use]
    pub fn yzyy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.y)
    }

    /// Swizzle `yzyz`.
    #[inline]
    #[must_use]
    pub fn yzyz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.z)
    }

    /// Swizzle `yzzx`.
    #[inline]
    #[must_use]
    pub fn yzzx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.x)
    }

    /// Swizzle `yzzy`.
    #[inline]
    #[must_use]
    pub fn yzzy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.y)
    }

    /// Swizzle `yzzz`.
    #[inline]
    #[must_use]
    pub fn yzzz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.z)
    }

    /// Swizzle `zxxx`.
    #[inline]
    #[must_use]
    pub fn zxxx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.x)
    }

    /// Swizzle `zxxy`.
    #[inline]
    #[must_use]
    pub fn zxxy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.y)
    }

    /// Swizzle `zxxz`.
    #[inline]
    #[must_use]
    pub fn zxxz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.z)
    }

    /// Swizzle `zxyx`.
    #[inline]
    #[must_use]
    pub fn zxyx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.x)
    }

    /// Swizzle `zxyy`.
    #[inline]
    #[must_use]
    pub fn zxyy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.y)
    }

    /// Swizzle `zxyz`.
    #[inline]
    #[must_use]
    pub fn zxyz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.z)
    }

    /// Swizzle `zxzx`.
    #[inline]
    #[must_use]
    pub fn zxzx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.x)
    }

    /// Swizzle `zxzy`.
    #[inline]
    #[must_use]
    pub fn zxzy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.y)
    }

    /// Swizzle `zxzz`.
    #[inline]
    #[must_use]
    pub fn zxzz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.z)
    }

    /// Swizzle `zyxx`.
    #[inline]
    #[must_use]
    pub fn zyxx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.x)
    }

    /// Swizzle `zyxy`.
    #[inline]
    #[must_use]
    pub fn zyxy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.y)
    }

    /// Swizzle `zyxz`.
    #[inline]
    #[must_use]
    pub fn zyxz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.z)
    }

    /// Swizzle `zyyx`.
    #[inline]
    #[must_use]
    pub fn zyyx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.x)
    }

    /// Swizzle `zyyy`.
    #[inline]
    #[must_use]
    pub fn zyyy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.y)
    }

    /// Swizzle `zyyz`.
    #[inline]
    #[must_use]
    pub fn zyyz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.z)
    }

    /// Swizzle `zyzx`.
    #[inline]
    #[must_use]
    pub fn zyzx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.x)
    }

    /// Swizzle `zyzy`.
    #[inline]
    #[must_use]
    pub fn zyzy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.y)
    }

    /// Swizzle `zyzz`.
    #[inline]
    #[must_use]
    pub fn zyzz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.z)
    }

    /// Swizzle `zzxx`.
    #[inline]
    #[must_use]
    pub fn zzxx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.x)
    }

    /// Swizzle `zzxy`.
    #[inline]
    #[must_use]
    pub fn zzxy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.y)
    }

    /// Swizzle `zzxz`.
    #[inline]
    #[must_use]
    pub fn zzxz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.z)
    }

    /// Swizzle `zzyx`.
    #[inline]
    #[must_use]
    pub fn zzyx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.x)
    }

    /// Swizzle `zzyy`.
    #[inline]
    #[must_use]
    pub fn zzyy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.y)
    }

    /// Swizzle `zzyz`.
    #[inline]
    #[must_use]
    pub fn zzyz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.z)
    }

    /// Swizzle `zzzx`.
    #[inline]
    #[must_use]
    pub fn zzzx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.x)
    }

    /// Swizzle `zzzy`.
    #[inline]
    #[must_use]
    pub fn zzzy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.y)
    }

    /// Swizzle `zzzz`.
    #[inline]
    #[must_use]
    pub fn zzzz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.z)
    }
}

impl Vec3A {
    /// Swizzle `xx`.
    #[inline]
    #[must_use]
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Swizzle `xy`.
    #[inline]
    #[must_use]
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Swizzle `xz`.
    #[inline]
    #[must_use]
    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Swizzle `yx`.
    #[inline]
    #[must_use]
    pub fn yx(self) -> Vec2 {
        Vec2::new(self.y, self.x)
    }

    /// Swizzle `yy`.
    #[inline]
    #[must_use]
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Swizzle `yz`.
    #[inline]
    #[must_use]
    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Swizzle `zx`.
    #[inline]
    #[must_use]
    pub fn zx(self) -> Vec2 {
        Vec2::new(self.z, self.x)
    }

    /// Swizzle `zy`.
    #[inline]
    #[must_use]
    pub fn zy(self) -> Vec2 {
        Vec2::new(self.z, self.y)
    }

    /// Swizzle `zz`.
    #[inline]
    #[must_use]
    pub fn zz(self) -> Vec2 {
        Vec2::new(self.z, self.z)
    }

    /// Swizzle `xxx`.
    #[inline]
    #[must_use]
    pub fn xxx(self) -> Vec3A {
        Vec3A::new(self.x, self.x, self.x)
    }

    /// Swizzle `xxy`.
    #[inline]
    #[must_use]
    pub fn xxy(self) -> Vec3A {
        Vec3A::new(self.x, self.x, self.y)
    }

    /// Swizzle `xxz`.
    #[inline]
    #[must_use]
    pub fn xxz(self) -> Vec3A {
        Vec3A::new(self.x, self.x, self.z)
    }

    /// Swizzle `xyx`.
    #[inline]
    #[must_use]
    pub fn xyx(self) -> Vec3A {
        Vec3A::new(self.x, self.y, self.x)
    }

    /// Swizzle `xyy`.
    #[inline]
    #[must_use]
    pub fn xyy(self) -> Vec3A {
        Vec3A::new(self.x, self.y, self.y)
    }

    /// Swizzle `xyz`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> Vec3A {
        Vec3A::new(self.x, self.y, self.z)
    }

    /// Swizzle `xzx`.
    #[inline]
    #[must_use]
    pub fn xzx(self) -> Vec3A {
        Vec3A::new(self.x, self.z, self.x)
    }

    /// Swizzle `xzy`.
    #[inline]
    #[must_use]
    pub fn xzy(self) -> Vec3A {
        Vec3A::new(self.x, self.z, self.y)
    }

    /// Swizzle `xzz`.
    #[inline]
    #[must_use]
    pub fn xzz(self) -> Vec3A {
        Vec3A::new(self.x, self.z, self.z)
    }

    /// Swizzle `yxx`.
    #[inline]
    #[must_use]
    pub fn yxx(self) -> Vec3A {
        Vec3A::new(self.y, self.x, self.x)
    }

    /// Swizzle `yxy`.
    #[inline]
    #[must_use]
    pub fn yxy(self) -> Vec3A {
        Vec3A::new(self.y, self.x, self.y)
    }

    /// Swizzle `yxz`.
    #[inline]
    #[must_use]
    pub fn yxz(self) -> Vec3A {
        Vec3A::new(self.y, self.x, self.z)
    }

    /// Swizzle `yyx`.
    #[inline]
    #[must_use]
    pub fn yyx(self) -> Vec3A {
        Vec3A::new(self.y, self.y, self.x)
    }

    /// Swizzle `yyy`.
    #[inline]
    #[must_use]
    pub fn yyy(self) -> Vec3A {
        Vec3A::new(self.y, self.y, self.y)
    }

    /// Swizzle `yyz`.
    #[inline]
    #[must_use]
    pub fn yyz(self) -> Vec3A {
        Vec3A::new(self.y, self.y, self.z)
    }

    /// Swizzle `yzx`.
    #[inline]
    #[must_use]
    pub fn yzx(self) -> Vec3A {
        Vec3A::new(self.y, self.z, self.x)
    }

    /// Swizzle `yzy`.
    #[inline]
    #[must_use]
    pub fn yzy(self) -> Vec3A {
        Vec3A::new(self.y, self.z, self.y)
    }

    /// Swizzle `yzz`.
    #[inline]
    #[must_use]
    pub fn yzz(self) -> Vec3A {
        Vec3A::new(self.y, self.z, self.z)
    }

    /// Swizzle `zxx`.
    #[inline]
    #[must_use]
    pub fn zxx(self) -> Vec3A {
        Vec3A::new(self.z, self.x, self.x)
    }

    /// Swizzle `zxy`.
    #[inline]
    #[must_use]
    pub fn zxy(self) -> Vec3A {
        Vec3A::new(self.z, self.x, self.y)
    }

    /// Swizzle `zxz`.
    #[inline]
    #[must_use]
    pub fn zxz(self) -> Vec3A {
        Vec3A::new(self.z, self.x, self.z)
    }

    /// Swizzle `zyx`.
    #[inline]
    #[must_use]
    pub fn zyx(self) -> Vec3A {
        Vec3A::new(self.z, self.y, self.x)
    }

    /// Swizzle `zyy`.
    #[inline]
    #[must_use]
    pub fn zyy(self) -> Vec3A {
        Vec3A::new(self.z, self.y, self.y)
    }

    /// Swizzle `zyz`.
    #[inline]
    #[must_use]
    pub fn zyz(self) -> Vec3A {
        Vec3A::new(self.z, self.y, self.z)
    }

    /// Swizzle `zzx`.
    #[inline]
    #[must_use]
    pub fn zzx(self) -> Vec3A {
        Vec3A::new(self.z, self.z, self.x)
    }

    /// Swizzle `zzy`.
    #[inline]
    #[must_use]
    pub fn zzy(self) -> Vec3A {
        Vec3A::new(self.z, self.z, self.y)
    }

    /// Swizzle `zzz`.
    #[inline]
    #[must_use]
    pub fn zzz(self) -> Vec3A {
        Vec3A::new(self.z, self.z, self.z)
    }

    /// Swizzle `xxxx`.
    #[inline]
    #[must_use]
    pub fn xxxx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.x)
    }

    /// Swizzle `xxxy`.
    #[inline]
    #[must_use]
    pub fn xxxy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.y)
    }

    /// Swizzle `xxxz`.
    #[inline]
    #[must_use]
    pub fn xxxz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.z)
    }

    /// Swizzle `xxyx`.
    #[inline]
    #[must_use]
    pub fn xxyx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.x)
    }

    /// Swizzle `xxyy`.
    #[inline]
    #[must_use]
    pub fn xxyy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.y)
    }

    /// Swizzle `xxyz`.
    #[inline]
    #[must_use]
    pub fn xxyz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.z)
    }

    /// Swizzle `xxzx`.
    #[inline]
    #[must_use]
    pub fn xxzx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.x)
    }

    /// Swizzle `xxzy`.
    #[inline]
    #[must_use]
    pub fn xxzy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.y)
    }

    /// Swizzle `xxzz`.
    #[inline]
    #[must_use]
    pub fn xxzz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.z)
    }

    /// Swizzle `xyxx`.
    #[inline]
    #[must_use]
    pub fn xyxx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.x)
    }

    /// Swizzle `xyxy`.
    #[inline]
    #[must_use]
    pub fn xyxy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.y)
    }

    /// Swizzle `xyxz`.
    #[inline]
    #[must_use]
    pub fn xyxz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.z)
    }

    /// Swizzle `xyyx`.
    #[inline]
    #[must_use]
    pub fn xyyx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.x)
    }

    /// Swizzle `xyyy`.
    #[inline]
    #[must_use]
    pub fn xyyy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.y)
    }

    /// Swizzle `xyyz`.
    #[inline]
    #[must_use]
    pub fn xyyz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.z)
    }

    /// Swizzle `xyzx`.
    #[inline]
    #[must_use]
    pub fn xyzx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.x)
    }

    /// Swizzle `xyzy`.
    #[inline]
    #[must_use]
    pub fn xyzy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.y)
    }

    /// Swizzle `xyzz`.
    #[inline]
    #[must_use]
    pub fn xyzz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.z)
    }

    /// Swizzle `xzxx`.
    #[inline]
    #[must_use]
    pub fn xzxx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.x)
    }

    /// Swizzle `xzxy`.
    #[inline]
    #[must_use]
    pub fn xzxy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.y)
    }

    /// Swizzle `xzxz`.
    #[inline]
    #[must_use]
    pub fn xzxz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.z)
    }

    /// Swizzle `xzyx`.
    #[inline]
    #[must_use]
    pub fn xzyx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.x)
    }

    /// Swizzle `xzyy`.
    #[inline]
    #[must_use]
    pub fn xzyy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.y)
    }

    /// Swizzle `xzyz`.
    #[inline]
    #[must_use]
    pub fn xzyz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.z)
    }

    /// Swizzle `xzzx`.
    #[inline]
    #[must_use]
    pub fn xzzx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.x)
    }

    /// Swizzle `xzzy`.
    #[inline]
    #[must_use]
    pub fn xzzy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.y)
    }

    /// Swizzle `xzzz`.
    #[inline]
    #[must_use]
    pub fn xzzz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.z)
    }

    /// Swizzle `yxxx`.
    #[inline]
    #[must_use]
    pub fn yxxx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.x)
    }

    /// Swizzle `yxxy`.
    #[inline]
    #[must_use]
    pub fn yxxy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.y)
    }

    /// Swizzle `yxxz`.
    #[inline]
    #[must_use]
    pub fn yxxz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.z)
    }

    /// Swizzle `yxyx`.
    #[inline]
    #[must_use]
    pub fn yxyx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.x)
    }

    /// Swizzle `yxyy`.
    #[inline]
    #[must_use]
    pub fn yxyy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.y)
    }

    /// Swizzle `yxyz`.
    #[inline]
    #[must_use]
    pub fn yxyz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.z)
    }

    /// Swizzle `yxzx`.
    #[inline]
    #[must_use]
    pub fn yxzx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.x)
    }

    /// Swizzle `yxzy`.
    #[inline]
    #[must_use]
    pub fn yxzy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.y)
    }

    /// Swizzle `yxzz`.
    #[inline]
    #[must_use]
    pub fn yxzz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.z)
    }

    /// Swizzle `yyxx`.
    #[inline]
    #[must_use]
    pub fn yyxx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.x)
    }

    /// Swizzle `yyxy`.
    #[inline]
    #[must_use]
    pub fn yyxy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.y)
    }

    /// Swizzle `yyxz`.
    #[inline]
    #[must_use]
    pub fn yyxz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.z)
    }

    /// Swizzle `yyyx`.
    #[inline]
    #[must_use]
    pub fn yyyx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.x)
    }

    /// Swizzle `yyyy`.
    #[inline]
    #[must_use]
    pub fn yyyy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.y)
    }

    /// Swizzle `yyyz`.
    #[inline]
    #[must_use]
    pub fn yyyz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.z)
    }

    /// Swizzle `yyzx`.
    #[inline]
    #[must_use]
    pub fn yyzx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.x)
    }

    /// Swizzle `yyzy`.
    #[inline]
    #[must_use]
    pub fn yyzy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.y)
    }

    /// Swizzle `yyzz`.
    #[inline]
    #[must_use]
    pub fn yyzz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.z)
    }

    /// Swizzle `yzxx`.
    #[inline]
    #[must_use]
    pub fn yzxx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.x)
    }

    /// Swizzle `yzxy`.
    #[inline]
    #[must_use]
    pub fn yzxy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.y)
    }

    /// Swizzle `yzxz`.
    #[inline]
    #[must_use]
    pub fn yzxz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.z)
    }

    /// Swizzle `yzyx`.
    #[inline]
    #[must_use]
    pub fn yzyx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.x)
    }

    /// Swizzle `yzyy`.
    #[inline]
    #[must_use]
    pub fn yzyy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.y)
    }

    /// Swizzle `yzyz`.
    #[inline]
    #[must_use]
    pub fn yzyz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.z)
    }

    /// Swizzle `yzzx`.
    #[inline]
    #[must_use]
    pub fn yzzx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.x)
    }

    /// Swizzle `yzzy`.
    #[inline]
    #[must_use]
    pub fn yzzy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.y)
    }

    /// Swizzle `yzzz`.
    #[inline]
    #[must_use]
    pub fn yzzz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.z)
    }

    /// Swizzle `zxxx`.
    #[inline]
    #[must_use]
    pub fn zxxx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.x)
    }

    /// Swizzle `zxxy`.
    #[inline]
    #[must_use]
    pub fn zxxy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.y)
    }

    /// Swizzle `zxxz`.
    #[inline]
    #[must_use]
    pub fn zxxz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.z)
    }

    /// Swizzle `zxyx`.
    #[inline]
    #[must_use]
    pub fn zxyx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.x)
    }

    /// Swizzle `zxyy`.
    #[inline]
    #[must_use]
    pub fn zxyy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.y)
    }

    /// Swizzle `zxyz`.
    #[inline]
    #[must_use]
    pub fn zxyz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.z)
    }

    /// Swizzle `zxzx`.
    #[inline]
    #[must_use]
    pub fn zxzx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.x)
    }

    /// Swizzle `zxzy`.
    #[inline]
    #[must_use]
    pub fn zxzy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.y)
    }

    /// Swizzle `zxzz`.
    #[inline]
    #[must_use]
    pub fn zxzz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.z)
    }

    /// Swizzle `zyxx`.
    #[inline]
    #[must_use]
    pub fn zyxx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.x)
    }

    /// Swizzle `zyxy`.
    #[inline]
    #[must_use]
    pub fn zyxy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.y)
    }

    /// Swizzle `zyxz`.
    #[inline]
    #[must_use]
    pub fn zyxz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.z)
    }

    /// Swizzle `zyyx`.
    #[inline]
    #[must_use]
    pub fn zyyx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.x)
    }

    /// Swizzle `zyyy`.
    #[inline]
    #[must_use]
    pub fn zyyy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.y)
    }

    /// Swizzle `zyyz`.
    #[inline]
    #[must_use]
    pub fn zyyz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.z)
    }

    /// Swizzle `zyzx`.
    #[inline]
    #[must_use]
    pub fn zyzx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.x)
    }

    /// Swizzle `zyzy`.
    #[inline]
    #[must_use]
    pub fn zyzy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.y)
    }

    /// Swizzle `zyzz`.
    #[inline]
    #[must_use]
    pub fn zyzz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.z)
    }

    /// Swizzle `zzxx`.
    #[inline]
    #[must_use]
    pub fn zzxx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.x)
    }

    /// Swizzle `zzxy`.
    #[inline]
    #[must_use]
    pub fn zzxy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.y)
    }

    /// Swizzle `zzxz`.
    #[inline]
    #[must_use]
    pub fn zzxz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.z)
    }

    /// Swizzle `zzyx`.
    #[inline]
    #[must_use]
    pub fn zzyx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.x)
    }

    /// Swizzle `zzyy`.
    #[inline]
    #[must_use]
    pub fn zzyy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.y)
    }

    /// Swizzle `zzyz`.
    #[inline]
    #[must_use]
    pub fn zzyz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.z)
    }

    /// Swizzle `zzzx`.
    #[inline]
    #[must_use]
    pub fn zzzx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.x)
    }

    /// Swizzle `zzzy`.
    #[inline]
    #[must_use]
    pub fn zzzy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.y)
    }

    /// Swizzle `zzzz`.
    #[inline]
    #[must_use]
    pub fn zzzz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.z)
    }
}

impl Vec4 {
    /// Swizzle `xx`.
    #[inline]
    #[must_use]
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Swizzle `xy`.
    #[inline]
    #[must_use]
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Swizzle `xz`.
    #[inline]
    #[must_use]
    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Swizzle `xw`.
    #[inline]
    #[must_use]
    pub fn xw(self) -> Vec2 {
        Vec2::new(self.x, self.w)
    }

    /// Swizzle `yx`.
    #[inline]
    #[must_use]
    pub fn yx(self) -> Vec2 {
        Vec2::new(self.y, self.x)
    }

    /// Swizzle `yy`.
    #[inline]
    #[must_use]
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Swizzle `yz`.
    #[inline]
    #[must_use]
    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Swizzle `yw`.
    #[inline]
    #[must_use]
    pub fn yw(self) -> Vec2 {
        Vec2::new(self.y, self.w)
    }

    /// Swizzle `zx`.
    #[inline]
    #[must_use]
    pub fn zx(self) -> Vec2 {
        Vec2::new(self.z, self.x)
    }

    /// Swizzle `zy`.
    #[inline]
    #[must_use]
    pub fn zy(self) -> Vec2 {
        Vec2::new(self.z, self.y)
    }

    /// Swizzle `zz`.
    #[inline]
    #[must_use]
    pub fn zz(self) -> Vec2 {
        Vec2::new(self.z, self.z)
    }

    /// Swizzle `zw`.
    #[inline]
    #[must_use]
    pub fn zw(self) -> Vec2 {
        Vec2::new(self.z, self.w)
    }

    /// Swizzle `wx`.
    #[inline]
    #[must_use]
    pub fn wx(self) -> Vec2 {
        Vec2::new(self.w, self.x)
    }

    /// Swizzle `wy`.
    #[inline]
    #[must_use]
    pub fn wy(self) -> Vec2 {
        Vec2::new(self.w, self.y)
    }

    /// Swizzle `wz`.
    #[inline]
    #[must_use]
    pub fn wz(self) -> Vec2 {
        Vec2::new(self.w, self.z)
    }

    /// Swizzle `ww`.
    #[inline]
    #[must_use]
    pub fn ww(self) -> Vec2 {
        Vec2::new(self.w, self.w)
    }

    /// Swizzle `xxx`.
    #[inline]
    #[must_use]
    pub fn xxx(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.x)
    }

    /// Swizzle `xxy`.
    #[inline]
    #[must_use]
    pub fn xxy(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.y)
    }

    /// Swizzle `xxz`.
    #[inline]
    #[must_use]
    pub fn xxz(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.z)
    }

    /// Swizzle `xxw`.
    #[inline]
    #[must_use]
    pub fn xxw(self) -> Vec3 {
        Vec3::new(self.x, self.x, self.w)
    }

    /// Swizzle `xyx`.
    #[inline]
    #[must_use]
    pub fn xyx(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.x)
    }

    /// Swizzle `xyy`.
    #[inline]
    #[must_use]
    pub fn xyy(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.y)
    }

    /// Swizzle `xyz`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Swizzle `xyw`.
    #[inline]
    #[must_use]
    pub fn xyw(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.w)
    }

    /// Swizzle `xzx`.
    #[inline]
    #[must_use]
    pub fn xzx(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.x)
    }

    /// Swizzle `xzy`.
    #[inline]
    #[must_use]
    pub fn xzy(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.y)
    }

    /// Swizzle `xzz`.
    #[inline]
    #[must_use]
    pub fn xzz(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.z)
    }

    /// Swizzle `xzw`.
    #[inline]
    #[must_use]
    pub fn xzw(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.w)
    }

    /// Swizzle `xwx`.
    #[inline]
    #[must_use]
    pub fn xwx(self) -> Vec3 {
        Vec3::new(self.x, self.w, self.x)
    }

    /// Swizzle `xwy`.
    #[inline]
    #[must_use]
    pub fn xwy(self) -> Vec3 {
        Vec3::new(self.x, self.w, self.y)
    }

    /// Swizzle `xwz`.
    #[inline]
    #[must_use]
    pub fn xwz(self) -> Vec3 {
        Vec3::new(self.x, self.w, self.z)
    }

    /// Swizzle `xww`.
    #[inline]
    #[must_use]
    pub fn xww(self) -> Vec3 {
        Vec3::new(self.x, self.w, self.w)
    }

    /// Swizzle `yxx`.
    #[inline]
    #[must_use]
    pub fn yxx(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.x)
    }

    /// Swizzle `yxy`.
    #[inline]
    #[must_use]
    pub fn yxy(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.y)
    }

    /// Swizzle `yxz`.
    #[inline]
    #[must_use]
    pub fn yxz(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.z)
    }

    /// Swizzle `yxw`.
    #[inline]
    #[must_use]
    pub fn yxw(self) -> Vec3 {
        Vec3::new(self.y, self.x, self.w)
    }

    /// Swizzle `yyx`.
    #[inline]
    #[must_use]
    pub fn yyx(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.x)
    }

    /// Swizzle `yyy`.
    #[inline]
    #[must_use]
    pub fn yyy(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.y)
    }

    /// Swizzle `yyz`.
    #[inline]
    #[must_use]
    pub fn yyz(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.z)
    }

    /// Swizzle `yyw`.
    #[inline]
    #[must_use]
    pub fn yyw(self) -> Vec3 {
        Vec3::new(self.y, self.y, self.w)
    }

    /// Swizzle `yzx`.
    #[inline]
    #[must_use]
    pub fn yzx(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.x)
    }

    /// Swizzle `yzy`.
    #[inline]
    #[must_use]
    pub fn yzy(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.y)
    }

    /// Swizzle `yzz`.
    #[inline]
    #[must_use]
    pub fn yzz(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.z)
    }

    /// Swizzle `yzw`.
    #[inline]
    #[must_use]
    pub fn yzw(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.w)
    }

    /// Swizzle `ywx`.
    #[inline]
    #[must_use]
    pub fn ywx(self) -> Vec3 {
        Vec3::new(self.y, self.w, self.x)
    }

    /// Swizzle `ywy`.
    #[inline]
    #[must_use]
    pub fn ywy(self) -> Vec3 {
        Vec3::new(self.y, self.w, self.y)
    }

    /// Swizzle `ywz`.
    #[inline]
    #[must_use]
    pub fn ywz(self) -> Vec3 {
        Vec3::new(self.y, self.w, self.z)
    }

    /// Swizzle `yww`.
    #[inline]
    #[must_use]
    pub fn yww(self) -> Vec3 {
        Vec3::new(self.y, self.w, self.w)
    }

    /// Swizzle `zxx`.
    #[inline]
    #[must_use]
    pub fn zxx(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.x)
    }

    /// Swizzle `zxy`.
    #[inline]
    #[must_use]
    pub fn zxy(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.y)
    }

    /// Swizzle `zxz`.
    #[inline]
    #[must_use]
    pub fn zxz(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.z)
    }

    /// Swizzle `zxw`.
    #[inline]
    #[must_use]
    pub fn zxw(self) -> Vec3 {
        Vec3::new(self.z, self.x, self.w)
    }

    /// Swizzle `zyx`.
    #[inline]
    #[must_use]
    pub fn zyx(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.x)
    }

    /// Swizzle `zyy`.
    #[inline]
    #[must_use]
    pub fn zyy(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.y)
    }

    /// Swizzle `zyz`.
    #[inline]
    #[must_use]
    pub fn zyz(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.z)
    }

    /// Swizzle `zyw`.
    #[inline]
    #[must_use]
    pub fn zyw(self) -> Vec3 {
        Vec3::new(self.z, self.y, self.w)
    }

    /// Swizzle `zzx`.
    #[inline]
    #[must_use]
    pub fn zzx(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.x)
    }

    /// Swizzle `zzy`.
    #[inline]
    #[must_use]
    pub fn zzy(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.y)
    }

    /// Swizzle `zzz`.
    #[inline]
    #[must_use]
    pub fn zzz(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.z)
    }

    /// Swizzle `zzw`.
    #[inline]
    #[must_use]
    pub fn zzw(self) -> Vec3 {
        Vec3::new(self.z, self.z, self.w)
    }

    /// Swizzle `zwx`.
    #[inline]
    #[must_use]
    pub fn zwx(self) -> Vec3 {
        Vec3::new(self.z, self.w, self.x)
    }

    /// Swizzle `zwy`.
    #[inline]
    #[must_use]
    pub fn zwy(self) -> Vec3 {
        Vec3::new(self.z, self.w, self.y)
    }

    /// Swizzle `zwz`.
    #[inline]
    #[must_use]
    pub fn zwz(self) -> Vec3 {
        Vec3::new(self.z, self.w, self.z)
    }

    /// Swizzle `zww`.
    #[inline]
    #[must_use]
    pub fn zww(self) -> Vec3 {
        Vec3::new(self.z, self.w, self.w)
    }

    /// Swizzle `wxx`.
    #[inline]
    #[must_use]
    pub fn wxx(self) -> Vec3 {
        Vec3::new(self.w, self.x, self.x)
    }

    /// Swizzle `wxy`.
    #[inline]
    #[must_use]
    pub fn wxy(self) -> Vec3 {
        Vec3::new(self.w, self.x, self.y)
    }

    /// Swizzle `wxz`.
    #[inline]
    #[must_use]
    pub fn wxz(self) -> Vec3 {
        Vec3::new(self.w, self.x, self.z)
    }

    /// Swizzle `wxw`.
    #[inline]
    #[must_use]
    pub fn wxw(self) -> Vec3 {
        Vec3::new(self.w, self.x, self.w)
    }

    /// Swizzle `wyx`.
    #[inline]
    #[must_use]
    pub fn wyx(self) -> Vec3 {
        Vec3::new(self.w, self.y, self.x)
    }

    /// Swizzle `wyy`.
    #[inline]
    #[must_use]
    pub fn wyy(self) -> Vec3 {
        Vec3::new(self.w, self.y, self.y)
    }

    /// Swizzle `wyz`.
    #[inline]
    #[must_use]
    pub fn wyz(self) -> Vec3 {
        Vec3::new(self.w, self.y, self.z)
    }

    /// Swizzle `wyw`.
    #[inline]
    #[must_use]
    pub fn wyw(self) -> Vec3 {
        Vec3::new(self.w, self.y, self.w)
    }

    /// Swizzle `wzx`.
    #[inline]
    #[must_use]
    pub fn wzx(self) -> Vec3 {
        Vec3::new(self.w, self.z, self.x)
    }

    /// Swizzle `wzy`.
    #[inline]
    #[must_use]
    pub fn wzy(self) -> Vec3 {
        Vec3::new(self.w, self.z, self.y)
    }

    /// Swizzle `wzz`.
    #[inline]
    #[must_use]
    pub fn wzz(self) -> Vec3 {
        Vec3::new(self.w, self.z, self.z)
    }

    /// Swizzle `wzw`.
    #[inline]
    #[must_use]
    pub fn wzw(self) -> Vec3 {
        Vec3::new(self.w, self.z, self.w)
    }

    /// Swizzle `wwx`.
    #[inline]
    #[must_use]
    pub fn wwx(self) -> Vec3 {
        Vec3::new(self.w, self.w, self.x)
    }

    /// Swizzle `wwy`.
    #[inline]
    #[must_use]
    pub fn wwy(self) -> Vec3 {
        Vec3::new(self.w, self.w, self.y)
    }

    /// Swizzle `wwz`.
    #[inline]
    #[must_use]
    pub fn wwz(self) -> Vec3 {
        Vec3::new(self.w, self.w, self.z)
    }

    /// Swizzle `www`.
    #[inline]
    #[must_use]
    pub fn www(self) -> Vec3 {
        Vec3::new(self.w, self.w, self.w)
    }

    /// Swizzle `xxxx`.
    #[inline]
    #[must_use]
    pub fn xxxx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.x)
    }

    /// Swizzle `xxxy`.
    #[inline]
    #[must_use]
    pub fn xxxy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.y)
    }

    /// Swizzle `xxxz`.
    #[inline]
    #[must_use]
    pub fn xxxz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.z)
    }

    /// Swizzle `xxxw`.
    #[inline]
    #[must_use]
    pub fn xxxw(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.x, self.w)
    }

    /// Swizzle `xxyx`.
    #[inline]
    #[must_use]
    pub fn xxyx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.x)
    }

    /// Swizzle `xxyy`.
    #[inline]
    #[must_use]
    pub fn xxyy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.y)
    }

    /// Swizzle `xxyz`.
    #[inline]
    #[must_use]
    pub fn xxyz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.z)
    }

    /// Swizzle `xxyw`.
    #[inline]
    #[must_use]
    pub fn xxyw(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.y, self.w)
    }

    /// Swizzle `xxzx`.
    #[inline]
    #[must_use]
    pub fn xxzx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.x)
    }

    /// Swizzle `xxzy`.
    #[inline]
    #[must_use]
    pub fn xxzy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.y)
    }

    /// Swizzle `xxzz`.
    #[inline]
    #[must_use]
    pub fn xxzz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.z)
    }

    /// Swizzle `xxzw`.
    #[inline]
    #[must_use]
    pub fn xxzw(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.z, self.w)
    }

    /// Swizzle `xxwx`.
    #[inline]
    #[must_use]
    pub fn xxwx(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.w, self.x)
    }

    /// Swizzle `xxwy`.
    #[inline]
    #[must_use]
    pub fn xxwy(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.w, self.y)
    }

    /// Swizzle `xxwz`.
    #[inline]
    #[must_use]
    pub fn xxwz(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.w, self.z)
    }

    /// Swizzle `xxww`.
    #[inline]
    #[must_use]
    pub fn xxww(self) -> Vec4 {
        Vec4::new(self.x, self.x, self.w, self.w)
    }

    /// Swizzle `xyxx`.
    #[inline]
    #[must_use]
    pub fn xyxx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.x)
    }

    /// Swizzle `xyxy`.
    #[inline]
    #[must_use]
    pub fn xyxy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.y)
    }

    /// Swizzle `xyxz`.
    #[inline]
    #[must_use]
    pub fn xyxz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.z)
    }

    /// Swizzle `xyxw`.
    #[inline]
    #[must_use]
    pub fn xyxw(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.x, self.w)
    }

    /// Swizzle `xyyx`.
    #[inline]
    #[must_use]
    pub fn xyyx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.x)
    }

    /// Swizzle `xyyy`.
    #[inline]
    #[must_use]
    pub fn xyyy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.y)
    }

    /// Swizzle `xyyz`.
    #[inline]
    #[must_use]
    pub fn xyyz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.z)
    }

    /// Swizzle `xyyw`.
    #[inline]
    #[must_use]
    pub fn xyyw(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.y, self.w)
    }

    /// Swizzle `xyzx`.
    #[inline]
    #[must_use]
    pub fn xyzx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.x)
    }

    /// Swizzle `xyzy`.
    #[inline]
    #[must_use]
    pub fn xyzy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.y)
    }

    /// Swizzle `xyzz`.
    #[inline]
    #[must_use]
    pub fn xyzz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.z)
    }

    /// Swizzle `xyzw`.
    #[inline]
    #[must_use]
    pub fn xyzw(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.w)
    }

    /// Swizzle `xywx`.
    #[inline]
    #[must_use]
    pub fn xywx(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.w, self.x)
    }

    /// Swizzle `xywy`.
    #[inline]
    #[must_use]
    pub fn xywy(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.w, self.y)
    }

    /// Swizzle `xywz`.
    #[inline]
    #[must_use]
    pub fn xywz(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.w, self.z)
    }

    /// Swizzle `xyww`.
    #[inline]
    #[must_use]
    pub fn xyww(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.w, self.w)
    }

    /// Swizzle `xzxx`.
    #[inline]
    #[must_use]
    pub fn xzxx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.x)
    }

    /// Swizzle `xzxy`.
    #[inline]
    #[must_use]
    pub fn xzxy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.y)
    }

    /// Swizzle `xzxz`.
    #[inline]
    #[must_use]
    pub fn xzxz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.z)
    }

    /// Swizzle `xzxw`.
    #[inline]
    #[must_use]
    pub fn xzxw(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.x, self.w)
    }

    /// Swizzle `xzyx`.
    #[inline]
    #[must_use]
    pub fn xzyx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.x)
    }

    /// Swizzle `xzyy`.
    #[inline]
    #[must_use]
    pub fn xzyy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.y)
    }

    /// Swizzle `xzyz`.
    #[inline]
    #[must_use]
    pub fn xzyz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.z)
    }

    /// Swizzle `xzyw`.
    #[inline]
    #[must_use]
    pub fn xzyw(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.y, self.w)
    }

    /// Swizzle `xzzx`.
    #[inline]
    #[must_use]
    pub fn xzzx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.x)
    }

    /// Swizzle `xzzy`.
    #[inline]
    #[must_use]
    pub fn xzzy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.y)
    }

    /// Swizzle `xzzz`.
    #[inline]
    #[must_use]
    pub fn xzzz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.z)
    }

    /// Swizzle `xzzw`.
    #[inline]
    #[must_use]
    pub fn xzzw(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.z, self.w)
    }

    /// Swizzle `xzwx`.
    #[inline]
    #[must_use]
    pub fn xzwx(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.w, self.x)
    }

    /// Swizzle `xzwy`.
    #[inline]
    #[must_use]
    pub fn xzwy(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.w, self.y)
    }

    /// Swizzle `xzwz`.
    #[inline]
    #[must_use]
    pub fn xzwz(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.w, self.z)
    }

    /// Swizzle `xzww`.
    #[inline]
    #[must_use]
    pub fn xzww(self) -> Vec4 {
        Vec4::new(self.x, self.z, self.w, self.w)
    }

    /// Swizzle `xwxx`.
    #[inline]
    #[must_use]
    pub fn xwxx(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.x, self.x)
    }

    /// Swizzle `xwxy`.
    #[inline]
    #[must_use]
    pub fn xwxy(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.x, self.y)
    }

    /// Swizzle `xwxz`.
    #[inline]
    #[must_use]
    pub fn xwxz(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.x, self.z)
    }

    /// Swizzle `xwxw`.
    #[inline]
    #[must_use]
    pub fn xwxw(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.x, self.w)
    }

    /// Swizzle `xwyx`.
    #[inline]
    #[must_use]
    pub fn xwyx(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.y, self.x)
    }

    /// Swizzle `xwyy`.
    #[inline]
    #[must_use]
    pub fn xwyy(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.y, self.y)
    }

    /// Swizzle `xwyz`.
    #[inline]
    #[must_use]
    pub fn xwyz(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.y, self.z)
    }

    /// Swizzle `xwyw`.
    #[inline]
    #[must_use]
    pub fn xwyw(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.y, self.w)
    }

    /// Swizzle `xwzx`.
    #[inline]
    #[must_use]
    pub fn xwzx(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.z, self.x)
    }

    /// Swizzle `xwzy`.
    #[inline]
    #[must_use]
    pub fn xwzy(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.z, self.y)
    }

    /// Swizzle `xwzz`.
    #[inline]
    #[must_use]
    pub fn xwzz(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.z, self.z)
    }

    /// Swizzle `xwzw`.
    #[inline]
    #[must_use]
    pub fn xwzw(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.z, self.w)
    }

    /// Swizzle `xwwx`.
    #[inline]
    #[must_use]
    pub fn xwwx(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.w, self.x)
    }

    /// Swizzle `xwwy`.
    #[inline]
    #[must_use]
    pub fn xwwy(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.w, self.y)
    }

    /// Swizzle `xwwz`.
    #[inline]
    #[must_use]
    pub fn xwwz(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.w, self.z)
    }

    /// Swizzle `xwww`.
    #[inline]
    #[must_use]
    pub fn xwww(self) -> Vec4 {
        Vec4::new(self.x, self.w, self.w, self.w)
    }

    /// Swizzle `yxxx`.
    #[inline]
    #[must_use]
    pub fn yxxx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.x)
    }

    /// Swizzle `yxxy`.
    #[inline]
    #[must_use]
    pub fn yxxy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.y)
    }

    /// Swizzle `yxxz`.
    #[inline]
    #[must_use]
    pub fn yxxz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.z)
    }

    /// Swizzle `yxxw`.
    #[inline]
    #[must_use]
    pub fn yxxw(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.x, self.w)
    }

    /// Swizzle `yxyx`.
    #[inline]
    #[must_use]
    pub fn yxyx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.x)
    }

    /// Swizzle `yxyy`.
    #[inline]
    #[must_use]
    pub fn yxyy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.y)
    }

    /// Swizzle `yxyz`.
    #[inline]
    #[must_use]
    pub fn yxyz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.z)
    }

    /// Swizzle `yxyw`.
    #[inline]
    #[must_use]
    pub fn yxyw(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.y, self.w)
    }

    /// Swizzle `yxzx`.
    #[inline]
    #[must_use]
    pub fn yxzx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.x)
    }

    /// Swizzle `yxzy`.
    #[inline]
    #[must_use]
    pub fn yxzy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.y)
    }

    /// Swizzle `yxzz`.
    #[inline]
    #[must_use]
    pub fn yxzz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.z)
    }

    /// Swizzle `yxzw`.
    #[inline]
    #[must_use]
    pub fn yxzw(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.z, self.w)
    }

    /// Swizzle `yxwx`.
    #[inline]
    #[must_use]
    pub fn yxwx(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.w, self.x)
    }

    /// Swizzle `yxwy`.
    #[inline]
    #[must_use]
    pub fn yxwy(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.w, self.y)
    }

    /// Swizzle `yxwz`.
    #[inline]
    #[must_use]
    pub fn yxwz(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.w, self.z)
    }

    /// Swizzle `yxww`.
    #[inline]
    #[must_use]
    pub fn yxww(self) -> Vec4 {
        Vec4::new(self.y, self.x, self.w, self.w)
    }

    /// Swizzle `yyxx`.
    #[inline]
    #[must_use]
    pub fn yyxx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.x)
    }

    /// Swizzle `yyxy`.
    #[inline]
    #[must_use]
    pub fn yyxy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.y)
    }

    /// Swizzle `yyxz`.
    #[inline]
    #[must_use]
    pub fn yyxz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.z)
    }

    /// Swizzle `yyxw`.
    #[inline]
    #[must_use]
    pub fn yyxw(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.x, self.w)
    }

    /// Swizzle `yyyx`.
    #[inline]
    #[must_use]
    pub fn yyyx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.x)
    }

    /// Swizzle `yyyy`.
    #[inline]
    #[must_use]
    pub fn yyyy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.y)
    }

    /// Swizzle `yyyz`.
    #[inline]
    #[must_use]
    pub fn yyyz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.z)
    }

    /// Swizzle `yyyw`.
    #[inline]
    #[must_use]
    pub fn yyyw(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.y, self.w)
    }

    /// Swizzle `yyzx`.
    #[inline]
    #[must_use]
    pub fn yyzx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.x)
    }

    /// Swizzle `yyzy`.
    #[inline]
    #[must_use]
    pub fn yyzy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.y)
    }

    /// Swizzle `yyzz`.
    #[inline]
    #[must_use]
    pub fn yyzz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.z)
    }

    /// Swizzle `yyzw`.
    #[inline]
    #[must_use]
    pub fn yyzw(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.z, self.w)
    }

    /// Swizzle `yywx`.
    #[inline]
    #[must_use]
    pub fn yywx(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.w, self.x)
    }

    /// Swizzle `yywy`.
    #[inline]
    #[must_use]
    pub fn yywy(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.w, self.y)
    }

    /// Swizzle `yywz`.
    #[inline]
    #[must_use]
    pub fn yywz(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.w, self.z)
    }

    /// Swizzle `yyww`.
    #[inline]
    #[must_use]
    pub fn yyww(self) -> Vec4 {
        Vec4::new(self.y, self.y, self.w, self.w)
    }

    /// Swizzle `yzxx`.
    #[inline]
    #[must_use]
    pub fn yzxx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.x)
    }

    /// Swizzle `yzxy`.
    #[inline]
    #[must_use]
    pub fn yzxy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.y)
    }

    /// Swizzle `yzxz`.
    #[inline]
    #[must_use]
    pub fn yzxz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.z)
    }

    /// Swizzle `yzxw`.
    #[inline]
    #[must_use]
    pub fn yzxw(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.x, self.w)
    }

    /// Swizzle `yzyx`.
    #[inline]
    #[must_use]
    pub fn yzyx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.x)
    }

    /// Swizzle `yzyy`.
    #[inline]
    #[must_use]
    pub fn yzyy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.y)
    }

    /// Swizzle `yzyz`.
    #[inline]
    #[must_use]
    pub fn yzyz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.z)
    }

    /// Swizzle `yzyw`.
    #[inline]
    #[must_use]
    pub fn yzyw(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.y, self.w)
    }

    /// Swizzle `yzzx`.
    #[inline]
    #[must_use]
    pub fn yzzx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.x)
    }

    /// Swizzle `yzzy`.
    #[inline]
    #[must_use]
    pub fn yzzy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.y)
    }

    /// Swizzle `yzzz`.
    #[inline]
    #[must_use]
    pub fn yzzz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.z)
    }

    /// Swizzle `yzzw`.
    #[inline]
    #[must_use]
    pub fn yzzw(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.z, self.w)
    }

    /// Swizzle `yzwx`.
    #[inline]
    #[must_use]
    pub fn yzwx(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.w, self.x)
    }

    /// Swizzle `yzwy`.
    #[inline]
    #[must_use]
    pub fn yzwy(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.w, self.y)
    }

    /// Swizzle `yzwz`.
    #[inline]
    #[must_use]
    pub fn yzwz(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.w, self.z)
    }

    /// Swizzle `yzww`.
    #[inline]
    #[must_use]
    pub fn yzww(self) -> Vec4 {
        Vec4::new(self.y, self.z, self.w, self.w)
    }

    /// Swizzle `ywxx`.
    #[inline]
    #[must_use]
    pub fn ywxx(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.x, self.x)
    }

    /// Swizzle `ywxy`.
    #[inline]
    #[must_use]
    pub fn ywxy(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.x, self.y)
    }

    /// Swizzle `ywxz`.
    #[inline]
    #[must_use]
    pub fn ywxz(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.x, self.z)
    }

    /// Swizzle `ywxw`.
    #[inline]
    #[must_use]
    pub fn ywxw(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.x, self.w)
    }

    /// Swizzle `ywyx`.
    #[inline]
    #[must_use]
    pub fn ywyx(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.y, self.x)
    }

    /// Swizzle `ywyy`.
    #[inline]
    #[must_use]
    pub fn ywyy(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.y, self.y)
    }

    /// Swizzle `ywyz`.
    #[inline]
    #[must_use]
    pub fn ywyz(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.y, self.z)
    }

    /// Swizzle `ywyw`.
    #[inline]
    #[must_use]
    pub fn ywyw(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.y, self.w)
    }

    /// Swizzle `ywzx`.
    #[inline]
    #[must_use]
    pub fn ywzx(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.z, self.x)
    }

    /// Swizzle `ywzy`.
    #[inline]
    #[must_use]
    pub fn ywzy(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.z, self.y)
    }

    /// Swizzle `ywzz`.
    #[inline]
    #[must_use]
    pub fn ywzz(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.z, self.z)
    }

    /// Swizzle `ywzw`.
    #[inline]
    #[must_use]
    pub fn ywzw(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.z, self.w)
    }

    /// Swizzle `ywwx`.
    #[inline]
    #[must_use]
    pub fn ywwx(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.w, self.x)
    }

    /// Swizzle `ywwy`.
    #[inline]
    #[must_use]
    pub fn ywwy(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.w, self.y)
    }

    /// Swizzle `ywwz`.
    #[inline]
    #[must_use]
    pub fn ywwz(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.w, self.z)
    }

    /// Swizzle `ywww`.
    #[inline]
    #[must_use]
    pub fn ywww(self) -> Vec4 {
        Vec4::new(self.y, self.w, self.w, self.w)
    }

    /// Swizzle `zxxx`.
    #[inline]
    #[must_use]
    pub fn zxxx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.x)
    }

    /// Swizzle `zxxy`.
    #[inline]
    #[must_use]
    pub fn zxxy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.y)
    }

    /// Swizzle `zxxz`.
    #[inline]
    #[must_use]
    pub fn zxxz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.z)
    }

    /// Swizzle `zxxw`.
    #[inline]
    #[must_use]
    pub fn zxxw(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.x, self.w)
    }

    /// Swizzle `zxyx`.
    #[inline]
    #[must_use]
    pub fn zxyx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.x)
    }

    /// Swizzle `zxyy`.
    #[inline]
    #[must_use]
    pub fn zxyy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.y)
    }

    /// Swizzle `zxyz`.
    #[inline]
    #[must_use]
    pub fn zxyz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.z)
    }

    /// Swizzle `zxyw`.
    #[inline]
    #[must_use]
    pub fn zxyw(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.y, self.w)
    }

    /// Swizzle `zxzx`.
    #[inline]
    #[must_use]
    pub fn zxzx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.x)
    }

    /// Swizzle `zxzy`.
    #[inline]
    #[must_use]
    pub fn zxzy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.y)
    }

    /// Swizzle `zxzz`.
    #[inline]
    #[must_use]
    pub fn zxzz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.z)
    }

    /// Swizzle `zxzw`.
    #[inline]
    #[must_use]
    pub fn zxzw(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.z, self.w)
    }

    /// Swizzle `zxwx`.
    #[inline]
    #[must_use]
    pub fn zxwx(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.w, self.x)
    }

    /// Swizzle `zxwy`.
    #[inline]
    #[must_use]
    pub fn zxwy(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.w, self.y)
    }

    /// Swizzle `zxwz`.
    #[inline]
    #[must_use]
    pub fn zxwz(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.w, self.z)
    }

    /// Swizzle `zxww`.
    #[inline]
    #[must_use]
    pub fn zxww(self) -> Vec4 {
        Vec4::new(self.z, self.x, self.w, self.w)
    }

    /// Swizzle `zyxx`.
    #[inline]
    #[must_use]
    pub fn zyxx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.x)
    }

    /// Swizzle `zyxy`.
    #[inline]
    #[must_use]
    pub fn zyxy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.y)
    }

    /// Swizzle `zyxz`.
    #[inline]
    #[must_use]
    pub fn zyxz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.z)
    }

    /// Swizzle `zyxw`.
    #[inline]
    #[must_use]
    pub fn zyxw(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.x, self.w)
    }

    /// Swizzle `zyyx`.
    #[inline]
    #[must_use]
    pub fn zyyx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.x)
    }

    /// Swizzle `zyyy`.
    #[inline]
    #[must_use]
    pub fn zyyy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.y)
    }

    /// Swizzle `zyyz`.
    #[inline]
    #[must_use]
    pub fn zyyz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.z)
    }

    /// Swizzle `zyyw`.
    #[inline]
    #[must_use]
    pub fn zyyw(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.y, self.w)
    }

    /// Swizzle `zyzx`.
    #[inline]
    #[must_use]
    pub fn zyzx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.x)
    }

    /// Swizzle `zyzy`.
    #[inline]
    #[must_use]
    pub fn zyzy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.y)
    }

    /// Swizzle `zyzz`.
    #[inline]
    #[must_use]
    pub fn zyzz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.z)
    }

    /// Swizzle `zyzw`.
    #[inline]
    #[must_use]
    pub fn zyzw(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.z, self.w)
    }

    /// Swizzle `zywx`.
    #[inline]
    #[must_use]
    pub fn zywx(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.w, self.x)
    }

    /// Swizzle `zywy`.
    #[inline]
    #[must_use]
    pub fn zywy(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.w, self.y)
    }

    /// Swizzle `zywz`.
    #[inline]
    #[must_use]
    pub fn zywz(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.w, self.z)
    }

    /// Swizzle `zyww`.
    #[inline]
    #[must_use]
    pub fn zyww(self) -> Vec4 {
        Vec4::new(self.z, self.y, self.w, self.w)
    }

    /// Swizzle `zzxx`.
    #[inline]
    #[must_use]
    pub fn zzxx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.x)
    }

    /// Swizzle `zzxy`.
    #[inline]
    #[must_use]
    pub fn zzxy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.y)
    }

    /// Swizzle `zzxz`.
    #[inline]
    #[must_use]
    pub fn zzxz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.z)
    }

    /// Swizzle `zzxw`.
    #[inline]
    #[must_use]
    pub fn zzxw(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.x, self.w)
    }

    /// Swizzle `zzyx`.
    #[inline]
    #[must_use]
    pub fn zzyx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.x)
    }

    /// Swizzle `zzyy`.
    #[inline]
    #[must_use]
    pub fn zzyy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.y)
    }

    /// Swizzle `zzyz`.
    #[inline]
    #[must_use]
    pub fn zzyz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.z)
    }

    /// Swizzle `zzyw`.
    #[inline]
    #[must_use]
    pub fn zzyw(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.y, self.w)
    }

    /// Swizzle `zzzx`.
    #[inline]
    #[must_use]
    pub fn zzzx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.x)
    }

    /// Swizzle `zzzy`.
    #[inline]
    #[must_use]
    pub fn zzzy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.y)
    }

    /// Swizzle `zzzz`.
    #[inline]
    #[must_use]
    pub fn zzzz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.z)
    }

    /// Swizzle `zzzw`.
    #[inline]
    #[must_use]
    pub fn zzzw(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.z, self.w)
    }

    /// Swizzle `zzwx`.
    #[inline]
    #[must_use]
    pub fn zzwx(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.w, self.x)
    }

    /// Swizzle `zzwy`.
    #[inline]
    #[must_use]
    pub fn zzwy(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.w, self.y)
    }

    /// Swizzle `zzwz`.
    #[inline]
    #[must_use]
    pub fn zzwz(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.w, self.z)
    }

    /// Swizzle `zzww`.
    #[inline]
    #[must_use]
    pub fn zzww(self) -> Vec4 {
        Vec4::new(self.z, self.z, self.w, self.w)
    }

    /// Swizzle `zwxx`.
    #[inline]
    #[must_use]
    pub fn zwxx(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.x, self.x)
    }

    /// Swizzle `zwxy`.
    #[inline]
    #[must_use]
    pub fn zwxy(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.x, self.y)
    }

    /// Swizzle `zwxz`.
    #[inline]
    #[must_use]
    pub fn zwxz(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.x, self.z)
    }

    /// Swizzle `zwxw`.
    #[inline]
    #[must_use]
    pub fn zwxw(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.x, self.w)
    }

    /// Swizzle `zwyx`.
    #[inline]
    #[must_use]
    pub fn zwyx(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.y, self.x)
    }

    /// Swizzle `zwyy`.
    #[inline]
    #[must_use]
    pub fn zwyy(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.y, self.y)
    }

    /// Swizzle `zwyz`.
    #[inline]
    #[must_use]
    pub fn zwyz(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.y, self.z)
    }

    /// Swizzle `zwyw`.
    #[inline]
    #[must_use]
    pub fn zwyw(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.y, self.w)
    }

    /// Swizzle `zwzx`.
    #[inline]
    #[must_use]
    pub fn zwzx(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.z, self.x)
    }

    /// Swizzle `zwzy`.
    #[inline]
    #[must_use]
    pub fn zwzy(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.z, self.y)
    }

    /// Swizzle `zwzz`.
    #[inline]
    #[must_use]
    pub fn zwzz(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.z, self.z)
    }

    /// Swizzle `zwzw`.
    #[inline]
    #[must_use]
    pub fn zwzw(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.z, self.w)
    }

    /// Swizzle `zwwx`.
    #[inline]
    #[must_use]
    pub fn zwwx(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.w, self.x)
    }

    /// Swizzle `zwwy`.
    #[inline]
    #[must_use]
    pub fn zwwy(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.w, self.y)
    }

    /// Swizzle `zwwz`.
    #[inline]
    #[must_use]
    pub fn zwwz(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.w, self.z)
    }

    /// Swizzle `zwww`.
    #[inline]
    #[must_use]
    pub fn zwww(self) -> Vec4 {
        Vec4::new(self.z, self.w, self.w, self.w)
    }

    /// Swizzle `wxxx`.
    #[inline]
    #[must_use]
    pub fn wxxx(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.x, self.x)
    }

    /// Swizzle `wxxy`.
    #[inline]
    #[must_use]
    pub fn wxxy(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.x, self.y)
    }

    /// Swizzle `wxxz`.
    #[inline]
    #[must_use]
    pub fn wxxz(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.x, self.z)
    }

    /// Swizzle `wxxw`.
    #[inline]
    #[must_use]
    pub fn wxxw(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.x, self.w)
    }

    /// Swizzle `wxyx`.
    #[inline]
    #[must_use]
    pub fn wxyx(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.y, self.x)
    }

    /// Swizzle `wxyy`.
    #[inline]
    #[must_use]
    pub fn wxyy(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.y, self.y)
    }

    /// Swizzle `wxyz`.
    #[inline]
    #[must_use]
    pub fn wxyz(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.y, self.z)
    }

    /// Swizzle `wxyw`.
    #[inline]
    #[must_use]
    pub fn wxyw(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.y, self.w)
    }

    /// Swizzle `wxzx`.
    #[inline]
    #[must_use]
    pub fn wxzx(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.z, self.x)
    }

    /// Swizzle `wxzy`.
    #[inline]
    #[must_use]
    pub fn wxzy(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.z, self.y)
    }

    /// Swizzle `wxzz`.
    #[inline]
    #[must_use]
    pub fn wxzz(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.z, self.z)
    }

    /// Swizzle `wxzw`.
    #[inline]
    #[must_use]
    pub fn wxzw(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.z, self.w)
    }

    /// Swizzle `wxwx`.
    #[inline]
    #[must_use]
    pub fn wxwx(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.w, self.x)
    }

    /// Swizzle `wxwy`.
    #[inline]
    #[must_use]
    pub fn wxwy(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.w, self.y)
    }

    /// Swizzle `wxwz`.
    #[inline]
    #[must_use]
    pub fn wxwz(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.w, self.z)
    }

    /// Swizzle `wxww`.
    #[inline]
    #[must_use]
    pub fn wxww(self) -> Vec4 {
        Vec4::new(self.w, self.x, self.w, self.w)
    }

    /// Swizzle `wyxx`.
    #[inline]
    #[must_use]
    pub fn wyxx(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.x, self.x)
    }

    /// Swizzle `wyxy`.
    #[inline]
    #[must_use]
    pub fn wyxy(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.x, self.y)
    }

    /// Swizzle `wyxz`.
    #[inline]
    #[must_use]
    pub fn wyxz(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.x, self.z)
    }

    /// Swizzle `wyxw`.
    #[inline]
    #[must_use]
    pub fn wyxw(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.x, self.w)
    }

    /// Swizzle `wyyx`.
    #[inline]
    #[must_use]
    pub fn wyyx(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.y, self.x)
    }

    /// Swizzle `wyyy`.
    #[inline]
    #[must_use]
    pub fn wyyy(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.y, self.y)
    }

    /// Swizzle `wyyz`.
    #[inline]
    #[must_use]
    pub fn wyyz(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.y, self.z)
    }

    /// Swizzle `wyyw`.
    #[inline]
    #[must_use]
    pub fn wyyw(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.y, self.w)
    }

    /// Swizzle `wyzx`.
    #[inline]
    #[must_use]
    pub fn wyzx(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.z, self.x)
    }

    /// Swizzle `wyzy`.
    #[inline]
    #[must_use]
    pub fn wyzy(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.z, self.y)
    }

    /// Swizzle `wyzz`.
    #[inline]
    #[must_use]
    pub fn wyzz(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.z, self.z)
    }

    /// Swizzle `wyzw`.
    #[inline]
    #[must_use]
    pub fn wyzw(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.z, self.w)
    }

    /// Swizzle `wywx`.
    #[inline]
    #[must_use]
    pub fn wywx(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.w, self.x)
    }

    /// Swizzle `wywy`.
    #[inline]
    #[must_use]
    pub fn wywy(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.w, self.y)
    }

    /// Swizzle `wywz`.
    #[inline]
    #[must_use]
    pub fn wywz(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.w, self.z)
    }

    /// Swizzle `wyww`.
    #[inline]
    #[must_use]
    pub fn wyww(self) -> Vec4 {
        Vec4::new(self.w, self.y, self.w, self.w)
    }

    /// Swizzle `wzxx`.
    #[inline]
    #[must_use]
    pub fn wzxx(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.x, self.x)
    }

    /// Swizzle `wzxy`.
    #[inline]
    #[must_use]
    pub fn wzxy(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.x, self.y)
    }

    /// Swizzle `wzxz`.
    #[inline]
    #[must_use]
    pub fn wzxz(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.x, self.z)
    }

    /// Swizzle `wzxw`.
    #[inline]
    #[must_use]
    pub fn wzxw(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.x, self.w)
    }

    /// Swizzle `wzyx`.
    #[inline]
    #[must_use]
    pub fn wzyx(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.y, self.x)
    }

    /// Swizzle `wzyy`.
    #[inline]
    #[must_use]
    pub fn wzyy(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.y, self.y)
    }

    /// Swizzle `wzyz`.
    #[inline]
    #[must_use]
    pub fn wzyz(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.y, self.z)
    }

    /// Swizzle `wzyw`.
    #[inline]
    #[must_use]
    pub fn wzyw(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.y, self.w)
    }

    /// Swizzle `wzzx`.
    #[inline]
    #[must_use]
    pub fn wzzx(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.z, self.x)
    }

    /// Swizzle `wzzy`.
    #[inline]
    #[must_use]
    pub fn wzzy(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.z, self.y)
    }

    /// Swizzle `wzzz`.
    #[inline]
    #[must_use]
    pub fn wzzz(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.z, self.z)
    }

    /// Swizzle `wzzw`.
    #[inline]
    #[must_use]
    pub fn wzzw(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.z, self.w)
    }

    /// Swizzle `wzwx`.
    #[inline]
    #[must_use]
    pub fn wzwx(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.w, self.x)
    }

    /// Swizzle `wzwy`.
    #[inline]
    #[must_use]
    pub fn wzwy(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.w, self.y)
    }

    /// Swizzle `wzwz`.
    #[inline]
    #[must_use]
    pub fn wzwz(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.w, self.z)
    }

    /// Swizzle `wzww`.
    #[inline]
    #[must_use]
    pub fn wzww(self) -> Vec4 {
        Vec4::new(self.w, self.z, self.w, self.w)
    }

    /// Swizzle `wwxx`.
    #[inline]
    #[must_use]
    pub fn wwxx(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.x, self.x)
    }

    /// Swizzle `wwxy`.
    #[inline]
    #[must_use]
    pub fn wwxy(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.x, self.y)
    }

    /// Swizzle `wwxz`.
    #[inline]
    #[must_use]
    pub fn wwxz(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.x, self.z)
    }

    /// Swizzle `wwxw`.
    #[inline]
    #[must_use]
    pub fn wwxw(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.x, self.w)
    }

    /// Swizzle `wwyx`.
    #[inline]
    #[must_use]
    pub fn wwyx(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.y, self.x)
    }

    /// Swizzle `wwyy`.
    #[inline]
    #[must_use]
    pub fn wwyy(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.y, self.y)
    }

    /// Swizzle `wwyz`.
    #[inline]
    #[must_use]
    pub fn wwyz(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.y, self.z)
    }

    /// Swizzle `wwyw`.
    #[inline]
    #[must_use]
    pub fn wwyw(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.y, self.w)
    }

    /// Swizzle `wwzx`.
    #[inline]
    #[must_use]
    pub fn wwzx(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.z, self.x)
    }

    /// Swizzle `wwzy`.
    #[inline]
    #[must_use]
    pub fn wwzy(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.z, self.y)
    }

    /// Swizzle `wwzz`.
    #[inline]
    #[must_use]
    pub fn wwzz(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.z, self.z)
    }

    /// Swizzle `wwzw`.
    #[inline]
    #[must_use]
    pub fn wwzw(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.z, self.w)
    }

    /// Swizzle `wwwx`.
    #[inline]
    #[must_use]
    pub fn wwwx(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.w, self.x)
    }

    /// Swizzle `wwwy`.
    #[inline]
    #[must_use]
    pub fn wwwy(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.w, self.y)
    }

    /// Swizzle `wwwz`.
    #[inline]
    #[must_use]
    pub fn wwwz(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.w, self.z)
    }

    /// Swizzle `wwww`.
    #[inline]
    #[must_use]
    pub fn wwww(self) -> Vec4 {
        Vec4::new(self.w, self.w, self.w, self.w)
    }
}

