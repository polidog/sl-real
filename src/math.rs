//! 最小限の 3D 数学ライブラリ。

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    pub const ZERO: V3 = v3(0.0, 0.0, 0.0);

    pub fn splat(v: f32) -> V3 {
        v3(v, v, v)
    }
    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: V3) -> V3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn norm(self) -> V3 {
        let l = self.len();
        if l > 1e-9 {
            self / l
        } else {
            V3::ZERO
        }
    }
    pub fn lerp(self, o: V3, t: f32) -> V3 {
        self + (o - self) * t
    }
    /// 成分ごとの積（色の乗算に使う）。
    pub fn mul3(self, o: V3) -> V3 {
        v3(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for V3 {
    fn add_assign(&mut self, o: V3) {
        *self = *self + o;
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Div<f32> for V3 {
    type Output = V3;
    fn div(self, s: f32) -> V3 {
        v3(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        v3(-self.x, -self.y, -self.z)
    }
}

/// 行優先 4x4 行列。列ベクトルに左から掛ける（`m * v`）。
#[derive(Clone, Copy, Debug)]
pub struct M4(pub [[f32; 4]; 4]);

impl M4 {
    pub fn identity() -> M4 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            m[i][i] = 1.0;
        }
        M4(m)
    }

    pub fn translate(t: V3) -> M4 {
        let mut m = M4::identity();
        m.0[0][3] = t.x;
        m.0[1][3] = t.y;
        m.0[2][3] = t.z;
        m
    }

    pub fn scale(s: V3) -> M4 {
        let mut m = M4::identity();
        m.0[0][0] = s.x;
        m.0[1][1] = s.y;
        m.0[2][2] = s.z;
        m
    }

    pub fn rot_x(a: f32) -> M4 {
        let (s, c) = a.sin_cos();
        let mut m = M4::identity();
        m.0[1][1] = c;
        m.0[1][2] = -s;
        m.0[2][1] = s;
        m.0[2][2] = c;
        m
    }

    pub fn rot_y(a: f32) -> M4 {
        let (s, c) = a.sin_cos();
        let mut m = M4::identity();
        m.0[0][0] = c;
        m.0[0][2] = s;
        m.0[2][0] = -s;
        m.0[2][2] = c;
        m
    }

    pub fn rot_z(a: f32) -> M4 {
        let (s, c) = a.sin_cos();
        let mut m = M4::identity();
        m.0[0][0] = c;
        m.0[0][1] = -s;
        m.0[1][0] = s;
        m.0[1][1] = c;
        m
    }

    pub fn mul(&self, o: &M4) -> M4 {
        let mut r = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.0[i][k] * o.0[k][j];
                }
                r[i][j] = s;
            }
        }
        M4(r)
    }

    /// 位置ベクトル（w=1）を変換する。
    pub fn xf_point(&self, p: V3) -> V3 {
        let m = &self.0;
        v3(
            m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3],
        )
    }

    /// 方向ベクトル（w=0）を変換する。非等方スケールが無い前提。
    pub fn xf_dir(&self, p: V3) -> V3 {
        let m = &self.0;
        v3(
            m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z,
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z,
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z,
        )
    }

    /// 同次座標へ変換して (x, y, z, w) を返す。
    pub fn xf_h(&self, p: V3) -> [f32; 4] {
        let m = &self.0;
        [
            m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3],
            m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3],
        ]
    }

    pub fn look_at(eye: V3, target: V3, up: V3) -> M4 {
        let f = (target - eye).norm();
        let s = f.cross(up).norm();
        let u = s.cross(f);
        M4([
            [s.x, s.y, s.z, -s.dot(eye)],
            [u.x, u.y, u.z, -u.dot(eye)],
            [-f.x, -f.y, -f.z, f.dot(eye)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// 右手系・深度は w に持たせる（z の値は使わない）。
    pub fn perspective(fov_y: f32, aspect: f32, near: f32) -> M4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        M4([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0 * near],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }
}

pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

pub fn saturate(v: f32) -> f32 {
    clamp(v, 0.0, 1.0)
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = saturate((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}
