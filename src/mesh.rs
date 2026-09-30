//! イミディエイトモードのジオメトリ生成。
//! 変換行列スタックを持ち、プリミティブをそのままラスタライザへ流す。

use crate::math::*;
use crate::render::{Grime, Material, Renderer, Vtx};

pub struct Builder<'a> {
    pub r: &'a mut Renderer,
    xf: M4,
    stack: Vec<M4>,
    pub mat: Material,
    pub color: V3,
    /// 0 で新品、1 で煤と錆にまみれた質感。
    pub grime: f32,
    pub grime_scale: f32,
    pub grime_tint: V3,
    /// 分割数の倍率。遠い物体は粗くする。
    pub detail: f32,
}

impl<'a> Builder<'a> {
    pub fn new(r: &'a mut Renderer) -> Builder<'a> {
        Builder {
            r,
            xf: M4::identity(),
            stack: Vec::with_capacity(16),
            mat: Material::default(),
            color: V3::splat(0.8),
            grime: 0.0,
            grime_scale: 0.7,
            grime_tint: v3(0.12, 0.10, 0.09),
            detail: 1.0,
        }
    }

    /// LOD を掛けた分割数。
    #[inline]
    fn sg(&self, n: usize) -> usize {
        if self.detail >= 0.999 {
            return n;
        }
        ((n as f32 * self.detail).round() as usize).max(3)
    }

    pub fn push(&mut self, m: M4) {
        self.stack.push(self.xf);
        self.xf = self.xf.mul(&m);
    }
    pub fn pop(&mut self) {
        self.xf = self.stack.pop().expect("unbalanced push/pop");
    }

    #[inline]
    fn wp(&self, p: V3) -> V3 {
        self.xf.xf_point(p)
    }
    #[inline]
    fn wn(&self, n: V3) -> V3 {
        self.xf.xf_dir(n).norm()
    }

    /// いまの素材色と汚れ。上向きの面ほど煤が溜まる想定は
    /// 呼び出し側で grime の値を変えて表現する。
    #[inline]
    fn surface(&self) -> (V3, Grime) {
        if self.grime <= 0.0 {
            return (self.color, Grime::default());
        }
        // 遠い物体は汚れの模様が見えないので、平均的に暗くするだけにする。
        if self.detail < 0.65 {
            return (self.color * (1.0 - self.grime * 0.30), Grime::default());
        }
        let g = Grime {
            amount: self.grime,
            scale: self.grime_scale,
            tint: self.grime_tint,
        };
        (self.color, g)
    }

    /// スムーズ法線付きの三角形（ローカル座標）。
    pub fn tri_n(&mut self, p: [V3; 3], n: [V3; 3]) {
        let (m, (c, g)) = (self.mat, self.surface());
        let vs: [Vtx; 3] = std::array::from_fn(|i| Vtx {
            p: self.wp(p[i]),
            n: self.wn(n[i]),
            c,
        });
        self.r.tri(&vs[0], &vs[1], &vs[2], &m, g);
    }

    /// フラット法線の三角形。
    pub fn tri_flat(&mut self, a: V3, b: V3, c: V3) {
        let (wa, wb, wc) = (self.wp(a), self.wp(b), self.wp(c));
        let n = (wb - wa).cross(wc - wa).norm();
        let (m, (c, g)) = (self.mat, self.surface());
        let va = Vtx { p: wa, n, c };
        let vb = Vtx { p: wb, n, c };
        let vc = Vtx { p: wc, n, c };
        self.r.tri(&va, &vb, &vc, &m, g);
    }

    /// フラット法線の四角形。頂点は順に並んでいること。
    pub fn quad(&mut self, a: V3, b: V3, c: V3, d: V3) {
        self.tri_flat(a, b, c);
        self.tri_flat(a, c, d);
    }

    /// スムーズ法線の四角形。
    pub fn quad_n(&mut self, p: [V3; 4], n: [V3; 4]) {
        self.tri_n([p[0], p[1], p[2]], [n[0], n[1], n[2]]);
        self.tri_n([p[0], p[2], p[3]], [n[0], n[2], n[3]]);
    }

    /// 軸に沿った直方体。
    pub fn bx(&mut self, lo: V3, hi: V3) {
        let (a, b) = (lo, hi);
        // -Z / +Z
        self.quad(
            v3(a.x, a.y, a.z),
            v3(a.x, b.y, a.z),
            v3(b.x, b.y, a.z),
            v3(b.x, a.y, a.z),
        );
        self.quad(
            v3(a.x, a.y, b.z),
            v3(b.x, a.y, b.z),
            v3(b.x, b.y, b.z),
            v3(a.x, b.y, b.z),
        );
        // -X / +X
        self.quad(
            v3(a.x, a.y, a.z),
            v3(a.x, a.y, b.z),
            v3(a.x, b.y, b.z),
            v3(a.x, b.y, a.z),
        );
        self.quad(
            v3(b.x, a.y, a.z),
            v3(b.x, b.y, a.z),
            v3(b.x, b.y, b.z),
            v3(b.x, a.y, b.z),
        );
        // -Y / +Y
        self.quad(
            v3(a.x, a.y, a.z),
            v3(b.x, a.y, a.z),
            v3(b.x, a.y, b.z),
            v3(a.x, a.y, b.z),
        );
        self.quad(
            v3(a.x, b.y, a.z),
            v3(a.x, b.y, b.z),
            v3(b.x, b.y, b.z),
            v3(b.x, b.y, a.z),
        );
    }

    /// Y 軸に沿った円錐台（r0 が y=0、r1 が y=h）。角度は `a0..a1`。
    pub fn cone_y(&mut self, r0: f32, r1: f32, h: f32, seg: usize, caps: bool) {
        self.cone_arc_y(r0, r1, h, seg, caps, 0.0, std::f32::consts::TAU);
    }

    /// 円錐台の一部（アーチや半円筒に使う）。
    pub fn cone_arc_y(
        &mut self,
        r0: f32,
        r1: f32,
        h: f32,
        seg: usize,
        caps: bool,
        a0: f32,
        a1: f32,
    ) {
        let seg = self.sg(seg).max(3);
        // 側面の傾きを法線に反映する。
        let slope = (r0 - r1) / h.max(1e-5);
        let nlen = (1.0 + slope * slope).sqrt();
        let mut prev: Option<(V3, V3, V3)> = None;
        for i in 0..=seg {
            let t = i as f32 / seg as f32;
            let a = a0 + (a1 - a0) * t;
            let (s, c) = a.sin_cos();
            let n = v3(c / nlen, slope / nlen, s / nlen);
            let p0 = v3(c * r0, 0.0, s * r0);
            let p1 = v3(c * r1, h, s * r1);
            if let Some((q0, q1, qn)) = prev {
                if r0 > 1e-6 || r1 > 1e-6 {
                    self.quad_n([q0, p0, p1, q1], [qn, n, n, qn]);
                }
                if caps {
                    if r0 > 1e-6 {
                        self.tri_n([V3::ZERO, p0, q0], [v3(0.0, -1.0, 0.0); 3]);
                    }
                    if r1 > 1e-6 {
                        self.tri_n([v3(0.0, h, 0.0), q1, p1], [v3(0.0, 1.0, 0.0); 3]);
                    }
                }
            }
            prev = Some((p0, p1, n));
        }
    }

    /// Y 軸に沿った円柱。
    pub fn cyl_y(&mut self, r: f32, h: f32, seg: usize, caps: bool) {
        self.cone_y(r, r, h, seg, caps);
    }

    /// 端点 `a`→`b` を結ぶ円柱（ロッドや配管に使う）。
    pub fn rod(&mut self, a: V3, b: V3, r: f32, seg: usize) {
        let d = b - a;
        let len = d.len();
        if len < 1e-6 {
            return;
        }
        let m = align_y(d / len);
        self.push(M4::translate(a).mul(&m));
        self.cyl_y(r, len, seg, true);
        self.pop();
    }

    /// 端点 `a`→`b` を結ぶ角断面の梁。`up` で断面の向きを決める。
    pub fn beam(&mut self, a: V3, b: V3, up: V3, hw: f32, hh: f32) {
        let d = b - a;
        let len = d.len();
        if len < 1e-6 {
            return;
        }
        let f = d / len;
        let r = f.cross(up).norm();
        let u = r.cross(f);
        let m = M4([
            [r.x, u.x, f.x, a.x],
            [r.y, u.y, f.y, a.y],
            [r.z, u.z, f.z, a.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        self.push(m);
        self.bx(v3(-hw, -hh, 0.0), v3(hw, hh, len));
        self.pop();
    }

    /// UV 球。
    pub fn sphere(&mut self, r: f32, seg: usize, rings: usize) {
        self.spheroid(v3(r, r, r), seg, rings, -1.0, 1.0);
    }

    /// 楕円体の一部。`y0`..`y1` は -1..1 の範囲で切り取る高さ。
    pub fn spheroid(&mut self, radii: V3, seg: usize, rings: usize, y0: f32, y1: f32) {
        let seg = self.sg(seg).max(3);
        let rings = self.sg(rings).max(2);
        let vert = |u: f32, v: f32| -> (V3, V3) {
            let phi = u * std::f32::consts::TAU;
            let sy = y0 + (y1 - y0) * v;
            let sy = clamp(sy, -1.0, 1.0);
            let rr = (1.0 - sy * sy).max(0.0).sqrt();
            let (sp, cp) = phi.sin_cos();
            let ln = v3(cp * rr, sy, sp * rr);
            let p = v3(ln.x * radii.x, ln.y * radii.y, ln.z * radii.z);
            let n = v3(ln.x / radii.x, ln.y / radii.y, ln.z / radii.z).norm();
            (p, n)
        };
        for j in 0..rings {
            let v0 = j as f32 / rings as f32;
            let v1 = (j + 1) as f32 / rings as f32;
            for i in 0..seg {
                let u0 = i as f32 / seg as f32;
                let u1 = (i + 1) as f32 / seg as f32;
                let (p00, n00) = vert(u0, v0);
                let (p10, n10) = vert(u1, v0);
                let (p11, n11) = vert(u1, v1);
                let (p01, n01) = vert(u0, v1);
                self.quad_n([p00, p10, p11, p01], [n00, n10, n11, n01]);
            }
        }
    }

    /// XZ 平面の円盤（法線は +Y）。
    pub fn disc_y(&mut self, r: f32, seg: usize, up: bool) {
        let seg = self.sg(seg).max(3);
        let n = if up {
            v3(0.0, 1.0, 0.0)
        } else {
            v3(0.0, -1.0, 0.0)
        };
        for i in 0..seg {
            let a0 = i as f32 / seg as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / seg as f32 * std::f32::consts::TAU;
            let p0 = v3(a0.cos() * r, 0.0, a0.sin() * r);
            let p1 = v3(a1.cos() * r, 0.0, a1.sin() * r);
            if up {
                self.tri_n([V3::ZERO, p1, p0], [n; 3]);
            } else {
                self.tri_n([V3::ZERO, p0, p1], [n; 3]);
            }
        }
    }
}

/// +Y をベクトル `d` に向ける回転行列。
pub fn align_y(d: V3) -> M4 {
    let up = if d.y.abs() > 0.999 {
        v3(1.0, 0.0, 0.0)
    } else {
        v3(0.0, 1.0, 0.0)
    };
    let x = up.cross(d).norm();
    let z = d.cross(x).norm();
    M4([
        [x.x, d.x, z.x, 0.0],
        [x.y, d.y, z.y, 0.0],
        [x.z, d.z, z.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}
