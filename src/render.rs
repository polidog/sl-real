//! ハーフブロック文字 (▀) を 1 セル 2 ピクセルとして使う、
//! Z バッファ付きソフトウェアラスタライザ。

use crate::math::*;
use std::fmt::Write as _;

/// 1 セルをどう埋めるか。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocks {
    /// `▀` だけを使う。1 セル = 縦 2 ピクセル。字形の対応は最も広い。
    Half,
    /// 四分割ブロックを使う。1 セル = 2x2 ピクセルで横解像度が倍になる。
    /// 1 セルに 2 色までなので、4 つの小画素を 2 色へ最適に分ける。
    Quad,
}

/// 端末に出す 1 セル。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct Cell {
    fg: [u8; 3],
    bg: [u8; 3],
    /// 前景として塗る小画素のビットマスク（下位から TL, TR, BL, BR）。
    mask: u8,
}

/// 面の質感。
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// 0 = 鏡面的につるつる, 1 = ざらざら。
    pub rough: f32,
    /// 金属らしさ。ハイライトが素材色に着色される。
    pub metal: f32,
    /// 自己発光（前照灯やかまどの火）。
    pub emissive: V3,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            rough: 0.6,
            metal: 0.0,
            emissive: V3::ZERO,
        }
    }
}

impl Material {
    pub const PAINT: Material = Material {
        rough: 0.45,
        metal: 0.0,
        emissive: V3::ZERO,
    };
    pub const IRON: Material = Material {
        rough: 0.5,
        metal: 0.85,
        emissive: V3::ZERO,
    };
    pub const MATTE: Material = Material {
        rough: 0.95,
        metal: 0.0,
        emissive: V3::ZERO,
    };
}

/// 頂点（ワールド空間）。
#[derive(Clone, Copy, Debug)]
pub struct Vtx {
    pub p: V3,
    pub n: V3,
    pub c: V3,
}

/// ライティングと大気の設定。
#[derive(Clone, Copy, Debug)]
pub struct Env {
    pub sun_dir: V3,
    pub sun_color: V3,
    pub sky_color: V3,
    /// 映り込み用の天頂色（環境光とは別に生の空の色を持つ）。
    pub zenith_color: V3,
    pub ground_color: V3,
    pub horizon_color: V3,
    pub fog_density: f32,
    pub exposure: f32,
    /// 夜間の前照灯。位置・向き・強さ。
    pub head_pos: V3,
    pub head_dir: V3,
    pub head_power: f32,
    /// かまどの赤い漏れ光。
    pub fire_pos: V3,
    pub fire_power: f32,
    /// 薄明視の強さ（夜に上げる）。
    pub scotopic: f32,
}

/// レンズに残る光の演出パラメータ。
pub struct Renderer {
    pub w: usize,
    pub h: usize,
    /// 1 ピクセルの横幅 / 高さ。Quad では横に細長い (0.5)。
    pub px_aspect: f32,
    blocks: Blocks,
    /// スーパーサンプリング倍率。ss 倍の解像度で描き、出力時に平均して縮小する。
    ss: usize,
    /// 縮小後の表示用バッファ。
    disp: Vec<V3>,
    pub color: Vec<V3>,
    pub depth: Vec<f32>,
    pub view_proj: M4,
    /// 視錐台の 6 平面（正規化済み）。カリングに使う。
    frustum: [[f32; 4]; 6],
    pub cam_pos: V3,
    pub cam_right: V3,
    pub cam_fwd: V3,
    pub cam_up: V3,
    /// 垂直画角の半分の tan。深度からワールド座標を戻すのに使う。
    pub tan_half: f32,
    pub env: Env,
    /// 影を落とす箱（ワールド軸に沿った AABB のリスト）。
    pub shadow_boxes: Vec<(V3, V3)>,
    /// 陰影付け待ちの三角形。flush() でまとめて並列処理する。
    tris: Vec<RawTri>,
    /// 積んだ半透明スプライト。flush_sprites() で帯ごとに並列処理する。
    sprites: Vec<Sprite>,
    /// ブルームの強さ（0 で無効）。
    pub bloom: f32,
    bloom_a: Vec<V3>,
    bloom_b: Vec<V3>,
    bw: usize,
    bh: usize,
    prev_cells: Vec<Cell>,
    prev_valid: bool,
}

/// クリップ空間の 3 頂点が同じ面の外側にそろっていれば捨てられる。
#[inline]
fn reject(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> bool {
    const NEAR_W: f32 = 0.05;
    if a[3] <= NEAR_W && b[3] <= NEAR_W && c[3] <= NEAR_W {
        return true;
    }
    (a[0] < -a[3] && b[0] < -b[3] && c[0] < -c[3])
        || (a[0] > a[3] && b[0] > b[3] && c[0] > c[3])
        || (a[1] < -a[3] && b[1] < -b[3] && c[1] < -c[3])
        || (a[1] > a[3] && b[1] > b[3] && c[1] > c[3])
}

/// 頂点を切り出したクリップ空間の一時データ。
#[derive(Clone, Copy)]
struct ClipV {
    pos: [f32; 4],
    c: V3,
}

/// 陰影付け前に溜めておく三角形。法線の縮退はこの時点で解決済み。
#[derive(Clone, Copy)]
struct RawTri {
    p: [V3; 3],
    n: [V3; 3],
    c: [V3; 3],
    clip: [[f32; 4]; 3],
    mat: Material,
}

/// 画面空間まで落とした半透明スプライト（煙の玉と光芒）。
/// 描く順に意味があるので、積んだ順を保ったまま帯ごとに処理する。
#[derive(Clone, Copy)]
struct Sprite {
    /// 0 = 煙の玉（アルファ合成）、1 = 光芒（加算）。
    kind: u8,
    sx: f32,
    sy: f32,
    rx: f32,
    ry: f32,
    depth: f32,
    a: V3,
    b: V3,
    k: f32,
    seed: f32,
}

/// 画面空間まで落とした三角形。色は 1/w を掛けた遠近補正済みの値。
#[derive(Clone, Copy)]
struct ScreenTri {
    x: [f32; 3],
    y: [f32; 3],
    iw: [f32; 3],
    c: [V3; 3],
    y0: f32,
    y1: f32,
}

impl Renderer {
    /// 端末の桁数・行数と描画モードからフレームバッファを作る。
    /// `ss` はスーパーサンプリング倍率（1 で無効、2 で縦横 2 倍に描いて縮小）。
    pub fn for_terminal(cols: usize, rows: usize, blocks: Blocks, ss: usize) -> Renderer {
        let ss = ss.clamp(1, 3);
        let (dw, dh) = Self::dims(cols, rows, blocks);
        let mut r = Renderer::new(dw * ss, dh * ss);
        r.blocks = blocks;
        r.ss = ss;
        r.px_aspect = if blocks == Blocks::Quad { 0.5 } else { 1.0 };
        r.disp = vec![V3::ZERO; dw * dh];
        r.prev_cells = vec![Cell::default(); cols * rows];
        r
    }

    fn dims(cols: usize, rows: usize, blocks: Blocks) -> (usize, usize) {
        match blocks {
            Blocks::Half => (cols, rows * 2),
            Blocks::Quad => (cols * 2, rows * 2),
        }
    }

    /// 表示解像度（縮小後）。
    fn disp_dims(&self) -> (usize, usize) {
        (self.w / self.ss, self.h / self.ss)
    }

    /// ss x ss を平均して表示用バッファを作る。平均はリニア空間で行う。
    fn resolve(&mut self) {
        let (dw, dh) = self.disp_dims();
        if self.ss == 1 {
            self.disp.clear();
            self.disp.extend_from_slice(&self.color);
            return;
        }
        if self.disp.len() != dw * dh {
            self.disp = vec![V3::ZERO; dw * dh];
        }
        let inv = 1.0 / (self.ss * self.ss) as f32;
        let (w, ss) = (self.w, self.ss);
        let src: &[V3] = &self.color;
        let band = dh.div_ceil(threads()).max(1);
        std::thread::scope(|scope| {
            let mut y0 = 0usize;
            for part in self.disp.chunks_mut(dw * band) {
                let start = y0;
                y0 += band;
                scope.spawn(move || {
                    for (i, out) in part.iter_mut().enumerate() {
                        let (y, x) = (start + i / dw, i % dw);
                        let mut acc = V3::ZERO;
                        for sy in 0..ss {
                            let base = (y * ss + sy) * w + x * ss;
                            for sx in 0..ss {
                                acc += src[base + sx];
                            }
                        }
                        *out = acc * inv;
                    }
                });
            }
        });
    }

    pub fn new(w: usize, h: usize) -> Renderer {
        Renderer {
            w,
            h,
            px_aspect: 1.0,
            blocks: Blocks::Half,
            color: vec![V3::ZERO; w * h],
            depth: vec![f32::INFINITY; w * h],
            view_proj: M4::identity(),
            frustum: [[0.0; 4]; 6],
            cam_pos: V3::ZERO,
            cam_right: v3(1.0, 0.0, 0.0),
            cam_fwd: v3(0.0, 0.0, -1.0),
            cam_up: v3(0.0, 1.0, 0.0),
            tan_half: 0.4,
            env: Env {
                sun_dir: v3(0.4, 0.6, 0.3).norm(),
                sun_color: V3::splat(1.0),
                sky_color: v3(0.35, 0.55, 0.9),
                zenith_color: v3(0.04, 0.12, 0.44),
                ground_color: v3(0.25, 0.28, 0.18),
                horizon_color: v3(0.7, 0.78, 0.9),
                fog_density: 0.0016,
                exposure: 1.0,
                head_pos: V3::ZERO,
                head_dir: v3(1.0, 0.0, 0.0),
                head_power: 0.0,
                fire_pos: V3::ZERO,
                fire_power: 0.0,
                scotopic: 0.0,
            },
            shadow_boxes: Vec::new(),
            tris: Vec::new(),
            sprites: Vec::new(),
            bloom: 0.55,
            bloom_a: vec![V3::ZERO; (w / 2).max(1) * (h / 2).max(1)],
            bloom_b: vec![V3::ZERO; (w / 2).max(1) * (h / 2).max(1)],
            bw: (w / 2).max(1),
            bh: (h / 2).max(1),
            ss: 1,
            disp: Vec::new(),
            prev_cells: vec![Cell::default(); w * (h / 2)],
            prev_valid: false,
        }
    }

    /// PPM 書き出しなどでピクセルを正方形として扱いたいときに使う。
    pub fn set_px_aspect(&mut self, a: f32) {
        self.px_aspect = a;
    }

    /// 端末サイズの変化に追随する。
    pub fn resize_terminal(&mut self, cols: usize, rows: usize) {
        let (dw, dh) = Self::dims(cols, rows, self.blocks);
        let ss = self.ss;
        self.resize(dw * ss, dh * ss);
        self.ss = ss;
        self.disp = vec![V3::ZERO; dw * dh];
        self.prev_cells = vec![Cell::default(); cols * rows];
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.color = vec![V3::ZERO; w * h];
        self.depth = vec![f32::INFINITY; w * h];
        self.prev_cells = vec![Cell::default(); w * (h / 2)];
        self.prev_valid = false;
        self.bw = (w / 2).max(1);
        self.bh = (h / 2).max(1);
        self.bloom_a = vec![V3::ZERO; self.bw * self.bh];
        self.bloom_b = vec![V3::ZERO; self.bw * self.bh];
        self.ss = 1;
        self.disp = Vec::new();
    }

    /// ビュー射影行列を設定し、視錐台の平面を作り直す。
    pub fn set_view(&mut self, vp: M4) {
        self.view_proj = vp;
        let m = &vp.0;
        // clip = M * p。w±x, w±y, w>0 の 5 平面（遠方は切らない）。
        let rows = [
            [m[3][0] + m[0][0], m[3][1] + m[0][1], m[3][2] + m[0][2], m[3][3] + m[0][3]],
            [m[3][0] - m[0][0], m[3][1] - m[0][1], m[3][2] - m[0][2], m[3][3] - m[0][3]],
            [m[3][0] + m[1][0], m[3][1] + m[1][1], m[3][2] + m[1][2], m[3][3] + m[1][3]],
            [m[3][0] - m[1][0], m[3][1] - m[1][1], m[3][2] - m[1][2], m[3][3] - m[1][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ];
        for (i, r) in rows.iter().enumerate() {
            let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-9);
            self.frustum[i] = [r[0] / n, r[1] / n, r[2] / n, r[3] / n];
        }
    }

    /// 球が視錐台の外にあれば false。物体まるごと省くのに使う。
    pub fn sphere_visible(&self, c: V3, r: f32) -> bool {
        for p in self.frustum.iter().take(5) {
            if p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -r {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self) {
        for d in self.depth.iter_mut() {
            *d = f32::INFINITY;
        }
        self.shadow_boxes.clear();
        self.tris.clear();
        self.sprites.clear();
    }

    /// 画面の物理的な縦横比。ピクセルが正方形でない場合も込み。
    pub fn aspect(&self) -> f32 {
        self.w as f32 * self.px_aspect / self.h as f32
    }

    // ---------------------------------------------------------------- 陰影


    /// 距離に応じた大気による減衰。
    pub fn apply_fog(&self, c: V3, dist: f32, dir: V3) -> V3 {
        fog_of(&self.env, c, dist, dir)
    }

    // ---------------------------------------------------------------- 描画

    #[inline]
    fn to_clip(&self, p: V3) -> [f32; 4] {
        self.view_proj.xf_h(p)
    }

    /// 三角形を積む。実際の陰影付けとラスタライズは flush() でまとめて行う。
    pub fn tri(&mut self, a: &Vtx, b: &Vtx, c: &Vtx, m: &Material) {
        // 面法線が縮退していたら法線を補う。
        let fnorm = (b.p - a.p).cross(c.p - a.p);
        if fnorm.dot(fnorm) < 1e-16 {
            return;
        }

        // 画面の外なら、ここで捨てる。以降の計算に一切入らない。
        let clip = [self.to_clip(a.p), self.to_clip(b.p), self.to_clip(c.p)];
        if reject(&clip[0], &clip[1], &clip[2]) {
            return;
        }

        let fixup = |v: &Vtx| if v.n.dot(v.n) < 1e-8 { fnorm.norm() } else { v.n };
        self.tris.push(RawTri {
            p: [a.p, b.p, c.p],
            n: [fixup(a), fixup(b), fixup(c)],
            c: [a.c, b.c, c.c],
            clip,
            mat: *m,
        });
    }

    /// 溜めた三角形を陰影付けして描き切る。
    ///
    /// 頂点の陰影付けは三角形ごとに独立なので三角形の束で分割し、
    /// ラスタライズは書き込む行が重ならないよう横帯で分割する。
    /// どちらも全コアを使う。
    pub fn flush(&mut self) {
        if self.tris.is_empty() {
            return;
        }
        let threads = threads();

        // ---- 第 1 段：頂点の陰影付け → 画面空間の三角形へ。
        let env = self.env;
        let cam = self.cam_pos;
        let (w, h) = (self.w as f32, self.h as f32);
        let tris = std::mem::take(&mut self.tris);
        let chunk = tris.len().div_ceil(threads).max(1);

        let mut screen: Vec<ScreenTri> = Vec::with_capacity(tris.len() + 16);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for part in tris.chunks(chunk) {
                handles.push(scope.spawn(move || {
                    let mut out: Vec<ScreenTri> = Vec::with_capacity(part.len() + 4);
                    for t in part {
                        let mut v = [ClipV { pos: [0.0; 4], c: V3::ZERO }; 3];
                        for i in 0..3 {
                            // 影の箱は地面用。物体自身に当てると自分の影で真っ黒になる。
                            let lit =
                                shade_of(&env, cam, t.p[i], t.n[i], t.c[i], &t.mat, 0.0);
                            let d = t.p[i] - cam;
                            let dist = d.len();
                            v[i] = ClipV {
                                pos: t.clip[i],
                                c: fog_of(&env, lit, dist, d / dist.max(1e-6)),
                            };
                        }
                        clip_and_project(&v, w, h, &mut out);
                    }
                    out
                }));
            }
            for hnd in handles {
                screen.extend(hnd.join().unwrap_or_default());
            }
        });
        self.tris = tris;
        self.tris.clear();

        // ---- 第 2 段：横帯ごとに、その帯にかかる三角形だけを塗る。
        let bands = threads;
        let band_h = self.h.div_ceil(bands).max(1);
        let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); bands];
        for (i, t) in screen.iter().enumerate() {
            let b0 = (t.y0.max(0.0) as usize / band_h).min(bands - 1);
            let b1 = (t.y1.max(0.0) as usize / band_h).min(bands - 1);
            for b in bucket.iter_mut().take(b1 + 1).skip(b0) {
                b.push(i as u32);
            }
        }

        let sw = self.w;
        let scr: &[ScreenTri] = &screen;
        std::thread::scope(|scope| {
            let mut y0 = 0usize;
            let mut idx = 0usize;
            for (cc, dd) in self
                .color
                .chunks_mut(sw * band_h)
                .zip(self.depth.chunks_mut(sw * band_h))
            {
                let start = y0;
                y0 += band_h;
                let list: &[u32] = if idx < bucket.len() { &bucket[idx] } else { &[] };
                idx += 1;
                scope.spawn(move || {
                    let rows = cc.len() / sw;
                    for &i in list {
                        raster_band(&scr[i as usize], cc, dd, sw, start, rows);
                    }
                });
            }
        });
    }

    /// ワールド空間の球をソフトなスプライトとして積む（煙・蒸気用）。
    /// `tint` は線形の色、`alpha` は最大不透明度、`seed` は形のゆらぎ。
    pub fn puff(&mut self, center: V3, radius: f32, tint: V3, alpha: f32, seed: f32) {
        self.flush();
        let Some((sx, sy, r, depth)) = self.project_sprite(center, radius) else {
            return;
        };
        if r < 0.35 {
            return;
        }
        let d = center - self.cam_pos;
        let dist = d.len();
        let dir = d / dist.max(1e-6);

        // 太陽側の縁を明るくして立体感を出す。
        let lit_dir = self.env.sun_dir.dot(-dir);
        let base = tint.mul3(
            self.env.sky_color * 0.45
                + self.env.sun_color * (0.55 + saturate(lit_dir) * 0.45)
                + self.env.ground_color * 0.12,
        );
        let base = self.apply_fog(base, dist, dir);
        let rim = tint.mul3(self.env.sun_color) * 0.9;

        self.sprites.push(Sprite {
            kind: 0,
            sx,
            sy,
            rx: r,
            ry: r * self.px_aspect,
            depth,
            a: base,
            b: rim,
            k: alpha,
            seed,
        });
    }

    /// 光そのもの（前照灯のレンズなど）を加算合成で積む。
    pub fn glow(&mut self, center: V3, radius: f32, color: V3, power: f32) {
        self.flush();
        let Some((sx, sy, r, depth)) = self.project_sprite(center, radius) else {
            return;
        };
        let r = r.max(0.8);
        self.sprites.push(Sprite {
            kind: 1,
            sx,
            sy,
            rx: r,
            ry: (r * self.px_aspect).max(0.5),
            depth,
            a: color,
            b: V3::ZERO,
            k: power,
            seed: 0.0,
        });
    }

    /// スプライトの中心と画面上の半径を求める。画面外なら None。
    fn project_sprite(&self, center: V3, radius: f32) -> Option<(f32, f32, f32, f32)> {
        let cp = self.to_clip(center);
        if cp[3] <= 0.1 {
            return None;
        }
        let ep = self.to_clip(center + self.cam_right * radius);
        let iw = 1.0 / cp[3];
        let sx = (cp[0] * iw * 0.5 + 0.5) * self.w as f32;
        let sy = (0.5 - cp[1] * iw * 0.5) * self.h as f32;
        let iw2 = 1.0 / ep[3].max(0.1);
        let ex = (ep[0] * iw2 * 0.5 + 0.5) * self.w as f32;
        let ey = (0.5 - ep[1] * iw2 * 0.5) * self.h as f32;
        let r = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt();
        if r > 4000.0 {
            return None;
        }
        Some((sx, sy, r, cp[3]))
    }

    /// 積んだスプライトを、書き込む行が重ならない横帯に分けて描く。
    /// 帯の中では積んだ順を保つので、合成結果は逐次描いたときと同じ。
    pub fn flush_sprites(&mut self) {
        if self.sprites.is_empty() {
            return;
        }
        let (w, h) = (self.w, self.h);
        let bands = threads();
        let band_h = h.div_ceil(bands).max(1);
        let sprites = std::mem::take(&mut self.sprites);

        let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); bands];
        for (i, s) in sprites.iter().enumerate() {
            let lo = (s.sy - s.ry).max(0.0) as usize / band_h;
            let hi = ((s.sy + s.ry).max(0.0) as usize / band_h).min(bands - 1);
            if lo > hi {
                continue;
            }
            for b in bucket.iter_mut().take(hi + 1).skip(lo.min(bands - 1)) {
                b.push(i as u32);
            }
        }

        let depth: &[f32] = &self.depth;
        let sp: &[Sprite] = &sprites;
        std::thread::scope(|scope| {
            let mut y0 = 0usize;
            let mut idx = 0usize;
            for cc in self.color.chunks_mut(w * band_h) {
                let start = y0;
                y0 += band_h;
                let list: &[u32] = if idx < bucket.len() { &bucket[idx] } else { &[] };
                idx += 1;
                scope.spawn(move || {
                    let rows = cc.len() / w;
                    for &i in list {
                        draw_sprite(&sp[i as usize], cc, depth, w, start, rows);
                    }
                });
            }
        });
        self.sprites = sprites;
        self.sprites.clear();
    }

    // ---------------------------------------------------------------- 遮蔽

    /// 画面空間アンビエントオクルージョン。
    ///
    /// 深度バッファからワールド座標と法線を戻し、半球内に置いた
    /// サンプル点がどれだけ物陰に入るかを数える。車輪まわりや
    /// ランボードの下、車体と地面の接地部に陰りが出て、
    /// 平board な塗りが立体に見えるようになる。
    pub fn ssao(&mut self, strength: f32, radius: f32) {
        if strength <= 0.0 {
            return;
        }
        self.flush();
        let (w, h) = (self.w, self.h);
        let (cam, fwd, right, up) = (self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up);
        let (th, aspect, pxa) = (self.tan_half, self.aspect(), self.px_aspect);
        let depth: &[f32] = &self.depth;

        let ray = |x: f32, y: f32| -> V3 {
            let nx = (x + 0.5) / w as f32 * 2.0 - 1.0;
            let ny = 1.0 - (y + 0.5) / h as f32 * 2.0;
            (fwd + right * (nx * th * aspect) + up * (ny * th)).norm()
        };
        let world = |x: f32, y: f32, d: f32| -> V3 {
            let r = ray(x, y);
            cam + r * (d / r.dot(fwd).max(1e-4))
        };
        let at = |x: i64, y: i64| -> f32 {
            let x = x.clamp(0, w as i64 - 1) as usize;
            let y = y.clamp(0, h as i64 - 1) as usize;
            depth[y * w + x]
        };

        // 円周上に半径を変えて散らした固定パターン。回転で縞を散らす。
        const N: usize = 10;
        let mut ao = vec![0.0f32; w * h];
        let band = h.div_ceil(threads()).max(1);
        std::thread::scope(|scope| {
            let mut y0 = 0usize;
            for part in ao.chunks_mut(w * band) {
                let start = y0;
                y0 += band;
                let world = &world;
                let at = &at;
                scope.spawn(move || {
                    for (i, out) in part.iter_mut().enumerate() {
                        let (py, px) = (start + i / w, i % w);
                        let dc = depth[py * w + px];
                        if !dc.is_finite() {
                            continue;
                        }
                        let (fx, fy) = (px as f32, py as f32);
                        let p = world(fx, fy, dc);

                        // 隣の画素との差から法線を作る。深度が飛ぶ縁では諦める。
                        let dx = at(px as i64 + 1, py as i64);
                        let dy = at(px as i64, py as i64 + 1);
                        if !dx.is_finite() || !dy.is_finite() {
                            continue;
                        }
                        let pu = world(fx + 1.0, fy, dx) - p;
                        let pv = world(fx, fy + 1.0, dy) - p;
                        let mut n = pv.cross(pu).norm();
                        if n.dot(n) < 0.5 {
                            continue;
                        }
                        // 復元した法線は向きが定まらないので、カメラ側へ向ける。
                        if n.dot(cam - p) < 0.0 {
                            n = -n;
                        }

                        // 半径をワールド単位で決め、画面上の大きさに直す。
                        let ry = radius * (h as f32 * 0.5) / (th * dc.max(0.05));
                        if ry < 0.7 {
                            continue;
                        }
                        let ry = ry.min(h as f32 * 0.12);
                        let rx = ry / pxa;

                        let rot = crate::noise::hash1((px as u32) ^ (py as u32) << 16)
                            * std::f32::consts::TAU;
                        let mut occ = 0.0f32;
                        for k in 0..N {
                            let t = (k as f32 + 0.5) / N as f32;
                            let a = rot + t * 6.2 * std::f32::consts::PI;
                            let rr = t.sqrt();
                            let sx = fx + a.cos() * rx * rr;
                            let sy = fy + a.sin() * ry * rr;
                            let sd = at(sx as i64, sy as i64);
                            if !sd.is_finite() {
                                continue;
                            }
                            let q = world(sx, sy, sd);
                            let v = q - p;
                            let len = v.len();
                            if len < 1e-4 {
                                continue;
                            }
                            // 法線より手前にある点だけが遮る。
                            let cosw = v.dot(n) / len - 0.06;
                            if cosw <= 0.0 {
                                continue;
                            }
                            // 遠すぎる点は無関係。
                            let fall = 1.0 / (1.0 + (len / radius).powi(2));
                            occ += cosw * fall;
                        }
                        *out = saturate(occ / N as f32 * 2.6);
                    }
                });
            }
        });

        // サンプルのばらつきをならす。
        let mut tmp = vec![V3::ZERO; w * h];
        let mut a3: Vec<V3> = ao.iter().map(|&v| V3::splat(v)).collect();
        let rb = (2.0 / pxa).round() as usize;
        blur_h(&a3, &mut tmp, w, h, rb);
        blur_v(&tmp, &mut a3, w, h, 2);

        let band = h.div_ceil(threads()).max(1);
        let src: &[V3] = &a3;
        std::thread::scope(|scope| {
            let mut idx = 0usize;
            for part in self.color.chunks_mut(w * band) {
                let start = idx;
                idx += w * band;
                scope.spawn(move || {
                    for (i, c) in part.iter_mut().enumerate() {
                        let k = 1.0 - src[start + i].x * strength;
                        *c = *c * k.max(0.0);
                    }
                });
            }
        });
    }

    // ---------------------------------------------------------------- 後処理

    /// 明るいところがにじむブルーム。前照灯・火室・夕日がぐっと生きる。
    pub fn post(&mut self) {
        self.flush();
        self.flush_sprites();
        if self.bloom <= 0.0 {
            return;
        }
        let (bw, bh) = (self.bw, self.bh);
        let ex = self.env.exposure;
        // 1/2 解像度に落としつつ、しきい値を超えた分だけ取り出す。
        {
            let (w, h) = (self.w, self.h);
            let src: &[V3] = &self.color;
            let band = bh.div_ceil(threads()).max(1);
            std::thread::scope(|scope| {
                let mut y0 = 0usize;
                for part in self.bloom_a.chunks_mut(bw * band) {
                    let start = y0;
                    y0 += band;
                    scope.spawn(move || {
                        for (i, out) in part.iter_mut().enumerate() {
                            let (y, x) = (start + i / bw, i % bw);
                            let mut acc = V3::ZERO;
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let sx = (x * 2 + dx).min(w - 1);
                                    let sy = (y * 2 + dy).min(h - 1);
                                    acc += src[sy * w + sx];
                                }
                            }
                            let c = acc * 0.25;
                            let lum = (c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722) * ex;
                            let over = (lum - 0.85).max(0.0);
                            *out = if lum > 1e-5 { c * (over / lum) } else { V3::ZERO };
                        }
                    });
                }
            });
        }
        // 半径を変えた箱ぼかしを重ねて、広がりのあるにじみにする。
        // ピクセルが横に細いモードでは、横方向の半径を増やして等方に見せる。
        let hx = (1.0 / self.px_aspect).round().max(1.0) as usize;
        for &r in &[2usize, 5, 11] {
            blur_h(&self.bloom_a, &mut self.bloom_b, bw, bh, r * hx * self.ss);
            blur_v(&self.bloom_b, &mut self.bloom_a, bw, bh, r * self.ss);
        }
        // 薄明視。暗部の彩度を落として青へ寄せる。
        let scot = self.env.scotopic;
        if scot > 0.0 {
            let band = self.color.len().div_ceil(threads()).max(1);
            std::thread::scope(|scope| {
                for part in self.color.chunks_mut(band) {
                    scope.spawn(move || {
                        for c in part.iter_mut() {
                            let lum = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
                            let k = smoothstep(0.16, 0.004, lum * ex) * scot;
                            if k > 0.001 {
                                *c = c.lerp(v3(0.72, 0.92, 1.35) * lum, k);
                            }
                        }
                    });
                }
            });
        }

        // バイリニアで戻して加算。
        let k = self.bloom;
        let w = self.w;
        let src: &[V3] = &self.bloom_a;
        let band = self.h.div_ceil(threads()).max(1);
        std::thread::scope(|scope| {
            let mut ystart = 0usize;
            for part in self.color.chunks_mut(w * band) {
                let start = ystart;
                ystart += band;
                scope.spawn(move || {
                    for (i, out) in part.iter_mut().enumerate() {
                        let (y, x) = (start + i / w, i % w);
                        let fy = (y as f32 * 0.5 - 0.25).clamp(0.0, bh as f32 - 1.001);
                        let (y0, ty) = (fy as usize, fy.fract());
                        let y1 = (y0 + 1).min(bh - 1);
                        let fx = (x as f32 * 0.5 - 0.25).clamp(0.0, bw as f32 - 1.001);
                        let (x0, tx) = (fx as usize, fx.fract());
                        let x1 = (x0 + 1).min(bw - 1);
                        let a = src[y0 * bw + x0].lerp(src[y0 * bw + x1], tx);
                        let b = src[y1 * bw + x0].lerp(src[y1 * bw + x1], tx);
                        *out += a.lerp(b, ty) * k;
                    }
                });
            }
        });
    }

    // ---------------------------------------------------------------- 出力

    /// リニア HDR をトーンマップして sRGB の 8bit に落とす。
    #[inline]
    fn tonemap(&self, c: V3) -> [u8; 3] {
        let x = c * self.env.exposure;
        // ACES 近似（Narkowicz）。
        let f = |v: f32| {
            let v = v.max(0.0);
            let n = v * (2.51 * v + 0.03);
            let d = v * (2.43 * v + 0.59) + 0.14;
            saturate(n / d)
        };
        let (r, g, b) = (f(x.x), f(x.y), f(x.z));
        // ガンマ 2.2。
        let e = |v: f32| (v.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        [e(r), e(g), e(b)]
    }

    /// 2x2 の小画素を 2 色へ最適に分ける。返すのは (前景色, 背景色, マスク)。
    ///
    /// 1 セルに置ける色は 2 つだけなので、4 つの分け方を総当たりして
    /// 二乗誤差が最小になる組を選ぶ。TL を必ず背景側に固定すると
    /// 反転した重複が消えて 8 通りで済む。
    #[inline]
    fn quantize(px: [[u8; 3]; 4]) -> ([u8; 3], [u8; 3], u8) {
        let f = |c: [u8; 3]| [c[0] as i32, c[1] as i32, c[2] as i32];
        let p: [[i32; 3]; 4] = [f(px[0]), f(px[1]), f(px[2]), f(px[3])];

        let mut best = (i32::MAX, 0u8, [0i32; 3], [0i32; 3]);
        for m in [0u8, 2, 4, 6, 8, 10, 12, 14] {
            let mut sa = [0i32; 3];
            let mut sb = [0i32; 3];
            let mut na = 0i32;
            let mut nb = 0i32;
            for (i, q) in p.iter().enumerate() {
                if m >> i & 1 == 1 {
                    for k in 0..3 {
                        sa[k] += q[k];
                    }
                    na += 1;
                } else {
                    for k in 0..3 {
                        sb[k] += q[k];
                    }
                    nb += 1;
                }
            }
            let ma = if na > 0 { [sa[0] / na, sa[1] / na, sa[2] / na] } else { [0; 3] };
            let mb = if nb > 0 { [sb[0] / nb, sb[1] / nb, sb[2] / nb] } else { [0; 3] };
            let mut err = 0i32;
            for (i, q) in p.iter().enumerate() {
                let c = if m >> i & 1 == 1 { ma } else { mb };
                for k in 0..3 {
                    let d = q[k] - c[k];
                    err += d * d;
                }
            }
            if err < best.0 {
                best = (err, m, ma, mb);
            }
            if err == 0 {
                break;
            }
        }
        let (_, mask, ma, mb) = best;
        let to_u8 = |c: [i32; 3]| [c[0] as u8, c[1] as u8, c[2] as u8];
        (to_u8(ma), to_u8(mb), mask)
    }

    /// マスクに対応する四分割ブロック文字。ビットは TL, TR, BL, BR。
    #[inline]
    fn glyph(mask: u8) -> char {
        match mask {
            0b0000 => ' ',
            0b0001 => '\u{2598}', // ▘
            0b0010 => '\u{259D}', // ▝
            0b0011 => '\u{2580}', // ▀
            0b0100 => '\u{2596}', // ▖
            0b0101 => '\u{258C}', // ▌
            0b0110 => '\u{259E}', // ▞
            0b0111 => '\u{259B}', // ▛
            0b1000 => '\u{2597}', // ▗
            0b1001 => '\u{259A}', // ▚
            0b1010 => '\u{2590}', // ▐
            0b1011 => '\u{259C}', // ▜
            0b1100 => '\u{2584}', // ▄
            0b1101 => '\u{2599}', // ▙
            0b1110 => '\u{259F}', // ▟
            _ => '\u{2588}',      // █
        }
    }

    /// 差分のみを ANSI で書き出す。`full` なら全セルを再描画する。
    pub fn present(&mut self, out: &mut String, full: bool) {
        self.flush();
        self.flush_sprites();
        self.resolve();
        out.clear();
        let (dw, dh) = self.disp_dims();
        let rows = dh / 2;
        let cols = match self.blocks {
            Blocks::Half => dw,
            Blocks::Quad => dw / 2,
        };
        let full = full || !self.prev_valid;
        let mut cursor: Option<(usize, usize)> = None;
        let mut last_fg: Option<[u8; 3]> = None;
        let mut last_bg: Option<[u8; 3]> = None;

        for row in 0..rows {
            let top = row * 2 * dw;
            let bot = (row * 2 + 1) * dw;
            for col in 0..cols {
                let cell = match self.blocks {
                    Blocks::Half => {
                        let t = self.tonemap(self.disp[top + col]);
                        let b = self.tonemap(self.disp[bot + col]);
                        if t == b {
                            Cell { fg: t, bg: b, mask: 0 }
                        } else {
                            Cell { fg: t, bg: b, mask: 0b0011 }
                        }
                    }
                    Blocks::Quad => {
                        let x = col * 2;
                        let px = [
                            self.tonemap(self.disp[top + x]),
                            self.tonemap(self.disp[top + x + 1]),
                            self.tonemap(self.disp[bot + x]),
                            self.tonemap(self.disp[bot + x + 1]),
                        ];
                        let (fg, bg, mask) = Self::quantize(px);
                        Cell { fg, bg, mask }
                    }
                };

                let pi = row * cols + col;
                if !full && self.prev_cells[pi] == cell {
                    continue;
                }
                self.prev_cells[pi] = cell;

                if cursor != Some((row, col)) {
                    let _ = write!(out, "\x1b[{};{}H", row + 1, col + 1);
                    last_fg = None;
                    last_bg = None;
                }

                // 全部が背景色なら空白 1 文字で済む。
                if cell.mask == 0 {
                    if last_bg != Some(cell.bg) {
                        let _ = write!(
                            out,
                            "\x1b[48;2;{};{};{}m",
                            cell.bg[0], cell.bg[1], cell.bg[2]
                        );
                        last_bg = Some(cell.bg);
                    }
                    out.push(' ');
                    cursor = Some((row, col + 1));
                    continue;
                }

                // 前景と背景が両方変わるときは 1 つの SGR にまとめる。
                let (f, b) = (cell.fg, cell.bg);
                match (last_fg != Some(f), last_bg != Some(b)) {
                    (true, true) => {
                        let _ = write!(
                            out,
                            "\x1b[38;2;{};{};{};48;2;{};{};{}m",
                            f[0], f[1], f[2], b[0], b[1], b[2]
                        );
                        last_fg = Some(f);
                        last_bg = Some(b);
                    }
                    (true, false) => {
                        let _ = write!(out, "\x1b[38;2;{};{};{}m", f[0], f[1], f[2]);
                        last_fg = Some(f);
                    }
                    (false, true) => {
                        let _ = write!(out, "\x1b[48;2;{};{};{}m", b[0], b[1], b[2]);
                        last_bg = Some(b);
                    }
                    (false, false) => {}
                }
                out.push(Self::glyph(cell.mask));
                cursor = Some((row, col + 1));
            }
        }
        self.prev_valid = true;
    }

    /// PPM (P6) として書き出す。開発時の目視確認用。
    /// ピクセルが正方形でないモードでは、縦に伸ばして比率を合わせる。
    pub fn to_ppm(&mut self) -> Vec<u8> {
        self.flush();
        self.flush_sprites();
        self.resolve();
        let (dw, _) = self.disp_dims();
        let dh = self.disp.len() / dw.max(1);
        let rep = (1.0 / self.px_aspect).round().max(1.0) as usize;
        let mut out = format!("P6\n{} {}\n255\n", dw, dh * rep).into_bytes();
        for row in self.disp.chunks(dw) {
            let line: Vec<u8> = row.iter().flat_map(|c| self.tonemap(*c)).collect();
            for _ in 0..rep {
                out.extend_from_slice(&line);
            }
        }
        out
    }
}

/// 影の箱による遮蔽率。地面のシェーディングから直接呼べるようフリー関数にしてある。
///
/// 点 `p` から太陽へ向かう線分が箱を貫いているかを、
/// 2D のスラブ判定で求める。貫通長が長いほど濃い影になる。
pub fn shadow_of(boxes: &[(V3, V3)], sun: V3, p: V3) -> f32 {
    if boxes.is_empty() || sun.y < 0.06 {
        return 0.0;
    }
    let l = sun;
    let mut occ: f32 = 0.0;
    for (lo, hi) in boxes {
        // 箱の高さの範囲を通る t の区間。
        let mut tmin = ((lo.y - p.y) / l.y).max(0.0);
        let mut tmax = (hi.y - p.y) / l.y;
        if tmax <= tmin {
            continue;
        }
        // x と z のスラブで区間を削る。
        let mut ok = true;
        for (o, d, a, b) in [(p.x, l.x, lo.x, hi.x), (p.z, l.z, lo.z, hi.z)] {
            if d.abs() < 1e-6 {
                if o < a || o > b {
                    ok = false;
                    break;
                }
                continue;
            }
            let (t0, t1) = ((a - o) / d, (b - o) / d);
            tmin = tmin.max(t0.min(t1));
            tmax = tmax.min(t0.max(t1));
            if tmax <= tmin {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // 距離が離れるほど輪郭をぼかす（半影）。
        let soft = 0.20 + tmin * 0.10;
        occ = occ.max(smoothstep(0.0, soft, tmax - tmin));
    }
    saturate(occ) * 0.94
}

/// 大気による減衰。
pub fn fog_of(env: &Env, c: V3, dist: f32, dir: V3) -> V3 {
    let f = 1.0 - (-dist * env.fog_density).exp();
    // 太陽方向の霧はほんのり明るい。
    let glow = saturate(dir.dot(env.sun_dir)).powf(4.0);
    let fog_col = env.horizon_color.lerp(env.sun_color, glow * 0.5);
    c.lerp(fog_col, f)
}

/// 陰影計算の本体。スレッドから直接呼べるようフリー関数にしてある。
pub fn shade_of(
    env: &Env,
    cam_pos: V3,
    p: V3,
    n: V3,
    albedo: V3,
    m: &Material,
    shadow: f32,
) -> V3 {
    let e = env;
    let view = (cam_pos - p).norm();
    // 裏面が見えているときは法線を反転して両面ライティングにする。
    let n = if n.dot(view) < 0.0 { -n } else { n };

    let ndl = saturate(n.dot(e.sun_dir));
    let vis = 1.0 - shadow;
    let mut lit = albedo.mul3(e.sun_color) * (ndl * vis);

    // 半球状の環境光。上は空、下は地面の照り返し。
    let up = saturate(n.y * 0.5 + 0.5);
    let amb = e.sky_color * up + e.ground_color * (1.0 - up);
    lit += albedo.mul3(amb) * 0.55;

    // Blinn-Phong のハイライト。金属は素材色に着色する。
    let hv = (e.sun_dir + view).norm();
    let shin = 2.0 / (m.rough * m.rough + 1e-3) + 2.0;
    let spec = saturate(n.dot(hv)).powf(shin) * (1.0 - m.rough) * vis * ndl.max(0.0);
    let spec_tint = V3::splat(1.0).lerp(albedo, m.metal);
    lit += spec_tint.mul3(e.sun_color) * (spec * (0.32 + m.metal * 0.85));

    // 空と地面の映り込み。黒い車体に立体感を与えるのはほぼこれ。
    let refl = n * (2.0 * n.dot(view)) - view;
    let ry = refl.y;
    let env_col = if ry >= 0.0 {
        e.horizon_color.lerp(e.zenith_color, saturate(ry).powf(0.45))
    } else {
        e.horizon_color.lerp(e.ground_color, saturate(-ry).powf(0.45))
    };
    // ざらついた面ほど反射はぼやけ、彩度も落ちる。
    let env_lum = env_col.x * 0.2126 + env_col.y * 0.7152 + env_col.z * 0.0722;
    let env_col = env_col.lerp(V3::splat(env_lum), m.rough * 0.55);
    let fres = (1.0 - saturate(n.dot(view))).powi(5);
    let k = (0.018 + m.metal * 0.20) * (1.0 - m.rough * 0.65)
        + fres * (0.08 + m.metal * 0.40);
    lit += env_col.mul3(spec_tint) * k;

    // 前照灯（点光源 + スポット）。
    if e.head_power > 0.0 {
        let d = e.head_pos - p;
        let dist = d.len().max(0.5);
        let ld = d / dist;
        let spot = saturate((-ld).dot(e.head_dir));
        let spot = spot.powf(42.0);
        let att = e.head_power / (1.0 + dist * dist * 0.02);
        lit += albedo.mul3(v3(1.0, 0.93, 0.75)) * (saturate(n.dot(ld)) * att * spot);
    }
    // かまどの火の漏れ光。
    if e.fire_power > 0.0 {
        let d = e.fire_pos - p;
        let dist = d.len().max(0.4);
        let att = e.fire_power / (1.0 + dist * dist * 0.55);
        lit += albedo.mul3(v3(1.0, 0.38, 0.10)) * (saturate(n.dot(d / dist)) * att);
    }

    lit + m.emissive
}

/// 使えるコア数（上限 16）。
fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
}

/// 横方向の箱ぼかし。移動和で持つので、半径を広げても 1 画素あたりの
/// 手間は変わらない。端は clamp する。
fn blur_h(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    if w == 0 {
        return;
    }
    let r = r.min(w * 2);
    let inv = 1.0 / (r * 2 + 1) as f32;
    for y in 0..h {
        let row = y * w;
        let mut acc = src[row] * (r as f32 + 1.0);
        for i in 1..=r {
            acc += src[row + i.min(w - 1)];
        }
        for x in 0..w {
            dst[row + x] = acc * inv;
            let add = src[row + (x + r + 1).min(w - 1)];
            let sub = src[row + x.saturating_sub(r)];
            acc += add - sub;
        }
    }
}

/// 縦方向の箱ぼかし。
fn blur_v(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    if h == 0 {
        return;
    }
    let r = r.min(h * 2);
    let inv = 1.0 / (r * 2 + 1) as f32;
    for x in 0..w {
        let mut acc = src[x] * (r as f32 + 1.0);
        for i in 1..=r {
            acc += src[i.min(h - 1) * w + x];
        }
        for y in 0..h {
            dst[y * w + x] = acc * inv;
            let add = src[(y + r + 1).min(h - 1) * w + x];
            let sub = src[y.saturating_sub(r) * w + x];
            acc += add - sub;
        }
    }
}

/// 近クリップ面で切り、残りを扇状に分割して画面空間へ落とす。
fn clip_and_project(v: &[ClipV; 3], w: f32, h: f32, out: &mut Vec<ScreenTri>) {
    const NEAR_W: f32 = 0.05;
    let inside = |x: &ClipV| x.pos[3] > NEAR_W;
    let n_in = v.iter().filter(|x| inside(x)).count();
    if n_in == 0 {
        return;
    }
    if n_in == 3 {
        if let Some(t) = project(&v[0], &v[1], &v[2], w, h) {
            out.push(t);
        }
        return;
    }
    let mut poly: [ClipV; 4] = [ClipV { pos: [0.0; 4], c: V3::ZERO }; 4];
    let mut n = 0usize;
    for i in 0..3 {
        let cur = v[i];
        let nxt = v[(i + 1) % 3];
        let (ci, ni) = (inside(&cur), inside(&nxt));
        if ci {
            poly[n] = cur;
            n += 1;
        }
        if ci != ni {
            let t = (NEAR_W - cur.pos[3]) / (nxt.pos[3] - cur.pos[3]);
            let mut p = [0.0f32; 4];
            for k in 0..4 {
                p[k] = cur.pos[k] + (nxt.pos[k] - cur.pos[k]) * t;
            }
            poly[n] = ClipV { pos: p, c: cur.c.lerp(nxt.c, t) };
            n += 1;
        }
    }
    for i in 1..n.saturating_sub(1) {
        if let Some(t) = project(&poly[0], &poly[i], &poly[i + 1], w, h) {
            out.push(t);
        }
    }
}

/// クリップ空間 → 画面空間。面積がゼロなら捨てる。
fn project(a: &ClipV, b: &ClipV, c: &ClipV, w: f32, h: f32) -> Option<ScreenTri> {
    let sp = |v: &ClipV| -> (f32, f32, f32) {
        let iw = 1.0 / v.pos[3];
        (
            (v.pos[0] * iw * 0.5 + 0.5) * w,
            (0.5 - v.pos[1] * iw * 0.5) * h,
            iw,
        )
    };
    let (x0, y0, w0) = sp(a);
    let (x1, y1, w1) = sp(b);
    let (x2, y2, w2) = sp(c);
    let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if area.abs() < 1e-7 {
        return None;
    }
    Some(ScreenTri {
        x: [x0, x1, x2],
        y: [y0, y1, y2],
        iw: [w0, w1, w2],
        // 遠近補正のために色は 1/w を掛けた空間で補間する。
        c: [a.c * w0, b.c * w1, c.c * w2],
        y0: y0.min(y1).min(y2),
        y1: y0.max(y1).max(y2),
    })
}

/// 1 本の横帯だけを塗る。`buf`/`depth` は帯の先頭からのスライス。
fn raster_band(
    t: &ScreenTri,
    buf: &mut [V3],
    depth: &mut [f32],
    w: usize,
    band_y0: usize,
    band_rows: usize,
) {
    let (x0, x1, x2) = (t.x[0], t.x[1], t.x[2]);
    let (y0, y1, y2) = (t.y[0], t.y[1], t.y[2]);
    let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    let inv_area = 1.0 / area;

    let min_x = t.x[0].min(x1).min(x2).floor().max(0.0) as usize;
    let max_x = (t.x[0].max(x1).max(x2).ceil() as i64).min(w as i64 - 1);
    if max_x < 0 || min_x >= w {
        return;
    }
    let lo = (t.y0.floor().max(band_y0 as f32) as usize).max(band_y0);
    let hi = (t.y1.ceil() as i64).min((band_y0 + band_rows) as i64 - 1);
    if hi < lo as i64 {
        return;
    }

    let (ca, cb, cc) = (t.c[0], t.c[1], t.c[2]);
    let (w0, w1, w2) = (t.iw[0], t.iw[1], t.iw[2]);

    for py in lo..=(hi as usize) {
        let fy = py as f32 + 0.5;
        let row = (py - band_y0) * w;
        for px in min_x..=(max_x as usize) {
            let fx = px as f32 + 0.5;
            let mut l0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inv_area;
            let mut l1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inv_area;
            let mut l2 = 1.0 - l0 - l1;
            if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                // わずかな隙間を埋めるため、境界は少しだけ許容する。
                const EPS: f32 = -1e-4;
                if l0 < EPS || l1 < EPS || l2 < EPS {
                    continue;
                }
                l0 = l0.max(0.0);
                l1 = l1.max(0.0);
                l2 = l2.max(0.0);
            }
            let iw = l0 * w0 + l1 * w1 + l2 * w2;
            if iw <= 0.0 {
                continue;
            }
            let d = 1.0 / iw;
            let idx = row + px;
            if d >= depth[idx] {
                continue;
            }
            depth[idx] = d;
            buf[idx] = (ca * l0 + cb * l1 + cc * l2) * d;
        }
    }
}

/// 1 本の横帯にスプライトを 1 つ描く。`buf` は帯の先頭からのスライス、
/// `depth` は画面全体（スプライトは深度を書き換えない）。
fn draw_sprite(
    s: &Sprite,
    buf: &mut [V3],
    depth: &[f32],
    w: usize,
    band_y0: usize,
    band_rows: usize,
) {
    let x0 = (s.sx - s.rx).floor().max(0.0) as usize;
    let x1 = ((s.sx + s.rx).ceil() as i64).min(w as i64 - 1);
    if x1 < 0 || x0 >= w {
        return;
    }
    let lo = ((s.sy - s.ry).floor().max(0.0) as usize).max(band_y0);
    let hi = ((s.sy + s.ry).ceil() as i64).min((band_y0 + band_rows) as i64 - 1);
    if hi < lo as i64 {
        return;
    }
    let inv_rx = 1.0 / s.rx.max(1e-4);
    let inv_ry = 1.0 / s.ry.max(1e-4);

    for py in lo..=(hi as usize) {
        let dy = (py as f32 + 0.5 - s.sy) * inv_ry;
        let row = (py - band_y0) * w;
        let drow = py * w;
        for px in x0..=(x1 as usize) {
            let dx = (px as f32 + 0.5 - s.sx) * inv_rx;
            let rr = dx * dx + dy * dy;
            if rr > 1.0 {
                continue;
            }
            if s.kind == 1 {
                // 中心が強く、外へ向かって急速に落ちる光芒。
                let f = (1.0 - rr.sqrt()).powi(3);
                buf[row + px] += s.a * (f * s.k);
                continue;
            }
            if s.depth >= depth[drow + px] {
                continue;
            }
            let rl = rr.sqrt();
            // もこもこした輪郭。ノイズで半径を揺らす。
            let ang = dy.atan2(dx);
            let wob = crate::noise::noise2(
                ang.cos() * 2.6 + s.seed * 7.0,
                ang.sin() * 2.6 + s.seed * 3.0,
            ) - 0.5;
            let edge = 1.0 + wob * 0.55;
            let dens = smoothstep(edge, edge * 0.25, rl);
            let a = dens * s.k;
            if a <= 0.004 {
                continue;
            }
            // 中心は濃く、縁は太陽光を透過して明るい。
            let col = s.a.lerp(s.b, (1.0 - dens) * 0.55);
            let dst = buf[row + px];
            buf[row + px] = dst.lerp(col, saturate(a));
        }
    }
}
