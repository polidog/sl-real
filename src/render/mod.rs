//! ハーフブロック文字 (▀) を 1 セル 2 ピクセルとして使う、
//! Z バッファ付きソフトウェアラスタライザ。

use crate::math::*;
use crate::noise::fbm3;

mod post;
mod raster;
mod shade;
mod term;

use raster::{RawTri, Sprite};
pub use shade::{fog_of, shade_of, shadow_of};
use term::Cell;

/// 1 セルをどう埋めるか。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocks {
    /// `▀` だけを使う。1 セル = 縦 2 ピクセル。字形の対応は最も広い。
    Half,
    /// 四分割ブロックを使う。1 セル = 2x2 ピクセルで横解像度が倍になる。
    /// 1 セルに 2 色までなので、4 つの小画素を 2 色へ最適に分ける。
    Quad,
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

/// 煤と錆の汚れ。素材色に世界座標のノイズで乗せる。
///
/// ノイズは高いので、画面外の三角形を捨てた後に並列段でまとめて解く。
#[derive(Clone, Copy, Debug, Default)]
pub struct Grime {
    /// 0 で新品、1 で煤と錆にまみれた質感。
    pub amount: f32,
    pub scale: f32,
    pub tint: V3,
}

impl Grime {
    pub fn apply(&self, c: V3, p: V3) -> V3 {
        if self.amount <= 0.0 {
            return c;
        }
        let n = fbm3(p * self.scale, 2);
        let k = saturate(n * 1.35 - 0.15) * self.amount;
        c.lerp(self.tint, k * 0.85) * (1.0 - k * 0.25)
    }
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
    /// 縮小・トーンマップ後の表示用バッファ。
    ldr: Vec<[u8; 3]>,
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
            ldr: Vec::new(),
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
    }

    /// ビュー射影行列を設定し、視錐台の平面を作り直す。
    pub fn set_view(&mut self, vp: M4) {
        self.view_proj = vp;
        let m = &vp.0;
        // clip = M * p。w±x, w±y, w>0 の 5 平面（遠方は切らない）。
        let rows = [
            [
                m[3][0] + m[0][0],
                m[3][1] + m[0][1],
                m[3][2] + m[0][2],
                m[3][3] + m[0][3],
            ],
            [
                m[3][0] - m[0][0],
                m[3][1] - m[0][1],
                m[3][2] - m[0][2],
                m[3][3] - m[0][3],
            ],
            [
                m[3][0] + m[1][0],
                m[3][1] + m[1][1],
                m[3][2] + m[1][2],
                m[3][3] + m[1][3],
            ],
            [
                m[3][0] - m[1][0],
                m[3][1] - m[1][1],
                m[3][2] - m[1][2],
                m[3][3] - m[1][3],
            ],
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
}
