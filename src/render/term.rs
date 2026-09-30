//! 端末への書き出し（ANSI）と PPM 出力。

use super::*;
use rayon::prelude::*;
use std::fmt::Write as _;

/// 端末に出す 1 セル。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Cell {
    fg: [u8; 3],
    bg: [u8; 3],
    /// 前景として塗る小画素のビットマスク（下位から TL, TR, BL, BR）。
    mask: u8,
}

impl Renderer {
    /// ss x ss をリニア空間で平均し、トーンマップして表示用の 8bit バッファを作る。
    fn resolve(&mut self) {
        let (dw, dh) = self.disp_dims();
        let (w, ss, ex) = (self.w, self.ss, self.env.exposure);
        let inv = 1.0 / (ss * ss) as f32;
        let src: &[V3] = &self.color;
        self.ldr.resize(dw * dh, [0; 3]);
        self.ldr
            .par_chunks_mut(dw.max(1))
            .enumerate()
            .for_each(|(y, row)| {
                for (x, out) in row.iter_mut().enumerate() {
                    let mut acc = V3::ZERO;
                    for sy in 0..ss {
                        let base = (y * ss + sy) * w + x * ss;
                        for sx in 0..ss {
                            acc += src[base + sx];
                        }
                    }
                    *out = tonemap(acc * inv, ex);
                }
            });
    }

    /// 表示用バッファの 1 セル分を 2 色とマスクに落とす。
    fn cell_at(&self, row: usize, col: usize) -> Cell {
        let (dw, _) = self.disp_dims();
        let top = row * 2 * dw;
        let bot = top + dw;
        let l = &self.ldr;
        match self.blocks {
            Blocks::Half => {
                let (t, b) = (l[top + col], l[bot + col]);
                Cell {
                    fg: t,
                    bg: b,
                    mask: if t == b { 0 } else { 0b0011 },
                }
            }
            Blocks::Quad => {
                let x = col * 2;
                let (fg, bg, mask) =
                    quantize([l[top + x], l[top + x + 1], l[bot + x], l[bot + x + 1]]);
                Cell { fg, bg, mask }
            }
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

        // セルへの量子化は独立なので並列に解き、書き出しだけを順に行う。
        let cells: Vec<Cell> = (0..rows * cols)
            .into_par_iter()
            .map(|i| self.cell_at(i / cols, i % cols))
            .collect();

        for row in 0..rows {
            for col in 0..cols {
                let cell = cells[row * cols + col];
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
                out.push(glyph(cell.mask));
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
        let (dw, dh) = self.disp_dims();
        let rep = (1.0 / self.px_aspect).round().max(1.0) as usize;
        let mut out = format!("P6\n{} {}\n255\n", dw, dh * rep).into_bytes();
        for row in self.ldr.chunks(dw.max(1)) {
            for _ in 0..rep {
                out.extend(row.iter().flatten());
            }
        }
        out
    }
}

/// リニア HDR をトーンマップして sRGB の 8bit に落とす。
#[inline]
fn tonemap(c: V3, exposure: f32) -> [u8; 3] {
    let x = c * exposure;
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
        let ma = if na > 0 {
            [sa[0] / na, sa[1] / na, sa[2] / na]
        } else {
            [0; 3]
        };
        let mb = if nb > 0 {
            [sb[0] / nb, sb[1] / nb, sb[2] / nb]
        } else {
            [0; 3]
        };
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
