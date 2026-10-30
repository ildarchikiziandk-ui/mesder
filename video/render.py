#!/usr/bin/env python3
"""Рендер видео «Весь Python за один урок».

    python3 render.py                 # всё видео -> build/python_za_odin_urok.mp4
    python3 render.py --preview 5 12  # кадр сцены 5 на 12-й секунде -> build/preview.png
    python3 render.py --no-voice      # игнорировать файлы озвучки, тайминг по оценке

Озвучка берётся из voice/SS_KK.mp3 (сцена SS, фраза KK) — её делает voice.py,
либо можно положить туда собственные записи. Если файла нет, длительность
фразы оценивается по длине текста, и видео собирается с субтитрами и музыкой.
"""
import argparse
import builtins
import hashlib
import io
import json
import keyword
import math
import os
import re
import subprocess
import sys
import tempfile
import tokenize
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from script import SCENES

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
VOICE_DIR = os.path.join(ROOT, "voice")
OUT_NAME = "python_za_odin_urok.mp4"

W, H, FPS, SR = 1920, 1080, 30, 48000

CARD = 1.7      # заставка главы
LEAD = 0.35     # пауза после заставки перед первой фразой
GAP = 0.45      # пауза между фразами
TAIL = 1.1      # пауза в конце сцены
FADE = 0.45     # затухание сцены

# ---------------------------------------------------------------- цвета и шрифты

BG1, BG2 = (9, 12, 26), (20, 12, 38)
PANEL = (15, 18, 31)
TEXT = (192, 202, 245)
MUTED = (115, 122, 162)
DIM = (59, 66, 97)
BLUE = (122, 162, 247)
PY_BLUE = (55, 118, 171)
PY_YELLOW = (255, 212, 59)
GREEN = (158, 206, 106)
RED = (247, 118, 142)
PURPLE = (187, 154, 247)

C_KEYWORD = PURPLE
C_BUILTIN = (125, 207, 255)
C_STRING = GREEN
C_NUMBER = (255, 158, 100)
C_COMMENT = (86, 95, 137)
C_FUNC = BLUE
C_CLASS = (224, 175, 104)
C_OP = (137, 221, 255)
C_SELF = RED
C_CONST = (255, 158, 100)

FONT_DIR_INTER = "/usr/share/fonts/opentype/inter"
FONT_DIR_MONO = "/usr/share/fonts/truetype/jetbrains-mono"
F_INTER = {
    "medium": f"{FONT_DIR_INTER}/Inter-Medium.otf",
    "semibold": f"{FONT_DIR_INTER}/Inter-SemiBold.otf",
    "bold": f"{FONT_DIR_INTER}/Inter-Bold.otf",
    "display": f"{FONT_DIR_INTER}/InterDisplay-ExtraBold.otf",
    "black": f"{FONT_DIR_INTER}/InterDisplay-Black.otf",
}
F_MONO = {
    "regular": f"{FONT_DIR_MONO}/JetBrainsMono-Regular.ttf",
    "italic": f"{FONT_DIR_MONO}/JetBrainsMono-Italic.ttf",
    "bold": f"{FONT_DIR_MONO}/JetBrainsMono-Bold.ttf",
}
_fonts = {}


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


# ---------------------------------------------------------------- утилиты

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def fade_alpha(img, a):
    """Копия RGBA-картинки с прозрачностью, умноженной на a."""
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    return out


def paste(frame, img, x, y, a=1.0):
    if a <= 0.001:
        return
    img = fade_alpha(img, a)
    frame.paste(img, (int(round(x)), int(round(y))), img)


def gradient(w, h, c1, c2, horizontal=True):
    t = np.linspace(0, 1, w if horizontal else h, dtype=np.float32)
    t = t[None, :] if horizontal else t[:, None]
    arr = np.zeros((h, w, 3), np.float32)
    for i in range(3):
        arr[..., i] = c1[i] * (1 - t) + c2[i] * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def gradient_text(text, fnt, c1, c2):
    l, t, r, b = fnt.getbbox(text)
    w, h = r - l + 4, b - t + 4
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((-l + 2, -t + 2), text, font=fnt, fill=255)
    img = gradient(w, h, c1, c2).convert("RGBA")
    img.putalpha(mask)
    return img


def text_img(text, fnt, color, pad=2):
    l, t, r, b = fnt.getbbox(text)
    img = Image.new("RGBA", (max(1, r - l + 2 * pad), max(1, b - t + 2 * pad)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l + pad, -t + pad), text, font=fnt, fill=color)
    return img


def shadowed_panel(w, h, radius=22, fill=PANEL + (236,), blur=28, offset=18):
    """Панель со скруглёнными углами, мягкой тенью и тонкой рамкой."""
    m = blur * 2
    img = Image.new("RGBA", (w + 2 * m, h + 2 * m + offset), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((m, m + offset, m + w, m + h + offset), radius, fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(blur)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((m, m, m + w, m + h), radius, fill=fill, outline=(255, 255, 255, 22), width=2)
    return img, m


def sub_text(s):
    return re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\1", s)


def tts_text(s):
    return re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\2", s)


def wrap_words(text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if fnt.getlength(cand) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- озвучка и тайминг

def text_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def load_manifest():
    try:
        with open(os.path.join(VOICE_DIR, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def voice_file(si, k, text, manifest):
    """Файл озвучки фразы, если он есть и соответствует текущему тексту."""
    stem = f"{si:02d}_{k:02d}"
    for ext in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        p = os.path.join(VOICE_DIR, stem + ext)
        if os.path.exists(p):
            h = manifest.get(stem)
            if h is None or h == text_hash(tts_text(text)):
                return p
    return None


def media_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def estimate_duration(text):
    # нейросетевой диктор говорит примерно 14 символов в секунду
    return len(tts_text(text)) / 14.0 + 0.3


SENTINEL = "⁣STEP⁣"


def run_code(blocks, stdin):
    """Выполняет код сцены и возвращает вывод после каждого блока (или None)."""
    prelude = ""
    if stdin:
        prelude = (
            "import builtins as _b\n"
            f"_it = iter({stdin!r})\n"
            "def _input(prompt=''):\n"
            "    v = next(_it)\n"
            "    print(prompt + v)\n"
            "    return v\n"
            "_b.input = _input\n"
        )
    parts, marks = [], []
    for i, b in enumerate(blocks):
        parts.append(b)
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        # нельзя вставлять print между if/elif, внутрь класса и т.п.
        can_mark = nxt is None or not (nxt[:1].isspace() or
                                       re.match(r"(elif|else|except|finally|case)\b", nxt))
        marks.append(can_mark)
        if can_mark:
            parts.append(f'print("{SENTINEL}", flush=True)')
    src = prelude + "\n".join(parts) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run([sys.executable, "-u", "-c", src], cwd=tmp, capture_output=True,
                             text=True, timeout=30, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = (res.stdout + res.stderr).replace("\r", "")
    chunks = out.split(SENTINEL + "\n")
    outs, acc, ci = [], "", 0
    for m in marks:
        if m:
            acc += chunks[ci] if ci < len(chunks) else ""
            ci += 1
            outs.append(acc)
        else:
            outs.append(None)
    return outs


def build_timeline(use_voice=True):
    manifest = load_manifest()
    scenes, t_global, frame0 = [], 0.0, 0
    chapter_no = 0
    for si, raw in enumerate(SCENES):
        sc = dict(raw)
        sc["index"] = si
        if sc["kind"] != "title":
            chapter_no += 1
            sc["number"] = chapter_no
        t = 1.3 if sc["kind"] == "title" else CARD + LEAD
        steps = []
        for k, st in enumerate(sc["steps"]):
            st = dict(st)
            st["sub"] = sub_text(st["text"])
            vf = voice_file(si, k, st["text"], manifest) if use_voice else None
            st["voice"] = vf
            st["dur"] = media_duration(vf) if vf else estimate_duration(st["text"])
            st["start"] = t
            st["end"] = t + st["dur"]
            t = st["end"] + GAP
            steps.append(st)
        sc["steps"] = steps
        dur = t - GAP + TAIL + (1.2 if sc["kind"] == "title" else 0)
        sc["frames"] = int(round(dur * FPS))
        sc["dur"] = sc["frames"] / FPS
        sc["frame0"] = frame0
        sc["t0"] = frame0 / FPS
        frame0 += sc["frames"]
        if sc["kind"] == "code":
            prepare_code(sc)
        scenes.append(sc)
    return scenes, frame0 / FPS


# ---------------------------------------------------------------- подсветка синтаксиса

BUILTIN_NAMES = set(dir(builtins))
FSTRING_TOKENS = {getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
                  if hasattr(tokenize, n)}


def colorize(code):
    """Цвет и начертание для каждого символа кода."""
    colors = [TEXT] * len(code)
    italic = [False] * len(code)
    starts = [0]
    for line in code.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    toks = [t for t in tokenize.generate_tokens(io.StringIO(code + "\n").readline)]
    skip = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
            tokenize.ENDMARKER}
    sig = [t for t in toks if t.type not in skip]
    nxt_of = {id(t): (sig[i + 1].string if i + 1 < len(sig) else "") for i, t in enumerate(sig)}
    prv_of = {id(t): (sig[i - 1].string if i > 0 else "") for i, t in enumerate(sig)}
    first_on_line = set()
    seen_rows = set()
    for t in sig:
        if t.start[0] not in seen_rows:
            seen_rows.add(t.start[0])
            first_on_line.add(id(t))
    decorator_next = False
    for tok in toks:
        a = starts[tok.start[0] - 1] + tok.start[1]
        b = min(len(code), starts[tok.end[0] - 1] + tok.end[1])
        s, col, it = tok.string, None, False
        if tok.type == tokenize.COMMENT:
            col, it = C_COMMENT, True
        elif tok.type == tokenize.STRING or tok.type in FSTRING_TOKENS:
            col = C_STRING
        elif tok.type == tokenize.NUMBER:
            col = C_NUMBER
        elif tok.type == tokenize.NAME:
            nxt, prv = nxt_of.get(id(tok), ""), prv_of.get(id(tok), "")
            if decorator_next:
                col, decorator_next = C_CLASS, False
            elif s in ("True", "False", "None"):
                col = C_CONST
            elif keyword.iskeyword(s):
                col = C_KEYWORD
            elif s in ("match", "case") and id(tok) in first_on_line:
                col = C_KEYWORD
            elif prv == "def":
                col = C_FUNC
            elif prv == "class":
                col = C_CLASS
            elif s == "self":
                col = C_SELF
            elif s.startswith("__") and s.endswith("__"):
                col = C_BUILTIN
            elif s in BUILTIN_NAMES:
                col = C_CLASS if s[0].isupper() else C_BUILTIN
            elif nxt == "(":
                col = C_CLASS if s[0].isupper() else C_FUNC
            elif s[0].isupper():
                col = C_CLASS
        elif tok.type == tokenize.OP:
            if s == "@" and id(tok) in first_on_line:
                col, decorator_next = C_CLASS, True
            else:
                col = C_OP
        if col:
            for i in range(a, b):
                colors[i], italic[i] = col, it
    return colors, italic


def prepare_code(sc):
    blocks, code = [], ""
    for st in sc["steps"]:
        if st["code"] is None:
            continue
        if code:
            code += "\n" if st["cont"] else "\n\n"
        st["char_start"] = len(code)
        code += st["code"]
        st["char_end"] = len(code)
        blocks.append(st["code"])
        # печатаем с естественной скоростью, но успеваем до конца фразы
        n = st["char_end"] - st["char_start"]
        st["type_start"] = st["start"] + 0.15
        st["type_dur"] = max(0.3, min(n / 24.0, st["dur"] * 0.8))
        st["type_end"] = st["type_start"] + st["type_dur"]
    sc["code"] = code
    sc["lines"] = code.split("\n")
    sc["line_starts"] = [0]
    for line in sc["lines"]:
        sc["line_starts"].append(sc["line_starts"][-1] + len(line) + 1)
    sc["colors"], sc["italic"] = colorize(code)
    outs = run_code(blocks, sc.get("stdin"))
    sc["outputs"] = []  # (время появления, полный текст вывода)
    bi = 0
    for st in sc["steps"]:
        if st["code"] is None:
            continue
        o = outs[bi]
        bi += 1
        prev = sc["outputs"][-1][1] if sc["outputs"] else ""
        if o is not None and o.rstrip("\n") != prev:
            st["out_time"] = st["type_end"] + 0.3
            sc["outputs"].append((st["out_time"], o.rstrip("\n")))
    # шрифт подбирается так, чтобы весь код влез в окно
    max_cols = max(len(l) for l in sc["lines"])
    for size in range(30, 17, -1):
        f = font(F_MONO["regular"], size)
        cw = f.getlength("M")
        lh = round(size * 1.5)
        if cw * max_cols <= CODE_W - GUTTER - 50 and lh * len(sc["lines"]) <= CODE_H - TITLEBAR - 50:
            break
    sc["font_size"], sc["char_w"], sc["line_h"] = size, cw, lh


def char_pos_at(sc, t):
    """Сколько символов кода видно к моменту t и активен ли набор."""
    pos, typing = 0, False
    for st in sc["steps"]:
        if st["code"] is None:
            continue
        if t < st["type_start"]:
            break
        prog = clamp((t - st["type_start"]) / st["type_dur"])
        n = st["char_end"] - st["char_start"]
        pos = st["char_start"] + int(round(n * prog))
        typing = prog < 1
        if typing:
            break
    return pos, typing


# ---------------------------------------------------------------- раскладка

HEADER_Y = 44
CODE_X, CODE_Y, CODE_W, CODE_H = 80, 136, 1090, 790
TERM_X, TERM_Y, TERM_W, TERM_H = 1200, 136, 640, 790
TITLEBAR = 54
GUTTER = 70
SUB_Y = 1000


class Background:
    def __init__(self):
        bw, bh = W + 420, H + 320
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        g = (xx / bw + yy / bh) / 2
        arr = np.zeros((bh, bw, 3), np.float32)
        for i in range(3):
            arr[..., i] = BG1[i] * (1 - g) + BG2[i] * g
        blobs = [(0.18, 0.25, 520, PY_BLUE, 0.55), (0.85, 0.75, 600, (124, 58, 237), 0.40),
                 (0.75, 0.15, 380, PY_YELLOW, 0.10), (0.30, 0.90, 450, (14, 165, 233), 0.22)]
        for cx, cy, r, c, s in blobs:
            d = np.exp(-(((xx - cx * bw) ** 2 + (yy - cy * bh) ** 2) / (r * r)))
            for i in range(3):
                arr[..., i] += c[i] * s * d
        dots = ((xx % 46) < 2.2) & ((yy % 46) < 2.2)
        arr[dots] += 9
        self.img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def at(self, t):
        ox = 210 + 170 * math.sin(2 * math.pi * t / 53)
        oy = 160 + 120 * math.sin(2 * math.pi * t / 67 + 1.3)
        return self.img.crop((int(ox), int(oy), int(ox) + W, int(oy) + H))


def python_logo(size):
    """Простой логотип из двух скруглённых «змей» Python-цветов."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size
    r = s * 0.16
    d.rounded_rectangle((s * 0.08, s * 0.08, s * 0.62, s * 0.62), r, fill=PY_BLUE)
    d.rounded_rectangle((s * 0.38, s * 0.38, s * 0.92, s * 0.92), r, fill=PY_YELLOW)
    d.ellipse((s * 0.2, s * 0.17, s * 0.29, s * 0.26), fill=(255, 255, 255))
    d.ellipse((s * 0.71, s * 0.74, s * 0.8, s * 0.83), fill=(30, 30, 30))
    return img


# ---------------------------------------------------------------- рендер сцен

class SceneRenderer:
    def __init__(self, sc, total, chapters):
        self.sc, self.total, self.chapters = sc, total, chapters
        self.bg = Background()
        self.line_cache, self.term_cache, self.sub_cache = {}, {}, {}
        self.subs = {}
        kind = sc["kind"]
        if kind == "title":
            self.init_title()
        else:
            self.init_card()
            self.init_header()
            if kind == "code":
                self.init_code()
            else:
                self.init_bullets()

    # ---- общие элементы

    def init_card(self):
        sc = self.sc
        self.card_num = gradient_text(f"{sc['number']:02d}", font(F_INTER["black"], 260), PY_BLUE, PY_YELLOW)
        self.card_label = text_img("ГЛАВА", font(F_INTER["bold"], 30), MUTED)
        self.card_title = text_img(sc["chapter"], font(F_INTER["display"], 96), (240, 244, 255))

    def draw_card(self, frame, t):
        a_in = ease_out(t / 0.45)
        a_out = 1 - ease_in_out((t - (CARD - 0.4)) / 0.4)
        a = min(a_in, a_out)
        if a <= 0:
            return
        dx = (1 - a_in) * -80 + (1 - a_out) * 80
        cy = H // 2
        x = 260 + dx
        paste(frame, self.card_num, x, cy - self.card_num.height // 2 - 10, a)
        tx = x + self.card_num.width + 60
        paste(frame, self.card_label, tx + 4, cy - 95, a)
        paste(frame, self.card_title, tx, cy - 40, a)
        lw = int(520 * ease_out((t - 0.15) / 0.6))
        if lw > 0:
            bar = gradient(lw, 6, PY_BLUE, PY_YELLOW).convert("RGBA")
            paste(frame, bar, tx + 4, cy + self.card_title.height - 10, a)

    def init_header(self):
        sc = self.sc
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        label = f"ГЛАВА {sc['number']:02d}"
        f = font(F_INTER["bold"], 22)
        pw = int(f.getlength(label)) + 44
        pill = gradient(pw, 50, PY_BLUE, (124, 58, 237)).convert("RGBA")
        mask = Image.new("L", pill.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, 49), 25, fill=255)
        pill.putalpha(mask)
        layer.paste(pill, (CODE_X, HEADER_Y), pill)
        d.text((CODE_X + 22, HEADER_Y + 25), label, font=f, fill=(255, 255, 255), anchor="lm")
        d.text((CODE_X + pw + 22, HEADER_Y + 25), sc["chapter"], font=font(F_INTER["bold"], 40),
               fill=(240, 244, 255), anchor="lm")
        brand = "Весь Python за один урок"
        fb = font(F_INTER["semibold"], 24)
        bx = W - 80 - int(fb.getlength(brand))
        d.text((bx, HEADER_Y + 25), brand, font=fb, fill=MUTED, anchor="lm")
        logo = python_logo(40)
        layer.paste(logo, (bx - 54, HEADER_Y + 5), logo)
        self.header = layer

    def init_titlebar(self, layer, x, y, w, title):
        d = ImageDraw.Draw(layer)
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            cx = x + 30 + i * 26
            d.ellipse((cx - 8, y + TITLEBAR // 2 - 8, cx + 8, y + TITLEBAR // 2 + 8), fill=c)
        d.text((x + w // 2, y + TITLEBAR // 2), title, font=font(F_INTER["medium"], 21), fill=MUTED, anchor="mm")
        d.line((x + 2, y + TITLEBAR, x + w - 2, y + TITLEBAR), fill=(255, 255, 255, 18), width=2)

    def draw_progress(self, frame, t):
        g = (self.sc["t0"] + t) / self.total
        bar_y = H - 7
        frame.paste((30, 34, 56), (0, bar_y, W, H))
        fw = int(W * g)
        if fw > 0:
            frame.paste(self.progress_grad.crop((0, 0, fw, 7)), (0, bar_y))
        for ct in self.chapters:
            x = int(W * ct / self.total)
            frame.paste((9, 12, 26), (x, bar_y, x + 3, H))

    @property
    def progress_grad(self):
        if not hasattr(self, "_pg"):
            self._pg = gradient(W, 7, PY_BLUE, PY_YELLOW)
        return self._pg

    def sub_img(self, k):
        if k not in self.sub_cache:
            text = self.sc["steps"][k]["sub"]
            f = font(F_INTER["medium"], 34)
            lines = wrap_words(text, f, 1500)
            lh = 46
            tw = max(int(f.getlength(l)) for l in lines)
            pw, ph = tw + 64, lh * len(lines) + 28
            img = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, pw - 1, ph - 1), 18, fill=(5, 7, 16, 175))
            for i, l in enumerate(lines):
                d.text((pw // 2, 14 + lh * i + lh // 2), l, font=f, fill=(245, 247, 255), anchor="mm")
            self.sub_cache[k] = img
        return self.sub_cache[k]

    def draw_subtitle(self, frame, t, bottom=SUB_Y + 52):
        for k, st in enumerate(self.sc["steps"]):
            if st["start"] - 0.1 <= t <= st["end"] + GAP * 0.6:
                a = min(ease_out((t - st["start"] + 0.1) / 0.2), 1 - ease_out((t - st["end"]) / (GAP * 0.6)))
                img = self.sub_img(k)
                paste(frame, img, (W - img.width) // 2, bottom - img.height + (1 - a) * 8, a)

    def workspace_alpha(self, t):
        a_in = ease_out((t - (CARD - 0.25)) / 0.5)
        a_out = 1 - ease_in_out((t - (self.sc["dur"] - FADE)) / FADE)
        return clamp(min(a_in, a_out)), (1 - a_in) * 40

    # ---- кодовая сцена

    def init_code(self):
        layer = self.header.copy()
        for (x, y, w, h, title) in [(CODE_X, CODE_Y, CODE_W, CODE_H, "lesson.py"),
                                    (TERM_X, TERM_Y, TERM_W, TERM_H, "терминал")]:
            p, m = shadowed_panel(w, h)
            layer.alpha_composite(p, (x - m, y - m))
            self.init_titlebar(layer, x, y, w, title)
        d = ImageDraw.Draw(layer)
        # фон под номерами строк
        d.rounded_rectangle((CODE_X + 2, CODE_Y + TITLEBAR + 2, CODE_X + GUTTER, CODE_Y + CODE_H - 2), 20,
                            fill=(12, 14, 25, 120), corners=(False, False, False, True))
        self.static = layer
        sc = self.sc
        self.f_code = font(F_MONO["regular"], sc["font_size"])
        self.f_code_i = font(F_MONO["italic"], sc["font_size"])
        self.f_num = font(F_MONO["regular"], max(16, sc["font_size"] - 6))
        self.f_term = font(F_MONO["regular"], 23)
        self.term_cw = self.f_term.getlength("M")
        self.term_lh = 34
        self.term_cols = int((TERM_W - 56) / self.term_cw)
        self.term_rows = int((TERM_H - TITLEBAR - 50) / self.term_lh)
        self.code_x0 = CODE_X + GUTTER + 24
        self.code_y0 = CODE_Y + TITLEBAR + 26
        lh = sc["line_h"]
        self.hl_blue = Image.new("RGBA", (CODE_W - GUTTER - 4, lh), (122, 162, 247, 22))
        self.hl_yellow = Image.new("RGBA", (CODE_W - GUTTER - 4, lh), (255, 212, 59, 16))
        self.term_lines_per_output = [self.wrap_term(o) for _, o in sc["outputs"]]
        self.run_chip = self.make_run_chip()

    def make_run_chip(self):
        f = font(F_INTER["bold"], 18)
        w = int(f.getlength("ВЫПОЛНЕНО")) + 52
        img = Image.new("RGBA", (w, 32), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, 31), 16, fill=(158, 206, 106, 40), outline=GREEN + (160,), width=2)
        d.polygon([(16, 9), (16, 23), (28, 16)], fill=GREEN)
        d.text((36, 16), "ВЫПОЛНЕНО", font=f, fill=GREEN, anchor="lm")
        return img

    def wrap_term(self, text):
        lines = []
        for raw in text.split("\n"):
            while len(raw) > self.term_cols:
                lines.append(raw[:self.term_cols])
                raw = raw[self.term_cols:]
            lines.append(raw)
        return lines

    def code_line_img(self, li, n):
        key = (li, n)
        if key not in self.line_cache:
            sc = self.sc
            line = sc["lines"][li][:n]
            base = sc["line_starts"][li]
            img = Image.new("RGBA", (int(sc["char_w"] * max(1, n)) + 8, sc["line_h"]), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            y = sc["line_h"] // 2
            i = 0
            while i < len(line):
                c, it = sc["colors"][base + i], sc["italic"][base + i]
                j = i
                while j < len(line) and sc["colors"][base + j] == c and sc["italic"][base + j] == it:
                    j += 1
                d.text((i * sc["char_w"], y), line[i:j], font=self.f_code_i if it else self.f_code,
                       fill=c, anchor="lm", features=["-liga", "-calt"])
                i = j
            self.line_cache[key] = img
        return self.line_cache[key]

    def term_line_img(self, text, color):
        key = (text, color)
        if key not in self.term_cache:
            self.term_cache[key] = text_img(text, self.f_term, color, pad=0) if text.strip() else None
        return self.term_cache[key]

    def draw_code_workspace(self, frame, t, alpha, dy):
        sc = self.sc
        if alpha < 0.999:
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        else:
            layer = frame
        layer.paste(self.static, (0, 0), self.static) if layer is frame else layer.alpha_composite(self.static)

        pos, typing = char_pos_at(sc, t)
        lh = sc["line_h"]
        # активный блок / подсветка
        for st in sc["steps"]:
            if not (st["start"] <= t <= st["end"] + GAP):
                continue
            rng, img = None, None
            if st["hl"]:
                rng, img = (st["hl"][0] - 1, st["hl"][1] - 1), self.hl_yellow
            elif st["code"] is not None and t >= st["type_start"]:
                a = self.line_of(st["char_start"])
                b = self.line_of(st["char_end"])
                rng, img = (a, b), self.hl_blue
            if rng:
                for li in range(rng[0], rng[1] + 1):
                    y = self.code_y0 + li * lh
                    layer.paste(img, (CODE_X + GUTTER + 2, y), img)
                    layer.paste(PY_YELLOW if img is self.hl_yellow else BLUE, (CODE_X + GUTTER - 2, y, CODE_X + GUTTER + 2, y + lh))
        cur_line = self.line_of(pos)
        d = ImageDraw.Draw(layer)
        for li, line in enumerate(sc["lines"]):
            start = sc["line_starts"][li]
            if start > pos or (start == pos and li > 0 and pos == 0):
                break
            n = min(len(line), pos - start)
            y = self.code_y0 + li * lh
            d.text((CODE_X + GUTTER - 14, y + lh // 2), str(li + 1), font=self.f_num,
                   fill=MUTED if li == cur_line else DIM, anchor="rm")
            if n > 0:
                img = self.code_line_img(li, n)
                layer.paste(img, (self.code_x0, y), img)
        # курсор
        if sc["code"] and (typing or (t * 1.6) % 2 < 1.25):
            col = pos - sc["line_starts"][cur_line]
            cx = self.code_x0 + col * sc["char_w"]
            cy = self.code_y0 + cur_line * lh
            layer.paste(PY_YELLOW, (int(cx), cy + 6, int(cx) + 3, cy + lh - 6))

        # терминал
        tx, ty = TERM_X + 28, TERM_Y + TITLEBAR + 24
        prompt = self.term_line_img("$ python lesson.py", TEXT)
        layer.paste(prompt, (tx, ty), prompt)
        ImageDraw.Draw(layer).text((tx, ty), "$", font=self.f_term, fill=GREEN)
        cur, prev, t_show = None, [], None
        for i, (ot, _) in enumerate(sc["outputs"]):
            if t >= ot:
                prev = cur or []
                cur, t_show = self.term_lines_per_output[i], ot
        if cur:
            common = 0
            while common < min(len(prev), len(cur)) and prev[common] == cur[common]:
                common += 1
            rows = cur[-(self.term_rows - 1):]
            off = len(cur) - len(rows)
            for ri, text in enumerate(rows):
                li = off + ri
                a = 1.0 if li < common else ease_out((t - t_show - (li - common) * 0.06) / 0.3)
                if a <= 0:
                    continue
                color = RED if re.match(r"(Traceback|  File|\w+Error)", text) else TEXT
                img = self.term_line_img(text, color)
                if img:
                    y = ty + (ri + 1) * self.term_lh + (1 - a) * 10
                    paste(layer, img, tx, y + (self.term_lh - img.height) // 2 - 6, a)
            # индикатор запуска
            k = clamp(1 - (t - t_show) / 1.4)
            if k > 0:
                paste(layer, self.run_chip, TERM_X + TERM_W - self.run_chip.width - 18,
                      TERM_Y + (TITLEBAR - 32) // 2, ease_out(k * 3) if k < 0.34 else 1.0)

        if layer is not frame:
            layer = fade_alpha(layer, alpha)
            frame.paste(layer, (0, int(dy)), layer)

    def line_of(self, pos):
        ls = self.sc["line_starts"]
        for i in range(len(ls) - 1):
            if pos < ls[i + 1]:
                return i
        return len(ls) - 2

    # ---- тезисы

    def init_bullets(self):
        layer = self.header.copy()
        x, y, w, h = 260, 150, 1400, 740
        p, m = shadowed_panel(w, h, radius=28)
        layer.alpha_composite(p, (x - m, y - m))
        self.static = layer
        self.b_items = []
        f = font(F_INTER["semibold"], 46)
        fnum = font(F_INTER["bold"], 30)
        for k, st in enumerate(self.sc["steps"]):
            img = Image.new("RGBA", (1240, 120), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            badge = gradient(64, 64, PY_BLUE, (124, 58, 237)).convert("RGBA")
            mask = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, 63, 63), fill=255)
            badge.putalpha(mask)
            img.paste(badge, (0, 28), badge)
            d.text((32, 60), str(k + 1), font=fnum, fill=(255, 255, 255), anchor="mm")
            d.text((100, 60), st["bullet"] or "", font=f, fill=(240, 244, 255), anchor="lm")
            self.b_items.append(img)
        self.b_x, self.b_y = x + 90, y + (h - len(self.b_items) * 150) // 2 - 15

    def draw_bullets_workspace(self, frame, t, alpha, dy):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.alpha_composite(self.static)
        steps = self.sc["steps"]
        active = max([k for k, st in enumerate(steps) if t >= st["start"] - 0.1] or [-1])
        for k, st in enumerate(steps):
            a = ease_out((t - st["start"] + 0.1) / 0.5)
            if a <= 0:
                continue
            dim = 1.0 if k == active else 0.55
            paste(layer, self.b_items[k], self.b_x + (1 - a) * 60, self.b_y + k * 150, a * dim)
        layer = fade_alpha(layer, alpha)
        frame.paste(layer, (0, int(dy)), layer)

    # ---- заставка

    def init_title(self):
        sc = self.sc
        self.t_label = text_img(sc["label"], font(F_INTER["bold"], 30), (165, 175, 215))
        self.t_line1 = gradient_text(sc["line1"], font(F_INTER["black"], 170), (96, 165, 250), PY_YELLOW)
        self.t_line2 = text_img(sc["line2"], font(F_INTER["display"], 96), (240, 244, 255)) if sc["line2"] else None
        self.t_logo = python_logo(120)
        self.t_prompt_font = font(F_MONO["regular"], 40)
        self.t_prompt = sc["prompt"]
        pc, pi = colorize(self.t_prompt)
        self.t_prompt_colors = pc
        # блёклые обрывки кода на фоне
        snippets = ["for i in range(10):", "def main():", "import this", "x = [n ** 2 for n in nums]",
                    "class Robot:", "print(f\"{name}\")", "while True:", "return result", "try:",
                    "with open(path) as f:", "lambda x: x * 2", "yield value", "{'key': 'value'}"]
        fm = font(F_MONO["regular"], 30)
        rnd = np.random.default_rng(7)
        self.t_floats = []
        for i, s in enumerate(snippets):
            img = text_img(s, fm, (120, 140, 200, 40))
            x = rnd.uniform(30, 260) if i % 2 else rnd.uniform(1480, 1880 - img.width * 0.6)
            self.t_floats.append((img, x, rnd.uniform(0, H), rnd.uniform(14, 34),
                                  rnd.uniform(0, 6.28)))

    def draw_title(self, frame, t):
        sc = self.sc
        dur = sc["dur"]
        out = 1 - ease_in_out((t - (dur - 0.8)) / 0.8)
        for img, x, y, sp, ph in self.t_floats:
            yy = (y - t * sp) % (H + 100) - 50
            paste(frame, img, x + 20 * math.sin(t * 0.3 + ph), yy, out)
        cx = W // 2
        y0 = 250 if self.t_line2 else 330
        a = ease_out((t - 0.0) / 0.7) * out
        paste(frame, self.t_logo, cx - 60, y0 - 150 + (1 - a) * 20, a)
        a = ease_out((t - 0.2) / 0.7) * out
        paste(frame, self.t_label, cx - self.t_label.width // 2, y0 + (1 - a) * 20, a)
        a = ease_out((t - 0.35) / 0.8) * out
        paste(frame, self.t_line1, cx - self.t_line1.width // 2, y0 + 60 + (1 - a) * 40, a)
        y = y0 + 60 + self.t_line1.height + 30
        if self.t_line2:
            a = ease_out((t - 0.55) / 0.8) * out
            paste(frame, self.t_line2, cx - self.t_line2.width // 2, y + (1 - a) * 40, a)
            y += self.t_line2.height + 70
        else:
            y += 40
        # строка-подсказка с печатающимся кодом
        a = ease_out((t - 0.8) / 0.6) * out
        if a > 0:
            p = self.t_prompt
            f = self.t_prompt_font
            cw = f.getlength("M")
            full = ">>> " + p
            pw = int(cw * len(full)) + 80
            box = Image.new("RGBA", (pw, 84), (0, 0, 0, 0))
            d = ImageDraw.Draw(box)
            d.rounded_rectangle((0, 0, pw - 1, 83), 20, fill=PANEL + (220,), outline=(255, 255, 255, 30), width=2)
            n = int(clamp((t - 1.1) / (len(p) / 18)) * len(p))
            d.text((40, 42), ">>>", font=f, fill=MUTED, anchor="lm")
            for i in range(n):
                d.text((40 + cw * (4 + i), 42), p[i], font=f, fill=self.t_prompt_colors[i], anchor="lm")
            if (t * 1.6) % 2 < 1.25 or n < len(p):
                cx2 = 40 + cw * (4 + n)
                d.rectangle((cx2, 20, cx2 + 3, 64), fill=PY_YELLOW)
            paste(frame, box, cx - pw // 2, y + (1 - a) * 20, a)

    # ---- кадр

    def frame(self, t):
        frame = self.bg.at(self.sc["t0"] + t)
        kind = self.sc["kind"]
        if kind == "title":
            self.draw_title(frame, t)
            self.draw_subtitle(frame, t)
        else:
            if t < CARD:
                self.draw_card(frame, t)
            alpha, dy = self.workspace_alpha(t)
            if alpha > 0:
                if kind == "code":
                    self.draw_code_workspace(frame, t, alpha, dy)
                else:
                    self.draw_bullets_workspace(frame, t, alpha, dy)
            self.draw_subtitle(frame, t)
        if self.sc["index"] > 0:
            self.draw_progress(frame, t)
        return frame


# ---------------------------------------------------------------- видео

def render_scene(args):
    sc, total, chapters, path = args
    r = SceneRenderer(sc, total, chapters)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "19", "-tune", "animation", "-pix_fmt", "yuv420p", "-g", str(FPS * 2), path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(sc["frames"]):
        p.stdin.write(r.frame(i / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for scene {sc['index']}")
    return sc["index"]


# ---------------------------------------------------------------- звук

def decode_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def one_pole_lowpass(x, cutoffs):
    y = np.empty_like(x)
    acc = 0.0
    k = 1 - np.exp(-2 * np.pi * cutoffs / SR)
    for i in range(len(x)):
        acc += k[i] * (x[i] - acc)
        y[i] = acc
    return y


def synth_click(rng):
    n = int(0.018 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 380)
    noise = np.diff(rng.standard_normal(n + 1))
    tone = np.sin(2 * np.pi * rng.uniform(1700, 2600) * t)
    body = np.sin(2 * np.pi * rng.uniform(140, 220) * t) * np.exp(-t * 160)
    return ((noise * 0.35 + tone * 0.25) * env + body * 0.5).astype(np.float32)


def synth_whoosh(rng, dur=1.0):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 2 * np.exp(-t * 1.5)
    cut = 300 + 3500 * np.sin(np.pi * t) ** 1.5
    x = one_pole_lowpass(rng.standard_normal(n).astype(np.float32), cut)
    x = one_pole_lowpass(x, cut)
    x = x / (np.abs(x).max() + 1e-9)
    return (x * env).astype(np.float32)


def synth_chime():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((1046.5, 0.6), (1568, 0.35), (2093, 0.15)))
    return (s * np.exp(-t * 7) * np.minimum(1, t * 400)).astype(np.float32)


def synth_music(total_s):
    """Спокойный эмбиент-пэд + мягкие арпеджио (Am – F – C – G)."""
    n = int(total_s * SR) + SR
    music = np.zeros(n, np.float32)
    bar = 4.0
    chords = [[110.0, 164.81, 220.0, 261.63], [87.31, 130.81, 174.61, 220.0],
              [130.81, 196.0, 261.63, 329.63], [98.0, 146.83, 196.0, 246.94]]
    seg = int((bar + 2.5) * SR)
    ts = np.arange(seg) / SR
    env = np.minimum(1, ts / 1.4) * np.clip((bar + 2.5 - ts) / 2.0, 0, 1)
    pluck_n = int(1.2 * SR)
    tp = np.arange(pluck_n) / SR
    pluck_env = np.exp(-tp * 4.5) * np.minimum(1, tp * 300)
    i = 0
    while i * bar < total_s + 1:
        ch = chords[i % 4]
        start = int(i * bar * SR)
        pad = np.zeros(seg, np.float32)
        for f in ch:
            for det in (-0.7, 0.0, 0.7):
                pad += np.sin(2 * np.pi * (f + det) * ts + det * 3) * (0.5 if det == 0 else 0.3)
            pad += 0.12 * np.sin(2 * np.pi * f * 2 * ts)
        pad *= env * (0.55 + 0.45 * np.sin(2 * np.pi * ts / bar * 0.5) ** 2)
        end = min(n, start + seg)
        music[start:end] += pad[:end - start] * 0.12
        # арпеджио на октаву-две выше
        for j, mult in enumerate([4, 3, 2, 3, 4, 3, 2, 3]):
            note = ch[(j * 3 + i) % 4] * (2 if mult >= 3 else 4)
            ps = start + int(j * bar / 8 * SR)
            pe = min(n, ps + pluck_n)
            if ps >= n:
                break
            tone = np.sin(2 * np.pi * note * tp) + 0.25 * np.sin(2 * np.pi * note * 2 * tp)
            music[ps:pe] += (tone * pluck_env)[:pe - ps] * 0.05
        i += 1
    # простое «эхо» для объёма
    d = int(0.33 * SR)
    music[d:] += music[:-d] * 0.3
    return music / (np.abs(music).max() + 1e-9)


def build_audio(scenes, total, path):
    rng = np.random.default_rng(1)
    n = int(total * SR) + SR
    voice = np.zeros(n, np.float32)
    fx = np.zeros(n, np.float32)
    duck = np.zeros(n, np.float32)
    clicks = [synth_click(rng) for _ in range(6)]
    whoosh = synth_whoosh(rng)
    chime = synth_chime()
    has_voice = False

    def put(buf, x, t, gain=1.0):
        s = int(t * SR)
        e = min(n, s + len(x))
        if s < n:
            buf[s:e] += x[:e - s] * gain

    for sc in scenes:
        t0 = sc["t0"]
        if sc["kind"] != "title":
            put(fx, whoosh, t0 + 0.05, 0.22)
        for st in sc["steps"]:
            duck[int((t0 + st["start"]) * SR):int((t0 + st["end"]) * SR)] = 1
            if st["voice"]:
                has_voice = True
                v = decode_audio(st["voice"])
                rms = np.sqrt(np.mean(v ** 2)) + 1e-9
                put(voice, v * min(0.1 / rms, 3.0), t0 + st["start"])
        if sc["kind"] == "code":
            for st in sc["steps"]:
                if st["code"] is None:
                    continue
                text = sc["code"][st["char_start"]:st["char_end"]]
                last = -1.0
                for ci, ch in enumerate(text):
                    tt = st["type_start"] + st["type_dur"] * (ci + 1) / len(text)
                    if ch.isspace() or tt - last < 0.05:
                        continue
                    last = tt
                    put(fx, clicks[rng.integers(len(clicks))], t0 + tt, 0.05 * rng.uniform(0.6, 1.0))
                if "out_time" in st:
                    put(fx, chime, t0 + st["out_time"], 0.06)
    # ducking музыки под голос, со сглаживанием
    k = int(0.25 * SR)
    duck = np.convolve(duck, np.ones(k, np.float32) / k, mode="same")
    music = synth_music(total)[:n]
    base = 0.07 if has_voice else 0.2
    music *= base * (1 - 0.55 * duck) if has_voice else base
    fade = int(3 * SR)
    end = int(total * SR)
    music[:fade] *= np.linspace(0, 1, fade)
    music[end - fade:end] *= np.linspace(1, 0, fade)
    music[end:] = 0
    mix = voice + fx + music
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    pcm = (np.clip(mix[:end], -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return has_voice


# ---------------------------------------------------------------- субтитры и главы

def fmt_ts(t, sep=","):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)) % 1000:03d}"


def write_extras(scenes):
    with open(os.path.join(BUILD, "subtitles.srt"), "w", encoding="utf-8") as f:
        i = 1
        for sc in scenes:
            for st in sc["steps"]:
                f.write(f"{i}\n{fmt_ts(sc['t0'] + st['start'])} --> {fmt_ts(sc['t0'] + st['end'])}\n{st['sub']}\n\n")
                i += 1
    with open(os.path.join(BUILD, "chapters.txt"), "w", encoding="utf-8") as f:
        for sc in scenes:
            t = int(sc["t0"])
            name = sc.get("chapter") or ("Вступление" if sc["index"] == 0 else "Финал")
            f.write(f"{t // 60:d}:{t % 60:02d} {name}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs=2, type=float, metavar=("SCENE", "SEC"))
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--scenes", type=str, help="например 0-3 — только эти сцены (для проверки)")
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    scenes, total = build_timeline(use_voice=not args.no_voice)
    chapters = [sc["t0"] for sc in scenes[1:]]
    n_voice = sum(1 for sc in scenes for st in sc["steps"] if st["voice"])
    n_steps = sum(len(sc["steps"]) for sc in scenes)
    print(f"Сцен: {len(scenes)}, длительность: {total / 60:.1f} мин, озвучено фраз: {n_voice}/{n_steps}")

    if args.preview:
        sc = scenes[int(args.preview[0])]
        img = SceneRenderer(sc, total, chapters).frame(args.preview[1])
        out = os.path.join(BUILD, f"preview_{int(args.preview[0]):02d}_{args.preview[1]:g}.png")
        img.save(out)
        print(out)
        return

    sel = scenes
    if args.scenes:
        a, b = map(int, args.scenes.split("-"))
        sel = scenes[a:b + 1]
    seg_dir = os.path.join(BUILD, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    jobs = [(sc, total, chapters, os.path.join(seg_dir, f"{sc['index']:02d}.mp4")) for sc in sel]
    jobs.sort(key=lambda j: -j[0]["frames"])  # длинные сцены — первыми
    with Pool(args.jobs) as pool:
        for i, idx in enumerate(pool.imap_unordered(render_scene, jobs), 1):
            print(f"  сцена {idx:02d} готова ({i}/{len(jobs)})", flush=True)

    audio = os.path.join(BUILD, "audio.wav")
    sel_total = sum(sc["dur"] for sc in sel)
    if args.scenes:
        # для частичного рендера сдвигаем сцены к нулю
        shift = sel[0]["t0"]
        sel = [dict(sc, t0=sc["t0"] - shift) for sc in sel]
    has_voice = build_audio(sel, sel_total, audio)
    write_extras(scenes)
    lst = os.path.join(seg_dir, "list.txt")
    with open(lst, "w") as f:
        for sc in sel:
            f.write(f"file '{sc['index']:02d}.mp4'\n")
    out = os.path.join(BUILD, OUT_NAME if not args.scenes else f"part_{args.scenes}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                    "-movflags", "+faststart", "-shortest", out], check=True)
    print(f"Готово: {out} ({'с озвучкой' if has_voice else 'без озвучки, с субтитрами'})")


if __name__ == "__main__":
    main()
