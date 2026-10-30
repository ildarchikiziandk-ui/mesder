#!/usr/bin/env python3
"""Генерация озвучки нейросетевым голосом Microsoft Edge (edge-tts).

    pip install edge-tts
    python3 voice.py                         # мужской голос Дмитрий
    python3 voice.py --voice ru-RU-SvetlanaNeural --rate +5%

Файлы ложатся в voice/SS_KK.mp3 (сцена SS, фраза KK). Уже готовые фразы
пропускаются, изменённые в script.py — переозвучиваются. Вместо TTS можно
положить туда свои записи (.wav/.mp3) с теми же именами.
"""
import argparse
import asyncio
import json
import os

import edge_tts

from render import VOICE_DIR, text_hash, tts_text
from script import SCENES


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ru-RU-DmitryNeural")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--pitch", default="+0Hz")
    args = ap.parse_args()
    os.makedirs(VOICE_DIR, exist_ok=True)
    mpath = os.path.join(VOICE_DIR, "manifest.json")
    try:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        manifest = {}
    for si, sc in enumerate(SCENES):
        for k, st in enumerate(sc["steps"]):
            stem = f"{si:02d}_{k:02d}"
            text = tts_text(st["text"])
            path = os.path.join(VOICE_DIR, stem + ".mp3")
            if os.path.exists(path) and manifest.get(stem) == text_hash(text):
                continue
            print(f"{stem}: {text[:70]}...")
            await edge_tts.Communicate(text, args.voice, rate=args.rate, pitch=args.pitch).save(path)
            manifest[stem] = text_hash(text)
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("Озвучка готова. Теперь: python3 render.py")


if __name__ == "__main__":
    asyncio.run(main())
