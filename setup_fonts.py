"""ML-Lab 字体安装脚本 — 在 Docker 构建时运行

策略：
1. 优先从 Noto GitHub Releases 下载 NotoSansSC OTF（最可靠）
2. 如果下载失败，尝试从系统 TTC 提取 SC 子字体
3. 如果都失败，安装 fonts-wqy-microhei 作为兜底
"""
import os, sys, subprocess, urllib.request, glob
from pathlib import Path

FONT_DIR = Path('/usr/share/fonts/opentype/noto')
FONT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f'[setup_fonts] {msg}', flush=True)

def run(cmd, **kw):
    log(f'Running: {cmd}')
    return subprocess.run(cmd, shell=True, **kw, capture_output=True, text=True)

downloaded = False

# ── 策略 1: 从 GitHub 下载 Noto Sans SC OTF ──
NOTO_SC_URL = 'https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJKsc.zip'

log(f'Trying to download from {NOTO_SC_URL}')
try:
    zip_path = '/tmp/noto_sc.zip'
    urllib.request.urlretrieve(NOTO_SC_URL, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('.otf') and 'Regular' in name:
                zf.extract(name, '/tmp/noto_extracted')
                src = Path('/tmp/noto_extracted') / name
                dst = FONT_DIR / 'NotoSansCJKSC-Regular.otf'
                src.rename(dst)
                log(f'Extracted {name} -> {dst}')
                downloaded = True
                break
        for name in zf.namelist():
            if name.endswith('.otf') and 'Bold' in name:
                zf.extract(name, '/tmp/noto_extracted')
                src = Path('/tmp/noto_extracted') / name
                dst = FONT_DIR / 'NotoSansCJKSC-Bold.otf'
                src.rename(dst)
                log(f'Extracted {name} -> {dst}')
                break
except Exception as e:
    log(f'Download failed: {e}')

# ── 策略 2: 从系统 TTC 提取 ──
if not downloaded:
    log('Download failed, trying TTC extraction...')
    try:
        from fontTools.ttLib import TTCollection
        ttc_candidates = glob.glob('/usr/share/fonts/**/NotoSansCJK*.ttc', recursive=True)
        if not ttc_candidates:
            ttc_candidates = glob.glob('/usr/share/fonts/**/*.ttc', recursive=True)
        log(f'TTC candidates: {ttc_candidates}')

        for ttc_path in ttc_candidates:
            if 'Noto' not in ttc_path and 'CJK' not in ttc_path:
                continue
            log(f'Processing {ttc_path}')
            try:
                ttc = TTCollection(ttc_path)
                log(f'  Subfonts: {len(ttc.fonts)}')
                for i, font in enumerate(ttc.fonts):
                    name = font.name_table.getDebugName(1) or 'unknown'
                    name4 = font.name_table.getDebugName(4) or 'unknown'
                    log(f'  [{i}] name1={name}, name4={name4}')

                sc_idx = None
                bold_sc_idx = None
                for i, font in enumerate(ttc.fonts):
                    n = (font.name_table.getDebugName(1) or '').lower()
                    n4 = (font.name_table.getDebugName(4) or '').lower()
                    if 'sc' in n or 'sc' in n4:
                        if 'bold' in n or 'bold' in n4:
                            bold_sc_idx = i
                        else:
                            sc_idx = i

                if sc_idx is not None:
                    ttc.fonts[sc_idx].save(str(FONT_DIR / 'NotoSansCJKSC-Regular.otf'))
                    log(f'Saved SC Regular from index {sc_idx}')
                    downloaded = True
                if bold_sc_idx is not None:
                    ttc.fonts[bold_sc_idx].save(str(FONT_DIR / 'NotoSansCJKSC-Bold.otf'))
                    log(f'Saved SC Bold from index {bold_sc_idx}')
            except Exception as e:
                log(f'Error processing {ttc_path}: {e}')
    except ImportError:
        log('fonttools not available, skipping TTC extraction')

# ── 策略 3: 安装 wqy-microhei 作为兜底 ──
if not downloaded:
    log('TTC extraction failed, installing wqy-microhei as fallback...')
    run('apt-get install -y --no-install-recommends fonts-wqy-microhei')

# ── 验证 ──
log('=== Font verification ===')
result = run('fc-list :lang=zh 2>/dev/null | head -20')
log(result.stdout)

log('=== OTF files ===')
for f in FONT_DIR.glob('*.otf'):
    log(f'  {f} ({f.stat().st_size} bytes)')

log('Setup complete!')
