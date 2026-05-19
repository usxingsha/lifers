#!/usr/bin/env python3
"""扩充弱支柱语料到100MB+ — 语音/KG/群体/仿真/遥测/仪表盘"""
import sys, os, random, time, importlib.util

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"

# Import domain data from gen_pillar_extra (sibling script)
_extra_path = Path(__file__).resolve().parent / "gen_pillar_extra.py"
spec = importlib.util.spec_from_file_location("gen_pillar_extra", str(_extra_path))
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)

ZH = extra.ZH_EXTRA_DOMAINS
EN = extra.EN_EXTRA_DOMAINS
TEMPLATES_ZH = extra.TEMPLATES_ZH
TEMPLATES_EN = extra.TEMPLATES_EN
INSIGHTS_ZH = extra.INSIGHTS_ZH
INSIGHTS_EN = extra.INSIGHTS_EN

# Target only the 6 weak domains
WEAK_ZH = {k: v for k, v in ZH.items() if k not in ['计算机视觉与感知']}
WEAK_EN = {k: v for k, v in EN.items() if k not in ['Computer Vision & Visual Intelligence', 'Reinforcement Learning & Decision Intelligence']}

def generate_entries(domains_dict, templates, insights, repeat_count):
    all_entries = []
    total_chars = 0
    for domain, topics in domains_dict.items():
        all_entries.append(f'\n\n# {domain}\n')
        for topic_name, topic_body in topics:
            for _ in range(repeat_count):
                tpl = random.choice(templates)
                ins = random.choice(insights)
                entry = tpl.format(topic=topic_name, body=topic_body, insight=ins)
                all_entries.append(entry)
                total_chars += len(entry)
        print(f'  {domain}: {len(topics)} topics x {repeat_count}')
    return '\n'.join(all_entries), total_chars

def main():
    print('=' * 60)
    print('  弱支柱语料扩充 — 目标: 每领域>=100MB')
    print('  语音/KG/群体/仿真/遥测/仪表盘')
    print('=' * 60)

    existing_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
    print(f'现有语料: {existing_mb:.0f}MB\n')

    # Write header
    header = '\n\n' + '=' * 60 + '\n# 弱支柱扩充语料 (Boost: 语音/KG/群体/仿真/遥测/仪表盘)\n' + '=' * 60 + '\n'
    with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
        f.write(header)

    total_new = 0
    seeds_zh = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177]
    seeds_en = [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178]

    # Chinese boost — 18 seeds x 1200 repeats
    print('[中文扩充] 18 seeds x 1200 repeats')
    for i, seed in enumerate(seeds_zh):
        random.seed(seed)
        zh_text, zh_chars = generate_entries(WEAK_ZH, TEMPLATES_ZH, INSIGHTS_ZH, 1200)
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            f.write(zh_text)
        total_new += zh_chars
        cur_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
        print(f'  Seed {seed}: +{zh_chars/1024/1024:.1f}MB => {cur_mb:.0f}MB')

    # English boost — 18 seeds x 1200 repeats
    print('\n[English 扩充] 18 seeds x 1200 repeats')
    for i, seed in enumerate(seeds_en):
        random.seed(seed)
        en_text, en_chars = generate_entries(WEAK_EN, TEMPLATES_EN, INSIGHTS_EN, 1200)
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            f.write(en_text)
        total_new += en_chars
        cur_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
        print(f'  Seed {seed}: +{en_chars/1024/1024:.1f}MB => {cur_mb:.0f}MB')

    final_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
    print(f'\n===== Boost 完成 =====')
    print(f'新增: {total_new/1024/1024:.1f}MB')
    print(f'总量: {final_mb:.0f}MB')

if __name__ == '__main__':
    main()
