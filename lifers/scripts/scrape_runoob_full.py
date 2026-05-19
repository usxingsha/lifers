#!/usr/bin/env python3
"""全面爬取 runoob.com 教程 - 并发下载，提取正文，追加到语料库"""
import requests
import re
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, NavigableString

# Set UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"
BASE_URL = "https://www.runoob.com"

# 要爬取的教程列表 (名称, URL路径, 估计页数)
TUTORIALS = [
    # ── 前端 ──
    ("HTML", "/html/html-tutorial.html", 30),
    ("CSS", "/css/css-tutorial.html", 30),
    ("CSS3", "/css3/css3-tutorial.html", 20),
    ("JavaScript", "/js/js-tutorial.html", 40),
    ("TypeScript", "/typescript/ts-tutorial.html", 20),
    ("jQuery", "/jquery/jquery-tutorial.html", 20),
    ("Bootstrap4", "/bootstrap4/bootstrap4-tutorial.html", 20),
    ("Bootstrap5", "/bootstrap5/bootstrap5-tutorial.html", 20),
    ("React", "/react/react-tutorial.html", 25),
    ("Angular", "/angularjs/angularjs-tutorial.html", 15),
    ("Node.js", "/nodejs/nodejs-tutorial.html", 25),
    ("Sass", "/sass/sass-tutorial.html", 15),
    ("Less", "/less/less-tutorial.html", 10),

    # ── 后端 ──
    ("Python3", "/python3/python3-tutorial.html", 30),
    ("Java", "/java/java-tutorial.html", 30),
    ("C", "/cprogramming/c-tutorial.html", 20),
    ("C++", "/cplusplus/cpp-tutorial.html", 30),
    ("C#", "/csharp/csharp-tutorial.html", 20),
    ("Go", "/go/go-tutorial.html", 20),
    ("Rust", "/rust/rust-tutorial.html", 15),
    ("Swift", "/swift/swift-tutorial.html", 20),
    ("PHP", "/php/php-tutorial.html", 25),
    ("Ruby", "/ruby/ruby-tutorial.html", 15),
    ("Scala", "/scala/scala-tutorial.html", 15),
    ("Kotlin", "/kotlin/kotlin-tutorial.html", 15),
    ("R", "/r/r-tutorial.html", 15),
    ("Lua", "/lua/lua-tutorial.html", 15),

    # ── 数据库 ──
    ("MySQL", "/mysql/mysql-tutorial.html", 20),
    ("MongoDB", "/mongodb/mongodb-tutorial.html", 15),
    ("Redis", "/redis/redis-tutorial.html", 15),
    ("PostgreSQL", "/postgresql/postgresql-tutorial.html", 15),
    ("SQLite", "/sqlite/sqlite-tutorial.html", 15),

    # ── 运维/工具 ──
    ("Linux", "/linux/linux-tutorial.html", 25),
    ("Docker", "/docker/docker-tutorial.html", 20),
    ("Git", "/git/git-tutorial.html", 15),
    ("Nginx", "/nginx/nginx-tutorial.html", 10),
    ("正则表达式", "/regexp/regexp-tutorial.html", 10),

    # ── 数据科学/AI ──
    ("NumPy", "/numpy/numpy-tutorial.html", 15),
    ("Pandas", "/pandas/pandas-tutorial.html", 15),
    ("Matplotlib", "/matplotlib/matplotlib-tutorial.html", 15),
    ("SciPy", "/scipy/scipy-tutorial.html", 10),
    ("Django", "/django/django-tutorial.html", 15),
    ("Flask", "/flask/flask-tutorial.html", 10),
    ("FastAPI", "/fastapi/fastapi-tutorial.html", 10),
    ("Scrapy", "/scrapy/scrapy-tutorial.html", 10),

    # ── 基础 ──
    ("算法", "/algorithm/algorithm-tutorial.html", 15),
    ("数据结构", "/data-structures/ds-tutorial.html", 15),
    ("设计模式", "/design-pattern/design-pattern-tutorial.html", 10),

    # ── 基础教程(新) ──
    ("Python基础", "/python/python-tutorial.html", 15),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

def get_tutorial_pages(tutorial_path):
    """获取教程所有子页面URL"""
    url = BASE_URL + tutorial_path
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 获取教程基础目录
        base_dir = tutorial_path.rsplit('/', 1)[0]  # e.g. /python3

        pages = []
        seen = set()
        # 查找侧边栏的教程链接
        sidebar = soup.find('div', class_='sidebar') or soup.find('div', id='leftcolumn')
        if sidebar:
            for a in sidebar.find_all('a', href=True):
                href = a['href'].strip()
                text = a.get_text(strip=True)
                if not text or len(text) < 2:
                    continue
                # 跳过 javascript: 和 #
                if href.startswith('javascript:') or href == '#':
                    continue
                # 构建完整URL
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('//'):
                    full_url = 'https:' + href
                elif href.startswith('/'):
                    full_url = BASE_URL + href
                else:
                    full_url = BASE_URL + base_dir + '/' + href

                if full_url not in seen and 'runoob.com' in full_url:
                    seen.add(full_url)
                    pages.append((full_url, text))

        return pages
    except Exception as e:
        print(f"  [错误] 获取页面列表失败: {e}")
        return []

def extract_content(html):
    """从HTML提取正文内容"""
    soup = BeautifulSoup(html, 'html.parser')

    # 找到主要内容区域
    content = soup.find('div', class_='article-body') or \
              soup.find('div', id='content') or \
              soup.find('div', class_='article') or \
              soup.find('article')

    if not content:
        return ""

    # 移除不需要的元素
    for tag in content.find_all(['script', 'style', 'ins', 'iframe', 'nav']):
        tag.decompose()

    # 提取文本
    lines = []
    for elem in content.descendants:
        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            if text and len(text) > 5:
                lines.append(text)
            continue

        name = elem.name.lower() if elem.name else ''
        if name in ('h1', 'h2', 'h3', 'h4'):
            text = elem.get_text(strip=True)
            if text:
                lines.append(f"\n## {text}")
        elif name == 'p':
            text = elem.get_text(strip=True)
            if text and len(text) > 10:
                lines.append(text)
        elif name == 'pre':
            code = elem.get_text()
            if code.strip():
                lines.append(f"\n```\n{code.strip()}\n```\n")
        elif name == 'li':
            text = elem.get_text(strip=True)
            if text and len(text) > 3:
                lines.append(f"- {text}")

    return '\n'.join(lines)

def scrape_tutorial(name, path, estimated_pages):
    """爬取单个教程"""
    print(f"\n{'='*60}")
    print(f"爬取: {name} ({estimated_pages} 页预估)")
    print(f"{'='*60}")

    pages = get_tutorial_pages(path)
    if not pages:
        print(f"  [跳过] 未找到子页面")
        return None

    print(f"  找到 {len(pages)} 个子页面")

    content_blocks = [f"\n\n# {name} 教程\n"]
    chars = 0
    scraped = 0

    for url, title in pages[:60]:  # 每个教程最多60页
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.encoding = 'utf-8'

            if resp.status_code != 200:
                continue

            text = extract_content(resp.text)
            if text and len(text) > 100:
                content_blocks.append(f"\n## {title}\n")
                content_blocks.append(text)
                chars += len(text)
                scraped += 1
                print(f"    [{scraped}] {title} ({len(text)}字符)")

            time.sleep(0.3)  # 礼貌延迟

        except Exception as e:
            continue

    if not content_blocks or scraped == 0:
        print(f"  [失败] 未提取到内容")
        return None

    result = '\n'.join(content_blocks)
    print(f"  [完成] {scraped}页, {chars:,}字符")
    return result

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Runoob 全站教程爬虫               ║")
    print(f"║   目标: {len(TUTORIALS)} 个教程系列        ║")
    print("╚══════════════════════════════════════╝")

    # 读取现有语料
    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
        print(f"现有语料: {len(existing):,} 字符, {existing.count(chr(10)):,} 行")

    total_new_chars = 0
    new_content_blocks = [f"\n\n{'='*60}\n# Runoob 教程合集 (第二批)\n{'='*60}\n"]

    for name, path, est_pages in TUTORIALS:
        result = scrape_tutorial(name, path, est_pages)
        if result:
            new_content_blocks.append(result)
            total_new_chars += len(result)
        time.sleep(1)  # 教程间隔延迟

    if not new_content_blocks:
        print("\n[失败] 未能爬取到任何新内容")
        return

    # 追加到语料库
    new_text = '\n'.join(new_content_blocks)
    combined = existing + new_text

    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\n{'='*60}")
    print(f"爬取完成!")
    print(f"  新增字符: {total_new_chars:,}")
    print(f"  语料库总大小: {len(combined):,} 字符 ({len(combined)/1024/1024:.1f}MB)")
    print(f"  语料库总行数: {combined.count(chr(10)):,} 行")
    print(f"  保存路径: {CORPUS_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
