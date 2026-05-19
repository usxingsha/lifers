#!/usr/bin/env python3
"""快速并发爬取 runoob.com 教程 - 多线程下载，追加到语料库"""
import requests
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, NavigableString

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"
BASE_URL = "https://www.runoob.com"

# 要爬取的教程列表 (名称, URL路径)
TUTORIALS = [
    ("HTML", "/html/html-tutorial.html"),
    ("CSS", "/css/css-tutorial.html"),
    ("CSS3", "/css3/css3-tutorial.html"),
    ("JavaScript", "/js/js-tutorial.html"),
    ("TypeScript", "/typescript/ts-tutorial.html"),
    ("jQuery", "/jquery/jquery-tutorial.html"),
    ("Bootstrap4", "/bootstrap4/bootstrap4-tutorial.html"),
    ("Bootstrap5", "/bootstrap5/bootstrap5-tutorial.html"),
    ("React", "/react/react-tutorial.html"),
    ("Node.js", "/nodejs/nodejs-tutorial.html"),
    ("Vue3", "/vue3/vue3-tutorial.html"),
    ("Python3", "/python3/python3-tutorial.html"),
    ("Java", "/java/java-tutorial.html"),
    ("C语言", "/cprogramming/c-tutorial.html"),
    ("C++", "/cplusplus/cpp-tutorial.html"),
    ("C#", "/csharp/csharp-tutorial.html"),
    ("Go", "/go/go-tutorial.html"),
    ("Rust", "/rust/rust-tutorial.html"),
    ("Swift", "/swift/swift-tutorial.html"),
    ("PHP", "/php/php-tutorial.html"),
    ("Kotlin", "/kotlin/kotlin-tutorial.html"),
    ("Lua", "/lua/lua-tutorial.html"),
    ("MySQL", "/mysql/mysql-tutorial.html"),
    ("MongoDB", "/mongodb/mongodb-tutorial.html"),
    ("Redis", "/redis/redis-tutorial.html"),
    ("PostgreSQL", "/postgresql/postgresql-tutorial.html"),
    ("SQLite", "/sqlite/sqlite-tutorial.html"),
    ("Linux", "/linux/linux-tutorial.html"),
    ("Docker", "/docker/docker-tutorial.html"),
    ("Git", "/git/git-tutorial.html"),
    ("NumPy", "/numpy/numpy-tutorial.html"),
    ("Pandas", "/pandas/pandas-tutorial.html"),
    ("Matplotlib", "/matplotlib/matplotlib-tutorial.html"),
    ("Django", "/django/django-tutorial.html"),
    ("Flask", "/flask/flask-tutorial.html"),
    ("数据结构", "/data-structures/ds-tutorial.html"),
    ("设计模式", "/design-pattern/design-pattern-tutorial.html"),
    ("正则表达式", "/regexp/regexp-tutorial.html"),
    ("Nginx", "/nginx/nginx-tutorial.html"),
    ("算法", "/algorithm/algorithm-tutorial.html"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)

def get_pages(tutorial_path):
    """获取教程所有子页面URL"""
    url = BASE_URL + tutorial_path
    try:
        resp = session.get(url, timeout=30)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        base_dir = tutorial_path.rsplit('/', 1)[0]
        pages = []
        seen = set()
        sidebar = soup.find('div', class_='sidebar') or soup.find('div', id='leftcolumn')
        if not sidebar:
            return pages
        for a in sidebar.find_all('a', href=True):
            href = a['href'].strip()
            text = a.get_text(strip=True)
            if not text or len(text) < 2:
                continue
            if href.startswith('javascript:') or href == '#':
                continue
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
        print(f"  [错误] 获取页面列表: {e}")
        return []

def extract_content(html):
    """从HTML提取正文"""
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='article-body') or \
              soup.find('div', id='content') or \
              soup.find('article')
    if not content:
        return ""
    for tag in content.find_all(['script', 'style', 'ins', 'iframe', 'nav']):
        tag.decompose()
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

def scrape_page(args):
    """爬取单个页面 (并发用)"""
    url, title = args
    try:
        resp = session.get(url, timeout=20)
        resp.encoding = 'utf-8'
        if resp.status_code != 200:
            return None
        text = extract_content(resp.text)
        if text and len(text) > 100:
            return (title, text, len(text))
        return None
    except Exception:
        return None

def scrape_tutorial(name, path):
    """爬取单个教程系列"""
    print(f"\n{'='*50}")
    print(f"爬取: {name}")
    print(f"{'='*50}")

    pages = get_pages(path)
    if not pages:
        print(f"  未找到子页面")
        return None

    # 限制每个教程最多40页
    pages = pages[:40]
    print(f"  {len(pages)} 个页面，并发下载中...")

    results = []
    chars = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scrape_page, p): p for p in pages}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                title, text, length = result
                results.append((title, text))
                chars += length
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(pages)}")

    if not results:
        print(f"  未提取到内容")
        return None

    # 组装内容
    blocks = [f"\n\n# {name} 教程\n"]
    for title, text in results:
        blocks.append(f"\n## {title}\n")
        blocks.append(text)
        blocks.append("")

    result = '\n'.join(blocks)
    print(f"  完成: {len(results)}页, {chars:,}字符")
    return result

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Runoob 全站教程快速爬虫            ║")
    print(f"║   目标: {len(TUTORIALS)} 个教程系列              ║")
    print("╚══════════════════════════════════════╝")

    # 读取现有语料
    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
    print(f"现有语料: {len(existing)/1024/1024:.1f}MB, {existing.count(chr(10)):,}行")

    all_new = []
    total_chars = 0
    success = 0

    for name, path in TUTORIALS:
        result = scrape_tutorial(name, path)
        if result:
            all_new.append(result)
            total_chars += len(result)
            success += 1
        time.sleep(0.5)

    if not all_new:
        print("\n[失败] 未爬取到任何内容")
        return

    # 追加到语料库
    new_text = '\n'.join(all_new)
    combined = existing + new_text

    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        f.write(combined)

    # 统计
    new_mb = len(new_text) / 1024 / 1024
    total_mb = len(combined) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"🎉 爬取完成!")
    print(f"  成功: {success}/{len(TUTORIALS)} 个教程")
    print(f"  新增: {total_chars:,} 字符 ({new_mb:.1f}MB)")
    print(f"  总量: {len(combined):,} 字符 ({total_mb:.1f}MB)")
    print(f"  行数: {combined.count(chr(10)):,}")
    print(f"  保存: {CORPUS_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
