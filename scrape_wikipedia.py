#!/usr/bin/env python3
"""维基百科大规模爬虫 — 中英文高质量语料"""
import requests, sys, os, time, random, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

sys.stdout.reconfigure(encoding='utf-8')
CORPUS_PATH = r"lifers\weights\training_corpus.txt"
session = requests.Session()
session.headers.update({
    "User-Agent": "LifersBot/2.0 (https://github.com/lifers; lifers@example.com)",
})

# 中英文维基百科API
ZH_API = "https://zh.wikipedia.org/w/api.php"
EN_API = "https://en.wikipedia.org/w/api.php"

# 要抓取的文章分类 - 中文
ZH_CATEGORIES = [
    "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "网络安全", "密码学", "数据库", "操作系统", "计算机网络",
    "数据结构", "算法", "程序设计", "软件工程", "计算机科学",
    "数学", "物理学", "化学", "生物学", "天文学",
    "地球科学", "医学", "心理学", "哲学", "经济学",
    "历史", "地理", "语言学", "社会学", "政治学",
    "文学", "艺术", "音乐", "电影", "建筑",
    "机器人学", "自动化", "电子工程", "通信技术", "能源",
    "环境科学", "材料科学", "航空航天", "量子力学", "相对论",
    "统计学", "概率论", "线性代数", "微积分", "图论",
    "中国历史", "世界历史", "中国古代文学", "世界文学", "中华文化",
    "机器学习算法", "深度学习框架", "编程语言", "Web开发", "移动开发",
]

# 英文
EN_CATEGORIES = [
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Natural language processing", "Computer vision", "Robotics",
    "Cybersecurity", "Cryptography", "Database", "Operating system",
    "Computer network", "Data structure", "Algorithm", "Software engineering",
    "Mathematics", "Physics", "Chemistry", "Biology", "Astronomy",
    "Medicine", "Psychology", "Philosophy", "Economics", "History",
    "Neuroscience", "Quantum computing", "Blockchain", "Internet of things",
    "Cloud computing", "Big data", "Statistics", "Probability theory",
    "Linear algebra", "Calculus", "Graph theory", "Game theory",
    "Reinforcement learning", "Computer programming", "Web development",
    "Electrical engineering", "Mechanical engineering", "Civil engineering",
]

def fetch_wiki_articles(api_url, category, lang, limit=50):
    """通过维基API获取分类相关文章"""
    articles = []
    try:
        params = {
            "action": "query", "format": "json",
            "list": "search", "srsearch": category,
            "srlimit": limit, "srprop": "snippet",
        }
        resp = session.get(api_url, params=params, timeout=30)
        data = resp.json()
        pages = data.get("query", {}).get("search", [])
        for p in pages:
            articles.append((p["title"], p["pageid"]))
    except Exception as e:
        print(f"  [{lang}] 搜索'{category}'失败: {e}")
    return articles

def fetch_article_content(api_url, pageid, lang):
    """获取文章完整内容"""
    try:
        params = {
            "action": "query", "format": "json",
            "prop": "extracts", "exintro": 0,
            "explaintext": 1, "pageids": pageid,
            "exsectionformat": "wiki",
        }
        resp = session.get(api_url, params=params, timeout=30)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        page = pages.get(str(pageid), {})
        return page.get("extract", ""), page.get("title", "")
    except Exception as e:
        return "", ""

def clean_wiki_text(text):
    """清理维基文本"""
    # Remove reference numbers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.I)
    # Remove excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Remove section numbering artifacts
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def scrape_wikipedia(api_url, categories, lang, max_per_cat=40):
    """抓取维基百科文章"""
    lang_name = "中文" if "zh" in api_url else "英文"
    all_content = []
    total_chars = 0
    article_count = 0

    print(f"\n{'='*50}")
    print(f"抓取{lang_name}维基百科: {len(categories)} 个分类")
    print(f"{'='*50}")

    for cat in categories:
        articles = fetch_wiki_articles(api_url, cat, lang, limit=max_per_cat)
        if not articles:
            continue

        cat_content = []
        for title, pageid in articles[:30]:
            text, real_title = fetch_article_content(api_url, pageid, lang)
            if text and len(text) > 500:
                cleaned = clean_wiki_text(text)
                cat_content.append(f"\n## {real_title}\n\n{cleaned}\n")
                total_chars += len(cleaned)
                article_count += 1

            time.sleep(0.2)  # Rate limiting

        if cat_content:
            all_content.append(f"\n# {lang_name}维基百科: {cat}\n" + '\n'.join(cat_content))
            print(f"  {cat}: {len(cat_content)}篇, {sum(len(c) for c in cat_content):,}字符")

        time.sleep(0.5)

    result = '\n'.join(all_content)
    print(f"{lang_name}完成: {article_count}篇文章, {total_chars:,}字符")
    return result, total_chars

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   维基百科大规模语料爬虫                ║")
    print(f"║   中英文各{len(ZH_CATEGORIES)}个分类                ║")
    print("╚══════════════════════════════════════════╝")

    # 读取现有语料
    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
    print(f"现有语料: {len(existing)/1024/1024:.1f}MB ({len(existing):,}字符)")

    all_new = []
    total_new = 0

    # 抓取中文维基
    zh_text, zh_chars = scrape_wikipedia(ZH_API, ZH_CATEGORIES, "zh", max_per_cat=50)
    all_new.append(zh_text)
    total_new += zh_chars

    # 抓取英文维基
    en_text, en_chars = scrape_wikipedia(EN_API, EN_CATEGORIES, "en", max_per_cat=50)
    all_new.append(en_text)
    total_new += en_chars

    # 保存
    new_combined = "\n\n" + "="*60 + "\n"
    new_combined += "# 维基百科知识语料 (中英文)\n"
    new_combined += "="*60 + "\n\n"
    new_combined += '\n\n'.join(all_new)

    # 保存到独立文件避免覆盖主语料
    wiki_output_path = r"lifers\weights\wikipedia_corpus.txt"
    with open(wiki_output_path, 'w', encoding='utf-8') as f:
        f.write(new_combined)
    print(f"Wikipedia内容已保存到: {wiki_output_path}")

    print(f"\n{'='*60}")
    print(f"Wikipedia抓取完成!")
    print(f"  新增: {total_new:,}字符 ({total_new/1024/1024:.1f}MB)")
    print(f"  总量: {len(final):,}字符 ({len(final)/1024/1024:.1f}MB)")
    print(f"  行数: {final.count(chr(10)):,}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
