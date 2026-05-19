#!/usr/bin/env python3
"""Download public domain books/texts — Project Gutenberg (EN) + Chinese classics"""
import requests, sys, os, re, time

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# Project Gutenberg: Top English books (public domain)
GUTENBERG_BOOKS = [
    (1342, "Pride and Prejudice by Jane Austen"),
    (84, "Frankenstein by Mary Shelley"),
    (11, "Alice's Adventures in Wonderland by Lewis Carroll"),
    (1661, "The Adventures of Sherlock Holmes by Arthur Conan Doyle"),
    (2701, "Moby Dick by Herman Melville"),
    (345, "Dracula by Bram Stoker"),
    (174, "The Picture of Dorian Gray by Oscar Wilde"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde by Robert Louis Stevenson"),
    (98, "A Tale of Two Cities by Charles Dickens"),
    (1400, "Great Expectations by Charles Dickens"),
    (2600, "War and Peace by Leo Tolstoy"),
    (4300, "Ulysses by James Joyce"),
    (1184, "The Count of Monte Cristo by Alexandre Dumas"),
    (1260, "Jane Eyre by Charlotte Bronte"),
    (768, "Wuthering Heights by Emily Bronte"),
    (2542, "A Doll's House by Henrik Ibsen"),
    (5200, "The Metamorphosis by Franz Kafka"),
    (408, "The Souls of Black Folk by W.E.B. Du Bois"),
    (20203, "Autobiography of Benjamin Franklin"),
    (28054, "The Brothers Karamazov by Fyodor Dostoyevsky"),
    (244, "A Study in Scarlet by Arthur Conan Doyle"),
    (2554, "Crime and Punishment by Fyodor Dostoyevsky"),
    (30254, "The Republic by Plato"),
    (1232, "The Prince by Niccolo Machiavelli"),
    (6130, "The Iliad by Homer"),
    (1727, "The Odyssey by Homer"),
    (3300, "An Inquiry into the Nature and Causes of the Wealth of Nations by Adam Smith"),
    (3600, "Essays of Michel de Montaigne"),
    (7370, "The Second Treatise of Civil Government by John Locke"),
    (3825, "Democracy in America by Alexis de Tocqueville"),
]

# Project Gutenberg: Science/Tech books
GUTENBERG_SCIENCE = [
    (5001, "On the Origin of Species by Charles Darwin"),
    (1228, "The Origin of Species by Charles Darwin"),
    (58585, "Relativity: The Special and General Theory by Albert Einstein"),
    (20417, "The Interpretation of Dreams by Sigmund Freud"),
    (3207, "Principia by Isaac Newton (excerpts)"),
    (15776, "A Brief History of the Internet"),
    (50500, "The Art of War by Sun Tzu (English translation)"),
    (132, "The Art of War by Sun Tzu"),
]


def download_gutenberg(book_id, title):
    """Download a book from Project Gutenberg"""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code != 200:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            resp = session.get(url, timeout=60)

        if resp.status_code == 200:
            text = resp.text
            # Strip header/footer
            start_markers = [
                "*** START OF THE PROJECT GUTENBERG EBOOK",
                "*** START OF THIS PROJECT GUTENBERG EBOOK",
                "***START OF THE PROJECT GUTENBERG EBOOK",
            ]
            end_markers = [
                "*** END OF THE PROJECT GUTENBERG EBOOK",
                "*** END OF THIS PROJECT GUTENBERG EBOOK",
                "***END OF THE PROJECT GUTENBERG EBOOK",
            ]

            start_pos = 0
            for marker in start_markers:
                pos = text.find(marker)
                if pos >= 0:
                    start_pos = text.find('\n', pos) + 1
                    break

            end_pos = len(text)
            for marker in end_markers:
                pos = text.find(marker)
                if pos >= 0:
                    end_pos = pos
                    break

            cleaned = text[start_pos:end_pos].strip()
            if len(cleaned) > 10000:
                return f"\n## {title}\n\n{cleaned[:200000]}\n", len(cleaned[:200000])
            return f"\n## {title}\n\n{cleaned}\n", len(cleaned)
    except Exception as e:
        print(f"  Error downloading {title}: {e}")
    return "", 0

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   Public Domain Books Downloader         ║")
    print(f"║   {len(GUTENBERG_BOOKS)} books + {len(GUTENBERG_SCIENCE)} science texts  ║")
    print("╚══════════════════════════════════════════╝")

    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
    print(f"Existing corpus: {len(existing)/1024/1024:.1f}MB")

    all_books = GUTENBERG_BOOKS + GUTENBERG_SCIENCE
    book_texts = []
    total_chars = 0
    success = 0

    for book_id, title in all_books:
        print(f"  Downloading: {title[:60]}...")
        text, chars = download_gutenberg(book_id, title)
        if text:
            book_texts.append(text)
            total_chars += chars
            success += 1
            print(f"    OK: {chars:,} chars")
        time.sleep(0.5)

    if book_texts:
        header = "\n\n" + "="*60 + "\n# Public Domain Literature & Science Corpus\n" + "="*60 + "\n"
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            f.write(header + '\n'.join(book_texts))

    print(f"\nDownloaded: {success}/{len(all_books)} books")
    print(f"New content: {total_chars:,} chars ({total_chars/1024/1024:.1f}MB)")

if __name__ == "__main__":
    main()
