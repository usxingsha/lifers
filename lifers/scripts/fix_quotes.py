#!/usr/bin/env python3
"""Fix Chinese double quotes in gen_industrial.py"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('gen_industrial.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Chinese double quotes that cause syntax errors
left_quote = '“'   # "
right_quote = '”'  # "
content = content.replace(left_quote, '「')   # " -> 「
content = content.replace(right_quote, '」')  # " -> 」

with open('gen_industrial.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
compile(content, 'gen_industrial.py', 'exec')
print('OK - Chinese quotes fixed')
