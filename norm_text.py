#!/usr/bin/env python3
"""
پاک‌سازی فایل result.txt:
- حذف خطوط خالی
- نرمال‌سازی (حذف کاراکترهای نامرئی، یکسان‌سازی حروف عربی/فارسی، حذف اعراب)
- حذف خطوط تکراری (بر اساس متن نرمال‌شده)
- ذخیره در فایل CSV برای استفاده در آموزش مدل
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path

# ------------------- تابع نرمال‌سازی -------------------
def normalize_text(text: str) -> str:
    # 1. حذف کاراکترهای نامرئی و فاصله‌های صفرعرض
    invisible = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060-\u2069\uFEFF\u00AD]')
    text = invisible.sub('', text)
    # 2. حذف اعراب عربی (فتحه، کسره، ضمه و …)
    text = re.sub(r'[\u064b-\u0652\u0670]', '', text)
    # 3. یکسان‌سازی حروف عربی به فارسی
    arabic_to_persian = str.maketrans({
        'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'ؤ': 'و',
        'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ى': 'ی', 'ٱ': 'ا'
    })
    text = text.translate(arabic_to_persian)
    # 4. نرمال‌سازی یونیکد (NFKC)
    text = unicodedata.normalize('NFKC', text)
    return text

# ------------------- بارگذاری و پاکسازی -------------------
input_file = Path('output.txt')   # فایل تبلیغات جمع‌آوری‌شده
output_file = Path('ads_negative_clean.csv')

# خواندن خطوط
with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
    lines = [line.strip() for line in f if line.strip()]  # حذف خطوط خالی

# تبدیل به DataFrame
df = pd.DataFrame(lines, columns=['text'])

# اعمال نرمال‌سازی
df['clean_text'] = df['text'].apply(normalize_text)

# حذف تکراری‌ها بر اساس متن نرمال‌شده
df = df.drop_duplicates(subset='clean_text').reset_index(drop=True)

# اضافه کردن برچسب (همه تبلیغ هستند)
df['label'] = 0

# ذخیره در فایل CSV (بدون ستون text اصلی، برای هماهنگی با داده‌های منفی)
df[['clean_text', 'label']].to_csv(output_file, index=False, encoding='utf-8')

print(f'تعداد خطوط اولیه: {len(lines)}')
print(f'تعداد نمونه‌های یکتا (مثبت): {len(df)}')
print(f'فایل تمیز در {output_file} ذخیره شد.')