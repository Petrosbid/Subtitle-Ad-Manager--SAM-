import pandas as pd

INPUT_POS  = "ads_positive_clean.csv"   # ممکن است تغییر نام داده باشید
INPUT_NEG  = "ads_negative_clean_filtered.csv"
OUTPUT_POS = "ads_positive_clean_final.csv"
OUTPUT_NEG = "ads_negative_clean_final.csv"

MIN_LEN = 15   # حداقل تعداد کاراکتر (پس از نرمال‌سازی)

def filter_short_lines(df, min_len):
    mask = df['clean_text'].str.len() >= min_len
    return df[mask]

# پردازش مثبت
df_pos = pd.read_csv(INPUT_POS)
print(f"Pos before: {len(df_pos)}")
df_pos = filter_short_lines(df_pos, MIN_LEN)
df_pos.to_csv(OUTPUT_POS, index=False, encoding='utf-8')
print(f"Pos after : {len(df_pos)}")

# پردازش منفی
df_neg = pd.read_csv(INPUT_NEG)
print(f"Neg before: {len(df_neg)}")
df_neg = filter_short_lines(df_neg, MIN_LEN)
df_neg.to_csv(OUTPUT_NEG, index=False, encoding='utf-8')
print(f"Neg after : {len(df_neg)}")