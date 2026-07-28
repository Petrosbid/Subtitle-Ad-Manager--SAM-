import os


def merge_txt_files(directory_path, output_file_path):
    """
    تمام فایل‌های txt موجود در دایرکتوری داده شده را در یک فایل ادغام می‌کند.

    پارامترها:
        directory_path (str): مسیر دایرکتوری حاوی فایل‌های txt.
        output_file_path (str): مسیر فایل خروجی که محتوا در آن ذخیره می‌شود.
    """
    # دریافت لیست تمام فایل‌های txt (ترتیب اهمیتی ندارد)
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

    if not txt_files:
        print("هیچ فایل txt ای در دایرکتوری پیدا نشد.")
        return

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for filename in txt_files:
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # خواندن محتوای فایل و نوشتن آن در فایل خروجی
                    outfile.write(infile.read())
                    # اضافه کردن یک خط خالی بین محتواهای فایل‌ها (اختیاری)
                    outfile.write("\n")
                print(f"اضافه شد: {filename}")
            except Exception as e:
                print(f"خطا در خواندن {filename}: {e}")

    print(f"تمامی فایل‌ها در {output_file_path} ادغام شدند.")


# مثال استفاده
if __name__ == "__main__":
    # مسیر دایرکتوری حاوی فایل‌های txt را وارد کنید
    مسیر_ورودی = os.path.dirname(os.path.abspath(__file__))  # این را به مسیر دلخواه خود تغییر دهید
    مسیر_خروجی = "./merged_all_neg.txt"  # نام فایل خروجی

    merge_txt_files(مسیر_ورودی, مسیر_خروجی)