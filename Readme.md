

# 🎬 Subtitle Ad Manager (SAM) | AI-Powered Subtitle Cleaner 🧹

Welcome to the **Subtitle Ad Manager**! Are you tired of distracting sponsor messages, Telegram channel links, and annoying promotional texts popping up in the middle of your favorite movies? This powerful, AI-driven tool is designed to automatically detect and remove (or replace) advertisement blocks from your subtitle files (`.srt`, `.vtt`, `.ass`).



---

## 🌟 Key Features

* **🧠 AI-Powered Detection:** Uses a calibrated Logistic Regression ML model with Character N-gram TF-IDF for highly accurate ad detection (far superior to standard Regex).
* **🛠️ Multi-Format Support:** Seamlessly handles `.srt`, `.vtt`, and advanced `.ass` formats (via `pysubs2`).
* **📂 Batch Processing:** Drag & Drop multiple files into the sleek PySide6 dark-themed GUI and clean them all simultaneously.
* **✏️ Custom Replacements:** Remove ad blocks entirely, replace them with a static string, or randomize replacements from a custom list.
* **📊 End-to-End ML Pipeline:** Includes scripts to extract, clean, normalize, and train your own custom ML models (`Train_model.py`, `norm_text.py`).

## 📸 Screenshots
<img width="1377" height="977" alt="img" src="https://github.com/user-attachments/assets/0d10c8f1-3e4a-4896-b486-c17634865d42" />
<img width="1377" height="977" alt="img_1" src="https://github.com/user-attachments/assets/333da4bf-ae18-4852-a3ab-a6113ad22b25" />
<img width="1377" height="977" alt="img_2" src="https://github.com/user-attachments/assets/3947ba84-636a-4820-a26d-dd2d3ee04608" />

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/subtitle-ad-manager.git
cd subtitle-ad-manager
pip install -r requirements.txt

```

*(Dependencies: `PySide6`, `scikit-learn`, `pandas`, `joblib`, `pysubs2`, `matplotlib`)*

### 2. Running the GUI

Launch the beautiful dark-mode interface:

```bash
python main.py

```

### 3. Training the AI Model (Optional)

If you want to improve the AI or train it on your own dataset:

1. Put your raw text in `ads_positive_clean.csv` and `ads_negative_clean.csv`.
2. Run data preparation scripts (`norm_text.py`, `tm.py`).
3. Execute the training script:

```bash
python Train_model.py

```

This will generate the `ad_classifier.pkl` and `ad_vectorizer.pkl` files in the `model/` directory.

---

## 💡 Ideas for Future Improvements (Roadmap)

* **Deep Learning Integration:** Upgrade the classic ML model to a lightweight Transformer-based model (like DistilBERT) for better contextual understanding, especially for multi-language ads.
* **Video Player Plugin:** Export the core backend as a plugin/extension for popular video players (VLC, PotPlayer, MPV) to block ads dynamically in real-time.
* **Auto-Sync & Timing Fixes:** Sometimes removing a large ad block disrupts the timing flow. Adding a feature to smartly bridge the gap between subtitle timestamps.
* **Crowdsourced Database:** Allow users to flag false positives/negatives in the GUI and automatically upload them to a central database to continuously improve the community ML model.
* **CLI Automation:** Enhance the command-line interface to easily integrate SAM into automated media server pipelines (e.g., Plex, Jellyfin, Sonarr).

---

---

# 🎬 مدیریت و پاکسازی زیرنویس (SAM) | حذف هوشمند تبلیغات 🧹

به **مدیریت زیرنویس (Subtitle Ad Manager)** خوش آمدید! آیا از دست پیام‌های اسپانسرها، لینک کانال‌های تلگرامی و متون تبلیغاتی که وسط فیلم روی صفحه ظاهر می‌شوند خسته شده‌اید؟ این ابزار قدرتمند و مبتنی بر هوش مصنوعی طراحی شده تا بلوک‌های تبلیغاتی را در فایل‌های زیرنویس (`.srt`, `.vtt`, `.ass`) به صورت خودکار شناسایی کرده و آن‌ها را حذف یا جایگزین کند.

---

## 🌟 ویژگی‌های کلیدی

* **🧠 تشخیص مبتنی بر هوش مصنوعی:** استفاده از مدل یادگیری ماشین (Logistic Regression) همراه با استخراج ویژگی TF-IDF برای شناسایی دقیق تبلیغات (بسیار هوشمندتر از Regex ساده).
* **🛠️ پشتیبانی از فرمت‌های مختلف:** پردازش بی‌نقص فایل‌های `.srt`، `.vtt` و فرمت پیشرفته `.ass`.
* **📂 پردازش دسته‌ای (Batch):** رابط کاربری تاریک و جذاب ساخته شده با PySide6 که امکان کشیدن و رها کردن (Drag & Drop) ده‌ها فایل برای پاکسازی همزمان را فراهم می‌کند.
* **✏️ جایگزینی دلخواه:** می‌توانید تبلیغات را کاملاً حذف کنید، یک متن ثابت جایگزین آن‌ها کنید، یا متون تصادفی از لیست دلخواه خود قرار دهید.
* **📊 خط لوله کامل یادگیری ماشین:** شامل اسکریپت‌های استخراج، پاکسازی، نرمال‌سازی حروف فارسی/عربی و آموزش مدل اختصاصی شما (`Train_model.py`, `norm_text.py`).

## 🚀 شروع سریع

### ۱. نصب و راه‌اندازی

پروژه را کلون کرده و پیش‌نیازها را نصب کنید:

```bash
git clone https://github.com/yourusername/subtitle-ad-manager.git
cd subtitle-ad-manager
pip install -r requirements.txt

```

### ۲. اجرای رابط کاربری (GUI)

برای باز کردن محیط گرافیکی نرم‌افزار، دستور زیر را اجرا کنید:

```bash
python main.py

```

### ۳. آموزش مدل هوش مصنوعی (اختیاری)

اگر قصد دارید هوش مصنوعی را با داده‌های خودتان آموزش دهید تا دقیق‌تر شود:
۱. داده‌های خود را در فایل‌های CSV آماده کنید.
۲. اسکریپت‌های پاکسازی (`norm_text.py`, `tm.py`) را اجرا کنید.
۳. مدل را آموزش دهید:

```bash
python Train_model.py

```

مدل‌های جدید در پوشه `model/` ذخیره خواهند شد و برنامه بلافاصله از آن‌ها استفاده خواهد کرد.

---

## 💡 ایده‌هایی برای بهبود پروژه در آینده

* **استفاده از یادگیری عمیق (Deep Learning):** ارتقای مدل به شبکه‌های مبتنی بر ترانسفورمر (مثل ParsBERT) برای درک بهتر مفهوم جملات و کاهش خطای تشخیص در زیرنویس‌های فارسی.
* **افزونه برای ویدیو پلیرها:** تبدیل هسته اصلی برنامه به یک افزونه (Plugin) برای پلیرهای معروف مانند PotPlayer یا VLC تا تبلیغات را به صورت زنده (Real-time) هنگام تماشای فیلم حذف کند.
* **اصلاح زمان‌بندی (Auto-Sync):** گاهی حذف یک زیرنویس تبلیغاتی طولانی باعث ایجاد سکوت طولانی در زیرنویس می‌شود. اضافه کردن قابلیتی برای اصلاح و پر کردن هوشمند این فواصل زمانی.
* **دیتابیس ابری مشارکتی:** افزودن دکمه‌ای در رابط کاربری که کاربران بتوانند تشخیص‌های اشتباه (False Positives) را ریپورت کنند تا در آپدیت‌های بعدی، مدل هوش مصنوعی با کمک جامعه کاربری قوی‌تر شود.
* **ادغام با سرورهای مدیا:** توسعه CLI برای اتصال راحت این ابزار به نرم‌افزارهای مدیریت مدیا (مانند Plex یا Radarr) تا فایل‌ها بلافاصله پس از دانلود، به صورت خودکار پاکسازی شوند.
