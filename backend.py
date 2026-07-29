import re
import random
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import joblib

try:
    import pysubs2
    HAS_PYSUBS2 = True
except ImportError:
    HAS_PYSUBS2 = False


model = joblib.load('ad_classifier.pkl')
vectorizer = joblib.load('ad_vectorizer.pkl')
with open('threshold.txt', 'r') as f:
    THRESHOLD = float(f.read())

# ------------------ Patterns & Normalization ------------------
AD_PATTERNS = [
    r'https?://\S+',
    r'www\.\S+',
    r'\b(?!www\.)([A-Za-z0-9-]+\.)+(com|ir|org|net|in|co|us|me|app|tv|xyz|info|online)\b',
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    r'@\w{2,}',
    r't\.me/\S+',
    r'telegram\s*(?:channel|group)?',
    r'discord\.gg/\S+',
    r'instagram\.com/\S+',
    r'twitter\.com/\S+',
    r'youtube\.com/\S+',
    r'fb\.com/\S+',
    r'whatsapp',
    r'\b(?:like|share|subscribe|follow)\s*(?:and|&)?\s*(?:like|share|subscribe|follow)?\b',
    r'\bjoin\s+(?:us|our|my|the)\b',
    r'\bcheck\s+out\b',
    r'\bclick\s+(?:the\s+)?link\b',
    r'\bvisit\s+(?:us|our|my)\b',
    r'\bdonate\b',
    r'\bpatreon\b',
    r'\b(?:free\s+)?download\b',
    r'\bbuy\s+now\b',
    r'\b(?:sponsor|advertisement|promo|ad\b|advert)\b',
    r'\b(?:subtitle|sub|translat)(?:ed|ion)?\s*by\b',
    r'\b(?:powered\s+by)\b',
    r'\b(?:subscene|opensubtitles|yifysubtitles|subdl|subtitlecat|tvsubtitles)\b',
    r'دانلود',
    r'زیرنویس\s*(?:فارسی|کامل|جدید)?',
    r'ترجمه\s*(?:شده)?\s*(?:توسط|از)?',
    r'کانال\s*تلگرام',
    r'تلگرام\s*(?:ما|کانال)?',
    r'اینستاگرام',
    r'سابسین',
    r'سابتایتل',
    r'خرید\s*(?:اینترنتی|آنلاین)?',
    r'تبلیغ',
    r'حمایت\s*(?:مالی|از\s*ما)?',
    r'لایک\s*و?\s*فالو',
    r'عضویت\s*در\s*کانال',
    r'آدرس\s*(?:سایت|وب)',
    r'برای\s*دسترسی\s*به\s*زیرنویس',
    r'با\s*کلیک\s*روی\s*لینک',
    r'کد\s*تخفیف',
    r'تقدیم\s*می\s*کند',
    r'دنبال\s*کنید',
    r'کاری\s*از',
    r'تی\s*وی',
    r'فیلم',
    r'مووی',
    r'زر',
    r'۳۰نما|30nama',
    r'CinamaSub',
    r'مترجم',
]
AD_REGEX = re.compile('|'.join(f'(?:{p})' for p in AD_PATTERNS), re.IGNORECASE)

SRT_TIME = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})')
VTT_TIME = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})')

ARABIC_TO_PERSIAN = str.maketrans({
    'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'ؤ': 'و',
    'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ى': 'ی', 'ٱ': 'ا',
})

def normalize_text(text: str, aggressive: bool = False) -> str:
    """Remove invisible chars, diacritics, unify Persian/Arabic letters."""
    invisible = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060-\u2069\uFEFF\u00AD]')
    text = invisible.sub('', text)
    text = re.sub(r'[\u064b-\u0652\u0670]', '', text)
    text = text.translate(ARABIC_TO_PERSIAN)
    text = unicodedata.normalize('NFKC', text)
    if aggressive:
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
    return text

# ------------------ Data Structures ------------------
@dataclass
class AdBlock:
    index: str
    start: str
    end: str
    text: str
    ass_event: Optional[object] = field(default=None, repr=False)

# ------------------ Single File Processor ------------------
class SubAdProcessor:
    def __init__(self):
        self.blocks: List[AdBlock] = []
        self.filepath: Optional[Path] = None
        self.format: str = 'srt'

    def load_file(self, path: str) -> int:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.filepath = path
        ext = path.suffix.lower()
        content = path.read_text(encoding='utf-8', errors='replace')
        if ext == '.vtt':
            self.blocks = self._parse_vtt(content)
            self.format = 'vtt'
        elif ext in ('.ass', '.ssa') and HAS_PYSUBS2:
            self.blocks = self._parse_ass(str(path))
            self.format = 'ass'
        else:
            self.blocks = self._parse_srt(content)
            self.format = 'srt'
        return len(self.blocks)

    @staticmethod
    def _parse_srt(content: str) -> List[AdBlock]:
        blocks = []
        raw = re.split(r'\n\s*\n', content.strip())
        for blk in raw:
            lines = blk.strip().splitlines()
            if len(lines) < 3 or not lines[0].strip().isdigit():
                continue
            m = SRT_TIME.search(lines[1])
            if not m:
                continue
            start, end = m.groups()
            text = '\n'.join(lines[2:])
            blocks.append(AdBlock(index=lines[0].strip(), start=start, end=end, text=text))
        return blocks

    @staticmethod
    def _parse_vtt(content: str) -> List[AdBlock]:
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.MULTILINE | re.DOTALL)
        raw = re.split(r'\n\s*\n', content.strip())
        blocks = []
        for blk in raw:
            lines = blk.strip().splitlines()
            if not lines:
                continue
            time_idx = None
            for i, line in enumerate(lines):
                if VTT_TIME.search(line):
                    time_idx = i
                    break
            if time_idx is None:
                continue
            m = VTT_TIME.search(lines[time_idx])
            start, end = m.groups()
            idx = lines[time_idx-1].strip() if (time_idx > 0 and lines[time_idx-1].strip().isdigit()) else ''
            text = '\n'.join(lines[time_idx+1:])
            blocks.append(AdBlock(index=idx, start=start, end=end, text=text))
        return blocks

    @staticmethod
    def _parse_ass(filepath: str) -> List[AdBlock]:
        subs = pysubs2.load(filepath, encoding='utf-8')
        blocks = []
        for i, event in enumerate(subs.events, 1):
            blocks.append(AdBlock(index=str(i), start=str(event.start),
                                  end=str(event.end), text=event.plaintext,
                                  ass_event=event))
        return blocks

    def is_ad_probability(self, text: str) -> float:
        prob = self.model.predict_proba([text])[0][1]
        return prob

    def detect_ads(self, aggressive: bool = False) -> List[Tuple[int, AdBlock]]:
        ads = []
        for i, blk in enumerate(self.blocks):
            norm = normalize_text(blk.text, aggressive)
            if AD_REGEX.search(norm):
                ads.append((i, blk))
        return ads

    def remove_blocks(self, indices: List[int]) -> None:
        for i in sorted(indices, reverse=True):
            del self.blocks[i]

    def replace_text(self, index: int, new_text: str) -> None:
        self.blocks[index].text = new_text
        if self.format == 'ass' and self.blocks[index].ass_event:
            self.blocks[index].ass_event.text = new_text

    def rebuild_content(self) -> str:
        if self.format == 'ass' and HAS_PYSUBS2:
            subs = pysubs2.SSAFile()
            for blk in self.blocks:
                ev = blk.ass_event or pysubs2.SSAEvent(start=blk.start, end=blk.end, text=blk.text)
                subs.append(ev)
            return subs.to_string('ass')
        elif self.format == 'vtt':
            return self._rebuild_vtt()
        else:
            return self._rebuild_srt()

    def _rebuild_srt(self) -> str:
        out = []
        for b in self.blocks:
            out.extend([b.index, f"{b.start} --> {b.end}", b.text, ''])
        return '\n'.join(out)

    def _rebuild_vtt(self) -> str:
        out = ['WEBVTT\n']
        for b in self.blocks:
            if b.index:
                out.append(b.index)
            out.append(f"{b.start} --> {b.end}")
            out.append(b.text)
            out.append('')
        return '\n'.join(out)

    def save(self, path: str) -> None:
        Path(path).write_text(self.rebuild_content(), encoding='utf-8')

# ------------------ Batch Processor ------------------
class BatchProcessor:
    def __init__(self):
        self.processors: Dict[str, SubAdProcessor] = {}

    def load_files(self, paths: List[str]) -> int:
        total = 0
        for p in paths:
            proc = SubAdProcessor()
            count = proc.load_file(p)
            self.processors[str(Path(p).resolve())] = proc
            total += count
        return total

    def clear(self):
        self.processors.clear()

    @property
    def files(self) -> List[str]:
        return list(self.processors.keys())

    @property
    def total_blocks(self) -> int:
        return sum(len(p.blocks) for p in self.processors.values())

    def detect_all_ads(self, aggressive: bool = False) -> List[Tuple[str, int, AdBlock]]:
        results = []
        for fpath, proc in self.processors.items():
            for idx, blk in proc.detect_ads(aggressive):
                results.append((fpath, idx, blk))
        return results

    def remove_blocks_for_file(self, filepath: str, indices: List[int]) -> None:
        if filepath in self.processors:
            self.processors[filepath].remove_blocks(indices)

    def replace_text_for_file(self, filepath: str, index: int, new_text: str) -> None:
        if filepath in self.processors:
            self.processors[filepath].replace_text(index, new_text)

    def save_all(self, output_dir: str = None, suffix: str = "_clean") -> List[str]:
        saved = []
        for fpath, proc in self.processors.items():
            orig = Path(fpath)
            if output_dir:
                dest = Path(output_dir) / (orig.stem + suffix + orig.suffix)
            else:
                dest = orig.with_name(orig.stem + suffix + orig.suffix)
            proc.save(str(dest))
            saved.append(str(dest))
        return saved

    def save_file(self, filepath: str, output_path: str) -> None:
        if filepath in self.processors:
            self.processors[filepath].save(output_path)


# ------------------ Command Line Interface ------------------
if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    parser = argparse.ArgumentParser(description="Subtitle Ad Manager - CLI")
    parser.add_argument('files', nargs='+', help='Subtitle files (.srt, .vtt, .ass)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Aggressive mode (letters only)')
    parser.add_argument('--list', action='store_true',
                        help='Only list detected ads (no changes)')
    parser.add_argument('--remove', action='store_true',
                        help='Remove all detected ad blocks')
    parser.add_argument('--replace', type=str,
                        help='Replace all ad texts with this single text')
    parser.add_argument('--replace-random', nargs='+',
                        help='Replace ad texts randomly with these texts (space separated)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for saved files')
    parser.add_argument('--suffix', type=str, default='_clean',
                        help='Suffix for output filenames')
    args = parser.parse_args()

    bp = BatchProcessor()
    total = bp.load_files(args.files)
    print(f"Loaded {len(args.files)} file(s), {total} blocks total.")

    ads = bp.detect_all_ads(aggressive=args.aggressive)

    if args.list or not (args.remove or args.replace or args.replace_random):
        for fpath, idx, blk in ads:
            print(f"{Path(fpath).name}: block {blk.index} [{blk.start} --> {blk.end}]")
            print(f"    {blk.text}\n")
        print(f"Total ad blocks: {len(ads)}")
    else:
        if args.remove:
            # Group ads by file and remove all indices
            file_groups: Dict[str, List[int]] = defaultdict(list)
            for fpath, idx, _ in ads:
                file_groups[fpath].append(idx)
            for fpath, indices in file_groups.items():
                bp.remove_blocks_for_file(fpath, indices)
            print(f"Removed {len(ads)} ad blocks across {len(file_groups)} file(s).")

        elif args.replace:
            for fpath, idx, _ in ads:
                bp.replace_text_for_file(fpath, idx, args.replace)
            print(f"Replaced {len(ads)} ad blocks with fixed text.")

        elif args.replace_random:
            texts = list(args.replace_random)
            for fpath, idx, _ in ads:
                new_text = random.choice(texts)
                bp.replace_text_for_file(fpath, idx, new_text)
            print(f"Randomly replaced {len(ads)} ad blocks from {len(texts)} candidates.")

        # Save
        saved = bp.save_all(output_dir=args.output_dir, suffix=args.suffix)
        print(f"\nSaved {len(saved)} cleaned file(s):")
        for s in saved:
            print(f"  {s}")