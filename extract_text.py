#!/usr/bin/env python3
"""
extract_subtitle_text.py

Reads a file containing SRT-like metadata (e.g., ad_list3.txt),
extracts the actual subtitle text from each "Text:" block,
removes HTML/ASS formatting tags, and saves the cleaned text into an output file.
"""

import re
import sys

def clean_text(text: str) -> str:
    """Remove ASS/HTML formatting tags from text."""
    # Remove ASS override tags like {\an8}
    text = re.sub(r'\{[^}]+\}', '', text)
    # Remove HTML tags like <font...> and </font>
    text = re.sub(r'<[^>]+>', '', text)
    # Remove any remaining extra spaces caused by tag removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_blocks(lines):
    """
    Parse lines and yield cleaned text blocks.
    Each block corresponds to one "Text:" entry.
    """
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for a line that starts with "Text:" (ignoring leading spaces)
        if stripped.startswith('Text:'):
            # Content after the colon on the same line (if any)
            _, after_colon = line.split(':', 1)
            after_colon = after_colon.lstrip()
            text_lines = [after_colon] if after_colon else []

            i += 1
            # Collect subsequent lines until an empty line or a new Index:/File: line
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()

                # Stop if we encounter the start of another entry or an empty line
                if next_stripped.startswith(('Index:', 'File:')):
                    break
                if next_stripped == '':
                    i += 1
                    break

                # Append non‑empty lines as part of the text
                if next_stripped:
                    text_lines.append(next_line.rstrip('\n'))
                i += 1

            full_text = '\n'.join(text_lines).strip()
            if full_text:
                yield clean_text(full_text)
        else:
            i += 1

def main():
    # Set default input and output filenames
    input_file = 'merged_all_pos.txt'
    output_file = 'output_pos.txt'

    # Allow user to specify input and output via command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    blocks = list(extract_text_blocks(lines))

    # Write cleaned text blocks to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, block in enumerate(blocks):
            f.write(block)
            if idx != len(blocks) - 1:
                f.write('\n\n')   # separate blocks with blank line

    print(f"Done! Cleaned text saved to '{output_file}'.")

if __name__ == '__main__':
    main()