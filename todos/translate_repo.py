import os
import re
import time
from pathlib import Path
from deep_translator import GoogleTranslator

# Configuration
TARGET_LANG = os.getenv("TARGET_LANG", "es")
ROOT_DIR = Path(".")
OUTPUT_DIR = ROOT_DIR / f"translated_{TARGET_LANG}"


def safe_translate(translator, text, retries=3, delay=1.5):
    """
    Translates text with retry mechanism and line break preservation.
    
    Args:
        translator (GoogleTranslator): Translator instance
        text (str): Input text to translate
        retries (int): Maximum number of retry attempts. Default: 3
        delay (float): Delay in seconds between retries. Default: 1.5
    
    Returns:
        str: Translated text or original text if translation fails
    """
    if not text.strip():
        return text

    for attempt in range(retries):
        try:
            translated = translator.translate(text)
            if translated:
                if text.endswith("\n") and not translated.endswith("\n"):
                    translated += "\n"
                return translated
        except Exception as e:
            print(f"[WARNING] Translation error: '{text[:40]}...' ({e}) - Attempt {attempt+1}/{retries}")
            time.sleep(delay)

    return text


def translate_markdown_line(line, translator):
    """
    Translates a single Markdown line while preserving structural elements.
    
    Translation rules:
    - Translates visible text in links [text](url)
    - Translates plain text content
    - Preserves URLs, inline code, and code blocks
    - Maintains line break formatting
    
    Args:
        line (str): Input Markdown line
        translator (GoogleTranslator): Translator instance
    
    Returns:
        str: Translated line with preserved Markdown syntax
    """
    # Pattern definitions for Markdown elements
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    code_pattern = re.compile(r'`[^`]+`|https?://\S+')

    # Process links: translate only visible text within brackets
    def replace_link(match):
        visible_text = match.group(1)
        url = match.group(2)
        translated_text = safe_translate(translator, visible_text)
        return f'[{translated_text}]({url})'

    line = link_pattern.sub(replace_link, line)

    # Split line to isolate code blocks and URLs
    parts = re.split(f'({code_pattern.pattern})', line)
    translated_parts = []
    
    for part in parts:
        if not part:
            translated_parts.append(part)
        elif code_pattern.match(part):
            translated_parts.append(part)
        else:
            translated_parts.append(safe_translate(translator, part))

    translated_line = ''.join(translated_parts)

    # Preserve line break if present in original
    if line.endswith("\n") and not translated_line.endswith("\n"):
        translated_line += "\n"

    return translated_line


def translate_file(filepath, translator):
    """
    Translates a single Markdown file and writes output to target directory.
    
    Args:
        filepath (Path): Path object pointing to source file
        translator (GoogleTranslator): Translator instance
    """
    rel_path = filepath.relative_to(ROOT_DIR)
    output_path = OUTPUT_DIR / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read file preserving original line structure
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()

    translated_lines = []
    for line in lines:
        translated_lines.append(translate_markdown_line(line, translator))

    # Write output maintaining exact line break structure
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(translated_lines)

    print(f"[SUCCESS] Translation completed (structure preserved): {filepath} -> {output_path}")


def translate_repo(root=ROOT_DIR):
    """
    Main translation workflow for repository Markdown files.
    
    Scans root directory recursively for .md files and translates each one
    to the target language specified in TARGET_LANG environment variable.
    
    Args:
        root (Path): Root directory path. Default: current directory
    """
    translator = GoogleTranslator(source="auto", target=TARGET_LANG)

    md_files = [
        f for f in Path(root).rglob("*.md")
        if not any(part.startswith("translated_") or part == "venv" for part in f.parts)  # Ignore 'venv' folder locally
    ]
    
    print(f"[INFO] Markdown files discovered: {len(md_files)}")

    for file in md_files:
        translate_file(file, translator)

    print(f"\n[SUCCESS] Translation process completed successfully. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    translate_repo()