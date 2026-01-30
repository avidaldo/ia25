#!/usr/bin/env python3
"""
Build script for converting course materials to a static site for GitHub Pages.
Converts markdown and Jupyter notebooks to HTML with proper link rewriting.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# GitHub-like CSS styling
GITHUB_STYLE_CSS = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #24292e;
    background-color: #ffffff;
    max-width: 980px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
    line-height: 1.25;
}

h1 {
    font-size: 2em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #eaecef;
}

h2 {
    font-size: 1.5em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #eaecef;
}

h3 { font-size: 1.25em; }
h4 { font-size: 1em; }
h5 { font-size: 0.875em; }
h6 { font-size: 0.85em; color: #6a737d; }

a {
    color: #0366d6;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

code {
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    font-size: 85%;
    margin: 0;
    padding: 0.2em 0.4em;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

pre {
    background-color: #f6f8fa;
    border-radius: 3px;
    font-size: 85%;
    line-height: 1.45;
    overflow: auto;
    padding: 16px;
}

pre code {
    background-color: transparent;
    border: 0;
    display: inline;
    line-height: inherit;
    margin: 0;
    overflow: visible;
    padding: 0;
    word-wrap: normal;
}

blockquote {
    border-left: 0.25em solid #dfe2e5;
    color: #6a737d;
    padding: 0 1em;
    margin: 0 0 16px 0;
}

table {
    border-collapse: collapse;
    border-spacing: 0;
    margin-bottom: 16px;
    width: 100%;
}

table th, table td {
    border: 1px solid #dfe2e5;
    padding: 6px 13px;
}

table th {
    background-color: #f6f8fa;
    font-weight: 600;
}

table tr:nth-child(2n) {
    background-color: #f6f8fa;
}

img {
    max-width: 100%;
    box-sizing: border-box;
}

ul, ol {
    margin-bottom: 16px;
    padding-left: 2em;
}

/* Notebook-specific styles */
.jp-Cell {
    margin-bottom: 1em;
}

.jp-InputArea {
    margin-bottom: 0.5em;
}

.jp-OutputArea {
    padding-top: 0.5em;
}

div.output_subarea {
    max-width: 100%;
    overflow-x: auto;
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
        font-size: 14px;
    }
    
    h1 { font-size: 1.75em; }
    h2 { font-size: 1.5em; }
    h3 { font-size: 1.25em; }
}
</style>
"""


def convert_markdown_to_html(md_file: Path, output_file: Path) -> bool:
    """Convert a markdown file to HTML using Python markdown library."""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        if HAS_MARKDOWN:
            html_content = markdown.markdown(
                md_content,
                extensions=['extra', 'codehilite', 'tables', 'fenced_code']
            )
        else:
            # Fallback: try using pandoc
            cmd = ["pandoc", str(md_file), "-o", str(output_file), "-s", "--standalone"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Warning: Failed to convert {md_file}: {result.stderr}")
                # As last resort, wrap the markdown in HTML
                html_content = f"<pre>{md_content}</pre>"
            else:
                return True
        
        # Create a complete HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_file.stem}</title>
{GITHUB_STYLE_CSS}
</head>
<body>
{html_content}
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        return True
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout converting {md_file}")
        return False
    except Exception as e:
        print(f"Warning: Error converting {md_file}: {e}")
        return False


def convert_notebook_to_html(ipynb_file: Path, output_file: Path, timeout: int = 600) -> bool:
    """Execute and convert a Jupyter notebook to HTML."""
    try:
        cmd = [
            "jupyter", "nbconvert",
            "--to", "html",
            "--execute",
            "--ExecutePreprocessor.timeout={}".format(timeout),
            "--template", "classic",
            str(ipynb_file),
            "--output", str(output_file.absolute())
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
        
        if result.returncode != 0:
            print(f"Warning: Failed to execute/convert {ipynb_file}: {result.stderr}")
            # Try without execution as fallback
            cmd = [
                "jupyter", "nbconvert",
                "--to", "html",
                "--template", "classic",
                str(ipynb_file),
                "--output", str(output_file.absolute())
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"Warning: Failed to convert {ipynb_file} without execution")
                return False
        
        return True
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout executing/converting {ipynb_file}")
        return False
    except Exception as e:
        print(f"Warning: Error converting {ipynb_file}: {e}")
        return False


def rewrite_links_in_html(html_file: Path) -> None:
    """Rewrite links in HTML file to point to .html versions."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Rewrite .md links to .html (but not external links)
        content = re.sub(
            r'href="(?!http://|https://|#)([^"]*?)\.md"',
            r'href="\1.html"',
            content
        )
        
        # Rewrite .ipynb links to .html (but not external links)
        content = re.sub(
            r'href="(?!http://|https://|#)([^"]*?)\.ipynb"',
            r'href="\1.html"',
            content
        )
        
        # Add GitHub-like styling if not already present
        if '<style>' not in content and GITHUB_STYLE_CSS not in content:
            # Insert style before </head> or after <head>
            if '</head>' in content:
                content = content.replace('</head>', GITHUB_STYLE_CSS + '\n</head>')
            elif '<head>' in content:
                content = content.replace('<head>', '<head>\n' + GITHUB_STYLE_CSS)
            else:
                # Add a head section if missing
                content = '<html>\n<head>\n' + GITHUB_STYLE_CSS + '\n</head>\n<body>\n' + content + '\n</body>\n</html>'
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Error rewriting links in {html_file}: {e}")


def should_ignore(path: Path) -> bool:
    """Check if a path should be ignored."""
    ignore_patterns = [
        '.git', '__pycache__', '.ipynb_checkpoints',
        '_site', 'node_modules', '.pytest_cache',
        '.venv', 'venv', 'env'
    ]
    
    path_str = str(path)
    for pattern in ignore_patterns:
        if pattern in path_str:
            return True
    
    return False


def build_site(source_dir: Path, output_dir: Path) -> None:
    """Build the static site from source directory."""
    print(f"Building site from {source_dir} to {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track converted files
    converted_files = []
    failed_files = []
    
    # Walk through source directory
    for root, dirs, files in os.walk(source_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(Path(root) / d)]
        
        root_path = Path(root)
        rel_path = root_path.relative_to(source_dir)
        output_root = output_dir / rel_path
        
        # Create corresponding output directory
        output_root.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            source_file = root_path / file
            
            # Skip ignored files
            if should_ignore(source_file):
                continue
            
            # Convert README.md to index.html
            if file == 'README.md':
                output_file = output_root / 'index.html'
                print(f"Converting {source_file} -> {output_file}")
                if convert_markdown_to_html(source_file, output_file):
                    rewrite_links_in_html(output_file)
                    converted_files.append(source_file)
                else:
                    failed_files.append(source_file)
            
            # Convert other .md files to .html
            elif file.endswith('.md'):
                output_file = output_root / (file[:-3] + '.html')
                print(f"Converting {source_file} -> {output_file}")
                if convert_markdown_to_html(source_file, output_file):
                    rewrite_links_in_html(output_file)
                    converted_files.append(source_file)
                else:
                    failed_files.append(source_file)
            
            # Execute and convert .ipynb files
            elif file.endswith('.ipynb'):
                output_file = output_root / (file[:-6] + '.html')
                print(f"Executing and converting {source_file} -> {output_file}")
                if convert_notebook_to_html(source_file, output_file):
                    rewrite_links_in_html(output_file)
                    converted_files.append(source_file)
                else:
                    failed_files.append(source_file)
            
            # Copy other files (images, PDFs, etc.)
            else:
                output_file = output_root / file
                try:
                    shutil.copy2(source_file, output_file)
                    print(f"Copied {source_file} -> {output_file}")
                except Exception as e:
                    print(f"Warning: Failed to copy {source_file}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Build complete!")
    print(f"Successfully converted: {len(converted_files)} files")
    print(f"Failed conversions: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print("="*60)


def main():
    """Main entry point."""
    # Get paths
    repo_root = Path(__file__).parent.parent.absolute()
    output_dir = repo_root / "_site"
    
    print(f"Repository root: {repo_root}")
    print(f"Output directory: {output_dir}")
    
    # Build the site
    build_site(repo_root, output_dir)
    
    print("\nSite built successfully in _site/")
    print("You can now deploy the _site/ directory to GitHub Pages")


if __name__ == "__main__":
    main()
