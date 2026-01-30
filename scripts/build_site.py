#!/usr/bin/env python3
"""
Build script for GitHub Pages deployment.
Converts Jupyter notebooks and markdown files to HTML while preserving directory structure.
"""

import os
import sys
import shutil
import subprocess
import re
from pathlib import Path
import markdown
from pygments.formatters import HtmlFormatter

# GitHub-like CSS styling
GITHUB_STYLE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            background-color: #ffffff;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{
            padding-bottom: 0.3em;
            font-size: 2em;
            border-bottom: 1px solid #eaecef;
        }}
        h2 {{
            padding-bottom: 0.3em;
            font-size: 1.5em;
            border-bottom: 1px solid #eaecef;
        }}
        code {{
            background-color: rgba(27, 31, 35, 0.05);
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
            margin: 0;
            padding: 0.2em 0.4em;
        }}
        pre {{
            background-color: #f6f8fa;
            border-radius: 3px;
            font-size: 85%;
            line-height: 1.45;
            overflow: auto;
            padding: 16px;
        }}
        pre code {{
            background-color: transparent;
            border: 0;
            display: inline;
            line-height: inherit;
            margin: 0;
            overflow: visible;
            padding: 0;
            word-wrap: normal;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        table {{
            border-collapse: collapse;
            border-spacing: 0;
            width: 100%;
            overflow: auto;
        }}
        table th {{
            font-weight: 600;
        }}
        table th, table td {{
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }}
        table tr {{
            background-color: #fff;
            border-top: 1px solid #c6cbd1;
        }}
        table tr:nth-child(2n) {{
            background-color: #f6f8fa;
        }}
        img {{
            max-width: 100%;
            box-sizing: border-box;
        }}
        blockquote {{
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0 0 16px 0;
        }}
        ul, ol {{
            padding-left: 2em;
        }}
        .jp-OutputArea-output {{
            padding: 10px;
        }}
        /* Code highlighting */
        {pygments_css}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def get_pygments_css():
    """Get Pygments CSS for code highlighting."""
    formatter = HtmlFormatter(style='github-dark')
    return formatter.get_style_defs('.highlight')


def fix_links_in_html(html_content):
    """
    Fix relative links in HTML content to point to HTML versions.
    Converts .ipynb and .md links to .html links.
    
    Args:
        html_content: HTML content as string
    
    Returns:
        str: HTML content with fixed links
    """
    # Fix links to .ipynb files
    html_content = re.sub(
        r'href="([^"]*?)\.ipynb"',
        r'href="\1.html"',
        html_content
    )
    
    # Fix links to .md files (but not README.md in root)
    html_content = re.sub(
        r'href="(?!\.\./)([^"]*?)\.md"',
        r'href="\1.html"',
        html_content
    )
    
    # Fix README.md links to index.html
    html_content = re.sub(
        r'href="(\.\./)README\.md"',
        r'href="\1index.html"',
        html_content
    )
    html_content = re.sub(
        r'href="README\.md"',
        r'href="index.html"',
        html_content
    )
    
    return html_content


def execute_notebook(notebook_path, output_path, timeout=600):
    """
    Execute a Jupyter notebook and convert it to HTML.
    
    Args:
        notebook_path: Path to the input notebook
        output_path: Path for the output HTML file
        timeout: Execution timeout in seconds (default 600)
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"Executing notebook: {notebook_path}")
        
        # Execute and convert notebook to HTML
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'html',
            '--execute',
            '--ExecutePreprocessor.timeout={}'.format(timeout),
            '--output', str(output_path),
            str(notebook_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60  # Add buffer to subprocess timeout
        )
        
        if result.returncode != 0:
            print(f"Error executing {notebook_path}:")
            print(result.stderr)
            return False
        
        # Fix links in the generated HTML
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            html_content = fix_links_in_html(html_content)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            print(f"Warning: Could not fix links in {output_path}: {str(e)}")
        
        print(f"Successfully converted: {notebook_path} -> {output_path}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Timeout executing notebook: {notebook_path}")
        return False
    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return False


def convert_markdown_to_html(md_path, output_path):
    """
    Convert a Markdown file to HTML with GitHub styling.
    
    Args:
        md_path: Path to the input markdown file
        output_path: Path for the output HTML file
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"Converting markdown: {md_path}")
        
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        
        # Fix links to point to HTML versions
        html_content = fix_links_in_html(html_content)
        
        # Get title from first heading or filename
        title = Path(md_path).stem.replace('_', ' ').title()
        lines = md_content.split('\n')
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        # Wrap in styled HTML
        full_html = GITHUB_STYLE.format(
            title=title,
            content=html_content,
            pygments_css=get_pygments_css()
        )
        
        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Successfully converted: {md_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {md_path}: {str(e)}")
        return False


def copy_static_file(src_path, dest_path):
    """
    Copy a static file (PDF, image, etc.) to the output directory.
    
    Args:
        src_path: Source file path
        dest_path: Destination file path
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")
        return True
    except Exception as e:
        print(f"Error copying {src_path}: {str(e)}")
        return False


def should_exclude(path):
    """
    Check if a path should be excluded from processing.
    
    Args:
        path: Path to check
    
    Returns:
        bool: True if should be excluded, False otherwise
    """
    exclude_patterns = [
        '.ipynb_checkpoints',
        '__pycache__',
        '.git',
        '.github',
        '_site',
        'scripts',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    return False


def build_site(repo_root, output_dir):
    """
    Build the GitHub Pages site.
    
    Args:
        repo_root: Root directory of the repository
        output_dir: Output directory for the built site
    """
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    
    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"Building site from {repo_root} to {output_dir}")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    # Process README.md as index.html
    readme_path = repo_root / 'README.md'
    if readme_path.exists():
        print("\nProcessing README.md as index.html...")
        if convert_markdown_to_html(readme_path, output_dir / 'index.html'):
            success_count += 1
        else:
            error_count += 1
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_root):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
        
        root_path = Path(root)
        rel_path = root_path.relative_to(repo_root)
        
        # Skip if current directory should be excluded
        if should_exclude(root_path):
            continue
        
        for file in files:
            file_path = root_path / file
            
            # Skip excluded files
            if should_exclude(file_path):
                continue
            
            # Skip README.md in root (already processed as index.html)
            if file == 'README.md' and root_path == repo_root:
                continue
            
            # Calculate output path
            if file.endswith('.ipynb'):
                # Convert notebook to HTML
                output_file_path = output_dir / rel_path / (file[:-6] + '.html')
                if execute_notebook(file_path, output_file_path):
                    success_count += 1
                else:
                    error_count += 1
                    
            elif file.endswith('.md'):
                # Convert markdown to HTML
                output_file_path = output_dir / rel_path / (file[:-3] + '.html')
                if convert_markdown_to_html(file_path, output_file_path):
                    success_count += 1
                else:
                    error_count += 1
                    
            elif file.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
                # Copy static files
                output_file_path = output_dir / rel_path / file
                if copy_static_file(file_path, output_file_path):
                    success_count += 1
                else:
                    error_count += 1
    
    print("\n" + "=" * 60)
    print(f"Build complete!")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_dir}")
    
    if error_count > 0:
        print("\nWarning: Some files failed to process. Check the logs above.")
        return 1
    
    return 0


def main():
    """Main entry point."""
    # Get repository root (parent of scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    output_dir = repo_root / '_site'
    
    print("GitHub Pages Site Builder")
    print("=" * 60)
    print(f"Repository root: {repo_root}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    exit_code = build_site(repo_root, output_dir)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
