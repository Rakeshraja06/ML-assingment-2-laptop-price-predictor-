import re
import subprocess
import os

def md_to_html(md_text):
    # Very simple markdown to html conversion for basic elements
    html = md_text
    # Headers
    html = re.sub(r'^# (.*)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    # Bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
    # Lists
    html = re.sub(r'^- (.*)', r'<li>\1</li>', html, flags=re.MULTILINE)
    # Horizontal rules
    html = re.sub(r'^---', r'<hr>', html, flags=re.MULTILINE)
    # Tables (minimal support)
    html = html.replace('|', '  ') # Just clean up pipes for basic readability
    # Paragraphs (basic)
    html = html.replace('\n\n', '</p><p>')
    return f"<html><body><p>{html}</p></body></html>"

with open('submission.md', 'r') as f:
    md_content = f.read()

html_content = md_to_html(md_content)

with open('temp_submission.html', 'w') as f:
    f.write(html_content)

# Use textutil to convert HTML to DOCX
try:
    subprocess.run(['textutil', '-convert', 'docx', 'temp_submission.html', '-output', 'submission.docx'], check=True)
    print("Successfully converted to submission.docx")
except Exception as e:
    print(f"Error during conversion: {e}")
finally:
    if os.path.exists('temp_submission.html'):
        os.remove('temp_submission.html')
