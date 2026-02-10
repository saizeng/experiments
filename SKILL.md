# PDF Skill

## Purpose
This skill provides utilities for reading, extracting, modifying, and generating PDF files.

Use this skill whenever a user asks to:
- read text from a PDF
- extract tables
- split pages
- merge PDFs
- create or rewrite PDFs
- convert PDFs to text

All file operations occur inside the local "workdir/" folder.


---

## How to execute actions

You MUST execute work by calling:

run_python_script

Arguments:
- skill: "pdf"
- script: the python filename inside scripts/
- args: list of CLI arguments


DO NOT explain code.
DO NOT simulate results.
ALWAYS call run_python_script when a PDF operation is required.


---

## Available Scripts

### extract_text.py
Purpose:
Extract all text from a PDF into a .txt file.

Arguments:
--in  input.pdf
--out output.txt

Example tool call:
skill="pdf"
script="extract_text.py"
args=["--in","report.pdf","--out","report.txt"]


---

### split_pdf.py
Purpose:
Extract a page range into a new PDF.

Arguments:
--in input.pdf
--start 1
--end 5
--out part.pdf

Example:
args=["--in","book.pdf","--start","1","--end","5","--out","part.pdf"]


---

### merge_pdf.py
Purpose:
Merge multiple PDFs into one.

Arguments:
--in a.pdf b.pdf c.pdf
--out merged.pdf

Example:
args=["--in","a.pdf","b.pdf","c.pdf","--out","merged.pdf"]


---

## Decision Rules

If user says:

"extract text"
→ use extract_text.py

"split pages" or "pages 1-5"
→ use split_pdf.py

"merge/combine"
→ use merge_pdf.py

If unsure:
choose the script that most directly performs the requested transformation.


---

## Output Rules

Scripts return JSON or text in stdout.

After running a script:
- read stdout
- summarize result to the user
- report created filenames


---

## Examples

User: Extract text from report.pdf
Action:
run extract_text.py

User: Split first 3 pages
Action:
run split_pdf.py

User: Merge a.pdf and b.pdf
Action:
run merge_pdf.py


---

## Safety Rules

- Only operate on files inside workdir/
- Never invent file contents
- Never pretend execution
- Always use run_python_script
