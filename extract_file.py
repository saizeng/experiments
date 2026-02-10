import argparse
import json
import pdfplumber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    text = []

    with pdfplumber.open(args.inp) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))

    print(json.dumps({"output_file": args.out}))


if __name__ == "__main__":
    main()
