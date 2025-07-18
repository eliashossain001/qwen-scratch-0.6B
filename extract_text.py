import pdfplumber

with pdfplumber.open("data/book.pdf") as pdf:
    text = []
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
with open("data/book.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(text))
