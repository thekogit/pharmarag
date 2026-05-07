import fitz  # PyMuPDF
import pdfplumber

class PDFIngestor:
    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> str:
        text_content = ""
        # PyMuPDF for raw speed
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_content += page.get_text()
        return text_content

    def extract_tables_sane(self, pdf_path: str) -> str:
        table_content = ""
        # pdfplumber for sanity
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    table_content += "\n" + "\n".join(["\t".join([str(cell).strip() if cell else "" for cell in row]) for row in table])
        return table_content

    def process(self, pdf_path: str) -> str:
        return self.extract_text(pdf_path) + "\n" + self.extract_tables_sane(pdf_path)
