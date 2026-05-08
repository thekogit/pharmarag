from typing import List
import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.router import RegulatoryRouter

class PDFIngestor:
    def __init__(self):
        self.router = RegulatoryRouter()
        self.narrative_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.dense_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", " ", ""]
        )

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
                    if table:
                        table_content += "\n" + "\n".join(["\t".join([str(cell).strip() if cell else "" for cell in row]) for row in table])
        return table_content

    def chunk_sections(self, sections: dict) -> List[dict]:
        all_chunks = []
        for section_name, text in sections.items():
            if section_name == "CLINICAL_STUDIES":
                splitter = self.narrative_splitter
            else:
                splitter = self.dense_splitter
            
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "text": f"[Section: {section_name}] {chunk}",
                    "metadata": {"section": section_name}
                })
        return all_chunks

    def process(self, pdf_path: str) -> List[dict]:
        text = self.extract_text(pdf_path)
        tables = self.extract_tables_sane(pdf_path)
        combined_text = text + "\n" + tables
        sections = self.router.parse(combined_text)
        return self.chunk_sections(sections)
