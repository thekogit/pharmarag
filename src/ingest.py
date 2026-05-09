from typing import List
import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.router import RegulatoryRouter

class PDFIngestor:
    def __init__(self):
        self.router = RegulatoryRouter()
        self.narrative_sections = {"INDICATIONS", "WARNINGS", "CONTRAINDICATIONS", "UNCLASSIFIED"}
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

    def chunk_sections(self, sections: dict, context: dict) -> List[dict]:
        all_chunks = []
        for section_name, text in sections.items():
            # Skip internal metadata or non-string sections
            if section_name.startswith("_") or not isinstance(text, str):
                continue
                
            if section_name in self.narrative_sections:
                splitter = self.narrative_splitter
            else:
                splitter = self.dense_splitter
            
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                metadata = context.copy()
                metadata.update({
                    "section": section_name,
                    "chunk_index": i
                })
                all_chunks.append({
                    "text": f"[Section: {section_name}] {chunk}",
                    "metadata": metadata
                })
        return all_chunks

    def process(self, pdf_path: str, context: dict) -> List[dict]:
        text = self.extract_text(pdf_path)
        tables = self.extract_tables_sane(pdf_path)
        
        if len(text) + len(tables) < 100:
            raise ValueError(f"Extraction too short (< 100 chars) for {pdf_path}")
            
        sections = self.router.parse(text)
        if tables.strip():
            sections["TABLES"] = tables.strip()
            
        return self.chunk_sections(sections, context)
