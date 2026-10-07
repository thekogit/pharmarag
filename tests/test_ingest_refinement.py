import pytest
from unittest.mock import MagicMock, patch
from src.ingest import PDFIngestor

def test_process_raises_value_error_on_short_content():
    ingestor = PDFIngestor()
    with patch.object(PDFIngestor, 'extract_text', return_value="short"):
        with patch.object(PDFIngestor, 'extract_tables_sane', return_value=""):
            with pytest.raises(ValueError, match="Extraction too short"):
                ingestor.process("mock.pdf", {})

def test_metadata_and_chunk_index_propagation():
    ingestor = PDFIngestor()
    context = {"source": "FDA", "drug": "Aspirin"}
    
    # Mocking router to return a simple section
    ingestor.router.parse = MagicMock(return_value={"INDICATIONS": "A" * 1500})
    
    with patch.object(PDFIngestor, 'extract_text', return_value="A" * 1500):
        with patch.object(PDFIngestor, 'extract_tables_sane', return_value=""):
            chunks = ingestor.process("mock.pdf", context)
            
            assert len(chunks) > 1
            for i, chunk in enumerate(chunks):
                assert chunk["metadata"]["source"] == "FDA"
                assert chunk["metadata"]["drug"] == "Aspirin"
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["text"].startswith("[Section: INDICATIONS]")

def test_splitter_selection_narrative():
    ingestor = PDFIngestor()
    # INDICATIONS should use narrative_splitter (chunk_size=1000)
    # 1500 chars should result in 2 chunks with 100 overlap
    text = "A" * 1500
    ingestor.router.parse = MagicMock(return_value={"INDICATIONS": text})
    
    with patch.object(PDFIngestor, 'extract_text', return_value=text):
        with patch.object(PDFIngestor, 'extract_tables_sane', return_value=""):
            chunks = ingestor.process("mock.pdf", {})
            # If narrative_splitter is used (size 1000, overlap 100):
            # Chunk 1: 0-1000
            # Chunk 2: 900-1500 (600 chars)
            assert len(chunks) == 2
            assert len(chunks[0]["text"]) > 1000 # [Section: INDICATIONS] + 1000

def test_splitter_selection_dense():
    ingestor = PDFIngestor()
    # CLINICAL_STUDIES should use dense_splitter (chunk_size=500)
    # 1500 chars should result in 3 chunks (no overlap)
    text = "A" * 1500
    ingestor.router.parse = MagicMock(return_value={"CLINICAL_STUDIES": text})
    
    with patch.object(PDFIngestor, 'extract_text', return_value=text):
        with patch.object(PDFIngestor, 'extract_tables_sane', return_value=""):
            chunks = ingestor.process("mock.pdf", {})
            # If dense_splitter is used (size 500, overlap 0):
            # Chunk 1: 500
            # Chunk 2: 500
            # Chunk 3: 500
            assert len(chunks) == 3

def test_tables_routed_to_tables_section_and_use_dense():
    ingestor = PDFIngestor()
    text = "Some normal text that is long enough to pass the guard." + "A" * 100
    tables = "Table data" * 100 # Long enough
    
    ingestor.router.parse = MagicMock(return_value={"UNCLASSIFIED": text})
    
    with patch.object(PDFIngestor, 'extract_text', return_value=text):
        with patch.object(PDFIngestor, 'extract_tables_sane', return_value=tables):
            chunks = ingestor.process("mock.pdf", {})
            
            table_chunks = [c for c in chunks if "[Section: TABLES]" in c["text"]]
            assert len(table_chunks) > 0
            # Check if dense splitter was likely used for TABLES
            # Tables use dense_splitter (size 500)
            for chunk in table_chunks:
                # [Section: TABLES] is 18 chars. 500 + 18 = 518
                assert len(chunk["text"]) <= 518
