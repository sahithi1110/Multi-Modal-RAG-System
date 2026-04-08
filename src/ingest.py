from pathlib import Path
from typing import Dict, List
import fitz
from rank_bm25 import BM25Okapi

from src.text_utils import clean_text, split_into_chunks
from src.image_utils import find_standalone_images, extract_images_from_pdf, read_image
from src.multimodal_embedder import MultiModalEmbedder
from src.storage import IndexStorage


class DataIngestor:
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path, storage: IndexStorage, embedder: MultiModalEmbedder):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.storage = storage
        self.embedder = embedder

    def run(self) -> Dict:
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        image_dump_folder = self.processed_data_dir / 'pdf_images'
        image_dump_folder.mkdir(parents=True, exist_ok=True)

        text_records: List[dict] = []
        image_records: List[dict] = []

        for file_path in self.raw_data_dir.rglob('*'):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix == '.txt':
                text_records.extend(self._read_text_file(file_path))
            elif suffix == '.pdf':
                pdf_text_records, pdf_image_records = self._read_pdf_file(file_path, image_dump_folder)
                text_records.extend(pdf_text_records)
                image_records.extend(pdf_image_records)

        for image_path in find_standalone_images(self.raw_data_dir):
            image_records.append(
                {
                    'item_id': f'image::{image_path.name}',
                    'item_type': 'image',
                    'source_name': image_path.name,
                    'page_number': None,
                    'text': f'Image file named {image_path.stem.replace("_", " ")}',
                    'path': str(image_path),
                }
            )

        self._build_text_assets(text_records)
        self._build_image_assets(image_records)

        return {
            'success': True,
            'message': 'Ingestion completed',
            'text_items': len(text_records),
            'image_items': len(image_records),
        }

    def _read_text_file(self, file_path: Path) -> List[dict]:
        text = clean_text(file_path.read_text(encoding='utf-8', errors='ignore'))
        chunks = split_into_chunks(text)
        return [
            {
                'item_id': f'text::{file_path.name}::{index}',
                'item_type': 'text',
                'source_name': file_path.name,
                'page_number': None,
                'text': chunk,
                'path': str(file_path),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]

    def _read_pdf_file(self, file_path: Path, image_dump_folder: Path):
        pdf_doc = fitz.open(file_path)
        text_records = []
        for page_index in range(len(pdf_doc)):
            page = pdf_doc[page_index]
            cleaned_page_text = clean_text(page.get_text())
            for chunk_index, chunk in enumerate(split_into_chunks(cleaned_page_text), start=1):
                text_records.append(
                    {
                        'item_id': f'text::{file_path.name}::page_{page_index + 1}::chunk_{chunk_index}',
                        'item_type': 'text',
                        'source_name': file_path.name,
                        'page_number': page_index + 1,
                        'text': chunk,
                        'path': str(file_path),
                    }
                )
        pdf_doc.close()

        raw_image_records = extract_images_from_pdf(file_path, image_dump_folder)
        image_records = []
        for item_index, image_info in enumerate(raw_image_records, start=1):
            image_records.append(
                {
                    'item_id': f'image::{file_path.name}::{item_index}',
                    'item_type': 'image',
                    'source_name': image_info['source_name'],
                    'page_number': image_info['page_number'],
                    'text': f'Image extracted from {file_path.name} page {image_info["page_number"]}',
                    'path': image_info['path'],
                }
            )
        return text_records, image_records

    def _build_text_assets(self, text_records: List[dict]) -> None:
        text_chunks = [item['text'] for item in text_records]
        vector_matrix = self.embedder.encode_text(text_chunks)
        self.storage.save_faiss_index('text_vectors', vector_matrix)
        tokenized_chunks = [chunk.lower().split() for chunk in text_chunks]
        bm25_model = BM25Okapi(tokenized_chunks) if tokenized_chunks else None
        if bm25_model is not None:
            self.storage.save_pickle('text_bm25', bm25_model)
        self.storage.save_metadata('text_items', text_records)

    def _build_image_assets(self, image_records: List[dict]) -> None:
        loaded_images = []
        kept_image_records = []
        for item in image_records:
            try:
                loaded_images.append(read_image(Path(item['path'])))
                kept_image_records.append(item)
            except Exception:
                continue
        vector_matrix = self.embedder.encode_images(loaded_images)
        self.storage.save_faiss_index('image_vectors', vector_matrix)
        self.storage.save_metadata('image_items', kept_image_records)
