from pathlib import Path
from typing import List, Dict
from PIL import Image
import fitz


image_suffixes = {'.png', '.jpg', '.jpeg', '.webp'}


def find_standalone_images(folder_path: Path) -> List[Path]:
    found = []
    for item in folder_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in image_suffixes:
            found.append(item)
    return found


def extract_images_from_pdf(pdf_path: Path, save_folder: Path) -> List[Dict]:
    save_folder.mkdir(parents=True, exist_ok=True)
    pdf_doc = fitz.open(pdf_path)
    saved_items = []

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images(full=True)
        for image_index, image_info in enumerate(image_list):
            xref = image_info[0]
            image_dict = pdf_doc.extract_image(xref)
            image_bytes = image_dict['image']
            image_ext = image_dict.get('ext', 'png')
            file_name = f'{pdf_path.stem}_page_{page_index + 1}_{image_index + 1}.{image_ext}'
            file_path = save_folder / file_name
            file_path.write_bytes(image_bytes)
            saved_items.append(
                {
                    'path': str(file_path),
                    'source_name': pdf_path.name,
                    'page_number': page_index + 1,
                }
            )
    pdf_doc.close()
    return saved_items


def read_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert('RGB')
