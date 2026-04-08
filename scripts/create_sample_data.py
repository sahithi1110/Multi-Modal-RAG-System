from pathlib import Path
import json
from PIL import Image, ImageDraw


project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / 'data' / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)

manual_text = """
The inspection platform supports both text-based defect logs and image-based defect validation.
Operators can upload visual evidence from the assembly line.
The retrieval system combines semantic matching with keyword search for better recall.
A re-ranking layer improves final ordering of evidence before answer generation.
The production API target is below 200 milliseconds for common cached requests.
Guardrails are used to reduce unsupported answers when evidence is weak.
""".strip()

architecture_text = """
The system uses a hybrid search stack.
Dense vectors are stored in a FAISS index.
Sparse keyword matching is handled with BM25.
Image retrieval uses CLIP embeddings.
Query rewriting expands abbreviations and improves the search phrase before retrieval.
""".strip()

(raw_dir / 'product_manual.txt').write_text(manual_text, encoding='utf-8')
(raw_dir / 'architecture_notes.txt').write_text(architecture_text, encoding='utf-8')

sample_questions = [
    {
        'question': 'How does the system reduce hallucinations?',
        'expected_topic': 'guardrails and evidence grounding'
    },
    {
        'question': 'Does the system support image retrieval?',
        'expected_topic': 'multi-modal retrieval with CLIP'
    }
]
(project_root / 'data' / 'sample_questions.json').write_text(json.dumps(sample_questions, indent=2), encoding='utf-8')

image = Image.new('RGB', (900, 520), color='white')
draw = ImageDraw.Draw(image)
draw.rectangle((60, 80, 840, 430), outline='black', width=4)
draw.text((90, 120), 'Visual Defect Inspection Flow', fill='black')
draw.text((90, 180), 'Camera capture -> image embeddings -> hybrid retrieval', fill='black')
draw.text((90, 240), 'Re-ranking -> grounded response -> lower hallucination risk', fill='black')
draw.text((90, 300), 'Real-time inference target: under 200ms', fill='black')
image.save(raw_dir / 'inspection_pipeline.png')

print('Sample data created in data/raw')
