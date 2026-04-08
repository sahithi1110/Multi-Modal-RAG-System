from fastapi import FastAPI, HTTPException
from app.schemas import AskRequest, AskResponse, IngestResponse
from app.dependencies import get_pipeline


app = FastAPI(title='Multi-Modal RAG System', version='1.0.0')


@app.get('/')
def home() -> dict:
    return {
        'message': 'Multi-Modal RAG System is running',
        'docs': '/docs',
        'health': '/health',
    }


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/ingest', response_model=IngestResponse)
def ingest_files() -> IngestResponse:
    pipeline = get_pipeline()
    summary = pipeline.ingest_all()
    return IngestResponse(**summary)


@app.post('/ask', response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    result = pipeline.answer_question(
        question=request.question,
        top_k=request.top_k,
        include_images=request.include_images,
    )
    if not result:
        raise HTTPException(status_code=500, detail='Could not build an answer')
    return AskResponse(**result)
