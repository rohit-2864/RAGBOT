"""
Multimodal Embedding Server
FastAPI server that loads multi-modal-embed-small and exposes REST endpoints
for text and image embedding generation.

Model: llm-semantic-router/multi-modal-embed-small (~120M params)
- Text encoder: MiniLM-L6-v2 (22M params) → 384-dim vectors
- Image encoder: SigLIP-base-patch16-512 (86M params) → 384-dim vectors

All modalities share the same 384-dimensional embedding space.
"""

import io
import logging
import os
import uuid
import fitz  # PyMuPDF
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
)
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definition (from the official model card)
# ---------------------------------------------------------------------------
class MultiModalEmbedder(nn.Module):
    """Standalone multimodal embedder matching the HF model card exactly."""

    def __init__(self):
        super().__init__()

        # Text encoder (384d, no projection needed)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Image encoder (768d -> 384d projection)
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-512"
        ).vision_model
        self.image_proj = nn.Linear(768, 384)


    @torch.no_grad()
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def encode_image(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)  # 768 -> 384
        return F.normalize(embeddings, p=2, dim=-1)



# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model() -> MultiModalEmbedder:
    """Download and load the multi-modal-embed-small model with trained weights."""
    logger.info("Initializing MultiModalEmbedder...")
    model = MultiModalEmbedder()

    logger.info("Downloading trained weights from HuggingFace Hub...")
    checkpoint_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small",
        filename="model.pt",
    )

    logger.info("Loading state dict...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load text encoder weights
    model.text_encoder.load_state_dict(
        {
            k.replace("text_encoder.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.encoder.")
        }
    )

    # Load image encoder and projection weights
    model.image_encoder.load_state_dict(
        {
            k.replace("image_encoder.vision_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.vision_encoder.")
        }
    )
    model.image_proj.load_state_dict(
        {
            k.replace("image_encoder.projection.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.projection.")
        }
    )


    model.eval()
    #logger.info("Model loaded successfully!")
    print("Model loaded successfully!")
    return model


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
_model: MultiModalEmbedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    global _model
    _model = load_model()
    yield
    # Cleanup
    _model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


app = FastAPI(
    title="Multimodal Embedding Server",
    description="Generates 384-dim embeddings for text and images using multi-modal-embed-small",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Express server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "multi-modal-embed-small",
        "embedding_dim": 384,
        "modalities": ["text", "image"],
    }


@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(req: TextRequest):
    """Generate a 384-dim embedding for a text string."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        embedding = _model.encode_text(req.text)
        vec = embedding.squeeze().tolist()
        return EmbeddingResponse(embedding=vec, dimensions=len(vec))
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate a 384-dim embedding for an uploaded image (JPG/PNG/WebP)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"Expected image file, got {file.content_type}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = _model.encode_image(image)
        vec = embedding.squeeze().tolist()
        return EmbeddingResponse(embedding=vec, dimensions=len(vec))
    except Exception as e:
        logger.error(f"Image embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for PDF processing
def format_markdown_table(grid):
    if not grid or not grid[0]:
        return ""
    cleaned_grid = []
    for r in grid:
        cleaned_row = []
        for cell in r:
            val = str(cell or "").strip().replace("\n", " ").replace("|", "\\|")
            cleaned_row.append(val)
        cleaned_grid.append(cleaned_row)
    
    headers = cleaned_grid[0]
    rows = cleaned_grid[1:]
    
    if not any(headers):
        return ""
        
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    markdown_rows = []
    for row in rows:
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        markdown_rows.append("| " + " | ".join(row) + " |")
        
    return "\n".join([header_line, separator_line] + markdown_rows)


def chunk_text_helper(text, max_words=500, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += (max_words - overlap)
    return chunks


class ProcessPDFRequest(BaseModel):
    filepath: str
    source: str


@app.post("/process-pdf")
async def process_pdf(req: ProcessPDFRequest):
    """
    Extract text, tables, and images from a PDF file.
    Generate embeddings for all extracted chunks and return them.
    """
    filepath = req.filepath
    source = req.source
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail=f"File not found: {filepath}")
        
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open PDF: {str(e)}")
        
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    chunks_to_encode = []
    
    try:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            
            # 1. Extract and format tables
            try:
                tables = page.find_tables()
                for t_idx, table in enumerate(tables):
                    grid = table.extract()
                    table_md = format_markdown_table(grid)
                    if table_md.strip():
                        chunks_to_encode.append({
                            "type": "table",
                            "text": table_md,
                            "page": page_num,
                            "storagePath": ""
                        })
            except Exception as table_err:
                logger.error(f"Error extracting tables on page {page_num}: {table_err}")
                
            # 2. Extract images
            try:
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        width, height = pil_img.size
                        if width < 100 or height < 100:
                            continue
                    except Exception:
                        continue
                    
                    filename = f"{uuid.uuid4()}.{image_ext}"
                    dest_path = os.path.join(UPLOAD_DIR, filename)
                    with open(dest_path, "wb") as f:
                        f.write(image_bytes)
                        
                    chunks_to_encode.append({
                        "type": "image",
                        "text": f"[Image on Page {page_num} of {source}]",
                        "page": page_num,
                        "storagePath": filename,
                        "pil_image": pil_img
                    })
            except Exception as img_err:
                logger.error(f"Error extracting images on page {page_num}: {img_err}")
                
            # 3. Extract text content
            try:
                text = page.get_text()
                if text.strip():
                    page_text_chunks = chunk_text_helper(text, max_words=500, overlap=100)
                    for chunk_txt in page_text_chunks:
                        chunks_to_encode.append({
                            "type": "text",
                            "text": chunk_txt,
                            "page": page_num,
                            "storagePath": ""
                        })
            except Exception as text_err:
                logger.error(f"Error extracting text on page {page_num}: {text_err}")
                
        # Batch-generate embeddings
        BATCH_SIZE = 16
        
        # 1. Text & Table Chunks
        text_and_table_chunks = [c for c in chunks_to_encode if c["type"] in ("text", "table")]
        if text_and_table_chunks:
            all_texts = [c["text"] for c in text_and_table_chunks]
            text_embeddings = []
            for i in range(0, len(all_texts), BATCH_SIZE):
                batch = all_texts[i:i + BATCH_SIZE]
                embeddings = _model.encode_text(batch)
                text_embeddings.extend(embeddings.tolist())
                
            for idx, chunk in enumerate(text_and_table_chunks):
                chunk["embedding"] = text_embeddings[idx]
                
        # 2. Image Chunks
        image_chunks = [c for c in chunks_to_encode if c["type"] == "image"]
        if image_chunks:
            all_images = [c["pil_image"] for c in image_chunks]
            image_embeddings = []
            for i in range(0, len(all_images), BATCH_SIZE):
                batch = all_images[i:i + BATCH_SIZE]
                embeddings = _model.encode_image(batch)
                image_embeddings.extend(embeddings.tolist())
                
            for idx, chunk in enumerate(image_chunks):
                chunk["embedding"] = image_embeddings[idx]
                del chunk["pil_image"]
                
        # Format response
        response_chunks = []
        for c in chunks_to_encode:
            response_chunks.append({
                "id": str(uuid.uuid4()),
                "text": c["text"],
                "source": source,
                "page": c["page"],
                "embedding": c["embedding"],
                "modality": "image" if c["type"] == "image" else "text",
                "storagePath": c["storagePath"]
            })
            
        return {"chunks": response_chunks}
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))





# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
