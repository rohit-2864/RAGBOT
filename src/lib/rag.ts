/**
 * RAG (Retrieval-Augmented Generation) Library
 * 
 * Handles text chunking and embedding generation via local servers.
 * Vector storage and retrieval are handled by the LanceDB backend.
 * All embeddings are 384-dimensional vectors in a shared semantic space.
 */

export interface DocumentChunk {
  id: string;
  text: string;
  source: string;
  page?: number;
  embedding?: number[];
  modality?: 'text' | 'image';
  storagePath?: string;
}

/**
 * Split text into overlapping chunks of ~chunkSize words.
 */
export function chunkText(text: string, source: string, chunkSize: number = 600, overlap: number = 100): DocumentChunk[] {
  const chunks: DocumentChunk[] = [];
  const words = text.split(/\s+/);
  
  let i = 0;
  while (i < words.length) {
    const chunkWords = words.slice(i, i + chunkSize);
    const chunkText = chunkWords.join(' ');
    
    chunks.push({
      id: crypto.randomUUID(),
      text: chunkText,
      source: source,
      page: Math.floor(i / 500) + 1,
      modality: 'text',
    });
    
    i += (chunkSize - overlap);
  }
  
  return chunks;
}

/**
 * Generate a 384-dim text embedding via the local embedding server.
 */
export async function generateEmbeddings(text: string): Promise<number[]> {
  const response = await fetch('/api/embed/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(`Text embedding failed: ${err.error || err.details || response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Generate a 384-dim image embedding via the local embedding server.
 */
export async function generateImageEmbedding(file: File): Promise<number[]> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/embed/image', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(`Image embedding failed: ${err.error || err.details || response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

