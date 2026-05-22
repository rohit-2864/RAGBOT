import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { PDFParse } from 'pdf-parse';
import * as lancedb from '@lancedb/lancedb';
import fs from 'fs-extra';
import { v4 as uuidv4 } from 'uuid';
import Tesseract from 'tesseract.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Ensure storage directories exist
const UPLOAD_DIR = process.env.UPLOAD_DIR || './uploads';
const LANCEDB_URI = process.env.LANCEDB_URI || './data/lancedb';
fs.ensureDirSync(UPLOAD_DIR);
fs.ensureDirSync(path.dirname(LANCEDB_URI));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${uuidv4()}${ext}`);
  }
});

const upload = multer({ storage: multer.memoryStorage() });
const diskUpload = multer({ storage: storage });

// Service URLs (local)
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const EMBED_SERVER_URL = process.env.EMBED_SERVER_URL || 'http://localhost:8000';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.2:3b';

async function startServer() {
  const app = express();
  const PORT = parseInt(process.env.PORT || '3000', 10);

  // Initialize LanceDB
  const db = await lancedb.connect(LANCEDB_URI);

  // Ensure table exists or create it
  // Table schema: id, vector, text, source, page, modality, storagePath
  let table: lancedb.Table;
  try {
    table = await db.openTable('chunks');
  } catch {
    // Create table with a dummy vector to define the schema if it doesn't exist
    table = await db.createTable('chunks', [
      {
        id: uuidv4(),
        vector: new Array(384).fill(0),
        text: 'initialization',
        source: 'system',
        page: 0,
        modality: 'text',
        storagePath: ''
      }
    ]);
    // Clean up the dummy entry
    await table.delete('source = "system"');
  }

  app.use(express.json({ limit: '50mb' }));
  app.use('/uploads', express.static(path.resolve(__dirname, UPLOAD_DIR)));

  // Health check
  app.get('/api/health', async (req, res) => {
    try {
      const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/tags`).catch(() => null);
      const ollamaOk = ollamaRes?.ok ?? false;
      const embedRes = await fetch(`${EMBED_SERVER_URL}/health`).catch(() => null);
      const embedOk = embedRes?.ok ?? false;

      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        services: {
          ollama: ollamaOk ? 'connected' : 'disconnected',
          embedServer: embedOk ? 'connected' : 'disconnected',
          lancedb: 'connected'
        },
        ollamaModel: OLLAMA_MODEL,
      });
    } catch {
      res.json({ status: 'ok', timestamp: new Date().toISOString() });
    }
  });

  // ---------------------------------------------------------------
  // File Management
  // ---------------------------------------------------------------

  // Upload file locally
  app.post('/api/upload', diskUpload.single('file'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    res.json({
      storagePath: req.file.filename,
      originalName: req.file.originalname
    });
  });

  // Get all documents (unique sources)
  app.get('/api/documents', async (req, res) => {
    try {
      // Query unique sources and metadata
      // LanceDB Node SDK doesn't have a direct "DISTINCT" yet, so we get all and filter in JS
      // or we can use a more efficient query if the dataset is large.
      const allMetadata = await table.query().select(['source', 'modality', 'storagePath']).toArray();

      const uniqueDocs = new Map();
      allMetadata.forEach(item => {
        if (!uniqueDocs.has(item.source)) {
          uniqueDocs.set(item.source, {
            id: item.source, // Using source as ID for simplicity in the list
            name: item.source,
            modality: item.modality,
            storagePath: item.storagePath
          });
        }
      });

      res.json(Array.from(uniqueDocs.values()));
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch documents' });
    }
  });

  // Add document chunks
  app.post('/api/documents', async (req, res) => {
    try {
      const { chunks } = req.body;
      if (!chunks || !Array.isArray(chunks)) {
        return res.status(400).json({ error: 'chunks array is required' });
      }

      const records = chunks.map(c => ({
        id: c.id || uuidv4(),
        vector: c.embedding,
        text: c.text,
        source: c.source,
        page: c.page || 0,
        modality: c.modality || 'text',
        storagePath: c.storagePath || ''
      }));

      await table.add(records);
      res.json({ status: 'success', count: records.length });
    } catch (error) {
      console.error('Failed to add chunks:', error);
      res.status(500).json({ error: 'Failed to save chunks to database' });
    }
  });

  // Process and ingest PDF documents (extract text, tables, images, and embeddings)
  app.post('/api/documents/process-pdf', async (req, res) => {
    try {
      const { storagePath, originalName } = req.body;
      if (!storagePath || !originalName) {
        return res.status(400).json({ error: 'storagePath and originalName are required' });
      }

      const filePath = path.resolve(__dirname, UPLOAD_DIR, storagePath);

      // Call Python embed server to process PDF layout, crop images, extract tables, and generate embeddings
      const embedServerRes = await fetch(`${EMBED_SERVER_URL}/process-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath: filePath, source: originalName }),
      });

      if (!embedServerRes.ok) {
        const errorMsg = await embedServerRes.text();
        throw new Error(`Python server error: ${errorMsg}`);
      }

      const { chunks } = await embedServerRes.json();
      
      if (!chunks || !Array.isArray(chunks)) {
        throw new Error('Invalid chunks returned from embedding server');
      }

      const records = chunks.map(c => ({
        id: c.id || uuidv4(),
        vector: c.embedding,
        text: c.text,
        source: c.source,
        page: c.page || 0,
        modality: c.modality || 'text',
        storagePath: c.storagePath || ''
      }));

      // Add to LanceDB
      await table.add(records);

      res.json({ status: 'success', count: records.length });
    } catch (error: any) {
      console.error('Failed to process PDF:', error);
      res.status(500).json({ error: 'Failed to process and save PDF document', details: error.message });
    }
  });

  // Delete document
  app.delete('/api/documents/:id', async (req, res) => {
    try {
      const source = req.params.id;

      // Find the storage path to delete the physical file
      const doc = await table.query().where(`source = "${source}"`).limit(1).toArray();
      if (doc.length > 0 && doc[0].storagePath) {
        const filePath = path.resolve(__dirname, UPLOAD_DIR, doc[0].storagePath);
        await fs.remove(filePath).catch(err => console.warn(`Failed to delete file: ${filePath}`, err));
      }

      await table.delete(`source = "${source}"`);
      res.json({ status: 'success' });
    } catch (error) {
      res.status(500).json({ error: 'Failed to delete document' });
    }
  });

  // ---------------------------------------------------------------
  // Vector Search
  // ---------------------------------------------------------------
  app.post('/api/search', async (req, res) => {
    try {
      const { queryVector, topK = 5 } = req.body;
      if (!queryVector || !Array.isArray(queryVector)) {
        return res.status(400).json({ error: 'queryVector is required' });
      }

      const results = await table.search(queryVector).limit(topK).toArray();

      res.json(results.map(r => ({
        id: r.id,
        text: r.text,
        source: r.source,
        page: r.page,
        modality: r.modality,
        storagePath: r.storagePath,
        score: r._distance // LanceDB returns distance (lower is better)
      })));
    } catch (error) {
      console.error('Search error:', error);
      res.status(500).json({ error: 'Search failed' });
    }
  });

  // ---------------------------------------------------------------
  // PDF Text Extraction
  // ---------------------------------------------------------------
  app.post('/api/extract', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

      let text = '';
      const mimetype = req.file.mimetype;

      if (mimetype === 'application/pdf') {
        const parser = new PDFParse({ data: req.file.buffer });
        const result = await parser.getText();
        text = result.text;
        await parser.destroy();
      } else if (mimetype.startsWith('image/')) {
        // OCR for images
        const { data: { text: ocrText } } = await Tesseract.recognize(
          req.file.buffer,
          'eng'
        );
        text = ocrText;
      } else {
        text = req.file.buffer.toString('utf-8');
      }

      res.json({ text, fileName: req.file.originalname });
    } catch (error) {
      console.error('Extraction error:', error);
      res.status(500).json({ error: 'Failed to extract text' });
    }
  });

  // ---------------------------------------------------------------
  // Chat via Ollama
  // ---------------------------------------------------------------
  app.post('/api/chat', async (req, res) => {
    try {
      const { prompt, images } = req.body;
      const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: OLLAMA_MODEL,
          prompt,
          images: images || [],
          stream: true,
          options: {

            temperature: 0.1,
          }
        }),
      });

      if (!ollamaRes.ok) throw new Error(`Ollama error: ${await ollamaRes.text()}`);

      // Pipe the stream from Ollama to the client
      if (!ollamaRes.body) throw new Error('Failed to get body from Ollama response');

      // Set headers for streaming
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      try {
        // @ts-ignore - body is a ReadableStream in Node.js fetch
        for await (const chunk of ollamaRes.body) {
          res.write(chunk);
        }
      } catch (streamErr) {
        console.error('Error during streaming:', streamErr);
      } finally {
        res.end();
      }
    } catch (error: any) {
      console.error('Chat error:', error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Chat failed', details: error.message });
      } else {
        res.end();
      }
    }
  });

  // ---------------------------------------------------------------
  // Embedding Proxies
  // ---------------------------------------------------------------
  app.post('/api/embed/text', async (req, res) => {
    try {
      const { text } = req.body;
      const embedRes = await fetch(`${EMBED_SERVER_URL}/embed/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      res.json(await embedRes.json());
    } catch (error) {
      res.status(500).json({ error: 'Text embedding failed' });
    }
  });

  app.post('/api/embed/image', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

      const formData = new FormData();
      // Pass the buffer as a Blob with the correct MIME type
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append('file', blob, req.file.originalname);

      const embedRes = await fetch(`${EMBED_SERVER_URL}/embed/image`, {
        method: 'POST',
        body: formData
      });

      if (!embedRes.ok) {
        const errorText = await embedRes.text();
        console.error('Embedding server error:', errorText);
        throw new Error(`Embedding server returned ${embedRes.status}: ${errorText}`);
      }

      res.json(await embedRes.json());
    } catch (error: any) {
      console.error('Image embedding proxy error:', error);
      res.status(500).json({ error: 'Image embedding failed', details: error.message });
    }
  });


  // Error handler
  app.use('/api', (err: any, req: any, res: any, next: any) => {
    res.status(500).json({ error: 'Internal Server Error', details: err.message });
  });

  // Vite middleware
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: 'spa' });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => res.sendFile(path.resolve(distPath, 'index.html')));
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
