import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Upload, 
  FileText, 
  Trash2, 
  Loader2, 
  MessageSquare, 
  Info, 
  ThumbsUp, 
  ThumbsDown,
  ChevronRight,
  Database,
  History,
  Image as ImageIcon,
  Paperclip,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  chunkText,
  generateEmbeddings,
  generateImageEmbedding,
  type DocumentChunk,
} from './lib/rag';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: DocumentChunk[];
  timestamp: Date;
  feedback?: 'up' | 'down';
}

interface Document {
  id: string;
  name: string;
  chunks: DocumentChunk[];
  storagePath?: string;
  modality: 'text' | 'image';
}

type ServiceStatus = 'checking' | 'connected' | 'disconnected';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const [ollamaStatus, setOllamaStatus] = useState<ServiceStatus>('checking');
  const [embedStatus, setEmbedStatus] = useState<ServiceStatus>('checking');

  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadUserData();
    checkServices();
  }, []);

  // Check health of local services
  const checkServices = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setOllamaStatus(data.services?.ollama === 'connected' ? 'connected' : 'disconnected');
        setEmbedStatus(data.services?.embedServer === 'connected' ? 'connected' : 'disconnected');
      }
    } catch {
      setOllamaStatus('disconnected');
      setEmbedStatus('disconnected');
    }
  };

  const loadUserData = async () => {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) throw new Error('Failed to load documents');
      const docsData = await res.json();
      
      setDocuments(docsData.map((d: any) => ({
        id: d.id,
        name: d.name,
        chunks: [], // Chunks are stored in LanceDB and retrieved via search
        storagePath: d.storagePath,
        modality: d.modality || 'text',
      })));
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  // ---------------------------------------------------------------
  // Detect file modality from MIME type
  // ---------------------------------------------------------------
  function detectModality(mimeType: string): 'text' | 'image' {
    if (mimeType.startsWith('image/')) return 'image';
    return 'text';
  }

  // ---------------------------------------------------------------
  // File Upload Handler (multimodal)
  // ---------------------------------------------------------------
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const modality = detectModality(file.type);
    
    try {
      let chunksWithEmbeddings: DocumentChunk[] = [];
      const fileName = file.name;

      // 1. Upload file locally
      setUploadProgress('Uploading file...');
      const uploadFormData = new FormData();
      uploadFormData.append('file', file);
      const uploadRes = await fetch('/api/upload', {
        method: 'POST',
        body: uploadFormData,
      });
      if (!uploadRes.ok) throw new Error('Upload failed');
      const { storagePath } = await uploadRes.json();

      // 2. Process and embed
      if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
        setUploadProgress('Processing PDF layout (extracting text, tables, and images)...');
        const processRes = await fetch('/api/documents/process-pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ storagePath, originalName: fileName }),
        });

        if (!processRes.ok) {
          const err = await processRes.json().catch(() => ({ error: 'Processing failed' }));
          throw new Error(err.error || err.details || 'Failed to process PDF');
        }
      } else {
        if (modality === 'text') {
          setUploadProgress('Extracting text...');
          const extractFormData = new FormData();
          extractFormData.append('file', file);

          const response = await fetch('/api/extract', {
            method: 'POST',
            body: extractFormData,
          });

          if (!response.ok) throw new Error('Failed to extract text from file.');
          const { text } = await response.json();
          
          setUploadProgress('Chunking and Embedding...');
          const chunks = chunkText(text, fileName);
          
          for (let i = 0; i < chunks.length; i++) {
            setUploadProgress(`Embedding chunk ${i + 1}/${chunks.length}...`);
            const embedding = await generateEmbeddings(chunks[i].text);
            chunksWithEmbeddings.push({ ...chunks[i], embedding, storagePath });
          }

        } else if (modality === 'image') {
          setUploadProgress('Extracting text (OCR)...');
          const extractFormData = new FormData();
          extractFormData.append('file', file);
          
          let ocrText = '';
          try {
            const extractRes = await fetch('/api/extract', {
              method: 'POST',
              body: extractFormData,
            });
            if (extractRes.ok) {
              const data = await extractRes.json();
              ocrText = data.text;
            }
          } catch (e) {
            console.warn("OCR failed, continuing with embedding only", e);
          }

          setUploadProgress('Generating image embedding...');
          const embedding = await generateImageEmbedding(file);
          chunksWithEmbeddings = [{
            id: crypto.randomUUID(),
            text: ocrText || `[Image: ${fileName}]`,
            source: fileName,
            page: 1,
            embedding,
            modality: 'image',
            storagePath
          }];

        }
        // 3. Save to LanceDB
        setUploadProgress('Saving to knowledge base...');
        const dbRes = await fetch('/api/documents', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ chunks: chunksWithEmbeddings }),
        });

        if (!dbRes.ok) throw new Error('Failed to save to database');
      }

      setDocuments(prev => [...prev, { 
        id: fileName, 
        name: fileName, 
        chunks: [], // We don't keep chunks in state anymore
        storagePath,
        modality,
      }]);
      
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Failed to process ${modality} file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeDocument = async (id: string, storagePath?: string) => {
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setDocuments(prev => prev.filter(doc => doc.id !== id));
      }
    } catch (error) {
      console.error("Failed to delete document:", error);
    }
  };

  // ---------------------------------------------------------------
  // Chat via Ollama
  // ---------------------------------------------------------------
  const handleSend = async () => {
    if (!input.trim() || isThinking) return;

    const query = input;
    setInput('');
    setIsThinking(true);

    try {
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: query,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMessage]);

      let context = '';
      let relevantChunks: DocumentChunk[] = [];

      if (documents.length > 0) {
        // 1. Generate query embedding
        const queryEmbedding = await generateEmbeddings(query);

        // 2. Search LanceDB via backend
        const searchRes = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ queryVector: queryEmbedding, topK: 5 }),
        });

        if (searchRes.ok) {
          relevantChunks = await searchRes.json();
          context = relevantChunks.map(c => {
            if (c.modality === 'image' && c.storagePath) {
              return `[Source Image: ${c.source}, Page: ${c.page}, Reference: /uploads/${c.storagePath}]\nThis is a relevant visual diagram/image from the document. If the user asks for diagram, image, visual details, or the matched topic, you MUST render this image inline in your response by outputting exactly this markdown tag: ![Image from ${c.source}](/uploads/${c.storagePath})`;
            }
            const modalityLabel = c.modality ? ` (${c.modality})` : '';
            return `[Source: ${c.source}${modalityLabel}, Page: ${c.page}]\n${c.text}`;
          }).join('\n\n');
        }
      }

      const historyContext = messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n');

      const prompt = `You are a helpful and accurate RAG chatbot. Use the provided context and conversation history to answer the user's query.

CONSTRAINTS:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say "I don't know. The provided documents do not contain information about this."
3. Be concise and professional.
4. Maintain a coherent conversation based on the history.
5. For image sources, describe what was matched and why it's relevant.
6. Crucially, if the user asks to see an image, diagram, or chart, and there is a relevant "[Source Image]" in the RETRIEVED CONTEXT, you MUST show it inline in your response using the exact markdown image tag provided in the context (e.g. ![Image from source](/uploads/path.png)). Do NOT say you cannot show images.

CONVERSATION HISTORY:
${historyContext}

RETRIEVED CONTEXT:
${context || 'No documents uploaded yet.'}

USER QUERY:
${query}

ASSISTANT RESPONSE:`;

      let base64Images: string[] = [];
      const imageChunks = relevantChunks.filter(c => c.modality === 'image' && c.storagePath);
      
      for (const chunk of imageChunks) {
        try {
          // Fetch image from local backend
          const res = await fetch(`/uploads/${chunk.storagePath}`);
          if (res.ok) {
            const blob = await res.blob();
            const reader = new FileReader();
            const base64 = await new Promise<string>((resolve) => {
              reader.onloadend = () => resolve(reader.result as string);
              reader.readAsDataURL(blob);
            });
            base64Images.push(base64.split(',')[1]); // Only the data part
          }
        } catch (e) {
          console.error("Failed to fetch local image for vision model:", e);
        }
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, images: base64Images }),
      });

      if (!response.ok) throw new Error('Chat request failed');

      // Create an empty assistant message to start streaming into
      const assistantMessageId = crypto.randomUUID();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        sources: relevantChunks,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          // Ollama sends JSON objects separated by newlines
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const data = JSON.parse(line);
              if (data.response) {
                fullResponse += data.response;
                // Update the last message content
                setMessages(prev => prev.map(m => 
                  m.id === assistantMessageId ? { ...m, content: fullResponse } : m
                ));
              }
            } catch (e) {
              console.warn("Failed to parse stream chunk:", e);
            }
          }
        }
      }

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get a response.'}`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleFeedback = (messageId: string, feedback: 'up' | 'down') => {
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback } : m));
  };

  function ModalityIcon({ modality, className }: { modality: string; className?: string }) {
    switch (modality) {
      case 'image': return <ImageIcon className={cn("w-4 h-4", className)} />;
      default: return <FileText className={cn("w-4 h-4", className)} />;
    }
  }

  function StatusDot({ status }: { status: ServiceStatus }) {
    if (status === 'checking') return <Loader2 className="w-3 h-3 animate-spin text-yellow-500" />;
    if (status === 'connected') return <CheckCircle2 className="w-3 h-3 text-green-500" />;
    return <XCircle className="w-3 h-3 text-red-500" />;
  }

  return (
    <div className="flex h-screen w-full bg-[#F8FAFC] text-slate-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 flex flex-col bg-white border-r border-slate-200 overflow-hidden flex-shrink-0 shadow-sm z-20">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-200">
              <Database className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight text-slate-800 leading-tight">Knowledge Base</h1>
              <p className="text-[10px] text-indigo-600 font-bold uppercase tracking-widest">Multimodal RAG</p>
            </div>
          </div>

          <div className="flex gap-3 mb-6 text-[10px] font-medium">
            <div className="flex items-center gap-2 px-2.5 py-1.5 bg-slate-50 rounded-lg border border-slate-100">
              <StatusDot status={ollamaStatus} />
              <span className="text-slate-600 font-semibold tracking-tight uppercase">Ollama</span>
            </div>
            <div className="flex items-center gap-2 px-2.5 py-1.5 bg-slate-50 rounded-lg border border-slate-100">
              <StatusDot status={embedStatus} />
              <span className="text-slate-600 font-semibold tracking-tight uppercase">Embeddings</span>
            </div>
          </div>
          
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 py-3.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 active:scale-[0.98] transition-all duration-200 disabled:opacity-50 shadow-md shadow-indigo-100"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="text-xs font-bold uppercase tracking-wider">Upload File</span>
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept=".pdf,.txt,.jpg,.jpeg,.png,.webp"
          />
          {isUploading && (
            <div className="mt-3 text-[10px] font-bold text-indigo-600 animate-pulse text-center bg-indigo-50 py-2 rounded-lg border border-indigo-100">
              {uploadProgress}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar bg-slate-50/30">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-4 px-2">Knowledge Sources</div>
          {documents.length === 0 ? (
            <div className="px-4 py-12 text-center border-2 border-dashed border-slate-200 rounded-2xl bg-white/50">
              <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Info className="w-6 h-6 text-slate-300" />
              </div>
              <p className="text-xs text-slate-400 italic">No documents uploaded yet.</p>
            </div>
          ) : (
            documents.map((doc) => (
              <motion.div 
                key={doc.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="group flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-white hover:border-indigo-300 hover:shadow-lg hover:shadow-indigo-500/5 transition-all duration-200"
              >
                <div className="flex items-center gap-3 overflow-hidden">
                  <div className="p-2 bg-slate-50 rounded-lg group-hover:bg-indigo-50 transition-colors">
                    <ModalityIcon modality={doc.modality} className="text-slate-400 group-hover:text-indigo-600" />
                  </div>
                  <div className="overflow-hidden">
                    <span className="text-xs font-bold text-slate-700 truncate block">{doc.name}</span>
                    <span className="text-[9px] text-slate-400 uppercase font-black tracking-widest">{doc.modality}</span>
                  </div>
                </div>
                <button 
                  onClick={() => removeDocument(doc.id, doc.storagePath)}
                  className="p-1.5 text-slate-300 hover:text-rose-500 hover:bg-rose-50 rounded-lg transition-all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </motion.div>
            ))
          )}
        </div>

        <div className="p-5 border-t border-slate-200 bg-white">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-[9px] text-slate-400 font-black uppercase tracking-widest mb-1">Messages</div>
              <div className="text-lg font-bold text-slate-800 leading-tight">{messages.length}</div>
            </div>
            <div className="text-center border-x border-slate-100">
              <div className="text-[9px] text-slate-400 font-black uppercase tracking-widest mb-1">Docs</div>
              <div className="text-lg font-bold text-slate-800 leading-tight">{documents.length}</div>
            </div>
            <div className="text-center">
              <div className="text-[9px] text-slate-400 font-black uppercase tracking-widest mb-1">System</div>
              <div className="text-[10px] font-black mt-1 text-emerald-500">READY</div>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col bg-white overflow-hidden relative">
        <header className="h-16 border-b border-slate-200 flex items-center px-10 justify-between bg-white/80 backdrop-blur-xl z-10 sticky top-0">
          <div className="flex items-center gap-4">
            <div className={cn(
              "w-2.5 h-2.5 rounded-full ring-4 ring-white shadow-sm",
              ollamaStatus === 'connected' ? "bg-emerald-500 animate-pulse shadow-emerald-200" : "bg-rose-500 shadow-rose-200"
            )} />
            <h2 className="font-bold text-slate-800 tracking-tight flex items-center gap-2">
              RAGBOT 
              <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded-full text-slate-500 font-black tracking-widest">v1.0</span>
            </h2>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">{new Date().toLocaleDateString()}</div>
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-10 space-y-8 scroll-smooth custom-scrollbar bg-[#F8FAFC]">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center max-w-md mx-auto text-center">
              <div className="p-4 bg-indigo-50 rounded-3xl mb-6">
                <MessageSquare className="w-10 h-10 text-indigo-600" />
              </div>
              <h3 className="font-bold text-2xl text-slate-800 mb-2">Welcome to RAGBOT</h3>
              <p className="text-sm text-slate-500 leading-relaxed">
                Your private, local multimodal assistant. Upload PDFs, images, or text to start a contextual conversation.
              </p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div 
                key={msg.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn("flex flex-col max-w-[85%]", msg.role === 'user' ? "ml-auto items-end" : "mr-auto items-start")}
              >
                <div className={cn(
                  "px-5 py-3.5 rounded-2xl shadow-sm relative group",
                  msg.role === 'user' 
                    ? "chat-gradient text-white rounded-tr-none" 
                    : "bg-white text-slate-800 border border-slate-100 rounded-tl-none"
                )}>
                  {msg.role === 'user' ? (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="text-sm leading-relaxed markdown-container">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                  
                  {msg.role === 'assistant' && (
                    <div className="absolute -right-12 top-0 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => handleFeedback(msg.id, 'up')} className={cn("p-1.5 rounded-lg border border-slate-200 transition-all hover:border-indigo-300", msg.feedback === 'up' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-400 hover:text-indigo-600")}>
                        <ThumbsUp className="w-3 h-3" />
                      </button>
                      <button onClick={() => handleFeedback(msg.id, 'down')} className={cn("p-1.5 rounded-lg border border-slate-200 transition-all hover:border-rose-300", msg.feedback === 'down' ? "bg-rose-500 text-white border-rose-500" : "bg-white text-slate-400 hover:text-rose-500")}>
                        <ThumbsDown className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 w-full">
                    <div className="flex items-center gap-2 mb-2 px-1">
                      <ChevronRight className="w-3 h-3 text-indigo-500" />
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Sources Found</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {msg.sources.map((source, idx) => (
                        <div key={idx} className="p-2.5 bg-white/50 border border-slate-200/50 rounded-xl text-[10px] shadow-sm flex flex-col justify-between">
                          <div>
                            <div className="flex justify-between mb-1 gap-1">
                              <span className="font-bold text-indigo-600 truncate max-w-[120px]" title={source.source}>{source.source}</span>
                              <span className="text-[8px] px-1.5 py-0.5 bg-slate-100 rounded-md font-black text-slate-500 uppercase flex-shrink-0">
                                {source.modality}{source.page ? ` • P. ${source.page}` : ''}
                              </span>
                            </div>
                            {source.modality === 'image' && source.storagePath ? (
                              <div className="mt-2 rounded-lg overflow-hidden border border-slate-100 bg-black/5 flex items-center justify-center">
                                <img 
                                  src={`/uploads/${source.storagePath}`} 
                                  alt={source.text || "source image"} 
                                  className="max-h-32 w-auto object-contain cursor-pointer hover:scale-105 transition-transform duration-200" 
                                  onClick={() => window.open(`/uploads/${source.storagePath}`, '_blank')}
                                />
                              </div>
                            ) : source.text.trim().startsWith('|') ? (
                              <div className="markdown-container overflow-x-auto text-[9px] mt-1 bg-white p-1 rounded-md border border-slate-100 max-h-32">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                  {source.text}
                                </ReactMarkdown>
                              </div>
                            ) : (
                              <p className="text-slate-500 line-clamp-4 leading-tight italic mt-1">"{source.text}"</p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          {isThinking && (
            <div className="flex items-center gap-3 text-xs text-indigo-500 font-medium animate-pulse ml-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>Analysing context...</span>
            </div>
          )}
        </div>

        <div className="p-6 bg-white/30 backdrop-blur-md border-t border-slate-200/50">
          <div className="max-w-4xl mx-auto relative group">
            <div className="absolute left-4 top-1/2 -translate-y-1/2 flex gap-2">
              <button 
                onClick={() => fileInputRef.current?.click()} 
                disabled={isUploading} 
                className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all"
                title="Attach file"
              >
                <Paperclip className="w-5 h-5" />
              </button>
            </div>
            <input 
              type="text" value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask a question about your documents..."
              className="w-full pl-14 pr-16 py-4 bg-white border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 shadow-sm transition-all text-sm placeholder:text-slate-400"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2">
              <button 
                onClick={handleSend} 
                disabled={!input.trim() || isThinking} 
                className={cn(
                  "p-2.5 rounded-xl transition-all duration-200",
                  !input.trim() || isThinking 
                    ? "bg-slate-100 text-slate-300" 
                    : "bg-indigo-600 text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700 active:scale-[0.95]"
                )}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
          <p className="text-[9px] text-center text-slate-400 mt-4 font-medium uppercase tracking-[0.2em]">Local Processing • Encrypted Retrieval</p>
        </div>
      </main>
    </div>
  );
}
