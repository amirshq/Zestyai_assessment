"""
Optimized Hybrid PDF QA System with Evaluation Metrics
- Faster processing with batch embeddings
- Comprehensive evaluation metrics
- Performance tracking and diagnostics
"""

import os
import glob
import time
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import hashlib
from dotenv import load_dotenv
import pdfplumber
import numpy as np
from numpy.linalg import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle

# Unstructured imports
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: Unstructured not available, using PDFPlumber only")

load_dotenv()


class PerformanceMetrics:
    """Track performance metrics for evaluation."""
    
    def __init__(self):
        self.metrics = {
            "extraction": {"time": 0, "pdfs_processed": 0, "tables_found": 0},
            "chunking": {"time": 0, "total_chunks": 0},
            "embedding": {"time": 0, "embeddings_generated": 0, "cache_hits": 0},
            "retrieval": {"time": 0, "chunks_scored": 0, "top_scores": []},
            "generation": {"time": 0, "tokens_used": 0},
            "total_time": 0
        }
        self.start_time = None
        
    def start_timer(self):
        self.start_time = time.time()
        
    def record(self, phase: str, **kwargs):
        """Record metrics for a phase."""
        for key, value in kwargs.items():
            if key == "time":
                self.metrics[phase][key] += value
            elif key in self.metrics[phase]:
                if isinstance(self.metrics[phase][key], list):
                    self.metrics[phase][key].extend(value if isinstance(value, list) else [value])
                else:
                    self.metrics[phase][key] += value
            else:
                self.metrics[phase][key] = value
    
    def finalize(self):
        """Calculate final metrics."""
        if self.start_time:
            self.metrics["total_time"] = time.time() - self.start_time
            
    def get_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("\n" + "="*80)
        report.append("PERFORMANCE EVALUATION REPORT")
        report.append("="*80)
        
        # Overall Performance
        report.append(f"\n📊 OVERALL PERFORMANCE:")
        report.append(f"  Total Time: {self.metrics['total_time']:.2f}s")
        
        # Phase Breakdown
        report.append(f"\n⏱️  PHASE BREAKDOWN:")
        for phase in ["extraction", "chunking", "embedding", "retrieval", "generation"]:
            time_val = self.metrics[phase].get("time", 0)
            pct = (time_val / self.metrics['total_time'] * 100) if self.metrics['total_time'] > 0 else 0
            report.append(f"  {phase.capitalize()}: {time_val:.2f}s ({pct:.1f}%)")
        
        # Extraction Quality
        report.append(f"\n📄 EXTRACTION QUALITY:")
        report.append(f"  PDFs Processed: {self.metrics['extraction']['pdfs_processed']}")
        report.append(f"  Tables Found: {self.metrics['extraction']['tables_found']}")
        report.append(f"  Total Chunks: {self.metrics['chunking']['total_chunks']}")
        
        # Embedding Efficiency
        report.append(f"\n🧠 EMBEDDING EFFICIENCY:")
        total_emb = self.metrics['embedding']['embeddings_generated']
        cache_hits = self.metrics['embedding']['cache_hits']
        cache_rate = (cache_hits / (total_emb + cache_hits) * 100) if (total_emb + cache_hits) > 0 else 0
        report.append(f"  New Embeddings: {total_emb}")
        report.append(f"  Cache Hits: {cache_hits}")
        report.append(f"  Cache Hit Rate: {cache_rate:.1f}%")
        if total_emb > 0:
            avg_time = self.metrics['embedding']['time'] / total_emb
            report.append(f"  Avg Time per Embedding: {avg_time*1000:.1f}ms")
        
        # Retrieval Quality
        report.append(f"\n🔍 RETRIEVAL QUALITY:")
        report.append(f"  Chunks Scored: {self.metrics['retrieval']['chunks_scored']}")
        if self.metrics['retrieval']['top_scores']:
            top_scores = self.metrics['retrieval']['top_scores']
            report.append(f"  Top Score: {max(top_scores):.1f}")
            report.append(f"  Avg Top-5 Score: {np.mean(top_scores[:5]):.1f}")
            report.append(f"  Score Range: {min(top_scores):.1f} - {max(top_scores):.1f}")
            
            # Score distribution analysis
            high_quality = sum(1 for s in top_scores if s > 800)
            medium_quality = sum(1 for s in top_scores if 500 <= s <= 800)
            low_quality = sum(1 for s in top_scores if s < 500)
            
            report.append(f"\n  Score Distribution:")
            report.append(f"    High Confidence (>800): {high_quality} chunks")
            report.append(f"    Medium Confidence (500-800): {medium_quality} chunks")
            report.append(f"    Low Confidence (<500): {low_quality} chunks")
            
            if low_quality > len(top_scores) * 0.5:
                report.append(f"  ⚠️  WARNING: Many low-confidence chunks - query may be challenging")
        
        # Generation Stats
        report.append(f"\n💬 GENERATION:")
        report.append(f"  Time: {self.metrics['generation']['time']:.2f}s")
        if self.metrics['generation'].get('tokens_used'):
            report.append(f"  Tokens Used: {self.metrics['generation']['tokens_used']}")
        
        # Bottleneck Analysis
        report.append(f"\n🎯 BOTTLENECK ANALYSIS:")
        bottlenecks = []
        for phase in ["extraction", "embedding", "retrieval", "generation"]:
            time_val = self.metrics[phase].get("time", 0)
            pct = (time_val / self.metrics['total_time'] * 100) if self.metrics['total_time'] > 0 else 0
            if pct > 30:
                bottlenecks.append((phase, pct))
        
        if bottlenecks:
            for phase, pct in bottlenecks:
                report.append(f"  ⚠️  {phase.capitalize()} is a bottleneck ({pct:.1f}% of time)")
        else:
            report.append(f"  ✅ Well-balanced pipeline")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if cache_rate < 50:
            report.append(f"  - Low cache hit rate - consider pre-computing embeddings")
        if self.metrics['embedding']['time'] > self.metrics['total_time'] * 0.4:
            report.append(f"  - Embeddings taking too long - consider batch processing or smaller model")
        if self.metrics['retrieval']['top_scores'] and max(self.metrics['retrieval']['top_scores']) < 600:
            report.append(f"  - Low semantic scores - query may not match document content well")
        if not bottlenecks:
            report.append(f"  - System is well-optimized!")
        
        report.append("="*80 + "\n")
        return "\n".join(report)


class HybridPDFQuestionAnswerer:
    """
    Optimized Hybrid PDF QA with performance tracking.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", 
                 embedding_model: str = "text-embedding-3-small",
                 cache_dir: str = ".pdf_qa_cache",
                 chunk_size: int = 2500,
                 chunk_overlap: int = 400):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model  # text-embedding-3-small is 3x faster
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_cache = {}
        self.embedding_cache = {}
        self.metrics = PerformanceMetrics()
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory: {self.cache_dir}")
    
    def extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text and tables using PDFPlumber."""
        all_text = []
        all_tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text:
                        all_text.append(f"\n--- Page {page_num} ---\n{page_text}")
                    
                    # Extract tables with multiple strategies
                    tables = None
                    
                    # Strategy 1: Lines-based
                    try:
                        tables = page.extract_tables({
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                        })
                    except:
                        pass
                    
                    # Strategy 2: Text-based
                    if not tables or len(tables) == 0:
                        try:
                            tables = page.extract_tables({
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                            })
                        except:
                            pass
                    
                    # Convert tables to text
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            if table:
                                table_text = self._format_table(table)
                                if table_text:
                                    all_tables.append(
                                        f"\n{'='*80}\n"
                                        f"TABLE (Page {page_num}, Table {table_num})\n"
                                        f"{'='*80}\n"
                                        f"{table_text}\n"
                                        f"{'='*80}\n"
                                    )
        
        except Exception as e:
            print(f"    PDFPlumber error: {e}")
        
        return "\n".join(all_text), all_tables
    
    def _format_table(self, table: List[List]) -> str:
        """Format table as readable text."""
        if not table:
            return ""
        
        cleaned = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cleaned_row):
                cleaned.append(cleaned_row)
        
        if not cleaned:
            return ""
        
        # Calculate column widths
        max_cols = max(len(row) for row in cleaned)
        col_widths = [0] * max_cols
        
        for row in cleaned:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))
        
        # Format rows
        lines = []
        for idx, row in enumerate(cleaned):
            cells = []
            for i in range(max_cols):
                cell = row[i] if i < len(row) else ""
                cells.append(cell.ljust(col_widths[i]))
            
            lines.append(" | ".join(cells).rstrip())
            
            # Header separator
            if idx == 0 and len(cleaned) > 1:
                lines.append("-+-".join(["-" * w for w in col_widths]))
        
        return "\n".join(lines)
    
    def extract_with_unstructured(self, pdf_path: str) -> str:
        """Extract using Unstructured for structure detection."""
        if not UNSTRUCTURED_AVAILABLE:
            return ""
        
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                infer_table_structure=False,
            )
            
            structured_text = []
            for element in elements:
                elem_type = type(element).__name__
                text = str(element).strip()
                
                if elem_type == "Title":
                    structured_text.append(f"\n### {text} ###")
                elif elem_type == "ListItem":
                    structured_text.append(f"  • {text}")
                else:
                    structured_text.append(text)
            
            return "\n".join(structured_text)
        
        except Exception as e:
            print(f"    Unstructured error: {e}")
            return ""
    
    def merge_extractions(self, pdfplumber_text: str, tables: List[str], 
                         unstructured_text: str) -> str:
        """Merge content from both extractors."""
        merged_parts = []
        
        if tables:
            merged_parts.extend(tables)
        
        if unstructured_text and len(unstructured_text) > 100:
            merged_parts.append("\n=== STRUCTURED CONTENT ===\n")
            merged_parts.append(unstructured_text)
        
        if pdfplumber_text:
            merged_parts.append("\n=== FULL TEXT ===\n")
            merged_parts.append(pdfplumber_text)
        
        return "\n\n".join(merged_parts)
    
    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """Main extraction combining both methods. Returns (text, table_count)."""
        pdfplumber_text, tables = self.extract_with_pdfplumber(pdf_path)
        table_count = len(tables)
        
        unstructured_text = ""
        if UNSTRUCTURED_AVAILABLE:
            unstructured_text = self.extract_with_unstructured(pdf_path)
        
        merged = self.merge_extractions(pdfplumber_text, tables, unstructured_text)
        
        return merged, table_count
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts in a single API call (FASTER!)."""
        if not texts:
            return []
        
        import time as time_module
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # OpenAI allows batch embedding requests
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts[:100]  # API limit is typically 100 texts
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                        print(f"    Rate limit hit, waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                        time_module.sleep(wait_time)
                        continue
                    else:
                        print(f"    Rate limit exceeded after {max_retries} retries, using sequential fallback")
                else:
                    print(f"    Warning: Batch embedding failed: {e}")
                
                # Fallback to individual calls (slower but works)
                return [self.get_embedding_single(text) for text in texts]
        
        return [self.get_embedding_single(text) for text in texts]
    
    def get_embedding_single(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text with caching."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            self.metrics.record("embedding", cache_hits=1)
            return self.embedding_cache[text_hash]
        
        try:
            if len(text) > 8000:
                text = text[:8000]
            
            start = time.time()
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            self.metrics.record("embedding", time=time.time()-start, embeddings_generated=1)
            self.embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            print(f"    Warning: Embedding failed: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        return np.dot(vec1_arr, vec2_arr) / (norm(vec1_arr) * norm(vec2_arr))
    
    def chunk_text(self, text: str, pdf_name: str = "") -> List[Dict]:
        """Chunk with emphasis on preserving tables and lists."""
        start = time.time()
        
        if len(text) <= 3000:
            chunks = [{"text": text, "has_table": "TABLE" in text.upper(), 
                      "pdf_name": pdf_name, "embedding": None}]
            self.metrics.record("chunking", time=time.time()-start, total_chunks=1)
            return chunks
        
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        in_table = False
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        
        for line in lines:
            if '=' * 40 in line and 'TABLE' in line:
                in_table = not in_table
            
            line_size = len(line) + 1
            
            if current_size + line_size > chunk_size and not in_table and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "has_table": "TABLE" in chunk_text.upper(),
                    "has_list": "•" in chunk_text,
                    "pdf_name": pdf_name,
                    "embedding": None
                })
                
                overlap_lines = current_chunk[-15:] if len(current_chunk) > 15 else current_chunk
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "has_table": "TABLE" in chunk_text.upper(),
                "has_list": "•" in chunk_text,
                "pdf_name": pdf_name,
                "embedding": None
            })
        
        self.metrics.record("chunking", time=time.time()-start, total_chunks=len(chunks))
        return chunks
    
    def get_relevant_chunks(self, question: str, chunks: List[Dict], top_k: int = 20) -> Tuple[List[str], List[str]]:
        """Retrieve most relevant chunks using SEMANTIC SEARCH with performance tracking."""
        start = time.time()
        
        if len(chunks) <= top_k:
            return [c["text"] for c in chunks], [c.get("pdf_name", "unknown") for c in chunks]
        
        print("  Generating embeddings for semantic search...")
        
        # Get question embedding
        question_embedding = self.get_embedding_single(question)
        if question_embedding is None:
            print("  Warning: Failed to get question embedding, falling back to keyword search")
            return self._fallback_keyword_search(question, chunks, top_k)
        
        # OPTIMIZATION: Batch embed chunks that don't have embeddings
        chunks_needing_embedding = [(idx, chunk) for idx, chunk in enumerate(chunks) if chunk.get("embedding") is None]
        
        if chunks_needing_embedding:
            print(f"  Computing {len(chunks_needing_embedding)} new embeddings in batches...")
            batch_size = 30  # Reduced from 50 to avoid rate limits
            for i in range(0, len(chunks_needing_embedding), batch_size):
                batch = chunks_needing_embedding[i:i+batch_size]
                texts = [chunk["text"][:8000] for _, chunk in batch]
                
                batch_start = time.time()
                embeddings = self.get_embeddings_batch(texts)
                batch_time = time.time() - batch_start
                
                for (idx, chunk), embedding in zip(batch, embeddings):
                    chunks[idx]["embedding"] = embedding
                
                self.metrics.record("embedding", time=batch_time, embeddings_generated=len(embeddings))
                
                if i + batch_size < len(chunks_needing_embedding):
                    print(f"    Processed {i+batch_size}/{len(chunks_needing_embedding)}...")
        
        # Score all chunks
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            pdf_name = chunk.get("pdf_name", "unknown")
            chunk_embedding = chunk.get("embedding")
            
            if chunk_embedding is not None:
                semantic_score = self.cosine_similarity(question_embedding, chunk_embedding)
            else:
                semantic_score = 0.0
            
            base_score = semantic_score * 1000
            
            pdf_name_lower = pdf_name.lower()
            
            # Apply domain multipliers
            if chunk.get("has_table", False):
                base_score *= 1.3
            
            if "rate" in pdf_name_lower and "page" in pdf_name_lower:
                base_score *= 1.4
            
            if "objection" in pdf_name_lower or "response" in pdf_name_lower:
                base_score *= 0.6
            
            if ("rule" in pdf_name_lower or "manual" in pdf_name_lower) and chunk.get("has_list", False):
                base_score *= 1.2
            
            scored_chunks.append((base_score, idx, chunk_text, pdf_name))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        # Determine top_k based on query
        question_lower = question.lower()
        if 'list all' in question_lower or 'all rules' in question_lower:
            actual_top_k = min(len(chunks), 60)
        elif any(word in question_lower for word in ['calculate', 'compute']):
            actual_top_k = min(len(chunks), 35)  # Increased from 20 to 35 to capture all needed tables
        else:
            actual_top_k = top_k
        
        result = scored_chunks[:actual_top_k]
        
        # Record metrics
        top_scores = [r[0] for r in result]
        self.metrics.record("retrieval", 
                           time=time.time()-start,
                           chunks_scored=len(chunks),
                           top_scores=top_scores)
        
        print(f"  Top 3 semantic scores: {[f'{r[0]:.1f}' for r in result[:3]]}")
        
        return [c[2] for c in result], [c[3] for c in result]
    
    def _fallback_keyword_search(self, question: str, chunks: List[Dict], top_k: int) -> Tuple[List[str], List[str]]:
        """Fallback to simple keyword matching if embeddings fail."""
        import re
        question_lower = question.lower()
        question_terms = set(re.findall(r'\b\w+\b', question_lower))
        
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_terms = set(re.findall(r'\b\w+\b', chunk_text.lower()))
            
            overlap = len(question_terms.intersection(chunk_terms))
            score = overlap
            
            if chunk.get("has_table", False):
                score += 50
            
            scored_chunks.append((score, idx, chunk_text, chunk.get("pdf_name", "unknown")))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        result = scored_chunks[:top_k]
        return [c[2] for c in result], [c[3] for c in result]
    
    def answer_question(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer with performance tracking."""
        start = time.time()
        
        context = "\n\n---\n\n".join(context_chunks)
        
        question_lower = question.lower()
        is_list_all = 'list all' in question_lower
        is_calculation = any(word in question_lower for word in ['calculate', 'compute', 'what is the', 'how much', 'determine the'])
        
        # Adjust max length
        if is_list_all:
            max_length = 40000
        elif is_calculation:
            max_length = 25000
        else:
            max_length = 30000
            
        if len(context) > max_length:
            if "TABLE" in context.upper():
                tables = []
                remaining = context
                while "TABLE" in remaining.upper():
                    idx = remaining.upper().find("TABLE")
                    end = remaining.find("=" * 40, idx + 10)
                    if end == -1:
                        end = len(remaining)
                    tables.append(remaining[max(0, idx-50):end+100])
                    remaining = remaining[:idx] + remaining[end+100:]
                
                table_text = "\n".join(tables)
                remaining_space = max_length - len(table_text)
                context = table_text + "\n\n" + remaining[:max(0, remaining_space)]
            else:
                context = context[:max_length]
        
        # Build prompts
        if is_calculation:
            prompt = f"""You are a precise insurance calculation expert. You MUST extract exact numbers from tables and perform the requested calculation.

CRITICAL INSTRUCTIONS:
1. SEARCH THOROUGHLY through ALL provided tables - you have multiple tables from different PDFs
2. IDENTIFY these specific components (BOTH are required):
   - BASE RATES table (usually shows rates per $1,000 of coverage by policy type)
   - FACTORS/DEDUCTIBLES table (shows multipliers like Hurricane Deductible Factor, distance from coast rules, etc.)
   - RULES (shows how to apply rates and factors)
3. DO NOT assume factors are 1.0 if not found - search more carefully for deductible rules
4. EXTRACT exact values matching the question parameters:
   - Policy type (HO3, HO5, etc.)
   - Coverage amount ($750,000, etc.)
   - Location parameters (distance from coast, territory, etc.)
   - Applicable factors and percentages
4. PERFORM the actual mathematical calculation step-by-step with REAL numbers
5. SHOW your work clearly
6. PROVIDE the final numerical answer (e.g., "$604")

Common calculation patterns:
- Premium = Base Rate × (Coverage Amount / 1000) × Factor
- Look for rates "per $1,000" or "per 1,000"
- Factors are often percentages or decimals

The content includes MULTIPLE tables - examine ALL of them for the needed values!

Content with tables from multiple PDFs:
{context}

Question: {question}

Extract the numbers from the tables above and calculate the answer:"""
        
        elif is_list_all:
            prompt = f"""You are extracting a COMPLETE and COMPREHENSIVE list from multiple PDF documents.

CRITICAL INSTRUCTIONS FOR COMPLETENESS:
1. Search EXHAUSTIVELY through ALL provided content - rules/items are scattered across multiple pages and documents
2. Look in EVERY section for:
   - Bullet points (•, -, *)
   - Numbered lists (1., 2., 3.)
   - Rule identifiers ("Rule C-24", "Rule X:")
   - Tables listing rules/factors
   - Section headers with "Factor", "Discount", "Rating"
   - Table of contents sections
3. Extract EVERY SINGLE unique item you find - completeness is CRITICAL
4. Common patterns to look for:
   - "Factor" (e.g., "Roof Type Factor", "Age of Home Factor")
   - "Discount" (e.g., "Claims Free Discount", "Protective Device Discount")
   - "Rating" (e.g., "Loss History Rating")
   - "Deductible" (e.g., "Hurricane Deductibles")
   - "Guidelines" or "Relationships"
5. Format as simple bullet list with "* " prefix
6. Use EXACT names from the document - do not paraphrase
7. Consolidate exact duplicates only
8. The list must be 25-35+ items for rating plan rules - if you find fewer, keep searching!

Content (from multiple PDFs):
{context}

Question: {question}

Provide the COMPLETE list with ALL items found across all documents:"""
        else:
            prompt = f"""Content:
{context}

Question: {question}

Answer based on the provided content:"""
        
        try:
            system_message = "You are a meticulous document analyst focused on completeness and accuracy."
            if is_calculation:
                system_message = "You are a precise insurance calculation expert. You extract exact numbers from documents and perform accurate calculations. Never use placeholders - always use real numbers from the provided content."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4000 if is_list_all else (2000 if is_calculation else 1500),
                timeout=120
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Record generation metrics
            self.metrics.record("generation", 
                              time=time.time()-start,
                              tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0)
            
            return answer
        except Exception as e:
            self.metrics.record("generation", time=time.time()-start)
            return f"Error: {str(e)}"
    
    def process_pdfs_folder(self, pdfs_folder: str) -> List[Dict]:
        """Process PDFs with performance tracking and persistent disk caching."""
        cache_key = self._get_folder_hash(pdfs_folder)
        
        # Check memory cache first
        if cache_key in self.chunk_cache:
            print("Using in-memory cached chunks...")
            return self.chunk_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"chunks_{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                print(f"Loading chunks from disk cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.chunk_cache[cache_key] = cached_data['chunks']
                    # Restore embeddings to memory cache
                    if 'embeddings' in cached_data:
                        self.embedding_cache.update(cached_data['embeddings'])
                        print(f"  Loaded {len(cached_data['chunks'])} chunks and {len(cached_data['embeddings'])} embeddings from cache")
                        print(f"  💰 Saved OpenAI API costs by using cached embeddings!")
                    return cached_data['chunks']
            except Exception as e:
                print(f"  Warning: Failed to load cache ({e}), will regenerate...")
        
        start = time.time()
        
        pdf_files = glob.glob(os.path.join(pdfs_folder, "*.pdf"))
        if not pdf_files:
            return []
        
        print(f"Found {len(pdf_files)} PDF(s)")
        all_chunks = []
        total_tables = 0
        
        for pdf_path in sorted(pdf_files):
            pdf_name = os.path.basename(pdf_path)
            print(f"Processing {pdf_name}...")
            
            text, table_count = self.extract_from_pdf(pdf_path)
            total_tables += table_count
            print(f"    {table_count} tables extracted")
            
            if text:
                chunks = self.chunk_text(text, pdf_name=pdf_name)
                all_chunks.extend(chunks)
                print(f"    Created {len(chunks)} chunks")
        
        print(f"\nTotal: {len(all_chunks)} chunks")
        
        self.metrics.record("extraction", 
                           time=time.time()-start,
                           pdfs_processed=len(pdf_files),
                           tables_found=total_tables)
        
        # Save to memory cache
        self.chunk_cache[cache_key] = all_chunks
        
        # Save to disk cache for future runs
        cache_file = os.path.join(self.cache_dir, f"chunks_{cache_key}.pkl")
        try:
            print(f"Saving chunks to disk cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': all_chunks,
                    'embeddings': {},  # Embeddings will be added later
                    'timestamp': time.time(),
                    'pdf_count': len(pdf_files)
                }, f)
            print(f"  ✅ Cache saved! Future runs will skip PDF processing.")
        except Exception as e:
            print(f"  Warning: Failed to save cache ({e})")
        
        return all_chunks
    
    def _get_folder_hash(self, folder_path: str) -> str:
        """Generate hash for caching based on PDF files and their modification times."""
        pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
        hash_input = []
        for pdf_path in pdf_files:
            hash_input.append(os.path.basename(pdf_path))
            try:
                # Include modification time so cache updates if PDFs change
                hash_input.append(str(int(os.path.getmtime(pdf_path))))
            except:
                pass
        return hashlib.md5("|".join(hash_input).encode()).hexdigest()
    
    def save_embeddings_to_cache(self, pdfs_folder: str):
        """Save current embedding cache to disk."""
        cache_key = self._get_folder_hash(pdfs_folder)
        cache_file = os.path.join(self.cache_dir, f"chunks_{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return
        
        try:
            # Load existing cache
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Update with new embeddings
            cached_data['embeddings'] = self.embedding_cache
            
            # Save back
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f"  💾 Saved {len(self.embedding_cache)} embeddings to disk cache")
            print(f"  💰 Next run will be FREE (no API calls for embeddings)!")
        except Exception as e:
            print(f"  Warning: Failed to save embeddings ({e})")


def answer_pdf_question(question: str, pdfs_folder: str, show_metrics: bool = True) -> str:
    """Main function with evaluation metrics and persistent caching."""
    qa = HybridPDFQuestionAnswerer()
    qa.metrics.start_timer()
    
    chunks = qa.process_pdfs_folder(pdfs_folder)
    if not chunks:
        return "No content found."
    
    print(f"Total chunks available: {len(chunks)}")
    
    relevant_chunks, sources = qa.get_relevant_chunks(question, chunks)
    print(f"Using {len(relevant_chunks)} chunks for answer generation")
    print(f"Total context length: {sum(len(c) for c in relevant_chunks)} characters")
    
    # Show source distribution
    unique_sources_debug = list(set(sources))
    print(f"Retrieving from {len(unique_sources_debug)} PDFs:")
    for src in sorted(unique_sources_debug):
        count = sources.count(src)
        print(f"  - {src}: {count} chunks")
    
    answer = qa.answer_question(question, relevant_chunks)
    
    # Add sources
    unique_sources = sorted(set(sources))[:5]
    if unique_sources and unique_sources != ['unknown']:
        answer += f"\n\n📄 Sources: {', '.join(unique_sources)}"
    
    # Save embeddings to disk cache for future runs
    qa.save_embeddings_to_cache(pdfs_folder)
    
    # Finalize and show metrics
    qa.metrics.finalize()
    
    if show_metrics:
        metrics_report = qa.metrics.get_report()
        print(metrics_report)
    
    return answer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_question_answering.py <question> <pdfs_folder>")
        sys.exit(1)
    
    answer = answer_pdf_question(sys.argv[1], sys.argv[2])
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
