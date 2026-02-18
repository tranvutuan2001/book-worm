import pdfplumber
import argparse
import time
from typing import List, Optional
from app.infra.llm_connector.llm_client import embed_text, complete_chat
from app.util import write_json_file
from app.domain.entity.message import Message
from app.constant import Role
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PreAnalyzeDocumentService:
    def _read_document_by_page(self, path: str) -> List[str]:
        text_by_page: List[str] = []
        with pdfplumber.open(path) as pdf:
            for _, page in enumerate(pdf.pages):
                try:
                    txt = page.extract_text()
                    if txt:
                        text_by_page.append(txt)
                except Exception:
                    # skip pages that fail to extract gracefully
                    continue

        return text_by_page


    def _embed_text_by_llm(self, text: str) -> List[float]:
        last_exception: Optional[Exception] = None

        for _ in range(3):
            try:
                embeded_text = embed_text(text)
                return embeded_text
            except Exception as exc:
                # capture the first failure and try again before surfacing it
                last_exception = exc

        raise RuntimeError("Failed to embed text after 3 attempts.") from last_exception


    def _embed_chunk(self, chunks: List[str]) -> List[List[float]]:
        res: List[List[float]] = []
        for index, chunk in enumerate(chunks):
            print(f"Embedding chunk {index + 1}/{len(chunks)}...")
            vector = self._embed_text_by_llm(text=chunk)
            res.append(vector)

        return res

    def _embed_section_summary(self, section_summary: List[str]) -> List[List[float]]:
        res: List[List[float]] = []
        for index, summary in enumerate(section_summary):
            print(f"Embedding section summary {index + 1}/{len(section_summary)}...")
            vector = self._embed_text_by_llm(text=summary)
            res.append(vector)
        
        return res


    def _embed_chapter_summary(self, chapter_summary: List[str]) -> List[List[float]]:
        res: List[List[float]] = []
        for index, summary in enumerate(chapter_summary):
            print(f"Embedding chapter summary {index + 1}/{len(chapter_summary)}...")
            vector = self._embed_text_by_llm(text=summary)
            res.append(vector)
        
        return res

    def _build_section_summaries(self, all_chunks: List[str]) -> List:
        numb_chunk_per_section = 10
        section_summary = []
        for i in range(0, len(all_chunks), numb_chunk_per_section):
            print(f"Building section summary for chunks {i + 1} to {min(i + numb_chunk_per_section, len(all_chunks))}...")
            section_text = "\n".join(all_chunks[i : i + numb_chunk_per_section])
            system_prompt = """You are an expert document analyzer that creates comprehensive section summaries for enabling whole-document question answering across diverse document types (novels, scientific papers, technical reports, business documents, etc.). Your summaries should adapt to the document type while capturing:

    1. STRUCTURAL ELEMENTS: 
    - For academic/technical: Key concepts, methodologies, findings, arguments
    - For narrative: Plot points, character development, themes, setting changes
    - For business: Processes, decisions, stakeholders, objectives, outcomes
    - Document organization and how this section relates to overall structure

    2. CONTENT PROGRESSION:
    - Evolution of ideas, arguments, or narrative across the section
    - Changes in tone, perspective, or approach
    - Key transitions or turning points

    3. CRITICAL INFORMATION:
    - Facts, data, evidence, or events essential to understanding
    - Important names, dates, locations, or technical terms
    - Decisions, conclusions, or revelations
    - Dependencies or prerequisites from other sections

    4. CONTEXTUAL SIGNIFICANCE:
    - How this section supports or contradicts other parts
    - Elements likely needed for cross-document analysis
    - Information that answers "why" and "how" questions about the whole work

    Adapt your analysis style to the document type while maintaining comprehensive coverage for future question-answering needs."""
            user_prompt = f"Analyze and summarize the following document section, focusing on its role within the larger work:\n\n{section_text}"
            
            # Create proper Message objects
            system_message = Message(
                id="system",
                content=system_prompt,
                role=Role.SYSTEM,
                timestamp=int(time.time())
            )
            user_message = Message(
                id="user", 
                content=user_prompt,
                role=Role.USER,
                timestamp=int(time.time())
            )
            message_list = [system_message, user_message]
            
            summary = complete_chat(
                message_list=message_list, system_prompt=system_prompt, tools=[]
            )
            section_summary.append(summary)

        return section_summary


    def _build_chapter_summary(self, section_summaries: List[str]) -> List[str]:
        sections_per_chapter = 5
        chapter_summaries = []
        
        for i in range(0, len(section_summaries), sections_per_chapter):
            print(f"Building chapter summary for sections {i + 1} to {min(i + sections_per_chapter, len(section_summaries))}...")
            chapter_sections = section_summaries[i:i + sections_per_chapter]
            combined_sections = "\n\n".join([f"Section {j+1}: {summary}" for j, summary in enumerate(chapter_sections)])
            
            system_prompt = """You are an expert document analyzer creating chapter-level summaries from section summaries. Your goal is to synthesize multiple section summaries into a coherent chapter overview that:

    1. SYNTHESIZES KEY THEMES: Identify overarching themes, concepts, or narrative elements that span across the sections
    2. TRACKS PROGRESSION: Show how ideas, arguments, or story elements develop throughout the chapter
    3. HIGHLIGHTS CONNECTIONS: Demonstrate relationships and transitions between sections
    4. MAINTAINS CRITICAL DETAILS: Preserve essential facts, events, or findings that are important for the whole document
    5. CONTEXTUALIZES SIGNIFICANCE: Explain how this chapter fits into and contributes to the larger document structure

    Create a comprehensive chapter summary that captures the essence and progression of the constituent sections while being suitable for answering questions about the broader document."""
            
            user_prompt = f"Create a comprehensive chapter summary from the following section summaries:\n\n{combined_sections}"
            
            # Create proper Message objects
            system_message = Message(
                id="system",
                content=system_prompt,
                role=Role.SYSTEM,
                timestamp=int(time.time())
            )
            user_message = Message(
                id="user",
                content=user_prompt,
                role=Role.USER,
                timestamp=int(time.time())
            )
            message_list = [system_message, user_message]
            
            print(f"Building chapter summary {len(chapter_summaries) + 1}...")
            chapter_summary = complete_chat(
                message_list=message_list, system_prompt=system_prompt, tools=[]
            )
            chapter_summaries.append(chapter_summary)
        
        return chapter_summaries


    def pre_analyze_document(self, pdf_path: str = None, document_name: str = None, process_levels: List[str] = None) -> None:
        """
        Pre-analyze a document by processing chunks, sections, and/or chapters.
        
        Args:
            pdf_path: Path to the PDF file. If None, uses the default book_name path.
            document_name: Name to use for output files. If None, uses book_name.
            process_levels: List of levels to process ['chunks', 'sections', 'chapters'].
                        If None, processes all levels.
        """
        if process_levels is None:
            process_levels = ["chunks", "sections", "chapters"]
        
        if pdf_path is None:
            pdf_path = f"./0_data/{book_name}.pdf"
            
        if document_name is None:
            document_name = book_name
        
        try:
            document_by_page = self._read_document_by_page(pdf_path)
            print(f"Extracted {len(document_by_page)} pages from the document.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=100
            )
            all_chunks = text_splitter.split_text("".join(document_by_page))
            print(f"Split document into {len(all_chunks)} chunks.")
            
            # Process chunks if requested
            if "chunks" in process_levels:
                print("Processing chunks...")
                write_json_file(all_chunks, f"./0_data/{document_name}/{document_name}_chunks.json")
                embeded_chunks = self._embed_chunk(all_chunks)
                write_json_file(embeded_chunks, f"./0_data/{document_name}/{document_name}_chunk_embeddings.json")
            
            # Process sections if requested
            section_summary = None
            if "sections" in process_levels:
                print("Processing sections...")
                section_summary = self._build_section_summaries(all_chunks)
                write_json_file(section_summary, f"./0_data/{document_name}/{document_name}_section_summaries.json")
                embeded_section_summaries = self._embed_section_summary(section_summary)
                write_json_file(embeded_section_summaries, f"./0_data/{document_name}/{document_name}_section_summary_embeddings.json")
            
            # Process chapters if requested
            if "chapters" in process_levels:
                print("Processing chapters...")
                # Load section summaries if not already created in this run
                if section_summary is None:
                    try:
                        import json
                        with open(f"./0_data/{document_name}/{document_name}_section_summaries.json", 'r') as f:
                            section_summary = json.load(f)
                        print("Loaded existing section summaries.")
                    except FileNotFoundError:
                        print("Section summaries not found. Creating them first...")
                        section_summary = self._build_section_summaries(all_chunks)
                        write_json_file(section_summary, f"./0_data/{document_name}/{document_name}_section_summaries.json")
                
                chapter_summaries = self._build_chapter_summary(section_summary)
                write_json_file(chapter_summaries, f"./0_data/{document_name}/{document_name}_chapter_summaries.json")
                embeded_chapter_summaries = self._embed_chapter_summary(chapter_summaries)
                write_json_file(embeded_chapter_summaries, f"./0_data/{document_name}/{document_name}_chapter_summary_embeddings.json")

        except RuntimeError as e:
            print(f"Text extraction error: {e}")

pre_analyze_document_service = PreAnalyzeDocumentService()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pre-analyze document by processing chunks, sections, and/or chapters"
    )
    parser.add_argument(
        "--level",
        choices=["chunks", "sections", "chapters"],
        nargs="+",
        help="Specify which levels to process. Can specify multiple levels (e.g., --level chunks sections). If not specified, all levels will be processed."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.level:
        print(f"Processing levels: {', '.join(args.level)}")
        pre_analyze_document_service.pre_analyze_document(args.level)
    else:
        print("Processing all levels: chunks, sections, chapters")
        pre_analyze_document_service.pre_analyze_document()