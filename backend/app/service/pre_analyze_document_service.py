import pdfplumber
import time
import logging
import traceback
from typing import List, Optional
from fastapi import Depends
from app.infra.llm_connector.llm_client import LLMService, get_llm_service
from app.infra.llm_connector.mlx_chat import MLXChatModel
from app.infra.llm_connector.mlx_embedding import MLXEmbeddingModel
from app.util import write_json_file
from app.domain.entity.message import Message
from app.constant import Role
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger('app.service')

_CHAT_MODEL_PATH = "models/chat/mlx-community/Qwen3.5-35B-A3B-4bit"
_EMBEDDING_MODEL_PATH = "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


class PreAnalyzeDocumentService:
    def __init__(self, llm_service: LLMService) -> None:
        """
        Args:
            llm_service: The LLM service used for text generation and embeddings.
                         Injected by FastAPI via ``Depends(get_llm_service)``.
        """
        self._llm_client = llm_service

    def _read_document_by_page(self, path: str) -> List[str]:
        text_by_page: List[str] = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        txt = page.extract_text()
                        if txt:
                            text_by_page.append(txt)
                    except Exception as e:
                        error_msg = f"Failed to extract text from page {page_num + 1}: {str(e)}"
                        logger.warning(error_msg)
                        print(f"⚠️  {error_msg}")
                        # skip pages that fail to extract gracefully
                        continue
        except FileNotFoundError as e:
            error_msg = f"PDF file not found: {path}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Failed to open or read PDF file {path}: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        return text_by_page


    def _prepare_models(self) -> None:
        """Unload all cached models then eagerly load the models needed for pre-analysis."""
        logger.info("Unloading all models from memory before pre-analysis")
        MLXChatModel._model_cache.clear()
        MLXEmbeddingModel._model_cache.clear()

        logger.info(f"Loading chat model: {_CHAT_MODEL_PATH}")
        print(f"🔄 Loading chat model: {_CHAT_MODEL_PATH}...")
        MLXChatModel._load_model(_CHAT_MODEL_PATH)

        logger.info(f"Loading embedding model: {_EMBEDDING_MODEL_PATH}")
        print(f"🔄 Loading embedding model: {_EMBEDDING_MODEL_PATH}...")
        MLXEmbeddingModel._load_model(_EMBEDDING_MODEL_PATH)
        print("✓ Models ready.")

    def _embed_text_by_llm(self, text: str) -> List[float]:
        last_exception: Optional[Exception] = None
        for attempt in range(3):
            try:
                return self._llm_client.embed_text(model_path=_EMBEDDING_MODEL_PATH, text=text)
            except Exception as exc:
                last_exception = exc
                error_msg = f"Embedding attempt {attempt + 1}/3 failed: {str(exc)}"
                logger.warning(error_msg)
                print(f"⚠️  {error_msg}")
                if attempt < 2:
                    time.sleep(1)
        error_msg = f"Failed to embed text after 3 attempts. Last error: {str(last_exception)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise RuntimeError("Failed to embed text after 3 attempts.") from last_exception

    def _embed_texts(self, texts: List[str], label: str = "text") -> List[List[float]]:
        """Embed a list of texts, logging progress with the given *label*."""
        result: List[List[float]] = []
        for index, text in enumerate(texts):
            try:
                print(f"Embedding {label} {index + 1}/{len(texts)}...")
                result.append(self._embed_text_by_llm(text=text))
            except Exception as e:
                logger.error(f"Failed to embed {label} {index + 1}/{len(texts)}: {e}")
                print(f"❌ Failed to embed {label} {index + 1}/{len(texts)}: {e}")
                raise
        return result

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
            # Note: system_prompt is passed separately to complete_chat, so only include
            # the user message here to avoid a duplicate system message in the conversation.
            user_message = Message(
                id="user", 
                content=user_prompt,
                role=Role.USER,
                timestamp=int(time.time())
            )
            message_list = [user_message]
            
            try:
                summary = self._llm_client.complete_chat(
                    message_list=message_list, system_prompt=system_prompt, tools=[], model_path=_CHAT_MODEL_PATH
                )
                section_summary.append(summary)
            except Exception as e:
                error_msg = f"Failed to generate section summary for chunks {i + 1} to {min(i + numb_chunk_per_section, len(all_chunks))}: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise

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
            # Note: system_prompt is passed separately to complete_chat, so only include
            # the user message here to avoid a duplicate system message in the conversation.
            user_message = Message(
                id="user",
                content=user_prompt,
                role=Role.USER,
                timestamp=int(time.time())
            )
            message_list = [user_message]
            
            print(f"Building chapter summary {len(chapter_summaries) + 1}...")
            try:
                chapter_summary = self._llm_client.complete_chat(
                    model_path=_CHAT_MODEL_PATH,
                    message_list=message_list, system_prompt=system_prompt, tools=[]
                )
                chapter_summaries.append(chapter_summary)
            except Exception as e:
                error_msg = f"Failed to generate chapter summary for sections {i + 1} to {min(i + sections_per_chapter, len(section_summaries))}: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise
        
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
        
        logger.info(f"Starting document pre-analysis for: {document_name}")
        print(f"\n🚀 Starting document pre-analysis for: {document_name}")
        self._prepare_models()
        
        try:
            document_by_page = self._read_document_by_page(pdf_path)
            logger.info(f"Successfully extracted {len(document_by_page)} pages from the document")
            print(f"✓ Extracted {len(document_by_page)} pages from the document.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=100
            )
            all_chunks = text_splitter.split_text("".join(document_by_page))
            logger.info(f"Successfully split document into {len(all_chunks)} chunks")
            print(f"✓ Split document into {len(all_chunks)} chunks.")
            
            # Process chunks if requested
            if "chunks" in process_levels:
                print("\n📝 Processing chunks...")
                logger.info("Starting chunk processing")
                try:
                    write_json_file(all_chunks, f"./0_data/{document_name}/{document_name}_chunks.json")
                    logger.info("Successfully saved chunks to file")
                    embeded_chunks = self._embed_texts(all_chunks, label="chunk")
                    write_json_file(embeded_chunks, f"./0_data/{document_name}/{document_name}_chunk_embeddings.json")
                    logger.info("Successfully saved chunk embeddings to file")
                    print(f"✓ Chunk processing completed")
                except Exception as e:
                    error_msg = f"Failed during chunk processing: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    raise
            
            # Process sections if requested
            section_summary = None
            if "sections" in process_levels:
                print("\n📚 Processing sections...")
                logger.info("Starting section processing")
                try:
                    section_summary = self._build_section_summaries(all_chunks)
                    write_json_file(section_summary, f"./0_data/{document_name}/{document_name}_section_summaries.json")
                    logger.info("Successfully saved section summaries to file")
                    embeded_section_summaries = self._embed_texts(section_summary, label="section summary")
                    write_json_file(embeded_section_summaries, f"./0_data/{document_name}/{document_name}_section_summary_embeddings.json")
                    logger.info("Successfully saved section summary embeddings to file")
                    print(f"✓ Section processing completed")
                except Exception as e:
                    error_msg = f"Failed during section processing: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    raise
            
            # Process chapters if requested
            if "chapters" in process_levels:
                print("\n📖 Processing chapters...")
                logger.info("Starting chapter processing")
                try:
                    # Load section summaries if not already created in this run
                    if section_summary is None:
                        try:
                            import json
                            with open(f"./0_data/{document_name}/{document_name}_section_summaries.json", 'r') as f:
                                section_summary = json.load(f)
                            logger.info("Loaded existing section summaries")
                            print("✓ Loaded existing section summaries.")
                        except FileNotFoundError:
                            logger.info("Section summaries not found. Creating them first...")
                            print("⚠️  Section summaries not found. Creating them first...")
                            section_summary = self._build_section_summaries(all_chunks)
                            write_json_file(section_summary, f"./0_data/{document_name}/{document_name}_section_summaries.json")
                        except Exception as e:
                            error_msg = f"Failed to load section summaries: {str(e)}"
                            logger.error(error_msg)
                            print(f"❌ {error_msg}")
                            raise
                    
                    chapter_summaries = self._build_chapter_summary(section_summary)
                    write_json_file(chapter_summaries, f"./0_data/{document_name}/{document_name}_chapter_summaries.json")
                    logger.info("Successfully saved chapter summaries to file")
                    embeded_chapter_summaries = self._embed_texts(chapter_summaries, label="chapter summary")
                    write_json_file(embeded_chapter_summaries, f"./0_data/{document_name}/{document_name}_chapter_summary_embeddings.json")
                    logger.info("Successfully saved chapter summary embeddings to file")
                    print(f"✓ Chapter processing completed")
                except Exception as e:
                    error_msg = f"Failed during chapter processing: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    raise

            logger.info(f"✅ Document pre-analysis completed successfully for: {document_name}")
            print(f"\n✅ Document pre-analysis completed successfully for: {document_name}")
            
        except FileNotFoundError as e:
            error_msg = f"File not found error during pre-analysis: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise
        except RuntimeError as e:
            error_msg = f"Runtime error during pre-analysis: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during pre-analysis: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise


def get_pre_analyze_document_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> PreAnalyzeDocumentService:
    """
    FastAPI dependency factory for ``PreAnalyzeDocumentService``.

    Inject this into route handlers with::

        from fastapi import Depends
        from app.service.pre_analyze_document_service import (
            PreAnalyzeDocumentService,
            get_pre_analyze_document_service,
        )

        async def my_route(
            service: PreAnalyzeDocumentService = Depends(get_pre_analyze_document_service)
        ):
            ...
    """
    return PreAnalyzeDocumentService(llm_service=llm_service)
