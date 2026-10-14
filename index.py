
"""
Agriculture Knowledge Graph RAG — PDF loader
Run: python index.py

Keep this file, .env, and your PDF files in the same folder.
"""

import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

VECTOR_INDEX_NAME = os.getenv(
    "NEO4J_VECTOR_INDEX", "farmer_knowledge_chunks"
)
MANIFEST_KEY = "farmer_agri_rag_pdf_fingerprint"


def require_env():
    required = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            "Missing .env values: "
            + ", ".join(missing)
            + f"\nExpected .env at: {BASE_DIR / '.env'}"
        )


def calculate_pdf_fingerprint(pdf_paths):
    """Identify whether the PDF collection has changed."""
    digest = hashlib.sha256()

    for pdf_path in sorted(pdf_paths, key=lambda p: p.name.lower()):
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(pdf_path.read_bytes())

    return digest.hexdigest()


def main():
    require_env()

    # 1. LOAD PDFs
    pdf_paths = sorted(
        [
            p for p in BASE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        ],
        key=lambda p: p.name.lower(),
    )

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {BASE_DIR}"
        )

    print(f"Project folder: {BASE_DIR}")
    print(f"PDF files found: {len(pdf_paths)}")

    fingerprint = calculate_pdf_fingerprint(pdf_paths)

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        refresh_schema=False,
    )

    # Skip processing when the PDF collection is unchanged.
    graph.query("""
        CREATE CONSTRAINT farmer_agri_manifest_key IF NOT EXISTS
        FOR (m:FarmerRAGManifest) REQUIRE m.key IS UNIQUE
    """)

    manifest_rows = graph.query(
        """
        MATCH (m:FarmerRAGManifest {key: $key})
        RETURN m.fingerprint AS fingerprint
        """,
        {"key": MANIFEST_KEY},
    )

    if (
        manifest_rows
        and manifest_rows[0].get("fingerprint") == fingerprint
    ):
        print(
            "SUCCESS: PDFs unchanged. "
            "Skipping database loading."
        )
        return

    # 2. EXTRACT TEXT FROM PDFs
    print("\n[1/5] Loading PDF pages...")

    pages = []

    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path.name}")

        pdf_pages = PyPDFLoader(str(pdf_path)).load()

        for page in pdf_pages:
            page.metadata["source"] = pdf_path.name

        pages.extend(pdf_pages)

    if not pages or not any(
        page.page_content.strip() for page in pages
    ):
        raise RuntimeError(
            "No extractable PDF text found. "
            "Scanned PDFs may require OCR."
        )

    print(f"Loaded {len(pages)} pages.")

    # 3. SPLIT TEXT INTO CHUNKS
    print("\n[2/5] Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents(pages)
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise RuntimeError("No text chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    # Generate stable IDs for vector records.
    chunk_ids = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page_num = str(chunk.metadata.get("page", 0))

        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()

        chunk_id = hashlib.sha256(
            f"{source}|{page_num}|{content_hash}".encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    # 4. EXTRACT ENTITIES AND RELATIONSHIPS
    print("\n[3/5] Extracting entities and relationships...")

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_retries=3,
        timeout=90,
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Crop",
            "Pest",
            "Disease",
            "Fertilizer",
            "SoilType",
            "Scheme",
            "Practice",
            "Organization",
            "Location",
            "Season",
            "IrrigationMethod",
            "Symptom",
            "Ingredient",
            "FarmerAdvice",
        ],
        allowed_relationships=[
            "AFFECTS",
            "TREATS",
            "PREVENTS",
            "REQUIRES",
            "APPLIES_TO",
            "GROWS_IN",
            "OCCURS_DURING",
            "CONTROLLED_BY",
            "RECOMMENDS",
            "PROVIDES",
            "ELIGIBLE_FOR",
            "CAUSES",
            "HAS_SYMPTOM",
            "IMPROVES",
            "RELATED_TO",
        ],
        strict_mode=False,
        node_properties=["description"],
        relationship_properties=["description"],
    )

    # IMPORTANT: chunks is defined above.
    # Keep this loop inside main().
    graph_documents = []

    for chunk in tqdm(
        chunks,
        desc="Extracting entities",
        unit="chunk",
    ):
        graph_document = transformer.process_response(chunk)
        graph_documents.append(graph_document)

    # 5. STORE KNOWLEDGE GRAPH
    print("\n[4/5] Saving graph to Neo4j Aura...")

    if graph_documents:
        graph.add_graph_documents(
            graph_documents,
            include_source=True,
            baseEntityLabel=True,
        )

    # STORE VECTOR EMBEDDINGS
    print("\n[5/5] Creating/updating vector index...")

    embeddings = OpenAIEmbeddings(
        model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        max_retries=3,
    )

    Neo4jVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name=VECTOR_INDEX_NAME,
        pre_delete_collection=False,
    )

    # Save manifest only after both storage operations succeed.
    graph.query(
        """
        MERGE (m:FarmerRAGManifest {key: $key})
        SET m.fingerprint = $fingerprint,
            m.pdf_count = $pdf_count,
            m.page_count = $page_count,
            m.chunk_count = $chunk_count,
            m.updated_at = datetime()
        """,
        {
            "key": MANIFEST_KEY,
            "fingerprint": fingerprint,
            "pdf_count": len(pdf_paths),
            "page_count": len(pages),
            "chunk_count": len(chunks),
        },
    )

    graph.refresh_schema()

    print("\n" + "=" * 55)
    print("SUCCESS: Knowledge graph and vector index stored!")
    print(f"PDFs: {len(pdf_paths)}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector index: {VECTOR_INDEX_NAME}")
    print("Run again only when the PDFs change.")
    print("=" * 55)


if __name__ == "__main__":
    main()
