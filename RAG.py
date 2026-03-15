import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai


class RAGPipeline:

    # ── Class-level constant (shared across all instances) ────────────────────
    GEMINI_MODEL = "gemini-2.0-flash"
    EMBED_MODEL  = "all-MiniLM-L6-v2"

    def __init__(self, api_key: str, chunk_size: int = 500, chunk_overlap: int = 50, top_k: int = 3):

        # ── Public variables (accessible from outside) ────────────────────────
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k
        self.chunks        = []          # stores all chunks after loading

        # ── Private variables (internal use only, prefix __) ──────────────────
        self.__api_key     = api_key     # sensitive — never expose
        self.__index       = None        # FAISS index — internal only
        self.__embed_model = None        # loaded once, reused

        # ── Setup ─────────────────────────────────────────────────────────────
        self.__configure_gemini()
        self.__load_embed_model()

    # =========================================================================
    # Private Methods — internal helpers, not meant to be called from outside
    # =========================================================================

    def __configure_gemini(self):
        """Configure Gemini API with the private API key."""
        genai.configure(api_key=self.__api_key)

    def __load_embed_model(self):
        """Load the sentence transformer model once during init."""
        print(f"Loading embedding model: {self.EMBED_MODEL}")
        self.__embed_model = SentenceTransformer(self.EMBED_MODEL)

    def __chunk_documents(self, documents: list) -> list:
        """Split documents into chunks. Private — called internally only."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = []
        for doc in documents:
            chunks.extend(splitter.split_text(doc))
        return chunks

    def __create_embeddings(self, texts: list) -> np.ndarray:
        """Encode texts into float32 embeddings. Private — internal only."""
        embeddings = self.__embed_model.encode(texts)
        return np.array(embeddings, dtype="float32")

    def __build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings. Private — internal only."""
        dims = embeddings.shape[1]
        index = faiss.IndexFlatL2(dims)
        index.add(embeddings)
        return index

    def __retrieve(self, query: str) -> list:
        """Retrieve top-k relevant chunks. Private — internal only."""
        query_embedding = self.__create_embeddings([query])
        distances, indices = self.__index.search(query_embedding, self.top_k)
        return [self.chunks[i] for i in indices[0]]

    def __build_prompt(self, query: str, context_chunks: list) -> str:
        """Construct the RAG prompt. Private — internal only."""
        context = "\n".join(context_chunks)
        return f"""You are a helpful AI assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

    def __call_gemini(self, prompt: str) -> str:
        """Call Gemini API with retry logic. Private — internal only."""
        model = genai.GenerativeModel(self.GEMINI_MODEL)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "ResourceExhausted" in str(e):
                    print(f"Rate limit hit. Retrying in 30s... ({attempt+1}/3)")
                    time.sleep(30)
                else:
                    raise e
        return "Error: Rate limit exceeded after 3 attempts."

    # =========================================================================
    # Public Methods — the interface for using this class from outside
    # =========================================================================

    def load_documents(self, documents: list):
        """
        Public — Load and index documents.
        Call this before asking any questions.
        """
        print(f"Loading {len(documents)} documents...")
        self.chunks          = self.__chunk_documents(documents)
        embeddings           = self.__create_embeddings(self.chunks)
        self.__index         = self.__build_index(embeddings)
        print(f"Indexed {len(self.chunks)} chunks successfully.")

    def ask(self, query: str) -> str:
        """
        Public — Ask a question against the loaded documents.
        Main entry point for querying the pipeline.
        """
        if self.__index is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        retrieved = self.__retrieve(query)

        print(f"\nQuery: {query}")
        print("-" * 50)
        print("Retrieved Chunks:")
        for i, chunk in enumerate(retrieved):
            print(f"  [{i+1}] {chunk}")

        prompt = self.__build_prompt(query, retrieved)
        answer = self.__call_gemini(prompt)

        print(f"\nGenerated Answer:\n{answer}")
        return answer

    def update_settings(self, chunk_size: int = None, chunk_overlap: int = None, top_k: int = None):
        """
        Public — Update pipeline settings after initialization.
        Note: changing chunk settings requires reloading documents.
        """
        if chunk_size:    self.chunk_size    = chunk_size
        if chunk_overlap: self.chunk_overlap = chunk_overlap
        if top_k:         self.top_k         = top_k
        print("Settings updated.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    DOCUMENTS = [
        "Artificial Intelligence is trying to match human intelligence",
        "Machine Learning is a subset of AI",
        "RAG stands for Retrieval Augmented Generation",
        "Solar Energy is a renewable energy source",
        "Deep Learning uses neural networks with many layers"
    ]

    # Create instance — public interface
    rag = RAGPipeline(
        api_key="YOUR_GEMINI_API_KEY",
        chunk_size=200,
        chunk_overlap=20,
        top_k=2
    )

    # Load documents — public method
    rag.load_documents(DOCUMENTS)

    # Ask questions — public method
    rag.ask("What is RAG?")
    rag.ask("What is Machine Learning?")

    # Update settings on the fly — public method
    rag.update_settings(top_k=3)
    rag.ask("Tell me about AI")