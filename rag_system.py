import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import base64
from io import BytesIO
from openai import OpenAI

class RAGSystem:
    def __init__(self, upload_folder='uploads', openai_api_key='', openai_model='gpt-5-mini'):
        self.upload_folder = upload_folder
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_client = None
        self.collection = None
        self.chunks = []
        self.df = None
        self.trained = False
        
    def is_trained(self):
        return self.trained and self.collection is not None
    
    def extract_text_from_pdf(self, path):
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return " ".join(text)

    def clean_text(self, t):
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def train(self):
        """Extract, chunk, and embed all PDFs in upload folder"""
        # Extract text from PDFs
        docs = {}
        for f in os.listdir(self.upload_folder):
            if f.lower().endswith(".pdf"):
                raw = self.extract_text_from_pdf(os.path.join(self.upload_folder, f))
                docs[f] = self.clean_text(raw)
        
        if not docs:
            raise Exception("No PDF files found in upload folder")
        
        # Chunk the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = []
        for name, text in docs.items():
            for ch in splitter.split_text(text):
                self.chunks.append({"source": name, "content": ch})
        
        # Create ChromaDB collection
        self.chroma_client = chromadb.Client()
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection("study_material")
        except:
            pass
            
        self.collection = self.chroma_client.create_collection("study_material")
        
        # Embed and store chunks
        for i, ch in enumerate(tqdm(self.chunks, desc="Embedding Chunks")):
            emb = self.embedder.embed_query(ch["content"])
            self.collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[emb],
                metadatas=[{"source": ch["source"]}],
                documents=[ch["content"]],
            )
        
        # Create DataFrame for analytics
        self.df = pd.DataFrame(self.chunks)
        self.df["length"] = self.df["content"].apply(len)
        
        self.trained = True
        
        return {
            'num_documents': len(docs),
            'num_chunks': len(self.chunks),
            'avg_chunk_length': self.df['length'].mean(),
            'documents': list(docs.keys())
        }
    
    def query_openai(self, prompt, max_tokens=500):
        """Query OpenAI API"""
        try:
            if not self.openai_client:
                return (
                    "⚠️ **OpenAI API Key Not Set**\n\n"
                    "Please set the OPENAI_API_KEY environment variable.\n\n"
                    "**Quick Fix:**\n"
                    "```bash\n"
                    "export OPENAI_API_KEY=your-api-key-here\n"
                    "python app.py\n"
                    "```\n\n"
                    "You can get an API key from: https://platform.openai.com/api-keys"
                )
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI study tutor assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error contacting OpenAI: {e}"

    def retrieve_context(self, q, k=3):
        """Retrieve relevant chunks for a query"""
        q_emb = self.embedder.embed_query(q)
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        docs = res["documents"][0]
        return docs, "\n\n".join(docs)

    def answer_question(self, q):
        """Answer a question using RAG"""
        if not self.is_trained():
            raise Exception("Model not trained. Please upload and train first.")
        
        docs, context = self.retrieve_context(q)
        prompt = f"""Answer the question using the provided study material if relevant.
If not found, answer from general knowledge.

Context:
{context}

Question: {q}
Answer:"""
        reply = self.query_openai(prompt)
        return reply.strip()
    
    def get_analytics(self):
        """Generate analytics data and visualizations"""
        if not self.is_trained():
            raise Exception("Model not trained")
        
        analytics = {}
        
        # Basic statistics
        analytics['chunk_stats'] = {
            'count': len(self.df),
            'mean_length': float(self.df['length'].mean()),
            'std_length': float(self.df['length'].std()),
            'min_length': int(self.df['length'].min()),
            'max_length': int(self.df['length'].max())
        }
        
        # Top keywords
        vectorizer = CountVectorizer(stop_words="english", max_features=20)
        X = vectorizer.fit_transform(self.df["content"])
        freq = np.asarray(X.sum(axis=0)).ravel()
        keywords = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": freq})
        keywords = keywords.sort_values("count", ascending=False)
        analytics['top_keywords'] = keywords.to_dict('records')
        
        # Generate visualizations as base64 images
        analytics['charts'] = {}
        
        # 1. Chunk length distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["length"], bins=20, kde=True)
        plt.title("Distribution of Chunk Lengths")
        plt.xlabel("Characters per Chunk")
        analytics['charts']['length_dist'] = self._fig_to_base64()
        
        # 2. Top keywords bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x="count", y="term", data=keywords.head(15), palette="viridis")
        plt.title("Top Keywords in Study Material")
        analytics['charts']['keywords'] = self._fig_to_base64()
        
        # 3. Word cloud
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(self.df["content"]))
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud of Combined Content")
        analytics['charts']['wordcloud'] = self._fig_to_base64()
        
        # 4. PCA visualization (sample)
        sample_size = min(200, len(self.df))
        embs = [self.embedder.embed_query(t) for t in self.df["content"][:sample_size]]
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embs)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, c='blue')
        plt.title("PCA of Embedding Space")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        analytics['charts']['pca'] = self._fig_to_base64()
        
        # 5. Clustering
        kmeans = KMeans(n_clusters=min(5, sample_size), random_state=42)
        labels = kmeans.fit_predict(reduced)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("K-Means Clustering of Document Chunks")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        analytics['charts']['clustering'] = self._fig_to_base64()
        
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().to_dict()
        analytics['cluster_distribution'] = {f"Cluster {k}": int(v) for k, v in cluster_counts.items()}
        
        return analytics
    
    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def parse_mcqs(self, raw_text):
        """Parse MCQs from model output"""
        mcqs = []
        blocks = re.split(r"\n(?=Q\d+:|\*\*Q\d+:|\d+\.\s)", raw_text.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            q_match = re.search(r"Q\d*[:.)-]?\s*(.+)", block)
            options = re.findall(r"[A-D]\)?[.)]?\s*(.+)", block)
            correct = re.search(r"(Correct Answer|Answer)\s*[:\-–]?\s*(.+)", block, re.IGNORECASE)
            explanation = re.search(r"(Explanation)\s*[:\-–]?\s*(.+)", block, re.IGNORECASE)

            if q_match and len(options) >= 4:
                mcq = {
                    "Question": q_match.group(1).strip(),
                    "A": options[0].strip() if len(options) > 0 else "",
                    "B": options[1].strip() if len(options) > 1 else "",
                    "C": options[2].strip() if len(options) > 2 else "",
                    "D": options[3].strip() if len(options) > 3 else "",
                    "Correct_Answer": correct.group(2).strip() if correct else "",
                    "Explanation": explanation.group(2).strip() if explanation else "",
                }
                mcqs.append(mcq)
        return mcqs

    def generate_mcqs_from_chunk(self, text, num_questions=3):
        """Generate MCQs from a text chunk"""
        prompt = f"""
You are an educational AI tutor. Based on the following study content, generate {num_questions} multiple-choice questions.
Each question must have:
1. A question statement.
2. Four options labeled A), B), C), D).
3. A clearly marked correct answer.
4. A one-line explanation.

Format strictly as:
Q1: <question>
A) ...
B) ...
C) ...
D) ...
Correct Answer: ...
Explanation: ...

Study Material:
{text[:3000]}
"""
        response = self.query_openai(prompt, max_tokens=800)
        return response

    def generate_quiz(self, num_chunks=3, questions_per_chunk=3):
        """Generate a quiz from random chunks"""
        if not self.is_trained():
            raise Exception("Model not trained")
        
        selected_chunks = self.df.sample(min(num_chunks, len(self.df)))["content"].tolist()
        all_mcqs = []

        for ch in selected_chunks:
            quiz_text = self.generate_mcqs_from_chunk(ch, num_questions=questions_per_chunk)
            print(f"[DEBUG] Quiz response:\n{quiz_text[:500]}...")  # Debug output
            parsed = self.parse_mcqs(quiz_text)
            print(f"[DEBUG] Parsed {len(parsed)} questions")  # Debug output
            all_mcqs.extend(parsed)

        quiz_df = pd.DataFrame(all_mcqs)
        
        # Save to CSV
        os.makedirs("output", exist_ok=True)
        quiz_df.to_csv("output/generated_quiz.csv", index=False)
        
        return quiz_df
