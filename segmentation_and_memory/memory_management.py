import redis
import faiss
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional, Tuple
import os
from transformers import AutoTokenizer, AutoModel
import torch

class MemoryManager:
    def __init__(self, faiss_dimension=768, redis_host='localhost', redis_port=6379, redis_db=0, index_file="faiss_index.bin"):
        self.index = faiss.IndexFlatL2(faiss_dimension)
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.dimension = faiss_dimension
        self.session_id_map = []
        self.index_file = index_file
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.load_index()  # Load existing index at initialization

    def generate_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().astype('float32')

        # Ensure embedding is 2D (n, d) where n=1 and d=768
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)  # shape (1, 768)

        # Debugging: print shape of the embedding
        logging.info(f"Generated embedding shape: {embedding.shape}")

        faiss.normalize_L2(embedding)  # Normalize the embedding
        return embedding

    def store_session(self, text: str, audio_file: str) -> str:
        # Generate the embedding for the text
        embedding = self.generate_text_embedding(text)
        
        # Debugging: print the shape of the embedding before adding to FAISS
        logging.info(f"Embedding shape before adding to FAISS: {embedding.shape}")

        # Add the embedding to FAISS
        try:
            self.index.add(embedding)  # Add the embedding to FAISS index
        except Exception as e:
            logging.error(f"Error adding embedding to FAISS: {str(e)}")
            return None  # Return None if adding to FAISS fails
        
        # Create a unique session ID
        session_id = f"session_{self.index.ntotal - 1}"  # Use the current index in FAISS
        
        # Store session data in Redis
        self.session_id_map.append(session_id)
        self.redis_client.hset(f"session:{session_id}", mapping={
            "text": text,
            "audio": audio_file,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save the FAISS index
        self.save_index()
        logging.info(f"Stored session: {session_id}, text: '{text[:50]}...'")
        
        return session_id

    def search_session(self, text: str, threshold: float = 0.1) -> Optional[Tuple[str, str]]:
        if self.index.ntotal == 0:
            logging.info("FAISS index empty")
            return None, None

        # Generate the query embedding for the text
        query_embedding = self.generate_text_embedding(text)

        # Debugging: print the shape of the query embedding before searching
        logging.info(f"Query embedding shape before searching: {query_embedding.shape}")

        # Perform the search in FAISS
        distances, indices = self.index.search(query_embedding, k=1)

        # If we find a match below the threshold
        if indices[0][0] >= 0 and distances[0][0] < threshold:
            session_id = self.session_id_map[indices[0][0]]
            audio_file = self.redis_client.hget(f"session:{session_id}", "audio")
            matched_text = self.redis_client.hget(f"session:{session_id}", "text")
            logging.info(f"Match found: {session_id}, text: '{matched_text}', distance: {distances[0][0]}")
            return matched_text, audio_file

        logging.info(f"No match for: '{text}', min distance: {distances[0][0]}")
        return None, None

    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.index_file + ".map", "w") as f:
            f.write("\n".join(self.session_id_map))
        logging.info(f"Saved FAISS index to {self.index_file}")

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.index_file + ".map", "r") as f:
                self.session_id_map = f.read().splitlines()
            logging.info(f"Loaded FAISS index from {self.index_file}, {self.index.ntotal} entries")
        else:
            logging.info("No existing FAISS index found")

    def clear_cache(self):
        self.index.reset()
        self.session_id_map = []
        for key in self.redis_client.scan_iter("session:*"):
            self.redis_client.delete(key)
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
            os.remove(self.index_file + ".map")
        logging.info("Cache cleared")
