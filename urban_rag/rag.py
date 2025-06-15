import re
import os
import json
import torch
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from datetime import datetime

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

@dataclass
class AnalysisMetadata:
    """Metadata for each analysis section"""
    section_id: str
    analysis_type: str
    timestamp: datetime
    location: str
    metrics: Dict[str, float]
    raw_text: str

class UrbanAnalysisRAG:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-mpnet-base-v2",
        llm_model: str = "google/gemma-3-4b-it",#google/gemma-2b-it (12b)
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Urban Analysis RAG system.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize models
        self.device = torch.device("cuda:1")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.llm = Gemma3ForConditionalGeneration.from_pretrained(
            llm_model, device_map=self.device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(llm_model, use_fast=True)
        
        # Storage for processed data
        self.analysis_sections: Dict[str, AnalysisMetadata] = {}
        self.embeddings = None
        self.text_chunks = []
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving meaningful boundaries.
        """
        #sections = re.split(r'(?:\n\s*\n|\={3,})', text)
        #return [s.strip() for s in sections if s.strip()]
        return [seccion.strip() for seccion in text.split("-" * 80)]

    def process_text_data(self, text: str) -> None:
        """
        Process raw text data and organize it into analyzed sections.
        """
        self.text_chunks = self._chunk_text(text)
        
        # Create embeddings for all chunks
        self.embeddings = self.embedding_model.encode(
            self.text_chunks,
            convert_to_tensor=True,
            device=self.device
        )

    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant text chunks based on query.
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        
        similarity_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings)
        top_k_indices = torch.topk(similarity_scores, min(top_k, len(self.text_chunks))).indices
        
        return [self.text_chunks[int(i)] for i in top_k_indices]

    def generate_response(self, query: str) -> str:
        """
        Generate a response using retrieved context and an LLM.
        """
        relevant_chunks = self.retrieve_relevant_context(query, top_k=5)
        
        if not relevant_chunks:
            return "No relevant information found in the dataset."
        
        context = "\n\n".join(relevant_chunks)
        text = f"""Below is the relevant information retrieved from the transportation database:

{context}

Now, based on this information, answer the following question: {query}

Guidelines:
1. Base your answer strictly on the provided data. Do not include external or speculative information.
2. If the available data is too limited to fully answer the question, acknowledge this clearly.
3. Highlight any inconsistencies, contradictions, or data gaps you detect.
4. If the question is general but only specific district-level information is available, feel free to draw conclusions **based on those districts**, but **explicitly name them** in your response.
5. When comparing districts, use concrete numbers rather than vague comparisons
6. Your answer should be concise, analytical, and limited to a single well-structured paragraph.
7. Focus exclusively on the information relevant to the question; do not associate unrelated data fields in your response.

Response:"""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert in urban transportation analysis. Your task is to examine and interpret data related to various districts in a city."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.llm.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7, repetition_penalty=1.1)
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        #return f"Retrieved Context:\n\n{context}\n\nResponse:\n" + decoded.strip()
        return decoded.strip(), relevant_chunks

def load_full_verbalizations(verbalization_type, dict_verbalizations, path):
    # Load and process text data
    text_data = ""
    for i in range (len(dict_verbalizations[verbalization_type])):
        with open(path+dict_verbalizations[verbalization_type][i], "r") as f:
            text = f.read()
        lines = text.splitlines()
        text = lines[5:]
        #text = [s for s in text if not s.startswith("##")]# Remove header
        text_data += '\n'.join(text)
    
    return text_data

# Example usage
def main(args):
    verbalization_type       = args.verbalization
    path_queries             = args.path_queries
    path_output              = "output_results/"
    verbalization_files      = "verbalizations.json"
    verbalization_path       = "../multimodal_verbalization/transport-experiment/verbalization_results/"
    out_file_name_txt        = verbalization_type + "_output_responses.txt"
    output_path_txt          = os.path.join(path_output, out_file_name_txt)
    output_path_json         = os.path.join(path_output, verbalization_type + "_output_responses.json")
    
    # Initialize the RAG system
    rag = UrbanAnalysisRAG()

    # Load queries
    with open(path_queries, 'r') as f:
        queries = json.load(f)

    # Load verbalization file names
    with open(verbalization_files, 'r') as f:
        dict_verbalizations = json.load(f)

    text_data = load_full_verbalizations(verbalization_type, dict_verbalizations, verbalization_path)
    
    rag.process_text_data(text_data)
    output_data = {}
    with open(output_path_txt, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# Output responses using: {verbalization_type}\n")
        outfile.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"# Source file: {path_queries}\n")
        outfile.write(f"# Total records: {len(queries)}\n\n")
        # Generate responses
        for key, value in tqdm(queries.items(), desc="Processing queries"):
            outfile.write(f"\nQuery {key}: {value}\n")
            outfile.write("-" * 100)
            response, context = rag.generate_response(value)
            output_data[key] = {"query": value,
                               "context": context,
                               "answer": response}
            outfile.write(f"\nResponse: {response}\n")
    
    with open(output_path_json, "w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalization", type=str, required=True)
    parser.add_argument("--path_queries", type=str, required=True)
    args = parser.parse_args()
    main(args)