import re
import os
import json
import torch
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

import google.generativeai as genai
genai.configure(api_key="ADD_API_KEY")

client = OpenAI(api_key="ADD_API_KEY")

def build_prompt(llm: str, query: str):
    if llm == "gemini":
        prompt = f"""You are an expert in demographic and transportation aspects of the city of Turin (Torino), Italy.\n
Based on detailed census data and transport-related statistics, your task is to provide accurate, very concise, and short answers to user questions regarding population, public transportation availability, connectivity, district statistics, and related metrics.
If the required information is not available, simply reply: 'I cannot answer the question due to lack of necessary data'.\n
Question: {query}"""
    elif llm == "gpt":
        base_content = f"""You are an expert in demographic and transportation aspects of the city of Turin (Torino), Italy.\n"""
        additional_content = f"""Based on detailed census data and transport-related statistics, your task is to provide accurate, very concise, and short answers to user questions regarding population, public transportation availability, connectivity, district statistics, and related metrics.
If the required information is not available, simply reply: 'I cannot answer the question due to lack of necessary data'.\n
Question: {query}"""
        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": base_content}],
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": additional_content}],
            }
        ]
    return prompt
    
def get_response(llm, prompt, model = None, tokens = 512):
    if llm == "gemini":
        response = model.generate_content(prompt,generation_config={
        "max_output_tokens": tokens})
        return response.text
    elif llm == "gpt":
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure the model name is correct
            messages=prompt,
            max_tokens=tokens,
        )
        return str(response.choices[0].message.content)
    return None

# Example usage
def main(args):
    llm                  = args.llm
    path_queries         = args.path_queries
    path_output          = "output_results/"
    out_file_name        = llm + "_output_responses.txt"
    output_path          = os.path.join(path_output, out_file_name)
    output_path_json     = os.path.join(path_output, llm + "_output_responses.json")
    
    # Load queries
    with open(path_queries, 'r') as f:
        queries = json.load(f)

    if llm == "gpt":
        model_id = "gpt-4o-mini"
        model = None
    elif llm == "gemini":
        model_id = "models/gemini-2.0-flash"
        model = genai.GenerativeModel(model_name=model_id)
    
    output_data = {}
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# Output responses using: {model_id}\n")
        outfile.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"# Source file: {path_queries}\n")
        outfile.write(f"# Total records: {len(queries)}\n\n")
        # Generate responses
        for key, value in tqdm(queries.items(), desc="Processing queries"):
            outfile.write(f"\nQuery {key}: {value}\n")
            outfile.write("-" * 100)
            prompt = build_prompt(llm, value)
            response = get_response(llm, prompt, model)
            outfile.write(f"\nResponse: {response}\n")
            output_data[key] = {"query": value,
                               "answer": response}
            time.sleep(5)
    
    with open(output_path_json, "w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--path_queries", type=str, required=True)
    args = parser.parse_args()
    main(args)
