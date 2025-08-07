import re
import os
import ast
import json
import torch
import time
import argparse
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

client = OpenAI(api_key="ADD_API_KEY")
device = torch.device("cuda:1")

def build_prompt(type_prompt: str, data: dict, sentences: list = None):
    base_content = f"""You are a helpful and precise assistant. Follow the instructions carefully to complete the task."""
    if type_prompt == "faithfulness_statements":
        base_prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
question: {data["query"]}
answer: {data["answer"]}
Note: Return the output strictly as a Python list of strings. Each element should be a standalone statement derived from the answer. Do not include any Python code blocks—only the raw list of strings."""
    elif type_prompt == "faithfulness_inferred":
        context = "\n".join(data["context"])
        base_prompt = f"""Given the following context:
{context}
And the list of statements:
{data["statements"]}
Determine whether each statement is supported by the information in the context. For each statement, provide a final verdict: 'Yes' if it is supported, or 'No' if it is not.
Important: Return your answer as a Python list of strings (e.g., ["Yes", "No", "Yes"]), preserving the order of the statements. Do not include any Python code blocks—only the raw list of strings. Do not deviate from the specified format."""
    elif type_prompt == "answer_relevance":
        base_prompt = f"""Generate a list of relevant questions based on the following answer:
Answer: {data["answer"]}
Important: Return the output strictly as a Python list of strings, preserving the order of the questions. Do not include any Python code blocks—only the plain list of strings. Stick exactly to this format."""
    elif type_prompt == "context_relevance":
        base_prompt = f"""Extract the most relevant sentences from the provided context:
Context: {sentences}
These sentences should be potentially helpful in answering the following question:
Question: {data["query"]}
You must extract as many sentences as possible, as long as they are relevant to the question. If no relevant sentences are found, return an empty list. Otherwise, return a Python list of strings containing only the relevant sentences.
Important constraints:
- Do not modify the sentences from the original context in any way.
- Return only the raw Python list of strings, preserving the original order of the sentences.
- Do not include any Python code blocks or additional formatting."""

    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": base_content}],
        },
        {
            "role": "system",
            "content": [{"type": "text", "text": base_prompt}],
        }
    ]
    return prompt
    
def get_response(prompt, model = None, tokens = 512):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure the model name is correct
        messages=prompt,
        max_tokens=tokens,
    )
    return str(response.choices[0].message.content)

def get_valid_list_from_prompt(prompt, max_retries=5):
    retries = 0
    while retries < max_retries:
        response = get_response(prompt)
        try:
            result = ast.literal_eval(response)
            if isinstance(result, list):
                return result
        except Exception:
            pass  # simplemente reintenta
        
        retries += 1
    
    return None

def contains_afirmation(s):
    s = s.strip().lower()
    return "yes" in s or "y" in s or "1" in s

def split_into_sentences(context):
    pattern = r'(?<!\d)\.(?!\d)\s+'
    return re.split(pattern, context)

def compute_faithfulness_score(inferred_statements: list):
    if inferred_statements is not None and len(inferred_statements) > 0:
        total = len(inferred_statements)
        afirmations = 0
        for verdict in inferred_statements:
            if contains_afirmation(verdict):
                afirmations += 1
        return round(afirmations/total, 3)
    else:
        return None

def compute_answer_relevance_score(original_question: str, questions: list):
    if questions is not None and len(questions) > 0:
        embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        original_embeddings =  embedding_model.encode(
            original_question,
            convert_to_tensor=True,
            device=device
        )
        sum_similarity = 0.0
        total = len(questions)
        for question in questions:
            qi_embeddings =  embedding_model.encode(
                question,
                convert_to_tensor=True,
                device=device
            )
            similarity_score = torch.cosine_similarity(original_embeddings, qi_embeddings, dim=0)
            sum_similarity += similarity_score.item()
        return round(sum_similarity/total, 3)
    else:
        return None

def compute_context_relevance_score(all_sentences: list, selected_sentences: list):
    if selected_sentences is not None and len(selected_sentences) > 0:
        return round(len(selected_sentences)/len(all_sentences), 3)
    else:
        return None

def main(args):
    path_answers            = "output_results/"
    answers_file            = args.answers_file
    faithfulness_score_list = []
    ar_score_list           = []
    cr_score_list           = []
    output_file             = args.output_file
    results                 = {}
    
    # Load answers
    with open(path_answers + answers_file, 'r') as f:
        answers = json.load(f)

    for key, value in tqdm(answers.items(), desc="Processing answers"):
        #Compute Faithfulness Score
        prompt = build_prompt("faithfulness_statements", value)
        list_statements = get_valid_list_from_prompt(prompt)
        value["statements"] = list_statements
        
        prompt = build_prompt("faithfulness_inferred", value)
        list_inferred_statements = get_valid_list_from_prompt(prompt)

        faithfulness_score = compute_faithfulness_score(list_inferred_statements)
        if faithfulness_score is not None:
            faithfulness_score_list.append(faithfulness_score)
        
        #Compute Answer Relevance
        prompt = build_prompt("answer_relevance", value)
        list_questions = get_valid_list_from_prompt(prompt)
        answer_relevance_score = compute_answer_relevance_score(value["query"], list_questions)
        if ar_score_list is not None:
            ar_score_list.append(answer_relevance_score)
        
        #Compute Context Relevance
        context = "\n".join(value["context"])
        sentences = split_into_sentences(context)
        prompt = build_prompt("context_relevance", value, sentences)
        relevant_sentences = get_valid_list_from_prompt(prompt)
        context_relevance_score = compute_context_relevance_score(sentences, relevant_sentences)
        if context_relevance_score is not None:
            cr_score_list.append(context_relevance_score)

        #Save data to dictionary
        results[key] = {"faithfulness": faithfulness_score,
                       "answer_relevance": answer_relevance_score,
                       "context_relevance": context_relevance_score}
        
        time.sleep(1)

    print(f"Mean Faithfulness Score: {round(statistics.mean(faithfulness_score_list), 3)}")
    print(f"Mean Answer Relevance Score: {round(statistics.mean(ar_score_list), 3)}")
    print(f"Mean Context Relevance Score: {round(statistics.mean(cr_score_list), 3)}")

    with open(path_answers + output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
