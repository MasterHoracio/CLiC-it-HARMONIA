# RAG for Inclusive Decision Making

## Baseline

This repository includes the `baseline.py` script, which is designed to evaluate language models using a set of predefined queries.

The `baseline.py` script takes two input parameters:

- `--llm`: Specifies the language model to use for the baseline. Supported values are:
  - `gpt` for OpenAI's GPT model (gpt-4o-mini)
  - `gemini` for Google's Gemini model (gemini-2.0-flash)

- `--path_queries`: The path to a `.json` file containing the list of queries that will be submitted to the selected language model.

The script loads the queries from the provided JSON file and submits them to the specified LLM, recording the responses for further evaluation.

To run the script with either GPT or Gemini as the LLM, use the following commands:

```bash
# Using GPT
python3 baseline.py --llm gpt --path_queries queries.json

# Using Gemini
python3 baseline.py --llm gemini --path_queries queries.json
```

## RAG

This repository also includes the `rag.py` script, which implements a Retrieval-Augmented Generation (RAG) approach for evaluating language model responses in inclusive decision-making tasks.

The `rag.py` script requires the following input parameters:

- `--path_queries`: The path to a `.json` file containing the list of queries to be submitted to the selected language model.
- `--verbalization`: Specifies the verbalization strategy to be used in the experiment. Supported values are:
  - `zero_shot_general_verbalization`
  - `few_shot_general_verbalization`
  - `perspective_verbalization`

To execute the RAG script with the available verbalization strategies, use the following commands:

```bash
# Using zero-shot general verbalization
python3 rag.py --verbalization zero_shot_general_verbalization --path_queries queries.json

# Using few-shot general verbalization
python3 rag.py --verbalization few_shot_general_verbalization --path_queries queries.json

# Using perspective verbalization
python3 rag.py --verbalization perspective_verbalization --path_queries queries.json
```

## Evaluación del sistema RAG

This repository provides instructions for evaluating the RAG (Retrieval-Augmented Generation) system using the `evaluate.py` script.

### How to Run the Evaluation

To run the evaluation, execute the `evaluate.py` script with the following two required arguments:

- `--answers_file`: the path to the JSON file containing the RAG system's answers.
- `--output_file`: the name of the file where the evaluation results will be saved.

The evaluation measures the following metrics:
- **Faithfulness**: the degree to which the answer is supported by the retrieved context.
- **Answer Relevance**: how well the answer addresses the question.
- **Context Relevance**: how relevant the retrieved context is to the question.

### Example Commands

Here are some example configurations for executing the evaluation:

```bash
python3 evaluate.py --answers_file zero_shot_general_verbalization_output_responses.json --output_file evaluation_results_zero_shot_general_verbalization.json

python3 evaluate.py --answers_file few_shot_general_verbalization_output_responses.json --output_file evaluation_results_few_shot_general_verbalization.json

python3 evaluate.py --answers_file perspective_verbalization_output_responses.json --output_file evaluation_results_perspective_verbalization.json
