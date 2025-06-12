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