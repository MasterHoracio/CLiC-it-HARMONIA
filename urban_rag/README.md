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
