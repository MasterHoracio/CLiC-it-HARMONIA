import torch
import json
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class UrbanDataVerbalizer:
    def __init__(self,
                 model_name = "meta-llama/Llama-3.1-8B-Instruct",
                 output_dir = "verbalization_results",
                 batch_size = 50,
                verbalization_level = "census"):
        """
        Initialize the verbalization pipeline with enhanced configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.verbalization_level = verbalization_level

        # Increased sequence length for comprehensive narratives
        self.max_length = 1024  # Increased from previous implementation (2048)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        # Updated generation pipeline with explicit truncation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def prepare_row_context(self, row):
        """
        Prepare contextual identifiers for the row, based on the verbalization level

        Args:
            row (pd.Series): DataFrame row

        Returns:
            str: Formatted context string
        """
        
        if self.verbalization_level == "zone":
            return (f"Year {row['year']}, Statistical Zone {row['zone_stat']} ({row['desc_zone']}), "
                    f"District {row['district']}: ")
        elif self.verbalization_level == "district":
            return (f"Year {row['year']}, District {row['district']}: ")
        elif self.verbalization_level == "census":
            return (f"Year {row['year']}, Census Area {row['cens']}, "
                    f"Statistical Zone {row['zone_stat']} ({row['desc_zone']}), "
                    f"District {row['district']}: ")

    def generate_comprehensive_narrative(self, row, field_description, example = None):
        """
        Generate a single-paragraph comprehensive narrative

        Args:
            row (pd.Series): DataFrame row with urban data

        Returns:
            str: Comprehensive verbalization of the row
        """
        try:
            # Prepare row context and facts
            row_context = self.prepare_row_context(row)
            facts = {key: row.get(key, 'Not available') for key in row.index}

            # Construct detailed prompt for single-paragraph narrative
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert urban data analyst. Convert census and transport data into clear narratives.
<|start_header_id|>user<|end_header_id|>\n
Generate a comprehensive, single-paragraph narrative about an urban area based on the following numeric data. 
The narrative must:
- Be concise, informative, cover all key aspects of the urban landscape, and limit to a single paragraph.
- Include and reflect the exact values as given in the Numeric Facts, without modification or approximation.
- Focus solely on describing the attributes defined in Field Descriptions, matching each field with its corresponding value.
- Avoid drawing conclusions, making assumptions, or interpreting the significance of the data.
- Avoid comparing the data to other entries, past values, or the example provided.

Unique Identifier: {row_context}
Field Descriptions: {field_description}"""
            if example is not None:
                prompt += f"""
Example:
{example}"""
            prompt += f"""
Numeric Facts: {json.dumps(facts)}
Generated Narrative: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
            
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            # Generate narrative with comprehensive settings
            response = self.generator(
                prompt,
                max_new_tokens=self.max_length,  # Increased sequence length
                num_return_sequences=1,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                truncation=True,  # Explicitly activate truncation
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_id,
                repetition_penalty=1.2
            )

            # Extract and combine context with generated text
            narrative = response[0]['generated_text'].replace(prompt, '').strip()
            return row_context + narrative

        except Exception as e:
            print(f"Error generating narrative: {e}")
            return f"Error processing row: {str(e)}"

    def verbalize_dataset(self, input_csv, output_file = 'verbalized_urban_data.txt', type_approach = "zero-shot"):
        """
        Verbalize entire dataset with progress tracking
        """
        df = pd.read_csv(input_csv)
        total_rows = len(df)

        output_path = os.path.join(self.output_dir, output_file)

        with open('../data/input/description_' + self.verbalization_level + '.json', 'r') as file:
            field_description = json.load(file)

        if type_approach == "few-shot":
            with open('../data/input/sample_' + self.verbalization_level + '.txt', 'r') as file:
                example = file.read()
        else:
            example = None

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Urban Data Verbalization Results ({type_approach})\n")
            outfile.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"# Source file: {input_csv}\n")
            outfile.write(f"# Total records: {total_rows}\n\n")
            
            with tqdm(total=total_rows, desc="Verbalizing Urban Data from " + self.verbalization_level) as pbar:
                for i in range(0, total_rows, self.batch_size):
                    batch = df.iloc[i:i+self.batch_size]

                    for _, row in batch.iterrows():
                        # Generate narrative
                        narrative = self.generate_comprehensive_narrative(row, field_description, example)

                        # Write to output file with metadata
                        if self.verbalization_level == "census":
                            record_id = f"Census {row['cens']}, Zone {row['zone_stat']}, District {row['district']}, Year {row['year']}"
                        elif self.verbalization_level == "zone":
                            record_id = f"Zone {row['zone_stat']}, District {row['district']}, Year {row['year']}"
                        elif self.verbalization_level == "district":
                            record_id = f"District {row['district']}, Year {row['year']}"
                        
                        outfile.write(f"## {record_id}\n\n")
                        outfile.write(narrative + "\n\n")
                        outfile.write("-" * 80 + "\n\n")

                        # Update progress bar
                        pbar.update(1)

        print(f"Verbalization complete. Results saved to {output_path}")

# Usage Example
def main(args):
    path_input           = "../data/input/"
    path_output          = "../verbalization_results/"
    verbalization_level  = args.verbalization_level # census, zone, district
    type_approach        = args.type_approach # "few-shot", "zero-shot"
    
    if verbalization_level == "zone" or verbalization_level == "district":
        in_file_name = "population-and-transport-" +  verbalization_level + ".csv"
    else:
        in_file_name = "population-and-transport.csv"

    if type_approach == "few-shot":
        out_file_name = "few_shot_verbalized_urban_data_" + verbalization_level + ".txt"
    elif type_approach == "zero-shot":
        out_file_name = "zero_shot_verbalized_urban_data_" + verbalization_level + ".txt"
    
    verbalizator = UrbanDataVerbalizer(batch_size = 50, verbalization_level = verbalization_level, output_dir = path_output)
    verbalizator.verbalize_dataset(path_input + in_file_name, out_file_name, type_approach)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalization_level", type=str, required=True)
    parser.add_argument("--type_approach", type=str, required=True)
    args = parser.parse_args()
    main(args)