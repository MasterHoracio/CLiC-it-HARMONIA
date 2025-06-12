import torch
import json
import os
import time
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

class UrbanDataVerbalizer:
    def __init__(self,
                 model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct",
                 output_dir = "verbalization_results",
                 batch_size = 50):
        """
        Initialize the verbalization pipeline with enhanced configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Increased sequence length for comprehensive narratives
        self.max_length = 1024  # Increased from previous implementation (2048)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model.tie_weights()

    def get_full_values(self, df, field, image):
        new_values = {}
        for index, row in df.iterrows():
            new_key = "Statisticas Zone: " + str(row["zone_stat"]) + " (name: " + row["desc_zone"] + ")"
            if len(image) == 2:#Percentage
                if row[image[1]] == 0:
                    new_v = round(((row[field]/(row[image[1]] + 1))*100), 2)
                else:
                    new_v = round(((row[field]/row[image[1]])*100), 2)
            else:
                new_v = row[field]
            if new_key not in new_values:
                new_values[new_key] = {row["year"]: new_v}
            else:
                new_values[new_key][row["year"]] = new_v
                new_values[new_key]["difference"] = new_values[new_key][2019] - new_values[new_key][2012]
        sorted_value = dict(sorted(new_values.items(), key=lambda item: item[1]['difference'], reverse=True))
        dif_values = []
        for item in sorted_value:
            if sorted_value[item]["difference"] > 0:
                if len(image) == 2:#Percentage
                    dif_values.append(item + " 2012: " + str(sorted_value[item][2012]) + "%, 2019: " + str(sorted_value[item][2019]) + "%, increase: " + str(round(sorted_value[item]['difference'], 2)) + "%")
                else:
                    dif_values.append(item + " 2012: " + str(sorted_value[item][2012]) + ", 2019: " + str(sorted_value[item][2019]) + ", increase: " + str(round(sorted_value[item]['difference'], 2)) )
            else:
                if len(image) == 2:#Percentage
                    dif_values.append(item + " 2012: " + str(sorted_value[item][2012]) + "%, 2019: " + str(sorted_value[item][2019]) + "%, decrease: " + str(round(sorted_value[item]['difference'], 2)) + "%")
                else:
                    dif_values.append(item + " 2012: " + str(sorted_value[item][2012]) + ", 2019: " + str(sorted_value[item][2019]) + ", decrease: " + str(round(sorted_value[item]['difference'], 2)) )
        return dif_values
        
    def generate_comprehensive_narrative(self, df, field, field_description, image):
        """
        Generate a single-paragraph comprehensive narrative

        Args:
            row (pd.Series): DataFrame row with urban data

        Returns:
            str: Comprehensive verbalization of the row
        """
        values = self.get_full_values(df, field, image)
        try:
            # Construct detailed prompt for single-paragraph narrative
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert urban data analyst. Your task is to generate clear and precise narratives based on census and transport data for the city of Turin.
<|start_header_id|>user<|end_header_id|>\n
Generate a comprehensive narrative that analyzes and compares the {field_description.lower()} across the statistical zones of Turin, based on the provided comparison maps. The image displays comparison data for the years 2012 and 2019.

<|image|>

Your narrative must:
- Be concise, informative, and clearly highlight key patterns and trends in the {field_description.lower()}, considering both temporal changes (between 2012 and 2019) and within-year variations, where relevant.
- Provide a Top-summary for each of the following:
    - The most common patterns observed across zones.
    - Zones with the highest increases in values from 2012 to 2019 (i.e., where 2019 value > 2012 value).
    - Zones with the largest decreases in values from 2012 to 2019 (i.e., where 2019 value < 2012 value).
- Use the exact numerical values provided for each statistical zone—do not round, estimate, or omit any data.
- Refrain from interpreting, inferring causes, or comparing with any external datasets or years outside of 2012 and 2019.

Below are the statistical zones with their respective values for the selected field in 2012 and 2019:\n{values}

Generated Narrative: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
            
            loaded_image = Image.open("img/"+image[0]).convert("RGB")
            
            inputs_question = self.processor(
                    text=prompt, 
                    images = loaded_image, 
                    add_special_tokens=False,
                    padding = True, 
                    return_tensors="pt"
                    ).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs_question, 
                                    max_new_tokens=1024, 
                                    temperature=0.6,
                                    top_p=0.9,
                                    do_sample=True)
            narrative = self.processor.decode(output[0])

            return narrative

        except Exception as e:
            print(f"Error generating narrative: {e}")
            return f"Error processing row: {str(e)}"

    def verbalize_dataset(self, input_csv, images_file, output_file = 'verbalized_urban_data.txt'):
        """
        Verbalize entire dataset with progress tracking
        """
        df = pd.read_csv(input_csv)
        
        output_path = os.path.join(self.output_dir, output_file)

        with open('../../data/input/description_zone.json', 'r') as file:
            field_description = json.load(file)

        with open(images_file, 'r') as file:
            images = json.load(file)
        total_rows = len(images)
        delimitator = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Urban Data Map Verbalization Results\n")
            outfile.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"# Source file: {input_csv}\n")
            outfile.write(f"# Total records: {total_rows}\n\n")
            
            for key, value in tqdm(images.items(), desc="Verbalizing Map Data from Statistical Z.", total=total_rows):
                # Generate narrative
                narrative = self.generate_comprehensive_narrative(df, key, field_description[key], value)
                narrative = narrative.split(delimitator)
                record_id = "Analysis of the " + field_description[key].lower() + " for the years 2012 and 2019."
                outfile.write(f"## {record_id}\n\n")
                outfile.write(narrative[1] + "\n\n")
                outfile.write("-" * 80 + "\n\n")

        print(f"Verbalization complete. Results saved to {output_path}")

# Usage Example
def main(args):
    path_input           = "../../data/input/"
    path_output          = "../../verbalization_results/"
    images_file          = args.images

    in_file_name  = "population-and-transport-zone.csv"
    out_file_name = "zero_shot_verbalized_map_urban_data.txt"
    
    verbalizator = UrbanDataVerbalizer(batch_size = 50, output_dir = path_output)
    verbalizator.verbalize_dataset(path_input + in_file_name, images_file, out_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    args = parser.parse_args()
    main(args)