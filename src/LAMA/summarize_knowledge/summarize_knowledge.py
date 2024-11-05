from datetime import datetime
import json
import os
import pickle
from typing import Dict, List, Tuple

from numpy import array
import torch


class Summarizer:

    def __init__(self, model_name: str, template: str, dataset: str):
        current_datetime = datetime.now()
        self.dataset = dataset
        self.template = template
        # Format it as a string
        #datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        self.output_file = "./_knowledge" + model_name + ".json"
        if not os.path.exists(self.output_file) and self.output_file.endswith('.json'):
            with open(self.output_file, 'w') as f:
                f.write('{"knowledge": [],}')
            return
        if not self.output_file.endswith('.json'):
            raise ValueError('Output file must be a JSON file')
        if not os.path.exists(self.output_file) or not self.output_file.endswith('.json'):
            raise ValueError('Output file must be a JSON file')
    
    def store_knowledge(self, 
                    samples: Dict[str, str|int| List[str]], 
                    filtered_log_probs_list: torch.tensor, 
                    vocab: List[str],
                    masked_indices: List[int],
                    index_list: List[int]):
        output: List[Dict[str, str|int| List[str]]] = []
        for i, sample in enumerate(samples):
            mask_idx = masked_indices[i][0]
            log_probs = filtered_log_probs_list[i][mask_idx]
            predictions_ = torch.topk(log_probs, 10, 0)
            predictions = [vocab[index_list[j]] for j in predictions_.indices]
            if sample['obj_label'] in predictions:
                output.append({
                    'sentence': sample['masked_sentences'][0],
                    'subject': sample['sub_label'],
                    'template:' : self.template,
                    'dataset': self.dataset,
                    'object_ground_truth_idx': predictions.index(sample['obj_label']),
                    'object_predicted': predictions[0],
                    'object_predicted_10': predictions,
                    'filtered_log_probs': predictions_.values.tolist(),
                })
        self.append_knowledge_to_json(output)

    def append_knowledge_to_json(self, output: List[Dict[str, str|int| List[str]]]):
        try:
            # Read existing data
            with open(self.output_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file is not found or empty, start with an empty list
            existing_data = {"knowledge": [],}
        
        # Append new data
        existing_data['knowledge'].extend(output)

        # Write back to the file
        with open(self.output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def add_summary_to_json(self, output):
        try:
            # Read existing data
            with open(self.output_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file is not found or empty, start with an empty list
            existing_data = {"knowledge": []}

        # Append new data
        output['template'] = self.template
        output['dataset'] = self.dataset
        existing_data['summary'] = [output] if 'summary' not in existing_data else existing_data['summary'].append(output)

        # Write back to the file
        with open(self.output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    

        