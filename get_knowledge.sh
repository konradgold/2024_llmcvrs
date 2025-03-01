#!/bin/bash
#SBATCH -A doellner-student
#SBATCH --partition sorcery
#SBATCH -C "GPU_MEM:40GB"
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name "baseline_knowledge"
#SBATCH --output baseline_knowledge.txt
#SBATCH --time 2:00:00



cd 2024_llmcvrs/src
source "/hpi/fs00/home/konrad.goldenbaum/miniconda3/etc/profile.d/conda.sh"

conda activate llmcvrs

souce .env

# First set of parameters

# Second set of parameters
NR_QUERIES2=400
MODEL_PATH2="baseline"
OUTPUT_KNOWLEDGE2="LAMA_knowledge_ext/results/baseline.json"
OUTPUT_SIMILARITY2="LAMA_knowledge_ext/results/baseline.json"


echo "Running script 1..."
python -m LAMA_knowledge_ext.get_knowledge \
  --nr_queries $NR_QUERIES2 \
  --model_path $MODEL_PATH2 \
  --output_knowledge $OUTPUT_KNOWLEDGE2 \
  --output_similarity $OUTPUT_SIMILARITY2 \
  --use_llm

OUTPUT_SENTENCES2="filter-openwebtext/filter_folder/knowledge_0.2.json"

echo "Running script 2..."
python -m filter-openwebtext.generate_filter_trainer \
  --similarity $OUTPUT_SIMILARITY2 \
  --texts  $OUTPUT_SENTENCES2


echo "Execution completed!"
