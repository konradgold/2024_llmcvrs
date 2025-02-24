#!/bin/bash
#SBATCH -A doellner-student
#SBATCH --partition sorcery
#SBATCH -C "GPU_MEM:40GB"
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name "icl_translation_attention"
#SBATCH --output sbatch_out.txt
#SBATCH --time 14:00:00



cd 2024_llmcvrs/src
source "/hpi/fs00/home/konrad.goldenbaum/miniconda3/etc/profile.d/conda.sh"

conda activate llmcvrs

eval $(python -m dotenv)


wandb login WANDB_API_KEY

N_ITER1=5
WIDTH1=0.05
MIN1=0.05

# Second set of parameters
N_ITER2=5
WIDTH2=0.2
MIN2=0.2

echo "Running script with first set of parameters..."
python3 reduce_model_weight.py --n_iter $N_ITER1 --width $WIDTH1 --min $MIN1

echo "Execution completed!"

# First set of parameters
NR_QUERIES1=400
MODEL_PATH1="models/finetuned_gpt_0.05.pt"
OUTPUT_KNOWLEDGE1="LAMA_knowledge_ext/results/knowledge_0.05.json"
OUTPUT_SIMILARITY1="LAMA_knowledge_ext/results/similarity_0.05.json"

# Second set of parameters
NR_QUERIES2=400
MODEL_PATH2="models/finetuned_gpt_0.2.pt"
OUTPUT_KNOWLEDGE2="LAMA_knowledge_ext/results/knowledge_0.2.json"
OUTPUT_SIMILARITY2="LAMA_knowledge_ext/results/similarity_0.2.json"


echo "Running script with second set of parameters..."
python3 -m LAMA_knowledge_ext.get_knowledge \
  --nr_queries $NR_QUERIES2 \
  --model_path $MODEL_PATH2 \
  --output_knowledge $OUTPUT_KNOWLEDGE2 \
  --output_similarity $OUTPUT_SIMILARITY2 \
  --use_llm

OUTPUT_SENTENCES1="filter-openwebtext/filter_folder/knowledge_0.05.json"
OUTPUT_SENTENCES2="filter-openwebtext/filter_folder/knowledge_0.2.json"


OUTPUT_STORE1="filter-openwebtext/filter_folder/train_0.05.bin"
OUTPUT_STORE2="filter-openwebtext/filter_folder/train_0.2.bin"

python3 -m filter-openwebtext.generate_filter_trainer \
  --similarity $OUTPUT_SIMILARITY2 \
  --texts  $OUTPUT_SENTENCES2

python3 -m filter-openwebtext.fit_lsa \
  --sentences $OUTPUT_SENTENCES2 \
  --dataset_store $OUTPUT_STORE2
echo "Execution completed! Final script to run is train_model.py"

python3 train_model.py \
  --train_dataset 'filter-openwebtext/filter_folder/train_0.2.bin' \
  --model_file 'models/finetuned_gpt_0.2.pt' \
  --output_dir 'out/out02'

echo "Execution completed!"
