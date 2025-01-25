

#!/bin/bash
# First set of parameters
N_ITER1=5
WIDTH1=0.2
MIN1=0.05

# Second set of parameters
N_ITER2=5
WIDTH2=0.2
MIN2=0.2

echo "Running script with first set of parameters..."
python3 reduce_model_weight.py --n_iter $N_ITER1 --width $WIDTH1 --min $MIN1

echo "Running script with second set of parameters..."
python3 reduce_model_weight.py --n_iter $N_ITER2 --width $WIDTH2 --min $MIN2

echo "Execution completed!"


# First set of parameters
NR_QUERIES1=200
MODEL_PATH1="models/finetuned_gpt_0.05.pt"
OUTPUT_KNOWLEDGE1="LAMA_knowledge_ext/results/knowledge_0.05.json"
OUTPUT_SIMILARITY1="LAMA_knowledge_ext/results/similarity_0.05.json"
USE_LLM1=true

# Second set of parameters
NR_QUERIES2=200
MODEL_PATH2="models/finetuned_gpt_0.2.pt"
OUTPUT_KNOWLEDGE2="LAMA_knowledge_ext/results/knowledge_0.2.json"
OUTPUT_SIMILARITY2="LAMA_knowledge_ext/results/similarity_0.2.json"
USE_LLM2=false

echo "Running script with first set of parameters..."
python3 -m LAMA_knowledge_ext.get_knowledge \
  --nr_queries $NR_QUERIES1 \
  --model_path $MODEL_PATH1 \
  --output_knowledge $OUTPUT_KNOWLEDGE1 \
  --output_similarity $OUTPUT_SIMILARITY1 \
  --use_llm

echo "Running script with second set of parameters..."
python3 -m LAMA_knowledge_ext.get_knowledge \
  --nr_queries $NR_QUERIES2 \
  --model_path $MODEL_PATH2 \
  --output_knowledge $OUTPUT_KNOWLEDGE2 \
  --output_similarity $OUTPUT_SIMILARITY2 \
  --use_llm


python3 -m filter-openwebtext.generate_filter_trainer
python3 -m filter-openwebtext.fit_lsa

#echo "Execution completed! Final script to run is train_model.py"

#python3 train_model.py

echo "Execution completed!"