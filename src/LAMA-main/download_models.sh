#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"


echo "BERT BASE LOWERCASED"
if [[ ! -f bert/uncased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
  unzip uncased_L-12_H-768_A-12.zip
  rm uncased_L-12_H-768_A-12.zip
  cd uncased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
  tar -xzf bert-base-uncased.tar.gz
  rm bert-base-uncased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "lowercase models"
echo "OpenAI GPT"
if [[ ! -f gpt/openai-gpt/config.json ]]; then
  rm -rf 'gpt/openai-gpt'
  mkdir -p 'gpt/openai-gpt'
  cd 'gpt/openai-gpt'
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json' -O vocab.json
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt' -O merges.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json' -O 'config.json'
  cd ../..
fi



cd "$ROOD_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_cased.txt" ]; then
  python -m lama.vocab_intersection
else
  echo 'Already exists. Run to re-build:'
  echo 'python util_KB_completion.py'
fi

