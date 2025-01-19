from scipy.stats import f
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
import re
import tiktoken

class SimilarityCalculator:

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.rogue = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def calculate_similarity(self, query, probs, predictions, truth, use_llm=False):
        """
        probs: (samples, max_new_tokens, self.config.vocab_size)
        tokens: (nr_samples, nr_new_tokens, self.config.vocab_size)
        """
        out_scores = {}
        reference = query + truth
        # Cosine Similarity between query and predictions
        potential_answers = []
        for prediction in predictions:
            potential_answers.append(query + prediction)
        
        # Rogue score
        out_scores['rogue'] = self._rogue(truth, predictions)

        # BLEU score
        out_scores['bleu'] = self._bleu_scores(reference, potential_answers)

        # Calculate probability of truth in predictions (using cross-entropy loss)
        out_scores["loss"] = self._calculate_loss(probs, truth)
        
        if use_llm:
            out_scores["llm_judgement"] = self._llm_judgement(reference, potential_answers)
        else:
            out_scores["llm_judgement"] = 0.

        return out_scores

    def _rogue(self, truth, predictions):
        rogue_scores = []
        for prediction in predictions:
            rogue_scores.append(self.rogue.score(prediction, truth))
        return rogue_scores

    def _bleu_scores(self, reference, potential_answers):
        bleu_scores = []
        for pot in potential_answers:
            assert isinstance(pot, str), f"Expected string, got {type(pot)}"
            assert isinstance(reference, str), f"Expected string, got {type(reference)}"
            bleu_scores.append(sentence_bleu(reference.split(), pot.split()))
        return bleu_scores

    
    def _calculate_loss(self, probs, truth):
        truth_idx = self.encode(truth)
        loss = 0.
        n, m = probs.shape[0], probs.shape[1]
        probs = probs.view(n * m, -1)
        for i in range(len(truth_idx)):
            t_idx = torch.tensor([truth_idx[i]], dtype=torch.long)
            truth_idx = torch.tensor([t_idx] * (n * m), dtype=torch.long)
            loss += self.loss(probs, truth_idx).item()
        assert len(truth_idx) > 0, f"Expected length > 0, got {len(truth_idx)}"
        loss /= len(truth_idx)
        return loss
    
    def _llm_judgement(self, reference, potential_answers):
        # LLM judgement
        llm_judgement = 0.
        for pot in potential_answers:
            prompt = f"""
            Imagine you are an expert teacher. You are asked to judge if the student's sentence is correct. The student's sentence is:
            {pot}
            The expected sentence is:
            {reference}
            On a scale from 1 to 5, how correct is the student's sentence?
            Examples: 
            1: The student's sentence is completely wrong. E.g. "The sky is green." "The sky is blue." -> 1
            4: The student's sentence is mostly correct, but has some mistakes. E.g. "Angela Merkel was born in germany." "Angela Merkel was born in Hamburg." -> 4
            """
            response = self.client.completions.create(
                model="gemma-2-9b-it",
                prompt=prompt,
            )
            judgement_score = response.choices[0].text if response.choices is not None else "0"
            judgement_score = re.search(r'\d+', judgement_score) if re.search(r'\d+', judgement_score) is not None else [0.]
            assert judgement_score is not None, f"Expected not None, got {judgement_score}"
            llm_judgement += float(judgement_score[0])
        llm_judgement /= len(potential_answers)
        return llm_judgement

