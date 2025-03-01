import re
import instructor
from openai import api_key
from pydantic import BaseModel
import google.generativeai as genai

class Judgement(BaseModel):
    #content: int = 0
    vocabulary: int = 0
    grammar: int = 0
    mechanics: int = 0
    
class JudgeInstructor:
    def __init__(self, judge_model: str = "models/gemini-1.5-flash-latest", judge_prompt: str = "", key: str = ""):
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        self.model_name_judge = judge_model
        self.judge_prompt = judge_prompt

    def judge_output(self, outputs: list, prompts: list):
        judgements = []
        for output, prompt in zip(outputs, prompts):
            if prompt in output:
                output = output.replace(prompt, "")
            cleaned_text = output.strip('"\n[]')
            cleaned_text = cleaned_text.rstrip(')')
            output = re.sub(r'\s+', ' ', cleaned_text)
            print(f'Prompt: {prompt}\nResponse: {output}')
            PROMPT = self.judge_prompt.format(response=output)
            judgements.append(self.client.messages.create(
                response_model=Judgement,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT
                    },
                ]
            ))
        final_judgement: Judgement = Judgement()
        for judgement in judgements:
            final_judgement.vocabulary += judgement.vocabulary
            #final_judgement.content += judgement.content
            final_judgement.grammar += judgement.grammar
            final_judgement.mechanics += judgement.mechanics
        

        final_judgement.vocabulary //= len(judgements)
        #final_judgement.content //= len(judgements)
        final_judgement.grammar //= len(judgements)
        final_judgement.mechanics //= len(judgements)

        return final_judgement
