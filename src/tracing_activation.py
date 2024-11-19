import torch
from nanoGPT import SampleMutableModel
import openai

model = SampleMutableModel()
model.get_activations()

activation_norms = {}

client = openai.OpenAI()


def generate_prompt():
    response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        {
            "role": "user",
            "content": "Write a creative, very short prompt to test a language model's ability to generate text.",
        },
    ],
    )
    return response.choices[0].message.content

criterion = False

def reduce_model(model, number_of_blocks=10, repetitions=5):
    reduce_list = []
    for _ in range(number_of_blocks):
        min_norm = ("", float("inf"))
        for _ in range(repetitions):
            prompt = generate_prompt()
            print(prompt)
            model.generate_verbose(prompt)
            for name, activation in model.activations.items():
                print(name, activation.shape)
                s = activation.shape[1]
                a = activation.view(1, s, 12, 64)
                a = a.permute(0, 2, 1, 3).contiguous()
                for j in range(12):
                    if name in activation_norms:
                        activation_norms[(name,j)] += float(a[:, j, :, :].view(1, s*64).norm(dim=-1))
                    else:
                        activation_norms[(name,j)] = float(a[:, j, :, :].view(1, s*64).norm(dim=-1))
        if criterion:
            break

        for name, norm_sum in activation_norms.items():
            if (name[0].split('.')[3], name[1]) in reduce_list:
                continue
            if min_norm[1] > norm_sum:
                min_norm = (name, norm_sum)
        reduce_list.append((int(min_norm[0][0].split('.')[3]), int(min_norm[0][1])))

        model.update_blocked(reduce_list)
    torch.save(model.model.state_dict(), 'model.pth')

reduce_model(model, 2, 2)

# Heat map of attentions
# Friday: Non-sensical text



