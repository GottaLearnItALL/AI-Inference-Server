from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key='x')

r = client.chat.completions.create(
    model="qwen",
    messages=[{"role":"user","content":"What is 2+2?"}],
    max_tokens=50,
)

print(r.choices[0].message.content)


for chunk in client.chat.completions.create(
     model="qwen",
    messages=[{"role":"user","content":"What is 2+2?"}],
    max_tokens=50,
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
    
            



