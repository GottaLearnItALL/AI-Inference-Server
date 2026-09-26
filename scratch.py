from mlx_lm import load,generate

model, tokenizer = load('mlx-community/Qwen3-4B-Instruct-2507-4bit')
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = generate(model,tokenizer,prompt=prompt, max_tokens = 200)

print(text)