from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tokenizer="meta-llama/Llama-2-7b-hf"
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=64,
    stop=[]  # No early stopping
)

# Clear, friendly prompt
prompt = "Write a short story about a robot learning to play the piano."

outputs = llm.generate([prompt], sampling_params=params)

print("Prompt:", prompt)
print("Response:", outputs[0].outputs[0].text.strip())
