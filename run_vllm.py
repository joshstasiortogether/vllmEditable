from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is a",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
    dtype="auto",
    max_model_len=8192
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generation = output.outputs[0].text.strip()
    print(f"{prompt}{generation}")
    print('-------------')
