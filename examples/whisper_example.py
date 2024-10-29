import vllm
import torch
import requests
from vllm import LLM
from datasets import Audio


def main():
    sr = 16000
    audio = Audio(sampling_rate=sr)
    llm = LLM(
        model="openai/whisper-large-v3",
        max_num_seqs = 1,
        max_model_len = 448,
        gpu_memory_utilization = 0.4,
        dtype = 'bfloat16',
        whisper_input_type = 'input_features',
        enforce_eager = True,
    )

    r = requests.get('https://github.com/mesolitica/malaya-speech/raw/master/speech/7021-79759-0004.wav')
    y = audio.decode_example(audio.encode_example(r.content))['array']

    output_lang = llm.generate({
        "prompt_token_ids": [50258],
        "whisper_data": y,
    }, sampling_params = SamplingParams(max_tokens = 1, temperature = 0))

    outputs = llm.generate({
        "prompt_token_ids": [50258, output_lang[0].outputs[0].token_ids[0], 50360],
        "whisper_data": y,
    }, sampling_params = SamplingParams(max_tokens = 100, temperature = 0))

    # ' without going to any such extreme as this we can easily see on reflection how vast an influence on the'
    print(outputs[0].outputs[0].text)



if __name__ == "__main__":
    main()
