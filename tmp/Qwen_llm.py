
import os
import sys
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class LocalLLM:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
        torch_dtype: str | torch.dtype = "auto",
        device_map: str | dict | None = "auto",
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        )
        self.model.eval()

    def chat(self, messages:List[List[Dict[str,str]]]):
        if not messages:
            raise ValueError("Messages must not empty")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize= False,
            add_generation_prompt = True
        )
        _first_real = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            None,
        )
        _device = _first_real.device if _first_real is not None else "cpu"
        model_inputs = self.tokenizer(
            [text],
            return_tensors = "pt",
        ).to(_device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature,
                do_sample = self.do_sample
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens = True,
        )[0]
        return response.strip()

def _auto_tensor_parallel_size(llm_parametres: dict | None) -> dict:
    """Fill in ``tensor_parallel_size`` from the visible GPU count.

    A large model (e.g. Qwen2.5-72B in bf16 ≈ 145 GB) cannot fit on a single
    GPU, so it must be sharded with tensor parallelism across every visible
    device. We only auto-set this when the caller hasn't pinned a value, and
    we copy the dict so the caller's config (often a shared cfg sub-dict) is
    never mutated.
    """
    params = dict(llm_parametres or {})
    if 'tensor_parallel_size' not in params:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            params['tensor_parallel_size'] = n_gpus
            print(f"Sharding model across {n_gpus} GPUs (tensor_parallel_size={n_gpus})")
    return params


class vllm_LLM:
    def __init__(self,
        model_path: str,
        sampling_parameters=None,
        llm_parametres=None
    ) -> None:
        self.sampling_params = SamplingParams(
            **sampling_parameters
        )
        self.llm = LLM(
            model = model_path,
            **_auto_tensor_parallel_size(llm_parametres)
        )
    def batch_chat(self, messages_list:List[List[Dict[str,str]]]):
        output = self.llm.chat(messages_list,self.sampling_params)
        return [o.outputs[0].text.strip() for o in output]
    def chat(self,message):
        return self.batch_chat([message])[0]

