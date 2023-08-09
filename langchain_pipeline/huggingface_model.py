from typing import Any, List, Mapping, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Extra
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import torch
from peft import PeftModel
from optimum.bettertransformer import BetterTransformer

DEFAULT_MODEL_ID = "psmathur/orca_mini_13b"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

class HuggingFacePipeline(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation` and `text2text-generation` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2", task="text-generation"
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        model_kwargs: Optional[dict] = None,
        peft_model: Optional[str] = None,
        use_gptq: bool = True,
        model_basename: Optional[str] = None,
        load_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}

        try:
            if use_gptq:
                model, tokenizer = load_model_gptq(model_id, model_basename, load_kwargs)
            else:
                model, tokenizer = load_model(model_id, peft_model)
        except ImportError as e:
            raise ValueError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        pipeline = hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **model_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt)
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt) :]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
    
def load_model(
    model_id, 
    finetuned=None, 
    mode_cpu=False,
    mode_mps=False,
    mode_full_gpu=True,
    mode_8bit=False,
    mode_4bit=False,
    force_download_ckpt=False,
    local_files_only=False,
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only,
        use_fast=True, 
    )
    tokenizer.padding_side = "left"
    
    if mode_cpu:
        print("cpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map={"": "cpu"}, 
            use_safetensors=False,
            local_files_only=local_files_only
        )
            
    elif mode_mps:
        print("mps mode")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=local_files_only
        )
            
    else:
        print("gpu mode")
        print(f"8bit = {mode_8bit}, 4bit = {mode_4bit}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            device_map="auto",
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=local_files_only
        )

        if not mode_8bit and not mode_4bit:
            model.half()

        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":

            model = PeftModel.from_pretrained(
                model, 
                finetuned, 
                # force_download=force_download_ckpt,
        )

    model = BetterTransformer.transform(model)
    return model, tokenizer

def load_model_gptq(
    model_id,
    model_basename,
    load_kwargs,
    **kwargs
):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    model_name_or_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            **load_kwargs)


    return model, tokenizer
