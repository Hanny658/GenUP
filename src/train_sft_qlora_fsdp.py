import torch

from argparse import ArgumentParser
from pathlib import Path
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2

"""
Usage:

ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 train_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.1-8B" \
    --max_length 16384 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-NYC-POI"
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--max_length", type=int, default=16384)  # 2**14
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_llama", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_qwen2", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_id", type=str, required=True)
    return parser.parse_args()


def main():
    cli_args = parse_args()

    model_id = f"{cli_args.model_checkpoint.split('/')[-1]}-{cli_args.dataset_id.split('/')[-1]}"
    output_dir = cli_args.output_dir or model_id

    resume_from_checkpoint = None
    if cli_args.resume_from_checkpoint:
        if Path(output_dir).is_dir():
            resume_from_checkpoint = get_last_checkpoint(output_dir)
        if resume_from_checkpoint is None:
            print(f"No checkpoint found in {output_dir}; starting training from scratch.")
        else:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    dataset = load_dataset(cli_args.dataset_id)

    tokenizer = AutoTokenizer.from_pretrained(cli_args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # NOTE: we've formatted the prompt to include the <s> token at the beginning of the prompt
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False

    response_template = "[/INST]" if "llama-2" in model_id.lower() else " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    max_seq_length = cli_args.max_length

    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    accelerator = Accelerator()
    device_index = accelerator.process_index
    # FSDP is only useful across multiple GPUs. With a single process it degrades to
    # NO_SHARD and then can't flatten QLoRA's mixed-dtype params (bf16 quantized base
    # + fp32 LoRA), raising "Must flatten tensors with uniform dtype". Gate it on the
    # world size: multi-GPU -> FSDP, single-GPU -> plain QLoRA (fits an 8B model fine).
    use_fsdp = accelerator.num_processes > 1

    model = AutoModelForCausalLM.from_pretrained(
        cli_args.model_checkpoint,
        use_cache=True if cli_args.gradient_checkpointing else False,
        attn_implementation="sdpa",  # alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": device_index},
    )

    if cli_args.apply_liger_kernel_to_llama:
        apply_liger_kernel_to_llama()

    if cli_args.apply_liger_kernel_to_qwen2:
        apply_liger_kernel_to_qwen2()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cli_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Only pass FSDP settings when actually sharding across >1 GPU (see use_fsdp above);
    # on a single GPU this dict stays empty so the Trainer runs plain (non-FSDP) QLoRA.
    fsdp_args = {}
    if use_fsdp:
        fsdp_args = {
            "fsdp": ["full_shard", "auto_wrap", "offload"],
            "fsdp_config": {
                "backward_prefetch": "backward_pre",
                "forward_prefetch": "false",
                "use_orig_params": "false",
            },
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=cli_args.save_steps,
        save_total_limit=cli_args.save_total_limit,
        learning_rate=cli_args.learning_rate,
        max_grad_norm=cli_args.max_grad_norm,
        warmup_steps=cli_args.warmup_steps,
        per_device_train_batch_size=cli_args.batch_size,
        gradient_checkpointing=cli_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True} if cli_args.gradient_checkpointing else None,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        dataloader_num_workers=cli_args.dataloader_num_workers,
        num_train_epochs=cli_args.num_epochs,
        optim="adamw_torch",
        report_to="tensorboard",
        **fsdp_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        dataset_text_field="llama_prompt",
        max_seq_length=max_seq_length,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir)
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)

    # Mark the run as fully complete so a resubmitted job can skip re-training
    # instead of resuming from the last checkpoint and re-running the final epochs.
    if trainer.accelerator.is_main_process:
        (Path(output_dir) / ".training_complete").touch()


if __name__ == "__main__":
    main()
