import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_log_path(log_dir, run_name):
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{run_name}_{stamp}.log"


def write_log(log_path, message):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_print(log_path, message):
    print(message)
    write_log(log_path, message)


class DPODataset(Dataset):
    # Stores raw strings; tokenisation is deferred to the collator

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class DPOCollator:
    # Tokenise chosen/rejected full sequences and record prompt length

    def __init__(self, tokenizer, max_length, max_prompt_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.device = device

    def encode(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def __call__(self, batch):
        # Concatenate prompt + response for each side
        chosen_texts = [b["prompt"] + " " + b["chosen"]   for b in batch]
        rejected_texts = [b["prompt"] + " " + b["rejected"] for b in batch]

        chosen = self.encode(chosen_texts)
        rejected = self.encode(rejected_texts)

        # Prompt-only length (used to build the completion mask)
        prompt_only = [b["prompt"] for b in batch]
        prompt_enc  = self.tokenizer(
            prompt_only,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        prompt_lens = torch.tensor(
            [min(len(ids), self.max_prompt_length) for ids in prompt_enc["input_ids"]],
            dtype=torch.long,
            device=self.device,
        )

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "prompt_lens": prompt_lens,
        }


def completion_mask(input_ids, attention_mask, prompt_lens):
    # Binary mask [B, T-1]: 1 for response tokens, 0 for prompt/padding.
    
    B, T = input_ids.shape
    pos = torch.arange(T - 1, device=input_ids.device)      
    # position t in the logit dimension corresponds to predicting token t+1
    after_prompt = pos.unsqueeze(0) >= prompt_lens.unsqueeze(1)   
    not_pad = attention_mask[:, 1:].bool()                   
    return (after_prompt & not_pad).float()


def sequence_logprob(model_logits, input_ids, attention_mask, prompt_lens):
    # Sum of log-probs over completion tokens.
    log_probs = F.log_softmax(model_logits[:, :-1, :], dim=-1)  
    targets = input_ids[:, 1:].clone()                        
    mask = completion_mask(input_ids, attention_mask, prompt_lens) 

    # Gather log-prob of the actual next token
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1) 
    selected = selected * mask
    return selected.sum(-1)                                      


def compute_dpo_loss(
    chosen_logprob,
    rejected_logprob,
    chosen_logprob_ref,
    rejected_logprob_ref,
    beta,
):
    # Batch-mean DPO loss.
    logits = beta * (
        (chosen_logprob   - rejected_logprob) -
        (chosen_logprob_ref - rejected_logprob_ref)
    )
    return -F.logsigmoid(logits).mean()


def cosine_schedule(step, total, min_ratio=0.1):
    if total <= 1:
        return 1.0
    p = step / max(1, total - 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser(description="Raw DPO training (no TRL)")
    parser.add_argument("--model-name", default="models/sft",
                        help="SFT checkpoint used as the trainable policy")
    parser.add_argument("--ref-model-name", default=None,
                        help="Frozen reference (defaults to --model-name)")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/dpo"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--log-dir", type=Path, default=Path("results/training_logs/raw_runs"))
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.log_steps <= 0:
        raise ValueError("--log-steps must be a positive integer")

    log_path = make_log_path(args.log_dir, "train_dpo_raw")
    log_print(log_path, f"Logging to {log_path}")
    write_log(log_path, json.dumps(vars(args), default=str, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()
    ref_name = args.ref_model_name or args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        eval_rows = eval_rows[: max(1, args.max_samples // 10)]

    collator = DPOCollator(tokenizer, args.max_length, args.max_prompt_length, device)

    train_loader = DataLoader(
        DPODataset(train_rows), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    eval_loader = DataLoader(
        DPODataset(eval_rows), batch_size=args.batch_size,
        shuffle=False, collate_fn=collator,
    )

    log_print(log_path, f"Train: {len(train_rows)} examples | Eval: {len(eval_rows)} examples")

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: cosine_schedule(s, total_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    step_loss_sum = 0.0
    step_margin_sum = 0.0
    step_batches = 0
    console_loss_sum = 0.0
    console_margin_sum = 0.0
    console_steps = 0

    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            c_ids  = batch["chosen_input_ids"]
            c_attn = batch["chosen_attention_mask"]
            r_ids  = batch["rejected_input_ids"]
            r_attn = batch["rejected_attention_mask"]
            plens  = batch["prompt_lens"]

            # Reference log-probs (no gradient)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                c_logits_ref = ref_model(input_ids=c_ids, attention_mask=c_attn).logits
                r_logits_ref = ref_model(input_ids=r_ids, attention_mask=r_attn).logits
            c_lp_ref = sequence_logprob(c_logits_ref, c_ids, c_attn, plens)
            r_lp_ref = sequence_logprob(r_logits_ref, r_ids, r_attn, plens)

            # Policy log-probs (with gradient)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                c_logits_pi = policy(input_ids=c_ids, attention_mask=c_attn).logits
                r_logits_pi = policy(input_ids=r_ids, attention_mask=r_attn).logits

            c_lp_pi = sequence_logprob(c_logits_pi, c_ids, c_attn, plens)
            r_lp_pi = sequence_logprob(r_logits_pi, r_ids, r_attn, plens)

            loss = compute_dpo_loss(c_lp_pi, r_lp_pi, c_lp_ref, r_lp_ref, args.beta)
            scaler.scale(loss / args.grad_accum).backward()

            with torch.no_grad():
                margin = args.beta * ((c_lp_pi - c_lp_ref) - (r_lp_pi - r_lp_ref))
                step_loss_sum += loss.item()
                step_margin_sum += margin.mean().item()
                step_batches += 1

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_loss = step_loss_sum / max(1, step_batches)
                step_margin = step_margin_sum / max(1, step_batches)
                write_log(
                    log_path,
                    f"train_step epoch {epoch} | step {global_step}/{total_steps} "
                    f"| loss {step_loss:.4f} | reward_margin {step_margin:.4f}"
                )
                console_loss_sum += step_loss
                console_margin_sum += step_margin
                console_steps += 1
                step_loss_sum = 0.0
                step_margin_sum = 0.0
                step_batches = 0

                if global_step % args.log_steps == 0:
                    log_print(
                        log_path,
                        f"epoch {epoch} | step {global_step}/{total_steps} "
                        f"| loss {console_loss_sum / max(1, console_steps):.4f} "
                        f"| reward_margin {console_margin_sum / max(1, console_steps):.4f}"
                    )
                    console_loss_sum = 0.0
                    console_margin_sum = 0.0
                    console_steps = 0

        if console_steps:
            log_print(
                log_path,
                f"epoch {epoch} | step {global_step}/{total_steps} "
                f"| loss {console_loss_sum / console_steps:.4f} "
                f"| reward_margin {console_margin_sum / console_steps:.4f}"
            )
            console_loss_sum = 0.0
            console_margin_sum = 0.0
            console_steps = 0

        # Eval
        policy.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                c_ids = batch["chosen_input_ids"]
                c_attn = batch["chosen_attention_mask"]
                r_ids = batch["rejected_input_ids"]
                r_attn = batch["rejected_attention_mask"]
                plens = batch["prompt_lens"]

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    c_lp_ref = sequence_logprob(
                        ref_model(input_ids=c_ids, attention_mask=c_attn).logits,
                        c_ids, c_attn, plens,
                    )
                    r_lp_ref = sequence_logprob(
                        ref_model(input_ids=r_ids, attention_mask=r_attn).logits,
                        r_ids, r_attn, plens,
                    )
                    c_lp_pi = sequence_logprob(
                        policy(input_ids=c_ids, attention_mask=c_attn).logits,
                        c_ids, c_attn, plens,
                    )
                    r_lp_pi = sequence_logprob(
                        policy(input_ids=r_ids, attention_mask=r_attn).logits,
                        r_ids, r_attn, plens,
                    )
                    eloss = compute_dpo_loss(c_lp_pi, r_lp_pi, c_lp_ref, r_lp_ref, args.beta)
                eval_losses.append(eloss.item())

        avg_eval = sum(eval_losses) / len(eval_losses)
        log_print(log_path, f"epoch {epoch} | eval_loss {avg_eval:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    log_print(log_path, f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
