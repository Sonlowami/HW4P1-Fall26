"""Response-only supervised fine-tuning for the P1 decoder."""
import json
import math
import yaml
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from ..decoding.sequence_generator import SequenceGenerator
from ..utils.webnlg import format_webnlg_prompt, score_webnlg


class SFTTrainer(BaseTrainer):
    """Train a decoder on prompt/response pairs.

    Complete the marked training, validation, and generation methods.
    Reuse your P1 model and decoding implementation.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        if model.num_classes != tokenizer.vocab_size:
            raise ValueError("Model and tokenizer vocabulary sizes differ")
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.best_metric = -float("inf")  # Higher BLEU is better.
        self.amp_enabled = bool(config["training"].get("amp", True)) and torch.device(self.device).type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # TODO: Use summed cross-entropy, ignoring labels equal to -100.
        self.criterion = None  # TODO

    def _init_experiment(self, run_name, config_file):
        """Provided: use a new output directory, or explicitly resume an existing run."""
        if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("run_name must be a single directory name")
        root = Path(self.config.get("output_dir", "expts")).expanduser().resolve() / run_name
        resume = self.config["training"].get("resume", False)
        if resume:
            if not root.is_dir():
                raise FileNotFoundError("Resume requires an existing experiment directory")
        else:
            root.mkdir(parents=True, exist_ok=False)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            (root / "model.txt").write_text(str(self.model) + "\n")
        checkpoint, attn, text = (root / part for part in ("checkpoints", "attn", "text"))
        for directory in (checkpoint, attn, text):
            directory.mkdir(exist_ok=True)
        if self.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config["training"].get("wandb_project", "hw4p2"),
                entity=self.config["training"].get("wandb_entity") or None,
                name=run_name, config=self.config,
                id=self.config["training"].get("wandb_run_id") or None,
                resume="allow" if resume else None, dir=str(root))
        return root, checkpoint, attn, text, checkpoint / "best.pth", checkpoint / "last.pth"

    def load_checkpoint(self, filename):
        """Provided: resume only the experiment described by the saved configuration."""
        from ..utils.sft_experiment import validate_model_config
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location="cpu", weights_only=True)
        saved = checkpoint.get("config", {})
        fields = ["model", "data", "base_checkpoint", "optimizer", "scheduler", "generation", "seed", "experiment"]
        if self.config.get("experiment") == "lora" or saved.get("experiment") == "lora":
            fields.append("lora")
        for field in fields:
            if saved.get(field) != self.config.get(field):
                raise ValueError(f"Resume config differs at {field}; use the saved config or a new run name")
        logging_fields = {"resume", "use_wandb", "wandb_project", "wandb_entity", "wandb_run_id",
                          "wandb_log_checkpoints", "progress", "num_workers"}
        old_training = {k: v for k, v in saved.get("training", {}).items() if k not in logging_fields}
        new_training = {k: v for k, v in self.config["training"].items() if k not in logging_fields}
        if old_training != new_training:
            raise ValueError("Resume training settings differ; use the saved config or a new run name")
        validate_model_config(self.model, saved.get("model", {}))
        del checkpoint
        super().load_checkpoint(filename)

    @staticmethod
    def _metrics(loss_sum, token_count):
        """Provided: perplexity here is response-token PPL, not character PPL."""
        if token_count <= 0:
            raise ValueError("No supervised response tokens")
        ce = loss_sum / token_count
        if not math.isfinite(ce):
            raise FloatingPointError("Non-finite response loss")
        return {"response_ce": ce, "response_ppl": math.exp(ce), "response_tokens": token_count}

    def _batch(self, batch):
        """Provided: move tensors and reject a batch without supervised targets."""
        inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching (B, T) shapes")
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError("Batch has no supervised response tokens")
        return inputs, labels, count

    def _train_epoch(self, dataloader):
        """Accumulate summed losses; normalize gradients by the cycle's token count.

        Handle a partial final accumulation cycle. Unscale before clipping.
        Advance the scheduler only when the optimizer step ran.
        """
        # TODO: Implement one training epoch using the instructions above.
        raise NotImplementedError("Complete _train_epoch")

    @torch.inference_mode()
    def _validate_epoch(self, dataloader):
        """Compute response CE from total NLL / total supervised tokens."""
        # TODO: Evaluate without gradients; exclude prompt and padding targets.
        raise NotImplementedError("Complete _validate_epoch")

    @torch.inference_mode()
    def generate(self, records):
        """Generate from the triples alone, using your P1 greedy decoder.

        Do not append references to prompts. Decode one unpadded prompt at a time.
        Limit total context and stop at generated EOS, excluding prompt text.
        """
        # TODO: Use SequenceGenerator and the model's score/score_cached methods.
        raise NotImplementedError("Complete generate")

    def evaluate(self, dataloader, records=None):
        """Provided: combine teacher-forced response loss with generated-text metrics."""
        metrics = self._validate_epoch(dataloader)
        if records is not None:
            predictions = self.generate(records)
            scores = score_webnlg(predictions, [row["references"] for row in records])
            metrics.update(bleu=scores["bleu"], chrf=scores["chrf"], examples=len(records))
        return metrics

    def train(self, train_dataloader, val_dataloader, dev_records):
        """Run epochs and select the best generated dev BLEU checkpoint.

        Save last.pth for resuming the next epoch and best.pth for prediction.
        Log token loss and generation scores as different quantities.
        """
        # TODO: Run training/evaluation, update best BLEU, log, and save checkpoints.
        raise NotImplementedError("Complete train")
