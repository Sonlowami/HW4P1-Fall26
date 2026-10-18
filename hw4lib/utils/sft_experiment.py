"""Provided checkpoint loading and local submission export for HW4P2."""
import json
import warnings
import zipfile
from pathlib import Path

import torch


def validate_model_config(model, saved):
    """Provided: validate shape-independent settings as well as weight shapes."""
    for key in ("num_layers", "d_model", "num_heads", "num_kv_heads", "max_len", "num_classes", "use_rope",
                "layer_drop_rate"):
        if key in saved:
            value = saved[key]
            if key == "num_kv_heads" and value is None:
                value = saved.get("num_heads", model.num_heads)
            if value != getattr(model, key):
                raise ValueError(f"Checkpoint model config differs at {key}")
    if "weight_tying" in saved:
        final = getattr(model.final_linear, "base", model.final_linear)
        tied = final.weight is model.target_embedding.weight
        if bool(saved["weight_tying"]) != tied:
            raise ValueError("Checkpoint model config differs at weight_tying")
    if "dropout" in saved and saved["dropout"] != model.dropout.p:
        raise ValueError("Checkpoint model config differs at dropout")
    if "d_ff" in saved and saved["d_ff"] != model.dec_layers[0].ffn.ffn[0].out_features:
        raise ValueError("Checkpoint model config differs at d_ff")
    if "rope_base" in saved and getattr(model, "use_rope", False):
        for layer in model.dec_layers:
            rope = layer.self_attn.mha.rope
            # Reuse the P1 implementation to check its nonpersistent tables.
            # No alternate RoPE implementation is supplied here.
            expected = type(rope)(rope.head_dim, rope.max_len, base=saved["rope_base"])
            if not (torch.equal(rope.cos, expected.cos.to(rope.cos))
                    and torch.equal(rope.sin, expected.sin.to(rope.sin))):
                raise ValueError("Checkpoint model config differs at rope_base")



def load_base_weights(model, path):
    """Load weights, not optimizer state. Never fall back to random initialization."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a state dictionary or a P1/P2 checkpoint")
    saved = payload.get("config", {}).get("model", {})
    if not saved:
        warnings.warn("Checkpoint has no model config: verify attention heads, RoPE and tying manually.")
    validate_model_config(model, saved)
    state = payload.get("model_state_dict", payload)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint weights do not match this model; check the architecture.") from exc
    model._hw4p2_weight_source = {"path": str(path.resolve()), "epoch": payload.get("epoch"),
                                 "step": payload.get("global_optimizer_step")}
    return saved


def evaluate_and_export(trainer, dataloader, records, *, available_inputs, seed, prefix=""):
    """Provided: record dev evaluation, or reuse a matching completed evaluation on resume."""
    import csv
    from .webnlg import score_webnlg, write_predictions
    if prefix not in ("", "base_"):
        raise ValueError("Evaluation prefix must be empty or base_")
    source = getattr(trainer.model, "_hw4p2_weight_source", None)
    if source is None:
        raise ValueError("Load the evaluated checkpoint with load_base_weights first")
    context = {"ids": [r["id"] for r in records], "generation": trainer.config.get("generation", {}),
               "weight_source": source, "available_inputs": available_inputs, "seed": seed}
    metrics_path = trainer.expt_root / f"{prefix}metrics.json"
    predictions_path = trainer.expt_root / ("base_predictions.csv" if prefix else "dev_predictions.csv")
    if metrics_path.exists() or predictions_path.exists():
        if not trainer.config["training"].get("resume", False):
            raise FileExistsError("Evaluation already exists; use explicit resume or a new run name")
        if not (metrics_path.is_file() and predictions_path.is_file()):
            raise ValueError("Incomplete evaluation files; preserve them and use a new run name")
        metrics = json.loads(metrics_path.read_text())
        with predictions_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if metrics.get("evaluation") != context or [r["id"] for r in rows] != context["ids"]:
            raise ValueError("Existing evaluation does not match this checkpoint, inputs or decoding settings")
        predictions = [row["prediction"] for row in rows]
        # Re-score the stored predictions; no new model generation is needed.
        scores = score_webnlg(predictions, [r["references"] for r in records])
        if any(metrics[key] != scores[key] for key in ("bleu", "chrf", "settings")):
            raise ValueError("Existing evaluation scores do not match saved predictions")
        return metrics, predictions
    metrics = trainer.evaluate(dataloader)
    predictions = trainer.generate(records)
    metrics.update(score_webnlg(predictions, [row["references"] for row in records]))
    metrics.update(split="dev", evaluated_inputs=len(records), available_inputs=available_inputs,
                   seed=seed, checkpoint=source["path"], evaluation=context)
    with metrics_path.open("x") as file:
        json.dump(metrics, file, indent=2)
    write_predictions(records, predictions, predictions_path)
    return metrics, predictions



def export_submission(project_dir, run_dirs, notebook_path, output_path):
    """Export code, the saved notebook and small evidence files, never weights.

    This local archive is for inspection; the final course submission contract
    will be supplied separately. Save the executed notebook before exporting.
    """
    project = Path(project_dir).resolve()
    notebook = Path(notebook_path).resolve()
    if not notebook.is_file():
        raise FileNotFoundError("Save the notebook before exporting")
    files = [(notebook, notebook.name)]
    for name in ("mytorch", "hw4lib", "tests"):
        source = project / name
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Submission source must not contain symlinks")
            if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts:
                files.append((path, str(path.relative_to(project))))
    # Only provided tokenizer assets, not arbitrary JSON files in the project.
    for path in sorted((project / "hw4lib/data/tokenizer_jsons").glob("*.json")):
        if path.is_symlink():
            raise ValueError("Tokenizer assets must not be symlinks")
        files.append((path, str(path.relative_to(project))))
    seen = set()
    for directory in map(Path, run_dirs):
        if not directory.is_dir() or directory.name in seen:
            raise ValueError("Evidence requires existing, uniquely named run directories")
        seen.add(directory.name)
        for name in ("config.yaml", "metrics.json", "base_metrics.json", "history.json",
                     "dev_predictions.csv", "base_predictions.csv", "parameter_counts.json"):
            path = directory / name
            if path.is_symlink():
                raise ValueError("Evidence must not contain symlinks")
            if path.is_file():
                files.append((path, f"evidence/{directory.name}/{name}"))
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
    return Path(output_path)
