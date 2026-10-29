from bisect import bisect_right
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SHARD_DTYPE = np.dtype("<u2")
MANIFEST_FILENAME = "manifest.json"
FORMAT_VERSION = 3

'''
Specification:
This dataset supplies fixed-length examples for next-token language modeling.
The text is tokenized ahead of time with GPT-2 BPE and stored in compact uint16
shards so that a large corpus does not need to be loaded into memory at once.

1. Dataset Layout:
   - Each partition contains a manifest and one or more token shards
   - The manifest records tokenizer identity and corpus statistics
   - Shards are opened lazily with NumPy memory maps

2. Training Examples:
   - Read context_length + 1 contiguous token IDs
   - Return the first context_length IDs as inputs
   - Return the same window shifted by one position as targets
   - Never cross a shard boundary

3. Dataset Properties:
   - Tracks total characters and serialized tokens for perplexity conversion
   - Supports deterministic prompt sampling for generation
   - Uses fixed-size examples, so PyTorch's default collator is sufficient

4. Validation:
   - Reads manifest fields and tokenizer metadata
   - Keeps partition and shard paths inside the selected dataset root
   - Rejects unusable partitions before training begins

Usage:
- Construct one LMDataset for each train/val/test partition
- Use a regular DataLoader or the provided collate_fn
- Use sample_prompts when checking model generations

TODO:
- Count valid fixed windows in every shard
- Map dataset indices to memory-mapped shard windows
- Return one-token-shifted input and target tensors
- Collate fixed-window batches
'''


class LMDataset(Dataset):
    """
    Dataset for fixed-window Language Model training/evaluation.

    Each item comes from ``context_length + 1`` contiguous token IDs::

        tokens = shard[start : start + context_length + 1]
        inputs = tokens[:-1]
        targets = tokens[1:]

    Shards are memory-mapped and samples never cross a shard boundary.  Since
    every item has the same shape, a regular PyTorch ``DataLoader`` can collate
    batches without a custom ``collate_fn``.
    """

    def __init__(
        self,
        root: str | Path,
        partition: str,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        """Initialize a partition of token shards.

        Args:
            root: Dataset root containing partition directories.
            partition: Partition directory name, such as ``train`` or ``val``.
            context_length: Number of input and target tokens in each example.
            stride: Distance between consecutive windows. The default uses
                non-overlapping input chunks.

        Raises:
            FileNotFoundError: If the manifest or a listed shard is missing.
            ValueError: If lengths, manifest fields, or shard sizes are invalid.
        """
        super().__init__()

        # Validate the context length
        if (
            not isinstance(context_length, Integral)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer")

        # Use non-overlapping input chunks when no stride is provided
        if stride is None:
            stride = int(context_length)

        # Validate the stride
        if (
            not isinstance(stride, Integral)
            or isinstance(stride, bool)
            or stride <= 0
        ):
            raise ValueError("stride must be a positive integer")

        # Validate the partition name
        if not isinstance(partition, str) or not partition:
            raise ValueError("partition must be one directory name inside root")
        partition_path = Path(partition)
        if (
            partition_path.is_absolute()
            or len(partition_path.parts) != 1
            or partition_path.name in (".", "..")
        ):
            raise ValueError("partition must be one directory name inside root")

        # Store configuration and other args
        self.root = Path(root).resolve()
        self.partition = partition
        self.context_length = int(context_length)
        self.stride = int(stride)

        # Set up data paths
        self.data_dir = (self.root / partition).resolve()
        if not self.data_dir.is_relative_to(self.root):
            raise ValueError("partition must stay inside root")
        self.manifest_path = self.data_dir / MANIFEST_FILENAME

        # Load the partition manifest
        manifest = self._load_manifest(self.manifest_path)

        # Validate corpus-level metadata before opening any shard
        self._validate_manifest_header(manifest)

        # Store tokenizer and corpus statistics
        self.vocab_size = int(manifest["vocab_size"])
        self.eos_id = int(manifest["eos_id"])
        self.total_chars = int(manifest["num_chars"])
        self.total_tokens = int(manifest["num_tokens"])
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens else 0.0
        )

        # Initialize lists that describe the token shards and their windows
        self.shard_paths: list[Path] = []
        self.shard_lengths: list[int] = []
        self.windows_per_shard: list[int] = []
        self.cumulative_windows: list[int] = []

        # Initialize tracking variables
        cumulative_windows = 0
        counted_tokens = 0

        # Count valid fixed windows in each shard
        for shard in manifest["shards"]:
            # Validate the shard entry and store its path and length
            shard_path, num_tokens = self._validate_shard_entry(shard)
            self.shard_paths.append(shard_path)
            self.shard_lengths.append(num_tokens)
            counted_tokens += num_tokens

            # Count windows using only tokens from this shard. Each example
            # needs context_length + 1 contiguous token IDs.
            if num_tokens < context_length + 1:
                num_windows = 0
            else:
                num_windows = (
                    (num_tokens - (context_length + 1)) // self.stride
                ) + 1

            # Build prefix sums for mapping global indices to shards
            self.windows_per_shard.append(num_windows)
            cumulative_windows += num_windows
            self.cumulative_windows.append(cumulative_windows)

        # Verify that shard token counts agree with the manifest total
        if counted_tokens != self.total_tokens:
            raise ValueError(
                "Manifest token total does not match its shard entries: "
                f"expected {self.total_tokens}, found {counted_tokens}"
            )
        # Verify that the selected context length produces at least one sample
        if cumulative_windows == 0:
            raise ValueError(
                "No shard contains enough tokens for one sample of length "
                f"{self.context_length}"
            )
        # A trainable corpus must contain source text
        if self.total_chars == 0:
            raise ValueError(
                "A trainable language-model dataset must contain source characters"
            )

        # Store the number of valid windows in the dataset.
        self.length = cumulative_windows

        # Initialize the lazy memory-map cache
        self._mmap_cache: dict[int, np.memmap] = {}

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        """Load a JSON dataset manifest.

        Args:
            path: Manifest path.

        Returns:
            Parsed manifest dictionary.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not a JSON object.
        """
        # Check that the manifest exists
        if not path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {path}")

        # Load the JSON manifest
        try:
            with path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dataset manifest is not valid JSON: {path}") from exc
        # Verify that the top-level JSON value is an object
        if not isinstance(manifest, dict):
            raise ValueError("Dataset manifest must contain a JSON object")
        loaded_manifest = manifest
        return loaded_manifest

    @staticmethod
    def _require_nonnegative_int(manifest: dict[str, Any], key: str) -> int:
        """Read one required nonnegative integer field.

        Args:
            manifest: Parsed manifest dictionary.
            key: Field name to read.

        Returns:
            Validated integer value.

        Raises:
            ValueError: If the field is missing or is not nonnegative.
        """
        # Read and validate the requested integer field
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Manifest field {key!r} must be a nonnegative integer")
        validated_value = value
        return validated_value

    def _validate_manifest_header(self, manifest: dict[str, Any]) -> None:
        """Validate fields that describe the whole partition.

        Args:
            manifest: Parsed manifest dictionary.

        Returns:
            ``None``.

        Raises:
            ValueError: If a required field is invalid.
        """
        # Check the file-format and tokenizer identifiers
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Manifest format_version must be {FORMAT_VERSION}")
        if manifest.get("dtype") != SHARD_DTYPE.str:
            raise ValueError(f"Manifest dtype must be {SHARD_DTYPE.str!r}")
        if manifest.get("tokenizer") != "gpt2":
            raise ValueError("Manifest tokenizer must be 'gpt2'")

        # Validate the corpus statistics
        vocab_size = self._require_nonnegative_int(manifest, "vocab_size")
        eos_id = self._require_nonnegative_int(manifest, "eos_id")
        self._require_nonnegative_int(manifest, "num_chars")
        self._require_nonnegative_int(manifest, "num_tokens")

        # Verify that token IDs fit in the on-disk uint16 representation
        if not 0 < vocab_size <= (1 << 16):
            raise ValueError("Manifest vocabulary does not fit in uint16")
        if eos_id >= vocab_size:
            raise ValueError("Manifest eos_id must be inside the vocabulary")
        if not isinstance(manifest.get("shards"), list):
            raise ValueError("Manifest field 'shards' must be a list")

    def _validate_shard_entry(self, shard: Any) -> tuple[Path, int]:
        """Read one shard path and its token count.

        Args:
            shard: Parsed shard entry.

        Returns:
            Validated shard path and token count.

        Raises:
            FileNotFoundError: If the shard file is missing.
            ValueError: If fields or paths are invalid.
        """
        # Each shard entry must be a JSON object
        if not isinstance(shard, dict):
            raise ValueError("Each manifest shard entry must be an object")

        # Read the required shard fields
        filename = shard.get("file")
        num_tokens = shard.get("num_tokens")
        # Validate the shard filename and token count
        if not isinstance(filename, str) or not filename:
            raise ValueError("Each shard must have a nonempty string 'file'")
        if (
            not isinstance(num_tokens, int)
            or isinstance(num_tokens, bool)
            or num_tokens < 0
        ):
            raise ValueError("Each shard must have a nonnegative integer 'num_tokens'")

        # Keep every shard path inside the selected partition
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Shard path must stay inside the partition: {filename}")
        shard_path = (self.data_dir / relative_path).resolve()
        data_directory = self.data_dir.resolve()
        if not shard_path.is_relative_to(data_directory):
            raise ValueError(f"Shard path must stay inside the partition: {filename}")
        if not shard_path.is_file():
            raise FileNotFoundError(f"Token shard not found: {shard_path}")

        actual_size = shard_path.stat().st_size
        expected_size = num_tokens * SHARD_DTYPE.itemsize
        if actual_size != expected_size:
            raise ValueError(
                f"Shard size mismatch for {filename}: "
                f"expected {expected_size} bytes, found {actual_size}"
            )

        declared_size = shard.get("size_bytes")
        if declared_size is not None:
            if (
                not isinstance(declared_size, int)
                or isinstance(declared_size, bool)
                or declared_size != actual_size
            ):
                raise ValueError(
                    f"Shard size_bytes does not match file size for {filename}"
                )

        validated_entry = (shard_path, num_tokens)
        return validated_entry

    def __len__(self) -> int:
        """Returns the number of fixed-length samples in the dataset.

        Returns:
            Number of examples.
        """
        # TODO: Return the number of fixed-window examples.
        return self.length

    def _get_shard(self, shard_index: int) -> np.memmap:
        """Open a shard lazily and reuse its memory map.

        Args:
            shard_index: Index into ``self.shard_paths``.

        Returns:
            Read-only memory map for the selected shard.
        """
        # Open the selected shard as a read-only memory map on first access.
        if shard_index not in self._mmap_cache:
            self._mmap_cache[shard_index] = np.memmap(
                self.shard_paths[shard_index],
                dtype=SHARD_DTYPE,
                mode="r",
                shape=(self.shard_lengths[shard_index],),
            )
        return self._mmap_cache[shard_index]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one next-token training example.

        Args:
            idx: Dataset example index.

        Returns:
            Input tokens and one-position-shifted target tokens.

        Raises:
            TypeError: If ``idx`` is not an integer.
            IndexError: If ``idx`` is outside the dataset.
        """
        # Add input validation
        if not isinstance(idx, Integral):
            raise TypeError("Dataset index must be an integer")
        index = int(idx)
        if index < 0 or index >= self.length:
            raise IndexError("Dataset index out of range")

        # Map the global index to a shard-local window.
        shard_index = bisect_right(self.cumulative_windows, index)
        previous_windows = (
            self.cumulative_windows[shard_index - 1] if shard_index > 0 else 0
        )
        local_index = index - previous_windows
        start = local_index * self.stride
        stop = start + self.context_length + 1

        # Copy the read-only memmap slice into an owned int64 array.
        shard = self._get_shard(shard_index)
        window_array = np.asarray(shard[start:stop], dtype=np.int64).copy()
        if np.any(window_array >= self.vocab_size):
            raise ValueError(
                f"Shard {self.shard_paths[shard_index].name} contains a token "
                "outside the manifest vocabulary"
            )
        # Split the window into next-token inputs and targets.
        window = torch.from_numpy(window_array).long()
        inputs = window[:-1]
        targets = window[1:]
        return inputs, targets

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack fixed-length examples into a batch.

        Args:
            batch: Input/target pairs returned by ``__getitem__``.

        Returns:
            Batched inputs and targets. PyTorch's default collator produces
            the same result, so this method is optional.
        """
        # Stack fixed-size input and target sequences along the batch axis.
        inputs, targets = zip(*batch)
        batched_inputs = torch.stack(inputs)
        batched_targets = torch.stack(targets)
        return batched_inputs, batched_targets

    def get_avg_chars_per_token(self) -> float:
        """Return source characters per serialized LM token.

        ``num_tokens`` includes document-boundary EOT tokens because those
        tokens are also targets in the language-model loss. This corpus-level
        ratio is an approximation for character-normalized window metrics:
        shard tails may be omitted, and non-default strides may overlap.

        Returns:
            Source characters per serialized token.
        """
        average = self.avg_chars_per_token
        return average

    def sample_prompts(
        self,
        num_samples: int,
        prompt_length: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample fixed prompts and their source chunks.

        Args:
            num_samples: Number of prompts to sample.
            prompt_length: Number of tokens in each prompt.
            seed: Optional seed for reproducible sampling.

        Returns:
            Stacked prompts and their complete source input chunks.

        Raises:
            ValueError: If counts or lengths are invalid.
        """
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise ValueError("num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if not isinstance(prompt_length, int) or isinstance(prompt_length, bool):
            raise ValueError("prompt_length must be a positive integer")
        if not 0 < prompt_length < self.context_length:
            raise ValueError("prompt_length must be between 1 and context_length - 1")

        # Use a local random generator so sampling does not alter global state.
        generator = np.random.default_rng(seed)

        # Sample random fixed-window examples.
        indices = generator.integers(0, len(self), size=num_samples)

        # Store complete input chunks as the original sequences.
        originals = [self[int(index)][0] for index in indices]

        # Get exactly prompt_length tokens from each sampled sequence.
        prompts = torch.stack(
            [original[:prompt_length].clone() for original in originals]
        )
        sampled_data = (prompts, originals)
        return sampled_data

    def __getstate__(self) -> dict[str, Any]:
        """Prepare dataset state for a DataLoader worker.

        Returns:
            Dataset state with an empty memory-map cache.
        """
        # Give each DataLoader worker its own empty memory-map cache
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state
