from torch.utils.data import DataLoader

'''
Specification:
This utility function verifies and displays key information about a PyTorch dataloader:

1. Basic Statistics:
   - Shows which data partition is being used (train/val/test)
   - Displays number of batches and batch size
   - Reports maximum sequence lengths

2. Shape Verification:
   - Checks the first batch to display tensor shapes
   - Handles both ASR datasets (with features) and fixed-window LM datasets
   - Shows shapes for features, shifted inputs, targets, and length information

3. Dataset Properties:
   - Reports maximum feature length (for ASR)
   - Reports the fixed context length (for language modeling)
   - Shows average characters per token (for perplexity calculation)

Usage:
- Called after dataloader creation to verify proper setup
- Helps debug shape mismatches and configuration issues
- Confirms proper batching before training begins
'''


def verify_dataloader(dataloader: DataLoader) -> None:
    '''
    Computes the maximum sequence lengths of features and text in the dataloader.
    Prints some basic statistics about the dataloader.

    Args:
        dataloader: ASR or fixed-window language-model DataLoader.

    Returns:
        ``None``.

    Raises:
        ValueError: If the DataLoader is empty or has an unknown batch format.
    '''
    def print_shapes(
        feats,
        transcripts_shifted=None,
        transcripts_golden=None,
        feat_lengths=None,
        transcript_lengths=None,
    ) -> None:
        """Print the shape of each available batch tensor.

        Args:
            feats: Optional speech feature batch.
            transcripts_shifted: Optional shifted token batch.
            transcripts_golden: Optional target token batch.
            feat_lengths: Optional speech feature lengths.
            transcript_lengths: Optional target token lengths.

        Returns:
            None.
        """
        values = (
            ("Feature Shape", feats),
            ("Shifted Transcript Shape", transcripts_shifted),
            ("Golden Transcript Shape", transcripts_golden),
            ("Feature Lengths Shape", feat_lengths),
            ("Transcript Lengths Shape", transcript_lengths),
        )
        for label, value in values:
            if value is not None:
                print(f"{label:<25}: {list(value.shape)}")

    print("="*50)
    print(f"{'Dataloader Verification':^50}")
    print("="*50)
    print(f"{'Dataloader Partition':<25}: {dataloader.dataset.partition}")
    print("-"*50)
    print(f"{'Number of Batches':<25}: {len(dataloader)}")
    print(f"{'Batch Size':<25}: {dataloader.batch_size}")
    print("-"*50)
    print(f"{'Checking shapes of the data...':<50}\n")

    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("dataloader must contain at least one batch") from exc

    max_feature_length = None
    if hasattr(dataloader.dataset, "feat_max_len"):
        # ASR Dataset
        if len(batch) != 5:
            raise ValueError("ASR batches must contain five tensors")
        features, shifted, targets, feature_lengths, target_lengths = batch
        print_shapes(
            features,
            shifted,
            targets,
            feature_lengths,
            target_lengths,
        )
        max_feature_length = dataloader.dataset.feat_max_len
        max_target_length = dataloader.dataset.text_max_len
    elif hasattr(dataloader.dataset, "context_length"):
        # Fixed-window Language Model Dataset
        if len(batch) != 2:
            raise ValueError("Language-model batches must contain two tensors")
        inputs, targets = batch
        print_shapes(None, inputs, targets)
        max_target_length = dataloader.dataset.context_length
    else:
        raise ValueError("Unsupported dataset type")

    print("-"*50)
    if max_feature_length is not None:
        print(f"{'Max Feature Length':<25}: {max_feature_length}")
    print(f"{'Max Transcript Length':<25}: {max_target_length}")
    if hasattr(dataloader.dataset, "get_avg_chars_per_token"):
        average = dataloader.dataset.get_avg_chars_per_token()
        print(f"{'Avg. Chars per Token':<25}: {average:.2f}")
    print("="*50)
