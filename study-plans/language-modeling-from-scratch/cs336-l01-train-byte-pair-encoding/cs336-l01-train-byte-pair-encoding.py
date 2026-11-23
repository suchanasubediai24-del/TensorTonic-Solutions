
def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """

    # Convert each string into UTF-8 byte token IDs
    sequences = [
        list(text.encode("utf-8"))
        for text in corpus
    ]

    # Initial vocabulary contains 256 byte tokens
    vocab = []
    merges = []

    next_id = 256

    # Repeat until target vocabulary size is reached
    while next_id < vocab_size:

        # Count adjacent pairs
        pair_counts = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Stop if no pairs exist
        if not pair_counts:
            break

        # Select most frequent pair
        # Tie-breaking uses lexicographically greater byte strings
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                bytes(
                    (vocab[pair[0] - 256][1]
                     if pair[0] >= 256 else [pair[0]])
                ),
                bytes(
                    (vocab[pair[1] - 256][1]
                     if pair[1] >= 256 else [pair[1]])
                )
            )
        )

        left, right = best_pair

        # Construct merged byte sequence
        left_bytes = (
            vocab[left - 256][1]
            if left >= 256 else [left]
        )

        right_bytes = (
            vocab[right - 256][1]
            if right >= 256 else [right]
        )

        merged_bytes = left_bytes + right_bytes

        # Assign new token ID
        new_id = next_id
        next_id += 1

        vocab.append([new_id, merged_bytes])
        merges.append([left, right, new_id])

        # Replace matching pairs from left to right
        new_sequences = []

        for seq in sequences:
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i < len(seq) - 1
                    and seq[i] == left
                    and seq[i + 1] == right
                ):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append(new_seq)

        sequences = new_sequences

    return {
        "vocab": vocab,
        "merges": merges
    }


# Test cases
print(train_bpe(["aaab"], 257))
print(train_bpe(["ab", "cd"], 257))
print(train_bpe(["abc"], 1000))
