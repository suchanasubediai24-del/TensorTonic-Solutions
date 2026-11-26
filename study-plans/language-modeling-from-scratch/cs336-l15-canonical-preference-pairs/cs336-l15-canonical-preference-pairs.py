
def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id, loser_id dictionaries.
    """
    pairs = []

    for record in records:
        prompt_id = record["prompt_id"]
        candidates = record["candidates"]

        # Sort candidates by rank (best candidate first)
        sorted_candidates = sorted(candidates, key=lambda x: x["rank"])

        # Generate all winner-loser combinations
        for i in range(len(sorted_candidates)):
            for j in range(i + 1, len(sorted_candidates)):
                pairs.append({
                    "prompt_id": prompt_id,
                    "winner_id": sorted_candidates[i]["id"],
                    "loser_id": sorted_candidates[j]["id"]
                })

    return {"pairs": pairs}
