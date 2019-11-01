import syllables


def estimate_syllables(haiku: str):
    """Estimate the number of syllables per each line of the given haiku."""
    lines = haiku.split("/")
    counts = []
    for line in lines:
        words = line.strip(" \t\n#").split()
        counts.append(sum(syllables.estimate(w) for w in words))
    return counts
