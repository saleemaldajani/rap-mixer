def trace_text(output) -> str:
    positives = "\n".join(f"+ {x.feature}: {x.amount:+.3f}" for x in output.positive) or "None"
    negatives = "\n".join(f"- {x.feature}: {x.amount:+.3f}" for x in output.negative) or "None"
    return f"### {output.name}: {output.score:.1f} ± {output.uncertainty:.1f}\n\nPositive contributors\n{positives}\n\nNegative contributors\n{negatives}\n\n`{output.trace['formula']}`"
