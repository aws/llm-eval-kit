"""
String similarity grader — fuzzy matching using edit distance or token overlap.

The grader framework is complete. The two algorithm functions below are stubs
for YOU to implement as Leetcode practice.
"""
from llm_eval_kit.graders.decorator import grader
from llm_eval_kit.models.results import EvaluateResult, MetricResult
from ._helpers import get_last_assistant_content


# ---------------------------------------------------------------------------
# YOUR TASK: Implement these two functions
# ---------------------------------------------------------------------------

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity between two strings.
    Return: 1.0 - (edit_distance / max(len(s1), len(s2)))

    LEETCODE CONNECTION: This is Leetcode #72 (Edit Distance).

    Algorithm:
    1. Build a 2D DP table of size (len(s1)+1) x (len(s2)+1)
    2. dp[i][j] = minimum edits to convert s1[:i] into s2[:j]
    3. Base cases: dp[i][0] = i, dp[0][j] = j
    4. Transition:
       - If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
       - Else: dp[i][j] = 1 + min(dp[i-1][j],      # delete
                                    dp[i][j-1],      # insert
                                    dp[i-1][j-1])    # replace
    5. edit_distance = dp[len(s1)][len(s2)]
    6. Normalize: 1.0 - (edit_distance / max(len(s1), len(s2)))

    Edge cases:
    - Both empty -> return 1.0
    - One empty -> return 0.0

    Space optimization (optional stretch goal):
    - You only need the previous row, so you can use O(min(m,n)) space
      instead of O(m*n). This is a common follow-up in interviews.
    """
    raise NotImplementedError("Implement levenshtein_similarity")


def token_f1_score(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.

    Algorithm:
    1. Tokenize: prediction.lower().split(), reference.lower().split()
    2. Use collections.Counter to count token frequencies (multiset)
    3. Overlap = sum of min counts for each token (Counter intersection)
       - In Python: sum((counter_pred & counter_ref).values())
    4. precision = overlap / len(predicted_tokens)
    5. recall = overlap / len(reference_tokens)
    6. F1 = 2 * precision * recall / (precision + recall)

    LEETCODE CONNECTION:
    - Counter intersection is related to array intersection problems
    - Using Counter (multiset) is key — plain set loses duplicate info
    - Think about: what if one string has "the the the" and the other
      has "the"? Plain set says full overlap, Counter says 1/3.

    Edge cases:
    - Both empty -> return 1.0
    - One empty -> return 0.0
    - No overlap -> return 0.0 (avoid division by zero in F1)
    """
    raise NotImplementedError("Implement token_f1_score")


# ---------------------------------------------------------------------------
# Grader (framework code — complete)
# ---------------------------------------------------------------------------

@grader(
    name="string_similarity",
    description="Fuzzy string matching via Levenshtein distance or token F1",
)
def string_similarity_grader(
    messages, ground_truth, *, strategy="levenshtein", **kwargs
):
    response = get_last_assistant_content(messages)
    if response is None:
        return EvaluateResult(
            score=0.0,
            is_valid=False,
            reason="No assistant message found",
        )

    expected = str(ground_truth) if ground_truth is not None else ""

    # Both empty is a perfect match
    if not response and not expected:
        return EvaluateResult(score=1.0, reason="Both empty")

    if strategy == "levenshtein":
        score = levenshtein_similarity(response, expected)
    elif strategy == "token_f1":
        score = token_f1_score(response, expected)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return EvaluateResult(
        score=score,
        reason=f"Similarity ({strategy}): {score:.4f}",
        metrics={
            strategy: MetricResult(
                score=score,
                reason=f"Computed via {strategy} strategy",
            ),
        },
    )
