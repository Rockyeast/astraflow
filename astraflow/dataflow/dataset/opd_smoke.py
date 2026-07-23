from datasets import Dataset

from astraflow.dataflow.dataset.utils import attach_query_ids


def get_opd_smoke_dataset(
    tokenizer=None,
    max_length: int | None = None,
) -> Dataset:
    """Return a tiny local prompt set for one-step OPD integration tests."""
    del tokenizer, max_length

    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {
                        "role": "user",
                        "content": "What is 17 + 25? Explain briefly.",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Which is larger, 3/4 or 2/3? Explain briefly.",
                    }
                ],
            ],
            "source": ["opd_smoke", "opd_smoke"],
        }
    )
    return attach_query_ids(dataset, "opd_smoke")
