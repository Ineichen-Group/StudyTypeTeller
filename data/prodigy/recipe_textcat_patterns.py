# reference code https://gist.github.com/rolisz/1b93e60f5b9a85fb5a5b79913fd0ad4c
import copy
from typing import Union, Iterable, Optional, List

import spacy
from prodigy import recipe, log, get_stream
from prodigy.models.matcher import PatternMatcher
from prodigy.types import RecipeSettingsType
from prodigy.util import get_labels


@recipe(
    "textcat.manual_patterns",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline or blank:lang (e.g. blank:en)", "positional", None, str),
    labels=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    patterns=("Path to match patterns file", "option", "pt", str),
    # fmt: on
)
def manual(
    dataset: str,
    source: Union[str, Iterable[dict]],
    spacy_model: str,
    labels: Optional[List[str]] = None,
    patterns: Optional[str] = None,
) -> RecipeSettingsType:
    """
    Manually annotate categories that apply to a text. If more than one label
    is specified, categories are added as multiple choice options. If the
    --exclusive flag is set, categories become mutually exclusive, meaning that
    only one can be selected during annotation.
    """
    if not labels:
    	labels = ["Human-systematic-review", "Human-RCT-drug-intervention", "Human-RCT-non-drug-intervention", "Human-RCT-non-intervention", "Human-case-report", "Human-non-RCT-drug-intervention", "Human-non-RCT-non-drug-intervention", "Animal-systematic-review", "Animal-drug-intervention", "Animal-non-drug-intervention", "Animal-other", "Non-systematic-review", "Remaining"]
    log("RECIPE: Starting recipe textcat.manual", locals())
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(source, rehash=True, dedup=True, input_key="text")
    nlp = spacy.load(spacy_model)

    matcher = PatternMatcher(
        nlp,
        prior_correct=5.0,
        prior_incorrect=5.0,
        label_span=False,
        label_task=False,
        filter_labels=labels,
        combine_matches=True,
        all_examples=True,
        task_hash_keys=("label",),
    )
    matcher = matcher.from_disk(patterns)
    stream = add_suggestions(stream, matcher, labels)

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": stream,
        "config": {
            "labels": labels,
            "choice_style": "single",
            "choice_auto_accept": False,
            "exclude_by": "task",
            "auto_count_stream": True,
        },
    }


def add_suggestions(stream, matcher, labels):
    matched_texts = set()
    options = [{"id": label, "text": label} for label in labels]

    for score, eg in matcher(stream):
        matched_texts.add(eg["text"])
        task = copy.deepcopy(eg)
        task["options"] = options
        if 'label' in task:
            task["accept"] = [task['label']]
            del task['label']
        yield task
