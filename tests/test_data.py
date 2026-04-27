import pytest
from transformers import AutoTokenizer

from src.data.dataset import LABEL_NAMES, LABEL2ID
from src.data.preprocessing import _align_single, tokenize_and_align


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-cased")


@pytest.fixture
def example_sentence():
    return {
        "tokens": [
            "Only",
            "France",
            "and",
            "Britain",
            "backed",
            "Fischler",
            "'s",
            "proposal",
            ".",
        ],
        "ner_tags": [0, 5, 0, 5, 0, 1, 0, 0, 0],
        # O, B-LOC, O, B-LOC, O, B-PER, O, O, O
    }


def test_label_names_count():
    assert len(LABEL_NAMES) == 9


def test_label2id_consistency():
    for idx, name in enumerate(LABEL_NAMES):
        assert LABEL2ID[name] == idx


def test_align_special_tokens(tokenizer, example_sentence):
    tokenized = tokenizer(
        example_sentence["tokens"],
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    aligned = _align_single(example_sentence["ner_tags"], word_ids)

    assert aligned[0] == -100
    assert aligned[-1] == -100


def test_align_bio_notation(tokenizer, example_sentence):
    tokenized = tokenizer(
        example_sentence["tokens"],
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    aligned = _align_single(example_sentence["ner_tags"], word_ids)

    tokens = tokenized.tokens()
    fischler_indices = [
        i for i, t in enumerate(tokens) if t in ("Fi", "##sch", "##ler")
    ]

    assert len(fischler_indices) == 3
    assert aligned[fischler_indices[0]] == LABEL2ID["B-PER"]
    assert aligned[fischler_indices[1]] == LABEL2ID["I-PER"]
    assert aligned[fischler_indices[2]] == LABEL2ID["I-PER"]


def test_align_length_matches_tokens(tokenizer, example_sentence):
    tokenized = tokenizer(
        example_sentence["tokens"],
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    aligned = _align_single(example_sentence["ner_tags"], word_ids)

    assert len(aligned) == len(tokenized.tokens())


def test_align_o_label_preserved(tokenizer, example_sentence):
    tokenized = tokenizer(
        example_sentence["tokens"],
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    aligned = _align_single(example_sentence["ner_tags"], word_ids)

    tokens = tokenized.tokens()
    only_idx = tokens.index("Only")
    assert aligned[only_idx] == LABEL2ID["O"]


def test_tokenize_and_align_batch(tokenizer, example_sentence):
    examples = {
        "tokens": [example_sentence["tokens"]],
        "ner_tags": [example_sentence["ner_tags"]],
    }
    result = tokenize_and_align(examples, tokenizer, max_length=128)

    assert "input_ids" in result
    assert "labels" in result
    assert len(result["labels"][0]) == len(result["input_ids"][0])
