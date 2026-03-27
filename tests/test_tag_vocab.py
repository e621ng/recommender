from recommender.model.tags import TagVocab


def test_same_string_returns_same_id():
    vocab = TagVocab()
    assert vocab.get_or_add("wolf") == vocab.get_or_add("wolf")


def test_ids_start_at_zero():
    vocab = TagVocab()
    assert vocab.get_or_add("first") == 0


def test_ids_are_contiguous():
    vocab = TagVocab()
    ids = [vocab.get_or_add(f"tag_{i}") for i in range(5)]
    assert sorted(ids) == list(range(5))


def test_len_counts_unique_tags():
    vocab = TagVocab()
    assert len(vocab) == 0
    vocab.get_or_add("a")
    assert len(vocab) == 1
    vocab.get_or_add("a")   # duplicate — no change
    assert len(vocab) == 1
    vocab.get_or_add("b")
    assert len(vocab) == 2


def test_roundtrip_preserves_ids():
    vocab = TagVocab()
    vocab.get_or_add("wolf")
    vocab.get_or_add("solo")
    vocab.get_or_add("male")
    restored = TagVocab.from_dict(vocab.to_dict())
    for tag in ("wolf", "solo", "male"):
        assert restored.get_or_add(tag) == vocab.get_or_add(tag)


def test_roundtrip_new_additions_do_not_collide():
    vocab = TagVocab()
    vocab.get_or_add("a")   # id=0
    vocab.get_or_add("b")   # id=1
    restored = TagVocab.from_dict(vocab.to_dict())
    new_id = restored.get_or_add("c")
    assert new_id not in {0, 1}


def test_to_dict_keys_are_strings():
    vocab = TagVocab()
    vocab.get_or_add("x")
    data = vocab.to_dict()
    assert all(isinstance(k, str) for k in data)


def test_from_dict_roundtrip_empty():
    vocab = TagVocab.from_dict({})
    assert len(vocab) == 0
    assert vocab.get_or_add("new") == 0
