from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from shrinkray import Reducer, sort_key
from shrinkray.cutting import (
    BracketCuttingStrategy,
    CuttingStrategy,
    TokenCuttingStrategy,
)


@st.composite
def interesting_strings(draw):
    initial_pool = draw(st.lists(st.binary(min_size=1), min_size=1))

    while draw(st.integers(0, 3)) > 0:
        a = draw(st.sampled_from(initial_pool))
        b = draw(st.sampled_from(initial_pool))
        initial_pool.append(a + b)

    initial_pool.sort(key=sort_key)

    parts = draw(st.lists(st.sampled_from(initial_pool), min_size=1))
    return b"".join(parts)


class DummyReducer(object):
    def __init__(self):
        self.log = []

    def debug(self, *args, **kwargs):
        self.log.append((args, kwargs))


@given(st.sampled_from(Reducer.CUTTING_STRATEGIES), interesting_strings())
def test_cutting_endpoints(cutting_class, target):
    cutting_strategy = cutting_class(DummyReducer(), target)

    for i in range(len(target)):
        for j in cutting_strategy.endpoints(i):
            assert i < j <= len(target)


def test_debugs_to_underlying_reducer():
    reducer = DummyReducer()
    cs = CuttingStrategy(reducer, b"foo")
    cs.debug("bar")
    assert reducer.log


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(Reducer.CUTTING_STRATEGIES), interesting_strings(), st.data())
def test_enlarge_improper_cut(cutting_class, target, data):
    cutting_strategy = cutting_class(DummyReducer(), target)
    i = data.draw(st.integers(0, len(target) - 1))
    j = data.draw(st.integers(0, len(target) - 1))
    assume(i != j)
    i, j = sorted((i, j))
    assume(j not in cutting_strategy.endpoints(i))

    count = 0

    def test(a, b):
        nonlocal count

        if a == i and b == j:
            return True
        return False
        key = (a, b)
        try:
            return results[key]
        except KeyError:
            pass

        if count < 5:
            count += 1
            result = data.draw(st.booleans())
        else:
            result = False

        return results.setdefault(key, result)

    i2, j2 = cutting_strategy.enlarge_cut(i, j, test)

    assert i2 <= i
    assert j <= j2
    assert test(i2, j2)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(Reducer.CUTTING_STRATEGIES), interesting_strings(), st.data())
def test_enlarge_cut(cutting_class, target, data):
    cutting_strategy = cutting_class(DummyReducer(), target)

    i = data.draw(
        st.sampled_from(range(len(target))).filter(
            lambda i: cutting_strategy.endpoints(i)
        )
    )

    endpoints = cutting_strategy.endpoints(i)
    assert endpoints

    j = data.draw(st.sampled_from(endpoints))

    results = {}

    count = 0

    def test(a, b):
        nonlocal count

        if a == i and b == j:
            return True
        return False
        key = (a, b)
        try:
            return results[key]
        except KeyError:
            pass

        if count < 5:
            count += 1
            result = data.draw(st.booleans())
        else:
            result = False

        return results.setdefault(key, result)

    i2, j2 = cutting_strategy.enlarge_cut(i, j, test)

    assert i2 <= i
    assert j <= j2
    assert test(i2, j2)


@given(st.sampled_from(Reducer.CUTTING_STRATEGIES), interesting_strings())
def test_enlarge_full(cutting_class, target):
    cutting_strategy = cutting_class(DummyReducer(), target)

    assert cutting_strategy.enlarge_cut(0, len(target), lambda a, b: True) == (
        0,
        len(target),
    )


def test_enlarge_to_tokens():
    target = b"foooo1234"
    cutting_strategy = TokenCuttingStrategy(DummyReducer(), target)
    assert cutting_strategy.tokens == [0, target.index(b"1"), len(target)]

    i, j = cutting_strategy.enlarge_cut(0, 1, lambda i, j: b"1" not in target[i:j])
    assert i == 0
    assert j == target.index(b"1")


STARTS = sorted(BracketCuttingStrategy.MATCHING)

NOISE = sorted(
    set(range(256)) - set(STARTS) - set(BracketCuttingStrategy.MATCHING.values())
)


@st.composite
def balanced_brackets(draw):

    result = []

    stack = []

    while True:
        if draw(st.booleans()):
            result.append(draw(st.sampled_from(NOISE)))
        if draw(st.booleans()):
            c = draw(st.sampled_from(STARTS))
            stack.append(c)
            result.append(c)
        elif not stack:
            break
        else:
            result.append(BracketCuttingStrategy.MATCHING[stack.pop()])

    assume(result)

    return bytes(result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(balanced_brackets(), st.data())
def test_enlarge_bracket_cut(target, data):
    cutting_strategy = BracketCuttingStrategy(DummyReducer(), target)

    i = data.draw(
        st.sampled_from(range(len(target))).filter(
            lambda i: cutting_strategy.endpoints(i)
        )
    )

    endpoints = cutting_strategy.endpoints(i)
    assert endpoints

    j = data.draw(st.sampled_from(endpoints))

    results = {}

    count = 0

    def test(a, b):
        nonlocal count

        if a == i and b == j:
            return True
        return False
        key = (a, b)
        try:
            return results[key]
        except KeyError:
            pass

        if count < 5:
            count += 1
            result = data.draw(st.booleans())
        else:
            result = False

        return results.setdefault(key, result)

    i2, j2 = cutting_strategy.enlarge_cut(i, j, test)

    assert i2 <= i
    assert j <= j2
    assert test(i2, j2)


def test_widen_bracket_to_parent():
    target = b"{" * 10 + b"}" * 10

    assert target[9] == ord(b"{")
    assert target[10] == ord(b"}")

    cutting_strategy = BracketCuttingStrategy(DummyReducer(), target)

    for i in range(9, 0, -1):
        assert cutting_strategy.parent(i) == i - 1
        assert cutting_strategy.matching_bracket(i) == 19 - i

    i = 9
    j = 11

    i, j = cutting_strategy.enlarge_cut(
        i, j, lambda a, b: b - a < len(target) and (i - a) == (b - j)
    )
    assert i == 1
    assert j == 19


def test_widen_bracket_to_parent():
    target = b"{" * 10 + b"foo" + b"}" * 10

    cutting_strategy = BracketCuttingStrategy(DummyReducer(), target)

    i = 10
    j = 13

    i, j = cutting_strategy.enlarge_cut(
        i, j, lambda a, b: b - a < len(target) and (i - a) == (b - j)
    )
    assert i == 1
    assert j == 22
