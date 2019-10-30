import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from shrinkray import InvalidArguments, Reducer, sort_key


@pytest.mark.parametrize("parallelism", [1, 2, 10])
def test_removes_all_elements(parallelism):
    reducer = Reducer(
        initial=b"01" * 100,
        predicate=lambda t: t.count(b"0") == 100,
        parallelism=parallelism,
    )

    reducer.run()

    assert reducer.target == b"0" * 100


@given(st.lists(st.integers(0, 255), unique=True, min_size=1))
def test_runs_to_fixation(ls):
    ordered = bytes(sorted(ls))

    reducer = Reducer(
        initial=bytes(ls),
        predicate=lambda t: t and ordered.startswith(bytes(sorted(t))),
        parallelism=1,
    )

    reducer.run()

    assert reducer.target == ordered[:1]


@given(
    initial=st.binary(min_size=1),
    n=st.integers(1, 10),
    parallelism=st.integers(1, 3),
    method=st.sampled_from(["run", "take_prefixes", "take_suffixes",]),
)
def test_reduces_to_min_size(initial, n, parallelism, method):
    assume(n <= len(initial))
    reducer = Reducer(
        initial=initial, predicate=lambda t: len(t) >= n, parallelism=parallelism,
    )

    getattr(reducer, method)()

    assert len(reducer.target) == n


@given(st.binary(min_size=1), st.data())
def test_explore_arbitrary_predicate(b, data):
    results = {}

    best = b

    def test(t):
        nonlocal best
        if not t:
            return False
        if t == b:
            return True
        try:
            return results[t]
        except KeyError:
            pass

        result = data.draw(st.booleans())
        results[t] = result
        if result:
            best = min(best, t, key=sort_key)
        return result

    class FakeRandom(object):
        def randrange(self, start, stop):
            if stop == start + 1:
                return start
            return data.draw(
                st.integers(start, stop - 1), label=f"randrange({start}, {stop})"
            )

    reducer = Reducer(initial=b, predicate=test, parallelism=1, random=FakeRandom(),)

    reducer.run()

    assert sort_key(reducer.target) <= sort_key(b)
    assert reducer.target == best


def const(t):
    def accept(*args):
        return t

    accept.__name__ = f"const({t})"
    accept.__qualname__ = f"const({t})"
    return accept


@pytest.mark.parametrize(
    "initial, test", [(b"", const(True)), (b"0", const(True)), (b"0", const(False)),]
)
def test_argument_validation(initial, test):
    with pytest.raises(InvalidArguments):
        Reducer(initial=initial, predicate=test)


def test_prints_debug_information(capsys):
    reducer = Reducer(
        initial=bytes(10), predicate=lambda x: len(x) > 3, parallelism=1, debug=True
    )
    reducer.run()
    captured = capsys.readouterr()
    assert not captured.out
    assert len(captured.err.splitlines()) > 2


def test_runs_callbacks():
    results = []

    reducer = Reducer(
        initial=bytes(10), predicate=lambda x: len(x) > 3, parallelism=1, debug=True
    )

    reducer.on_improve(results.append)

    reducer.run()

    assert results[-1] == reducer.target
