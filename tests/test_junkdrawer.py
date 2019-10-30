import pytest

from shrinkray.junkdrawer import LazySequenceCopy


def test_out_of_range():
    x = LazySequenceCopy([1, 2, 3])

    with pytest.raises(IndexError):
        x[3]

    with pytest.raises(IndexError):
        x[-4]


def test_pass_through():
    x = LazySequenceCopy([1, 2, 3])
    assert x[0] == 1
    assert x[1] == 2
    assert x[2] == 3


def test_can_assign_without_changing_underlying():
    underlying = [1, 2, 3]
    x = LazySequenceCopy(underlying)
    x[1] = 10
    assert x[1] == 10
    assert underlying[1] == 2


def test_pop():
    x = LazySequenceCopy([2, 3])
    assert x.pop() == 3
    assert x.pop() == 2

    with pytest.raises(IndexError):
        x.pop()
