def find_integer(f):
    """Finds a (hopefully large) integer n such that f(n) is True and f(n + 1)
    is False. Runs in O(log(n)).

    f(0) is assumed to be True and will not be checked. May not terminate unless
    f(n) is False for all sufficiently large n.
    """
    # We first do a linear scan over the small numbers and only start to do
    # anything intelligent if f(4) is true. This is because it's very hard to
    # win big when the result is small. If the result is 0 and we try 2 first
    # then we've done twice as much work as we needed to!
    for i in range(1, 5):
        if not f(i):
            return i - 1

    # We now know that f(4) is true. We want to find some number for which
    # f(n) is *not* true.
    # lo is the largest number for which we know that f(lo) is true.
    lo = 4

    # Exponential probe upwards until we find some value hi such that f(hi)
    # is not true. Subsequently we maintain the invariant that hi is the
    # smallest number for which we know that f(hi) is not true.
    hi = 5
    while f(hi):
        lo = hi
        hi *= 2

    # Now binary search until lo + 1 = hi. At that point we have f(lo) and not
    # f(lo + 1), as desired..
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo


def swap_and_pop(seq, i):
    """Remove the value at index ``i`` from ``seq`` in O(1) by swapping it to
    the end of the list and then popping it."""
    j = len(seq) - 1
    if i != j:
        seq[i], seq[j] = seq[j], seq[i]
    return seq.pop()


def pop_random(seq, random):
    return swap_and_pop(seq, random.randrange(0, len(seq)))


class LazySequenceCopy(object):
    """A "copy" of a sequence that works by inserting a mask in front
    of the underlying sequence, so that you can mutate it without changing
    the underlying sequence. Effectively behaves as if you could do list(x)
    in O(1) time. The full list API is not supported yet but there's no reason
    in principle it couldn't be."""

    def __init__(self, values):
        self.__values = values
        self.__len = len(values)
        self.__mask = None

    def __len__(self):
        return self.__len

    def pop(self):
        if len(self) == 0:
            raise IndexError("Cannot pop from empty list")
        result = self[-1]
        self.__len -= 1
        if self.__mask is not None:
            self.__mask.pop(self.__len, None)
        return result

    def __getitem__(self, i):
        i = self.__check_index(i)
        default = self.__values[i]
        if self.__mask is None:
            return default
        else:
            return self.__mask.get(i, default)

    def __setitem__(self, i, v):
        i = self.__check_index(i)
        if self.__mask is None:
            self.__mask = {}
        self.__mask[i] = v

    def __check_index(self, i):
        n = len(self)
        if i < -n or i >= n:
            raise IndexError("Index %d out of range [0, %d)" % (i, n))
        if i < 0:
            i += n
        assert 0 <= i < n
        return i
