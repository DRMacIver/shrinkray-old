import bisect
import unicodedata

from shrinkray.junkdrawer import find_integer


class CuttingStrategy(object):
    def __init__(self, reducer, target):
        self.reducer = reducer
        self.target = target
        self.__endpoints = {}
        self.__index = {}

    def indexes(self, c):
        if isinstance(c, int):
            c = bytes([c])
        try:
            return self.__index[c]
        except KeyError:
            pass

        target = self.target
        i = 0
        results = []
        while True:
            try:
                results.append(target.index(c, i))
            except ValueError:
                break
            i = results[-1] + 1
        return self.__index.setdefault(c, tuple(results))

    def endpoints(self, i):
        assert 0 <= i < len(self.target), (i, len(self.target))
        try:
            return self.__endpoints[i]
        except KeyError:
            return self.__endpoints.setdefault(i, tuple(sorted(self.calc_endpoints(i))))

    def enlarge_cut(self, i, j, predicate):
        target = self.target

        def can_cut(a, b):
            if not (0 <= a < b <= len(target)):
                return False
            return predicate(a, b)

        return widen_range(can_cut, i, j)

    def debug(self, *args, **kwargs):
        self.reducer.debug(*args, **kwargs)


class NGramCuttingStrategy(CuttingStrategy):
    def calc_endpoints(self, i):
        k = 1
        indices = []
        while True:
            substring = self.target[i : i + k]
            try:
                occurrence = self.target.index(substring, i + k)
            except ValueError:
                break
            if k > 1:
                indices.append(occurrence)
            any_iters = False
            while self.target[i : i + k] == self.target[occurrence : occurrence + k]:
                any_iters = True
                k += 1
            assert any_iters
        return indices

    def enlarge_cut(self, i, j, predicate):
        target = self.target

        def can_cut(a, b):
            if not (0 <= a < b <= len(target)):
                return False
            return predicate(a, b)

        assert can_cut(i, j)

        k = 0
        while i + k < j and j + k < len(target) and target[i + k] == target[j + k]:
            k += 1

        if k > 0:
            substring = target[i : i + k]

            self.debug(f"Cut with {substring}")

            assert substring

            indices = [0]
            while True:
                try:
                    indices.append(target.index(substring, indices[-1] + 1))
                except ValueError:
                    break

            indices.append(len(target))

            indices = [q for q in indices if abs(i - q) >= len(substring) or i == q]

            i_index = indices.index(i)
            j_index = indices.index(j)

            i_index, j_index = widen_range(
                lambda a, b: (0 <= a < b < len(indices))
                and can_cut(indices[a], indices[b]),
                i_index,
                j_index,
            )

            total = j_index - i_index

            if total > 1:
                self.debug(f"Deleted {total} runs of {substring}")

            i, j = indices[i_index], indices[j_index]

        return (i, j)


class BracketCuttingStrategy(CuttingStrategy):
    MATCHING = {l: r for l, r in [b"{}", b"[]", b"<>", b"()"]}

    def __init__(self, *args, **kwargs):
        super(BracketCuttingStrategy, self).__init__(*args, **kwargs)
        self.__parentage = {}

    def parent(self, i):
        try:
            return self.__parentage[i]
        except KeyError:
            pass

        l = self.target[i]
        result = None
        r = self.MATCHING[l]
        j = self.matching_bracket(i)
        if j is not None:
            indices = self.indexes(l)
            i_index = bisect.bisect_left(indices, i)
            assert indices[i_index] == i
            for candidate in reversed(indices[:i_index]):
                match = self.matching_bracket(candidate)
                if match is not None and match >= j:
                    result = candidate
                    break
        return self.__parentage.setdefault(i, result)

    def matching_bracket(self, i):
        if i < 0 or i >= len(self.target):
            return None
        l = self.target[i]
        try:
            r = BracketCuttingStrategy.MATCHING[l]
        except:
            return None
        counter = 1
        for j in range(i + 1, len(self.target)):
            if self.target[j] == l:
                counter += 1
            elif self.target[j] == r:
                counter -= 1
                if counter == 0:
                    return j
        return None

    def calc_endpoints(self, i):
        result = []

        match = self.matching_bracket(i)
        if match is not None:
            result.append(match + 1)

        match = self.matching_bracket(i - 1)
        if match is not None and match > i:
            result.append(match)

        return result

    def enlarge_cut(self, i, j, predicate):
        while self.target[i] in self.MATCHING:
            parent = self.parent(i)
            if parent is None:
                break
            match = self.matching_bracket(parent)
            assert match is not None
            end = max(match + 1, j)
            if predicate(parent, end):
                i = parent
                j = end
            else:
                break
        while i > 0 and self.target[i - 1] in self.MATCHING:
            parent = self.parent(i - 1)
            if parent is None:
                break
            match = self.matching_bracket(parent)
            assert match is not None
            end = max(match, j)
            if predicate(parent + 1, end):
                i = parent + 1
                j = end
            else:
                break
        return (i, j)


class ShortCuttingStrategy(CuttingStrategy):
    def calc_endpoints(self, i):
        return range(i + 1, min(len(self.target), i + 10))


class TokenCuttingStrategy(CuttingStrategy):
    __tokens = None

    @property
    def tokens(self):
        if self.__tokens is None:
            boundaries = []
            prev_cat = None

            for i, c in enumerate(self.target):
                cat = CHAR_CLASSES[c]
                if cat != prev_cat:
                    prev_cat = cat
                    boundaries.append(i)
            boundaries.append(len(self.target))
            self.__tokens = boundaries
        return self.__tokens

    def calc_endpoints(self, i):
        i_index = bisect.bisect_left(self.tokens, i)
        if i_index >= len(self.tokens) or self.tokens[i_index] != i:
            return ()
        return self.tokens[i_index + 1 : i_index + 10]

    def enlarge_cut(self, i, j, predicate):
        i_index = bisect.bisect_left(self.tokens, i)
        j_index = bisect.bisect_left(self.tokens, j)

        i2 = self.tokens[i_index]
        j2 = self.tokens[j_index]
        if not predicate(i2, j2):
            return (i, j)

        def can_cut(a, b):
            if not (0 <= a < b < len(self.tokens)):
                return False
            return predicate(self.tokens[a], self.tokens[b])

        i_index, j_index = widen_range(can_cut, i_index, j_index)

        return (self.tokens[i_index], self.tokens[j_index])


class CutRepetitions(CuttingStrategy):
    def calc_endpoints(self, i):
        for j in range(i + 1, len(self.target)):
            if self.target[i] == self.target[j]:
                yield j
            else:
                break

    def enlarge_cut(self, i, j, predicate):
        return (i, j)


class CharCuttingStrategy(CuttingStrategy):
    def calc_endpoints(self, i):
        c = self.target[i]
        indices = self.indexes(c)
        i_index = bisect.bisect_left(indices, i)
        try:
            return (indices[i_index + 1],)
        except IndexError:
            return (len(self.target),)

    def enlarge_cut(self, i, j, predicate):
        c = self.target[i]
        if j < len(self.target) and self.target[j] != c:
            return (i, j)

        indexes = self.indexes(c) + (len(self.target),)

        if self.target[0] != c:
            indexes = (0,) + indexes

        i_index = bisect.bisect_left(indexes, i)
        j_index = bisect.bisect_left(indexes, j)

        def can_cut(a, b):
            if not (0 <= a < b < len(indexes)):
                return False
            return predicate(indexes[a], indexes[b])

        i_index, j_index = widen_range(can_cut, i_index, j_index)

        return (indexes[i_index], indexes[j_index])


def widen_range(f, i, j):
    assert f(i, j)
    gap = j - i
    j = i + gap * find_integer(lambda k: f(i, i + k * gap))
    i = i - gap * find_integer(lambda k: f(i - k * gap, j))
    i -= find_integer(lambda k: f(i - k, j))
    j += find_integer(lambda k: f(i, j + k))
    return (i, j)


CHAR_CLASSES = [unicodedata.category(chr(i)) for i in range(256)]
