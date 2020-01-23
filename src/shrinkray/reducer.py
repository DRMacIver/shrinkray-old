import hashlib
import itertools
import math
import sys
from collections import Counter, deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from random import Random
from threading import Lock
import heapq
import re
import bisect
from bisect import bisect_left
from sortedcontainers import SortedList
from shrinkray.junkdrawer import (
    Stream,
    LazySequenceCopy,
    find_integer,
    pop_random,
    swap_and_pop,
)
import time


class InvalidArguments(Exception):
    pass


class Reducer(object):
    def __init__(
        self,
        initial,
        predicate,
        debug=False,
        parallelism=1,
        random=None,
        lexical=True,
        slow=False,
    ):
        if not initial:
            raise InvalidArguments("Initial example is empty")

        self.target = initial

        self.__debug = debug
        self.__cache = {}
        self.__parallelism = parallelism
        self.__predicate = predicate
        self.__lock = Lock()
        self.random = random or Random(0)
        self.__improvement_callbacks = []
        self.__lexical = lexical
        self.__slow = slow

        if parallelism > 1:
            self.__thread_pool = ThreadPoolExecutor(max_workers=parallelism)
        else:
            self.__thread_pool = None

        assert isinstance(initial, bytes)

        if not self.predicate(initial):
            raise InvalidArguments("Initial example does not satisfy test")
        if self.predicate(b""):
            raise InvalidArguments("Trivial example of zero bytes satisfies test")

    def debug(self, *args, **kwargs):
        if self.__debug:
            print(*args, **kwargs, file=sys.stderr)

    def on_improve(self, f):
        self.__improvement_callbacks.append(f)

    def predicate(self, value):
        key = cache_key(value)

        if self.__lock.acquire(blocking=False):
            try:
                return self.__cache[key]
            except KeyError:
                pass
            finally:
                self.__lock.release()

        result = self.__predicate(value)

        try:
            self.__lock.acquire(blocking=True)
            if key not in self.__cache:
                self.__cache[key] = result

                if result:
                    if len(value) < len(self.target):
                        self.debug(
                            f"Reduced best to {len(value)} bytes (removed {len(self.target) - len(value)} bytes)"
                        )
                    else:
                        self.debug(
                            f"Found satisfying example of size {len(value)} bytes"
                        )
                    if sort_key(value) < sort_key(self.target):
                        self.target = value
                        for f in self.__improvement_callbacks:
                            f(value)
            else:
                # We want to force consistency even when the test is flaky
                # (which can happen e.g. due to timeouts in calling the test
                # script).
                result = self.__cache[key]
        finally:
            self.__lock.release()

        return result

    def delete_matching_brackets(self, bracket):
        l, r = bracket

        target = self.target

        regions = []
        stack = []

        labelled_starts = defaultdict(list)

        for i, c in enumerate(target):
            if c == l:
                stack.append(i)
                labelled_starts[len(stack)].append(i)
            elif c == r and stack:
                regions.append((stack.pop(), i))

        if regions:
            self.debug(f"Deleting matching brackets {bracket}")
            self.attempt_merge_cuts(
                target,
                [(i, j + 1) for i, j in regions]
                + [(i + 1, j) for i, j in regions if i + 1 < j]
                + [t for v in labelled_starts.values() for t in zip(v, v[1:])],
            )

    def one_pass_delete(self, ls, test):
        """Returns a subsequence ``result`` of ``ls`` such
        that ``test(result)`` is True. If ``result == ls``
        then ``ls`` is 1-minimal: i.e. no element can be
        removed from it.
        """
        ls = list(ls)

        i = len(ls)
        while True:
            try:
                i = self.find_first(
                    lambda j: test(ls[:j] + ls[j + 1 :]), range(i, -1, -1),
                )
            except NotFound:
                break
            k = find_integer(lambda k: k <= i and test(ls[: i - k] + ls[i + 1 :]))
            del ls[i - k : i + 1]
            i -= k + 1

#       k = 2
#       while k < len(ls):
#           i = 0
#           while i + k <= len(ls):
#               try:
#                   i = self.find_first(
#                       lambda j: test(ls[:j] + ls[j + k:]),
#                       range(i, len(ls) - k),
#                   )
#               except NotFound:
#                   break
#               del ls[i:i+k]
#           k += 1

        return ls

    def delete_one_cuts(self):
        self.debug("Deleting 1-cuts")

        index = defaultdict(list)
        target = self.target
        for i, c in enumerate(target):
            index[c].append(i)

        parts = [sorted({0, len(target)} | set(v)) for v in index.values()]

        self.attempt_merge_cuts(
            target, [t for part in parts for t in zip(part, part[1:])]
        )

    def delete_by_chars(self):
        alphabet = set(self.target)

        target = self.target
        counts = Counter(target)

        while alphabet:
            c = min(alphabet, key=lambda c: (counts[c], c))
            alphabet.remove(c)
            if counts[c] == 0:
                continue
            sep = bytes([c])
            self.one_char_delete(sep)
            new_target = self.target
            if new_target != target:
                target = new_target
                counts = Counter(target)

    def one_char_delete(self, sep):
        target = self.target
        ls = target.split(sep)
        self.debug(f"Deleting by {sep}, split into {len(ls)} parts")
        self.one_pass_delete(ls, lambda t: self.predicate(sep.join(t)))
            

    def delete_repeated_intervals(self):
        self.debug("Deleting repeated intervals")

        target = self.target
        self.attempt_merge_cuts(target, repeated_intervals(target))

    def delete_short_ranges(self):
        self.debug("Deleting short intervals")
        self.attempt_merge_cuts(
            self.target,
            [
                (i, j)
                for i in range(len(self.target))
                for j in range(i + 1, min(len(self.target), i + 11))
            ],
        )

    def run(self):
        initial_time = time.monotonic()

        #       self.delete_re(rb'/\*.+?\*/')
        #       self.delete_re(rb'#.+?\n')
        #       self.delete_re(rb'//.+?\n')
        #       self.delete_re(rb'"[^"]*"', adjust=lambda i, j: (i + 1, j))
        #       self.delete_re(rb"'[^']*'", adjust=lambda i, j: (i + 1, j))

        prev = None
        initial_passes = True
        while prev is not self.target:
            prev = self.target

            self.delete_matching_brackets(b"{}")
            self.delete_matching_brackets(b"[]")
            self.delete_matching_brackets(b"()")
            self.delete_matching_brackets(b"<>")
            self.delete_repeated_intervals()
            self.delete_short_ranges()
            continue
            self.delete_lines()

            if prev is not self.target and initial_passes:
                continue

            self.delete_one_cuts()

            initial_passes = False

            self.delete_short_ranges()

        runtime = time.monotonic() - initial_time

        self.debug(f"Reduction completed in {runtime:.2f}s")

    def find_first(self, f, xs):
        for x in self.filter(f, xs):
            return x
        raise NotFound()

    def take_prefixes(self):
        self.debug("taking prefixes")
        target = self.target
        self.find_first(
            lambda i: self.predicate(target[:i]), range(len(target) + 1),
        )

    def take_suffixes(self):
        self.debug("taking suffixes")
        target = self.target
        self.find_first(
            lambda i: self.predicate(target[i:]), range(len(target), -1, -1),
        )

    def alphabet_reduce(self):
        if not self.__lexical:
            return
        target = self.target
        counts = Counter(target)
        alphabet = sorted(counts, key=counts.__getitem__, reverse=True)
        rewritten = list(range(256))

        self.debug("Rewriting alphabet")
        for a in alphabet:

            def can_lower(k):
                if k > a:
                    return False
                try:
                    rewritten[a] = a - k
                    attempt = bytes([rewritten[c] for c in target])
                    return self.predicate(attempt)
                finally:
                    rewritten[a] = a

            k = find_integer(can_lower)
            if k > 0:
                self.debug(f"Lowering all {bytes([a])} bytes to {bytes([a - k])}")
            rewritten[a] -= k

    def chaos_run(self, cutting_strategy_class):
        self.debug(f"Beginning chaos run for {cutting_strategy_class.__name__}")

        prev = float("inf")

        while True:
            target = self.target
            if len(target) >= prev * 0.99:
                break
            prev = len(target)
            cutting_strategy = cutting_strategy_class(self, target)
            sampler = iter(EndpointSampler(cutting_strategy))

            pending_cuts = []

            def cut_iterator():
                done = False
                while True:
                    if not done and len(pending_cuts) < 100:
                        batch = list(itertools.islice(sampler, 100))
                        if not batch:
                            done = True
                        else:
                            pending_cuts.extend(batch)
                            pending_cuts.sort(key=lambda t: (t[1] - t[0], t[0]))
                    if pending_cuts:
                        yield pending_cuts.pop()
                    else:
                        break

            initial_size = len(target)

            good_cuts = self.consume_many_cuts(cutting_strategy, cut_iterator())

            working_target = cut_all(target, good_cuts)

            self.debug(
                f"{len(good_cuts)} cuts succeeded deleting {initial_size - len(working_target)} bytes."
            )

            if len(good_cuts) <= 10:
                return

    def delete_re(self, expression, adjust=None):
        expression = re.compile(expression, re.DOTALL | re.MULTILINE)

        target = self.target

        cuts = []
        i = 0
        prev = -1
        while True:
            assert i > prev
            prev = i
            m = expression.match(target, pos=i)
            if m is None:
                break
            cuts.append(m.span())
            i = cuts[-1][0] + 1

        cuts = [m.span() for m in expression.finditer(target)]
        if adjust is not None:
            cuts = [adjust(*c) for c in cuts]
            cuts = [c for c in cuts if c is not None and c[0] < c[1]]

        if cuts:
            self.debug(f"Deleting matches of {expression}")
            self.attempt_merge_cuts(target, cuts)

    def attempt_merge_cuts(self, target, cuts):
        cuts = list(cuts)
        if not cuts:
            return
        DeletionState(self, target, cuts).run()
        return

        self.debug(f"Trying to delete {len(cuts)} cuts")
        good_cuts = []

        if self.__slow:
            for c in cuts:
                good_cuts.append(c)
                if not self.predicate(cut_all(target, good_cuts)):
                    good_cuts.pop()
            return good_cuts

        if self.predicate(cut_all(target, cuts)):
            return cuts

        for c in cuts:
            assert c[0] < c[1], c

        cuts.sort(key=lambda c: (c[1], -c[0]))

        prev = float("inf")
        while True:
            assert good_cuts == merged_cuts(good_cuts)
            current_best = cut_all(target, good_cuts)
            assert self.predicate(current_best)
            assert len(current_best) < prev
            prev = len(current_best)

            if good_cuts:
                cuts = [c for c in cuts if not cut_contained(good_cuts, c)]

            try:
                i = self.find_first(
                    lambda i: self.predicate(cut_all(target, good_cuts + [cuts[i]])),
                    range(len(cuts) - 1, -1, -1),
                )
            except NotFound:
                for c in shuffle_of(self.random, cuts):
                    assert not self.predicate(cut_all(target, good_cuts + [c]))
                break

            del cuts[i + 1 :]

            i -= find_integer(
                lambda k: k <= i
                and self.predicate(cut_all(target, good_cuts + cuts[i - k :]))
            )
            assert i >= 0

            to_merge = cuts[i:]

            assert to_merge
            good_cuts = merged_cuts(good_cuts + to_merge)
            del cuts[i:]

            self.debug(f"Merged {len(to_merge)} cuts")

        return good_cuts

    def map(self, f, ls):
        if self.__thread_pool is not None:
            it = iter(ls)

            block_size = self.__parallelism

            any_yields = True
            while any_yields:
                any_yields = False
                with ThreadPoolExecutor(max_workers=self.__parallelism) as tpe:
                    for v in tpe.map(f, itertools.islice(it, block_size)):
                        any_yields = True
                        yield v
                block_size *= 2
                block_size = min(block_size, 2 * self.__parallelism)
        else:
            for x in ls:
                yield f(x)

    def filter(self, f, ls):
        for x, r in self.map(lambda x: (x, f(x)), ls):
            if r:
                yield x

    def any(self, f, ls):
        ls = list(ls)
        self.random.shuffle(ls)
        for x, r in self.map(lambda x: (x, f(x)), ls):
            if r:
                return True
        return False


def merged_cuts(cuts):
    merged = []

    for t in sorted(cuts):
        i, j = t
        if not merged or merged[-1][-1] < i:
            merged.append(list(t))
        else:
            merged[-1][-1] = max(j, merged[-1][-1])

    return list(map(tuple, merged))


def cut_all(target, cuts):
    cuts = merged_cuts(cuts)
    target = memoryview(target)
    result = bytearray()
    prev = 0
    for u, v in cuts:
        result.extend(target[prev:u])
        prev = v
    result.extend(target[prev:])
    return bytes(result)


class NotFound(Exception):
    pass


def cache_key(s):
    assert isinstance(s, bytes), type(s)
    return hashlib.sha1(s).digest()


def sort_key(s):
    return (len(s), s)


def shuffled_range(random, lo, hi):
    return shuffle_of(range(lo, hi))


def shuffle_of(random, ls):
    values = LazySequenceCopy(ls)
    while values:
        yield pop_random(values, random)


def repeated_intervals(target):
    return [(i, j) for ls in repeated_intervals_with_ngram(target).values() for (i, j) in ls]


def repeated_intervals_with_ngram(target):
    index = defaultdict(list)
    for i, c in enumerate(target):
        index[c].append(i)

    queue = []
    results = defaultdict(list)

    for v in index.values():
        if len(v) > 1:
            queue.append((1, v))

    del index

    while queue:
        k, indices = queue.pop()

        end = indices[-1]
        while True:
            while indices and indices[-1] + k >= len(target):
                indices.pop()
            if not indices:
                break
            if len({target[i + k] for i in indices}) > 1:
                break
            k += 1

        if len(indices) <= 1 or (
            indices[0] > 0 and len({target[i - 1] for i in indices}) == 1
        ):
            continue

        start = indices[0]
        ngram = target[start:start+k]

        cuts = results[ngram]

        cuts.append((0, start + k))

        for i, start in enumerate(indices):
            j = i + 1
            while j < len(indices) and indices[j] <= i + k:
                j += 1
            if j < len(indices):
                results[ngram].append((start, indices[j]))

        cuts.append((indices[-1], len(target)))

        grouped = defaultdict(list)
        for i in indices:
            grouped[target[i + k]].append(i)
        for v in grouped.values():
            if len(v) > 1:
                queue.append((k + 1, v))

    return results


def cut_contained(cuts, candidate):
    i = max(0, bisect.bisect_left(cuts, candidate) - 1)
    while i < len(cuts) and cuts[i][0] <= candidate[0]:
        c = cuts[i]
        i += 1
        if (c[0] <= candidate[0] <= candidate[1] <= c[1] for c in cuts):
            return True
    return False


def cut_overlaps(cuts, candidate):
    i = max(0, bisect.bisect_left(cuts, candidate) - 1)
    while i < len(cuts) and cuts[i][0] < candidate[1]:
        c = cuts[i]
        i += 1
        if not (c[1] <= candidate[0] or candidate[1] <= c[0]):
            return True
    return False

def all_indices(string, ngram):
    indices = []
    i = 0
    while True:
        try:
            i = string.index(ngram, i)
        except ValueError:
            break
        indices.append(i)
        i += 1
    return indices


class DeletionState(object):
    def __init__(self, reducer, target, cuts):
        self.reducer = reducer
        self.target = target
        self.cuts = SortedList(cuts)
        self.good_cuts = []

        self.__done = set()
        self.__random = self.reducer.random
        self.__queue = deque()

    def can_cut(self, *cuts):
        return self.reducer.predicate(cut_all(self.target, self.good_cuts + list(cuts)))

    def already_cut(self, i):
        if not self.good_cuts:
            return False
        j = bisect_left(self.good_cuts, (i,))
        while j > 0 and self.good_cuts[j - 1][1] >= i:
            j -= 1
        while j < len(self.good_cuts):
            cut = self.good_cuts[j]
            if cut[0] > i:
                return False
            if cut[0] <= i < cut[1]:
                return True
            j += 1
        return False

    def debug(self, *args, **kwargs):
        self.reducer.debug(*args, **kwargs)

    def run(self):
        random = False
        failures = 0
        while self.cuts:
            if len(self.cuts) > 100:
                cuts = self.__random.sample(self.cuts, 100)
            else:
                cuts = self.cuts
            cuts = sorted(cuts, key=lambda x: (x[1] - x[0], x[0] ), reverse=True)

            for cut, good in zip(cuts, self.reducer.map(self.can_cut, cuts)):
                if good:
                    self.try_delete_cut(cut)
                    break
                else:
                    self.cuts.remove(cut)

    def delete_at(self, i):
        if not (0 <= i < len(self.target)):
            return
        if self.already_cut(i):
            return
        if i in self.__done:
            return
        self.__done.add(i)

        j = bisect_left(self.cuts, (i,))
        while j > 0 and self.cuts[j - 1][1] >= i:
            j -= 1
        candidates = []
        while j < len(self.cuts):
            cut = self.cuts[j]
            if cut[0] > i:
                break
            if cut[1] > i:
                assert cut[0] <= i < cut[1], (cut, i)
                candidates.append(cut)
            j += 1
        if not candidates:
            return

        self.debug(f"Trying {len(candidates)} cuts around {i}")

        candidates.sort(key=lambda c: (c[1] - c[0], c[0]), reverse=True)

        for cut, good in zip(candidates, self.reducer.map(self.can_cut, candidates)):
            if good:
                self.try_delete_cut(cut)
                return
            else:
                self.cuts.remove(cut)

    def try_delete_cut(self, cut):
        if cut not in self.cuts:
            return False

        if not self.can_cut(cut):
            return False

        i = self.cuts.index(cut)

        upper = find_integer(lambda k: i + k <= len(self.cuts) and self.can_cut(*self.cuts[i:i+k]))
        assert upper > 0
        lower = find_integer(lambda k: i - k >= 0 and self.can_cut(*self.cuts[i - k:i+upper]))
        self.good_cuts = merged_cuts(self.good_cuts + self.cuts[i-lower:i+upper])
        self.cuts = [c for c in self.cuts if not cut_contained(self.good_cuts, c)]

        del self.cuts[max(0, i - lower):i + upper + 1]
        total = upper + lower
        if total > 1:
            self.debug(f"Deleted {total} cuts")

        return True
