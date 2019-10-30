import hashlib
import itertools
import math
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from random import Random
from threading import Lock

from shrinkray.cutting import (
    BracketCuttingStrategy,
    CharCuttingStrategy,
    CutRepetitions,
    NGramCuttingStrategy,
    ShortCuttingStrategy,
    TokenCuttingStrategy,
)
from shrinkray.junkdrawer import (
    LazySequenceCopy,
    Stream,
    find_integer,
    pop_random,
    swap_and_pop,
)


class InvalidArguments(Exception):
    pass


class Reducer(object):

    CUTTING_STRATEGIES = [
        NGramCuttingStrategy,
        BracketCuttingStrategy,
        CharCuttingStrategy,
        TokenCuttingStrategy,
        ShortCuttingStrategy,
        CutRepetitions,
    ]

    def __init__(self, initial, predicate, debug=False, parallelism=1, random=None):
        if not initial:
            raise InvalidArguments("Initial example is empty")

        self.target = initial

        self.__debug = debug
        self.__cache = {}
        self.__parallelism = parallelism
        self.__predicate = predicate
        self.__lock = Lock()
        self.__random = random or Random(0)
        self.__improvement_callbacks = []

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

                if result and sort_key(value) < sort_key(self.target):
                    self.debug(
                        f"Reduced best to {len(value)} bytes (removed {len(self.target) - len(value)} bytes)"
                    )
                    self.target = value
                    for f in self.__improvement_callbacks:
                        f(value)
        finally:
            self.__lock.release()

        return result

    def run(self):
        for cs in Reducer.CUTTING_STRATEGIES:
            prev = float("inf")
            while len(self.target) <= 0.99 * prev:
                prev = len(self.target)
                self.chaos_run(cs)

        prev = None
        while prev is not self.target:
            prev = self.target

            for cs in Reducer.CUTTING_STRATEGIES:
                self.deterministic_run(cs)

            if prev is self.target:
                self.take_prefixes()
                self.take_suffixes()

    def __find_first(self, f, xs):
        if self.__thread_pool is None:
            for x in xs:
                if f(x):
                    return x
            raise NotFound()
        else:
            xs = iter(xs)

            chunk_size = 1

            while True:
                chunk = list(itertools.islice(xs, chunk_size))
                if not chunk:
                    raise NotFound()

                for v, result in zip(chunk, self.__thread_pool.map(f, chunk)):
                    if result:
                        return v
                chunk_size *= 2

    def take_prefixes(self):
        self.debug("taking prefixes")
        target = self.target
        self.__find_first(
            lambda i: self.predicate(target[:i]), range(len(target) + 1),
        )

    def take_suffixes(self):
        self.debug("taking suffixes")
        target = self.target
        self.__find_first(
            lambda i: self.predicate(target[i:]), range(len(target), -1, -1),
        )

    def chaos_run(self, cutting_strategy_class):
        self.debug(f"Beginning chaos run with {cutting_strategy_class.__name__}")

        target = self.target

        cutting_strategy = cutting_strategy_class(self, target)

        def random_cuts():
            indices = LazySequenceCopy(range(len(target)))
            endpoints_remaining = {}
            cuts = 0

            while indices and cuts < 1000:
                i_index = self.__random.randrange(0, len(indices))
                i = indices[i_index]
                try:
                    endpoints = endpoints_remaining[i]
                except KeyError:
                    endpoints = endpoints_remaining.setdefault(
                        i, LazySequenceCopy(cutting_strategy.endpoints(i))
                    )
                if not endpoints:
                    swap_and_pop(indices, i_index)
                    continue
                yield (i, pop_random(endpoints, self.__random))
                cuts += 1

        self.try_all_cuts(target, random_cuts(), cutting_strategy)

    def deterministic_run(self, cutting_strategy_class):
        self.debug(
            f"Beginning deterministic run with {cutting_strategy_class.__name__}"
        )

        i = len(self.target)
        while i > 0:
            target = self.target
            cutting_strategy = cutting_strategy_class(self, target)
            try:
                i, j = self.__find_first(
                    lambda t: self.predicate(target[: t[0]] + target[t[1] :]),
                    (
                        (a, b)
                        for a in range(min(i, len(target)) - 1, -1, -1)
                        for b in reversed(cutting_strategy.endpoints(a))
                    ),
                )
            except NotFound:
                break
            i, j = cutting_strategy.enlarge_cut(
                i, j, lambda a, b: self.predicate(target[:a] + target[b:])
            )

    def try_all_cuts(self, target, cuts, cutting_strategy):
        cuts = list(cuts)
        if not cuts:
            return

        cuts.sort(key=lambda t: (t[0] - t[1], t[0]))

        results = Stream(
            self.filter(lambda t: self.predicate(target[: t[0]] + target[t[1] :]), cuts)
        )

        good_cuts = []

        i = 0
        while results[i : i + 1]:

            def can_merge(k):
                parts = results[i : i + k]
                if len(parts) < k:
                    return False
                return self.predicate(cut_all(target, good_cuts + parts))

            k = find_integer(can_merge)
            good_cuts.extend(results[i : i + k])
            i += k + 1

        if not good_cuts:
            return

        loss_to_incompatibility = len(list(results)) - len(good_cuts)

        initial_length = len(good_cuts)
        good_cuts = merged_cuts(good_cuts)
        loss_to_merging = initial_length - len(good_cuts)

        loss = loss_to_incompatibility + loss_to_merging

        self.debug(
            f"Applied {len(good_cuts)} / {len(cuts)} cuts."
            + (
                f" Lost {loss_to_incompatibility} to incompatibility and {loss_to_merging} to merging."
                if loss > 0
                else ""
            )
        )

        for cut in list(good_cuts):
            enlarged = cutting_strategy.enlarge_cut(
                *cut, lambda i, j: self.predicate(cut_all(target, good_cuts + [(i, j)]))
            )
            if enlarged != cut:
                self.debug(
                    f"Enlarged cut {cut} to {enlarged} (increase of {enlarged[1] - enlarged[0] + cut[0] - cut[1]} bytes)"
                )

                good_cuts.append(enlarged)
                good_cuts = merged_cuts(good_cuts)

                assert self.predicate(cut_all(target, good_cuts))

    def __map(self, f, ls):
        if self.__thread_pool is not None:
            return self.__thread_pool.map(f, ls)
        else:
            return map(f, ls)

    def filter(self, f, ls):
        for x, r in self.__map(lambda x: (x, f(x)), ls):
            if r:
                yield x


def merged_cuts(cuts):
    endpoints = {i for t in cuts for i in t}

    merged = []

    for t in sorted(cuts):
        i, j = t
        if not merged or merged[-1][-1] < i:
            merged.append(list(t))
        else:
            merged[-1][-1] = max(j, merged[-1][-1])

    for t in merged:
        assert set(t).issubset(endpoints)

    return list(map(tuple, merged))


def cut_all(target, cuts):
    attempt = bytearray(target)

    for i, j in reversed(merged_cuts(cuts)):
        assert 0 <= i < j <= len(target), (i, j)
        del attempt[i:j]
    return bytes(attempt)


class NotFound(Exception):
    pass


def cache_key(s):
    assert isinstance(s, bytes), type(s)
    return hashlib.sha1(s).digest()


def sort_key(s):
    return (len(s), s)
