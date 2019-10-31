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
    ToCharsCuttingStrategy,
)
from shrinkray.junkdrawer import (
    LazySequenceCopy,
    find_integer,
    pop_random,
    swap_and_pop,
)


class InvalidArguments(Exception):
    pass


class Reducer(object):

    CUTTING_STRATEGIES = [
        BracketCuttingStrategy,
        NGramCuttingStrategy,
        CharCuttingStrategy,
        ToCharsCuttingStrategy,
        TokenCuttingStrategy,
        ShortCuttingStrategy,
        CutRepetitions,
    ]

    def __init__(
        self, initial, predicate, debug=False, parallelism=1, random=None, lexical=True
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
                    if len(value) < len(self.target):
                        self.debug(
                            f"Reduced best to {len(value)} bytes (removed {len(self.target) - len(value)} bytes)"
                        )
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

    def run(self):
        for cs in self.CUTTING_STRATEGIES:
            self.chaos_run(cs)

        self.alphabet_reduce()

        prev = None
        while prev is not self.target:
            prev = self.target

            self.deterministic_cutting()

            if prev is self.target:
                self.take_prefixes()
                self.take_suffixes()

            if prev is self.target:
                self.alphabet_reduce()

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

        failures = 0

        while failures < 3:
            target = self.target
            cutting_strategy = cutting_strategy_class(self, target)
            sampler = EndpointSampler(cutting_strategy)

            initial_size = len(target)

            good_cuts = []

            def can_cut(i, j):
                return self.predicate(cut_all(target, good_cuts + [(i, j)]))

            for i, j in self.filter(
                lambda t: self.predicate(target[: t[0]] + target[t[1] :]), sampler
            ):
                if not can_cut(i, j):
                    assert good_cuts
                    break
                good_cuts.append(cutting_strategy.enlarge_cut(i, j, can_cut))

            if not good_cuts:
                return

            target = cut_all(target, good_cuts)

            self.debug(
                f"{len(good_cuts)} cuts succeeded deleting {initial_size - len(target)} bytes."
            )

            if len(good_cuts) < 10:
                failures += 1
            else:
                failures = 0

    def deterministic_cutting(self):
        self.debug("Beginning deterministic cutting")
        i = len(self.target)
        while i > 0:
            target = self.target
            cutting_strategies = [
                cls(self, target) for cls in Reducer.CUTTING_STRATEGIES
            ]
            try:
                i, j, cs = self.__find_first(
                    lambda t: self.predicate(target[: t[0]] + target[t[1] :]),
                    (
                        (a, b, cs)
                        for a in range(min(i, len(target)) - 1, -1, -1)
                        for cs in cutting_strategies
                        for b in reversed(cs.endpoints(a))
                    ),
                )
                self.debug(
                    f"Successful cut at {i} / {len(target)} with {cs.__class__.__name__}"
                )

            except NotFound:
                break
            i, j = cs.enlarge_cut(
                i, j, lambda a, b: self.predicate(target[:a] + target[b:])
            )

    def try_all_cuts(self, target, cuts, cutting_strategy):
        cuts = list(cuts)
        if not cuts:
            return

    def map(self, f, ls):
        if self.__thread_pool is not None:
            it = iter(ls)

            block_size = self.__parallelism

            any_yields = True
            while any_yields:
                any_yields = False
                for v in self.__thread_pool.map(f, itertools.islice(it, block_size)):
                    any_yields = True
                    yield v
                block_size *= 2
        else:
            return map(f, ls)

    def filter(self, f, ls):
        for x, r in self.map(lambda x: (x, f(x)), ls):
            if r:
                yield x


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


class EndpointSampler(object):
    def __init__(self, cutting_strategy):
        self.cutting_strategy = cutting_strategy
        self.indices = LazySequenceCopy(range(len(cutting_strategy.target)))
        self.endpoints_remaining = {}

    def __iter__(self):
        cs = self.cutting_strategy
        random = cs.reducer.random
        indices = self.indices
        endpoints_remaining = self.endpoints_remaining

        while self.indices:
            i_index = random.randrange(0, len(indices))
            i = indices[i_index]
            try:
                endpoints = endpoints_remaining[i]
            except KeyError:
                endpoints = endpoints_remaining.setdefault(
                    i, LazySequenceCopy(cs.endpoints(i))
                )
            if not endpoints:
                swap_and_pop(indices, i_index)
                continue

            yield (i, pop_random(endpoints, random))
