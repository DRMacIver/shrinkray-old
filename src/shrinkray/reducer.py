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
import time


class InvalidArguments(Exception):
    pass


class Reducer(object):

    CUTTING_STRATEGIES = [
        BracketCuttingStrategy,
        NGramCuttingStrategy,
        CharCuttingStrategy,
        TokenCuttingStrategy,
        ShortCuttingStrategy,
        CutRepetitions,
    ]

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

    def slowly_reduce(self):
        for cs in reversed(self.CUTTING_STRATEGIES):
            prev = None
            while prev is not self.target:
                prev = self.target
                self.deterministic_cutting(cs, adaptive=False)

    def run(self):
        initial_time = time.monotonic()

        if self.__slow:
            self.slowly_reduce()

        for cs in self.CUTTING_STRATEGIES:
            self.chaos_run(cs)

        self.alphabet_reduce()

        prev = None
        while prev is not self.target:
            prev = self.target

            for cs in self.CUTTING_STRATEGIES:
                self.deterministic_cutting(cs)

            if prev is self.target:
                self.take_prefixes()
                self.take_suffixes()

            if prev is self.target:
                self.alphabet_reduce()

        runtime = time.monotonic() - initial_time

        self.debug(f"Reduction completed in {runtime:.2f}s")

    def __find_first(self, f, xs):
        for x in self.filter(f, xs):
            return x
        raise NotFound()

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

    def deterministic_cutting(self, cls, adaptive=True):
        self.debug(f"Beginning deterministic cutting with {cls.__name__}")
        i = len(self.target)
        while i > 0:
            target = self.target
            self.debug(f"Starting cuts from {i} / {len(target)}")

            cutting_strategy = cls(self, target)

            good_cuts = self.consume_many_cuts(
                cutting_strategy,
                (
                    (a, b)
                    for a in range(min(i, len(target)) - 1, -1, -1)
                    for b in reversed(cutting_strategy.endpoints(a))
                ),
                adaptive=adaptive,
            )

            if not good_cuts:
                return

            i = good_cuts[0][0]

    def consume_many_cuts(self, cutting_strategy, cuts, adaptive=True):
        target = cutting_strategy.target
        working_target = target

        good_cuts = []

        def can_cut(i, j):
            if not (0 <= i < j <= len(target)):
                return False
            if not good_cuts:
                return self.predicate(target[:i] + target[j:])
            if j <= good_cuts[0][0]:
                return self.predicate(working_target[:i] + working_target[j:])
            return self.predicate(cut_all(target, good_cuts + [(i, j)]))

        redundant_merges = 0

        for a, b in self.filter(
            lambda t: self.predicate(target[: t[0]] + target[t[1] :]), cuts,
        ):
            if not can_cut(a, b):
                # We've hit a conflict. Restart the cutting from the last
                # place we succeeded at cutting.
                break

            if adaptive:
                cut = cutting_strategy.enlarge_cut(a, b, can_cut)
            else:
                cut = (a, b)

            good_cuts.append(cut)
            good_cuts = merged_cuts(good_cuts)

            # We've gotten to a point where most of the cuts we're discovering
            # are a bit useless, probably because of a successful cut expansion
            # that includes them. If this happens then we bail early so as to
            # not waste time doing work that doesn't need doing.
            prev = working_target
            working_target = cut_all(target, good_cuts)
            if working_target == prev:
                redundant_merges += 1
                if redundant_merges >= 10:
                    break
        return good_cuts

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
                block_size = min(block_size, 10 * self.__parallelism)
        else:
            for x in ls:
                yield f(x)

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

            j = pop_random(endpoints, random)
            if j >= i + 100:
                yield (i, j)
