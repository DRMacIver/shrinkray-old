# The Design of Shrink Ray

These are some design notes for shrink ray that attempt to describe how it works.
They are far from complete, and as I'm currently tinkering with the implementation of Shrink Ray a fair bit they will tend to be a little out of date.

## Concurrency

The basic design of shrink ray is that it will make use of what parallelism is available to it,
but it will produce the same results regardless of its parallelism level.

Most of its speedups from parallelism will come when it is failing to make progress: In general *verifying* that each of `n` transformations fails to reduce the test case is embarrassingly parallel,
but applying `n` successful transformations is intrinsically sequential.
Shrink Ray will tend to go through alternating periods of parallel and sequential execution as it works.

## Cutting Down Test Cases

The core design of shrink ray is based on the idea of a *cut*.
A cut is any test case that is formed by removing a contiguous subsequence of the original sequence.
In Python syntax a cut of `t` is any `t[:i] + t[j:]` for some `0 <= i < j <= len(t)`.
A successful cut is one where the cut test case is still interesting.

An ideal test-case reducer would perform all successful cuts, but this is impractical as there are `O(n^2)` possible cuts,
and most test cases are large enough and most interestingness tests are slow enough that you can't possibly try them all.

Shrink Ray works by:

1. Identifying a set of cuts that are likely to work. This set is large but more or less linear in the size of the test case.
2. Using a variety of heuristics and clever tricks to apply the set of cuts it identifies as quickly as possible.

### Identifying Good Cuts

Shrink Ray uses the idea of a pluggable *cutting strategy*, which is designed around identifying and manipulating cuts of a particular type.
A cutting strategy implements two methods:

1. Given an index `i`, what values of `j` would cause `(i, j)` to be a good cut?
2. Given an cut `i, j` satisfying some predicate `p`, return a possibly larger cut `i2 <= i < j <= j2`.

The latter is useful because it allows us to use [adaptive test-case reduction methods](https://www.drmaciver.com/2017/06/adaptive-delta-debugging/) for Shrink Ray's cut based approach.

Shrink Ray currently implements the following cutting strategies:

* Look for cuts such that `t[i:]` and `t[j:]` start with some common ngram `s` and such that `s not in t[i:j]`
* Looking for balanced bracket pairs for a variety of different types of matching bracket and try cutting both their interiors and the whole brackets.
* Cut all short sequences of bytes
* Cut all short sequences of tokens (according to a fairly crude tokeniser that groups together "similar" bytes).
* Cut every `0, i` and every `i, len(t)`
* Cut every `i < j` such that `t[i:j]` is a repetition of the same character.

Each of these comes with a reasonably natural notion of cut expansion that allows us to grow interesting cuts to a larger size.
For example, ngram based cuts can be enlarged by skipping over multiple instances of the ngram,
and bracket based cuts can be enlarged by moving to an enclosing bracket.

These are not a particularly well thought out set of cuts, but they're all relatively cheap and cover some cases that none of the others do.

### "Chaos Mode" Reduction

For each cutting strategy we do an initial "chaos mode" run, which uses non-deterministic choices of cut to attempt to make large reductions to the size of the test case.
This seems to be much more effective than starting with a deterministic reduction strategy, because it is often the case that there are a lot of potential reductions but they are spread out across the test case.

Chaos mode proceeds as follows:

1. Generate up to `1000` random cuts. In parallel, determine which of these is successful.
2. Perform a *cut merging* step, which takes a set of individually successful cuts and attempts to apply all of them to the test case. This will typically only apply a subset of the successful cuts, as some of them may conflict.
3. For each cut that was selected in step 2, attempt to enlarge it subject to the condition that when applied along with all of the currently selected cuts.

In practice steps 2 and 3 are combined and can be run in parallel with step 1.

Cut merging takes \(n\) cuts and attempts to apply all of them to the target test case.
This can be done fairly straightforwardly by merging overlapping cuts and then building
the desired test case from the now non-overlapping list of cuts:

```python
def cut_all(target, cuts):
    """Form the test case that results from simultaneously combining all of
    these cuts."""
    merged_cuts = []

    for t in sorted(cuts):
        i, j = t
        if not merged_cuts or merged_cuts[-1][-1] < i:
            # This cut is separated from all previous cuts so should be added
            # as its own cut.
            merged_cuts.append(list(t))
        else:
            # These two cuts overlap or abut, so extend the right end of the last
            # cut to include the end of the current cut.
            merged_cuts[-1][-1] = max(j, merged_cuts[-1][-1])

    # Build the cut result by iterating through the original target, skipping
    # over any regions that we've cut.
    result = bytearray()
    prev = 0
    for u, v in cuts:
        result.extend(target[prev:u])
        prev = v
    result.extend(target[prev:])
    return bytes(result)
```

The difficulty is that the merge of all successful cuts may not itself produce an interesting test case.
Consider for example a test that fails if it has at least two bytes,
and an initial three byte test case. Any one byte cut succeeds, but no two of these cuts can be simultaneously applied.

The solution is to go through all successful cuts found in the initial parallel stage one by one and attempt to add them to a set of "good" cuts that can be simultaneously applied.
In an ideal case all will work, but some conflicts may occur.

In principle we could do this in fewer than `O(n)` attempts (e.g. we could first try applying all cuts and if that works just do that).
The reason we don't is that we need to do cut expansions for each good cut anyway.
We combine this with the cut merging step:
If we successfully add a cut to the list of good cuts, we immediately try to expand it subject to the condition that the expanded cut is still compatible with our set of known good cuts.

This is a greedy algorithm and as such doesn't necessarily choose the combination of cuts that minimises the test case - e.g. it might reject a larger cut that conflicts with already selected cuts.
In order to mitigate (but not prevent) this we add cuts in order to largest to smallest.

This merging and expansion stage is unfortunately intrinsically sequential. This is offset by the fact that the amount of work it does is more or less proportional to the amount of progress it makes,
adhering to the goal of only being sequential when making progress,
and by the fact that it can be run in parallel with the cut discovery phase.

Chaos mode reduction is run for each cutting strategy in turn, repeating a cutting strategy until chaos mode completes and has reduced the test case size by no more than 1%.
After that point it is deemed no longer worth running the cutting strategy in random mode,
as it is likely to be more efficient to run it deterministically.

### Deterministic Mode Reduction

Once chaos mode has stopped producing good results, Shrink Ray switches to a deterministic mode, that for each `i` tries every identified cut from `i`, and if it finds a successful one attempts to expand it.

The details of this are fairly straightforward, but two notable features are:

1. Values of `i` are tried from largest to smallest, as dependencies will tend to go in that direction. Going from smallest to largest will tend to "unlock" more on subsequent passes.
2. The cuts are found based on a parallel implementation of a function `find_first`, which finds the first element of some sequence satisfying a predicate. `find_first` can be implemented so that it scales reasonably well across multiple cores - if the index of result is much larger than the number of available cores it will almost perfectly use all the available parallelism - and can be designed to do not too much wasted work (no more than twice as many calls as should be required) when the index of the result is small.

## Pass Ordering

Currently Shrink Ray adopts a fairly naive and hand coded ordering of the various passes it tries. This is expected to change in future.
