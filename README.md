# Shrink Ray

Shrink Ray is a new test case reducer designed to work for any format, text
or binary. It's the successor to [structureshrink](https://github.com/DRMacIver/structureshrink),
a previous experimental test-case reducer of mine, and is significantly faster and more robust.
Additionally, unlike structureshrink it is designed to make extensive use of
parallelism, and should scale near linearly with the number of cores available.

## What is a Test-Case Reducer?

A test-case reducer is an automated tool for taking some file based test case
and creating a smaller version of it that satisfies some property of interest,
typically that it triggers a bug in some piece of software.

This is useful when you have a bug you want to understand or report but your
test case for reproducing it is large and messy. A test-case reducer automates
the process of turning it into something that is comparatively small and tidy,
so you don't have to waste your time doing that or waste anyone else's time
by reporting a bug which is hard to understand.

## Why use shrink ray?

Basically you should use shrink ray whenever [C-Reduce](https://github.com/csmith-project/creduce) is not working well on your problem.
Most of the time if your input format is some vaguely C-Like language you should consider just using C-Reduce instead.
When your input format is *actually* C you should definitely use C-Reduce instead of Shrink Ray.

Shrink Ray has a goal of producing smaller results than any test-case reducer that isn't C-Reduce or of a similar level of specialisation to it.
I can't currently promise it *achieves* that goal, but it is likely to do fairly well on most input formats - certainly better than a basic delta-debugging based approach -
and if you have any examples of it failing to do so I'd love to hear about them so I can fix them.

## Usage

Currently it is recommended you install Shrink Ray from git into a virtualenv
running Python 3.7+. Versions will be released to pypi, but not with any great
regularity.

This can be done as:

```bash
virtualenv shrinkray-venv
source shrinkray-venv/bin/activate
pip install git+https://github.com/DRMacIver/shrinkray.git
```    

Documentation, such as there is, is obtained by running
`shrinkray --help`, but basic usage is:

```bash
shrinkray ./test-script.sh sourcefile
```

The test script is some command that determines if the contents read from
stdin are interesting, and sourcefile should be some interesting file. A
smaller version will be written to (by default) ``sourcefile.reduced``.

## Supported platforms

Shrink Ray only runs on Python 3.7+ and has only been tested on Linux. It will
ideally work on any other unix, but this hasn't been tested so there are
probably significant problems. It will not work on Windows and no Windows
support is currently intended.
