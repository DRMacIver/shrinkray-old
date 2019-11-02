import hashlib
import os
import random
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import cpu_count
from shutil import which
from random import Random
from shrinkray.junkdrawer import find_integer

from tqdm import tqdm
import click

from shrinkray import InvalidArguments, Reducer


def validate_command(ctx, param, value):
    if value is None:
        return None
    parts = shlex.split(value)
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter("%s: command not found" % (command,))
        command = os.path.abspath(what)
    return [command] + parts[1:]


def signal_group(sp, signal):
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, signal)


def interrupt_wait_and_kill(sp):
    if sp.returncode is None:
        # In case the subprocess forked. Python might hang if you don't close
        # all pipes.
        for pipe in [sp.stdout, sp.stderr, sp.stdin]:
            if pipe:
                pipe.close()
        signal_group(sp, signal.SIGINT)
        for _ in range(10):
            if sp.poll() is not None:
                return
            time.sleep(0.1)
        signal_group(sp, signal.SIGKILL)


@click.command(
    help="""
shrinkray takes a file and a test command and attempts to produce a
minimal example such that the test command exits with 0.

TEST should be a command that takes its input on stdin and exits with 0 when
the input file is interesting. FILENAME should be some file such that TEST
exits with 0 when its contents are read.

The best result that shrink-ray can find will be written to the file specified
by --target.

If FILENAME is "-" then shrinkray will read its input from stdin.
""".strip()
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help=("Emit (extremely verbose) debug output while shrinking"),
)
@click.option(
    "--working-dir",
    default="",
    help=(
        """Directory to save results and intermediate data in. If empty, will
        default to creating a directory called shrinkray-(some string).
        """
    ),
)
@click.option(
    "--lexical/--size-only",
    default=False,
    help=(
        "Controls whether to enable lexical passes that don't directly "
        "attempt to reduce the size but may unlock further reductions."
    ),
)
@click.option(
    "--timeout",
    default=5,
    type=click.FLOAT,
    help=(
        "Time out subprocesses after this many seconds. If set to <= 0 then "
        "no timeout will be used."
    ),
)
@click.option(
    "--input-mode",
    type=click.Choice(["stdin", "file"]),
    default="stdin",
    help=(
        "How to pass input to the test program. If set to stdin, input will be passed to "
        "the program via its stdin. If set to file, a file of the same name as the original "
        "will be created in the temporary current working directly. This is mostly useful for "
        "compatibility with C-Reduce."
    ),
)
@click.argument("test", callback=validate_command)
@click.argument(
    "filename",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, allow_dash=True),
)
@click.option(
    "--parallelism",
    type=int,
    default=-1,
    help="""
Number of tests to run in parallel. If set to <= 0 will default to (1, n_cores - 1).
""",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="""
Set a random seed to use for nondeterministic parts of the reduction process.
""",
)
@click.option(
    "--also-interesting",
    type=int,
    default=30,
    help="""
If set to a non-zero return code, when the script exits with this return code
the variant tried will be saved for later inspection.
""",
)
@click.option(
    "--slow-mode/--fast-mode",
    default=False,
    help="""
If --slow-mode is past the reducer will run in a very timid mode that takes a
long time to make progress. You are unlikely to want this, but it can be useful
as an ersatz fuzzer.
""",
)
def reducer(
    debug,
    test,
    filename,
    timeout,
    working_dir,
    parallelism,
    seed,
    input_mode,
    lexical,
    also_interesting,
    slow_mode,
):

    if not working_dir:
        file_basename = os.path.basename(filename)
        if file_basename == "-":
            file_basename = "stdin"

        base = os.path.join(".shrinkray", file_basename)
        try:
            os.makedirs(base)
        except FileExistsError:
            pass

        i = 1

        while True:
            k = find_integer(lambda k: os.path.exists(os.path.join(base, str(i + k))))
            i += k + 1
            name = os.path.join(base, str(i))

            try:
                os.mkdir(name)
            except FileExistsError:
                continue
            working_dir = name
            break

        print(f"Saving results in {working_dir}")

    interesting_dir = os.path.join(working_dir, "interesting")

    if input_mode == "file" and filename == "-":
        raise click.UsageError(
            "Cannot combine --input-mode=file with reading from stdin."
        )

    if debug:

        def dump_trace(signum, frame):
            traceback.print_stack()

        signal.signal(signal.SIGQUIT, dump_trace)

    basename = os.path.basename(filename)

    first_call = True

    def classify_data(string):
        nonlocal first_call

        should_print = debug and first_call and False

        first_call = False

        with tempfile.TemporaryDirectory() as d:
            sp = subprocess.Popen(
                test,
                stdin=subprocess.PIPE,
                stdout=sys.stdout if should_print else subprocess.DEVNULL,
                stderr=sys.stderr if should_print else subprocess.DEVNULL,
                universal_newlines=False,
                preexec_fn=os.setsid,
                cwd=d,
            )

            original_string = string

            if input_mode == "file":
                with open(os.path.join(d, basename), "wb") as o:
                    o.write(string)
                string = b""

            try:
                sp.communicate(string, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            finally:
                interrupt_wait_and_kill(sp)

            if 0 != sp.returncode == also_interesting:
                key = hashlib.sha1(string).hexdigest()[:10]

                if filename == "-":
                    name = f"result-{key}"
                else:
                    base, ext = os.path.splitext(os.path.basename(filename))
                    name = f"{base}-{key}{ext}"

                try:
                    os.mkdir(interesting_dir)
                except FileExistsError:
                    pass

                path = os.path.join(interesting_dir, name)

                try:
                    with open(path, "xb") as o:
                        o.write(original_string)
                    if debug:
                        print(
                            f"Recording interesting variant in {path}", file=sys.stderr
                        )
                except FileExistsError:
                    pass

            return sp.returncode == 0

    if filename == "-":
        target = os.path.join(working_dir, "result")
    else:
        target = os.path.join(working_dir, os.path.basename(filename))

    timeout *= 10
    if timeout <= 0:
        timeout = None

    if filename == "-":
        initial = sys.stdin.buffer.read()
    else:
        with open(filename, "rb") as o:
            initial = o.read()

    if parallelism <= 0:
        parallelism = max(1, cpu_count() - 1)

    try:
        reducer = Reducer(
            initial,
            classify_data,
            debug=debug,
            parallelism=parallelism,
            random=Random(seed),
            lexical=lexical,
            slow=slow_mode,
        )
    except InvalidArguments as e:
        raise click.UsageError(e.args[0])

    pb = None

    prev = len(initial)

    @reducer.on_improve
    def _(s):
        if pb is not None:
            nonlocal prev
            pb.update(prev - len(s))
            prev = len(s)
        with open(target, "wb") as o:
            o.write(s)

    if debug:
        reducer.run()
    else:
        with tqdm(total=len(initial)) as pb:
            reducer.run()


if __name__ == "__main__":
    reducer()
