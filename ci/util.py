import argparse
import os
import platform
import shutil
import subprocess
import sys

verbose = True  # pylint: disable=invalid-name
DEFAULT_RATIO_THRESHOLD = 0.8


def printf(*args, **kwargs):
    excludes_env = {key: kwargs[key] for key in kwargs if key not in ['env']}
    if excludes_env:
        print(*args, excludes_env)
    else:
        print(*args)
    sys.stdout.flush()


def call(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    return subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    subprocess.check_call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    return subprocess.check_output(cmd, **kwargs)


def buildkite_upload(pattern, **kwargs):
    check_call(['buildkite-agent', 'artifact', 'upload', pattern], **kwargs)


def buildkite_download(pattern, destination, check=True, **kwargs):
    cmd = ['buildkite-agent', 'artifact', 'download', pattern, destination]
    if check:
        check_call(cmd, **kwargs)
    else:
        call(cmd, **kwargs)


def argument(*name_or_flags, **kwargs):
    """Convenience function to properly format arguments to pass to the
    subcommand decorator.
    """
    return name_or_flags, kwargs


# This comes from: https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f
def subcommand(parent, *subparser_args):
    """Decorator to define a new subcommand in a sanity-preserving way.
    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so::

        args = cli.parse_args()
        args.func(args)

    Usage example::

        @subcommand(argument("-d", help="Enable debug mode", action="store_true"))
        def subcommand(args):
            print(args)

    Then on the command line::

        $ python cli.py subcommand -d

    """

    def decorator(func):
        name = func.__name__[len('cmd_'):]
        parser = parent.add_parser(name, description=func.__doc__)
        for args, kwargs in subparser_args:
            parser.add_argument(*args, **kwargs)
        parser.set_defaults(func=func)

    return decorator


class CondaEnv(object):

    def __init__(self, path):
        self.path = path.absolute().resolve()
        if platform.system() == 'Windows':
            self.bin = self.path / 'Scripts'
            self.python = self.path / 'python.exe'
            self.paths = [
                str(self.path),
                str(self.path / 'Library' / 'mingw-64' / 'bin'),
                str(self.path / 'Library' / 'usr' / 'bin'),
                str(self.path / 'Library' / 'bin'),
                str(self.path / 'Scripts'),
                str(self.path / 'bin'),
            ]
        else:
            self.bin = self.path / 'bin'
            self.python = self.bin / 'python'
            self.paths = [str(self.bin)]

    def env(self):
        env = {
            'CONDA_DEFAULT_ENV': str(self.path),
            'PATH': os.pathsep.join(self.paths + os.getenv('PATH').split(os.pathsep)),
        }
        if platform.system() != 'Windows':
            env['JAVA_HOME'] = str(self.path)
        else:
            env['CONDA_EXE'] = str(self.path / 'Scripts' / 'conda.exe')
            env['JAVA_HOME'] = str(self.path / 'Library')
        return env

    def create(self, spec):
        try:
            if not self.path.exists():
                check_call(['conda', 'env', 'create', '-f', str(spec), '-p', str(self.path)])
            else:
                check_call(
                    ['conda', 'env', 'update', '--prune', '-f',
                     str(spec), '-p',
                     str(self.path)])
        except:
            if self.path.exists():
                shutil.rmtree(self.path)
            raise

    def clone(self, path):
        if path.exists():
            shutil.rmtree(path)
        check_call(['conda', 'create', '--clone', str(self.path), '-p', str(path)])
        return CondaEnv(path)

    def install(self, package):
        check_call([str(self.python), '-m', 'pip', 'install', str(package)])


class DictAction(argparse.Action):

    def __init__(self, **kwargs):

        def key_value(string):
            return string.split('=', 1)

        super(DictAction, self).__init__(default={}, type=key_value, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        k, v = values
        var = getattr(namespace, self.dest)
        var[k] = v
        setattr(namespace, self.dest, var)


def first(choices):
    for choice in choices:
        if choice is not None:
            return choice
    return None


class PlanOption(object):

    def __init__(self, suite, workload, platform):
        self._suite = suite
        self._workload = workload
        self._platform = platform

    def get(self, name, default=None):
        """
        precedence order for options:
        - platform_overrides
        - workload
        - suite
        - default
        - None
        """
        override = self._workload.get('platform_overrides', {}).get(self._platform, {}).get(name)
        return first([
            override,
            self._workload.get(name),
            self._suite.get(name),
            default,
        ])


class Platform(object):

    def __init__(self, full, gpu_flops):
        self.full = full
        self.compiler, self.runtime, self.gpu = full.split('-')
        self.engine = '{}_{}'.format(self.compiler, self.runtime)
        self.gpu_flops = gpu_flops.get(self.gpu)

    def __repr__(self):
        return '<Platform({})>'.format(self.full)


class TestInfo(object):

    def __init__(self, suite, workload, platform, batch_size, variant, popt, shards=1, shard_id=0):
        self.suite_name, self.suite = suite
        self.workload_name, self.workload = workload
        self.platform_name, self.platform = platform
        self.batch_size = batch_size
        self.variant = variant
        self.timeout = popt.get('timeout', 20)
        self.retry = popt.get('retry')
        self.soft_fail = popt.get('soft_fail')
        self.perf_threshold = popt.get('perf_threshold', DEFAULT_RATIO_THRESHOLD)
        self.shards = shards
        self.shard_id = shard_id
        if self.shards > 1:
            self.instance_name = '{}-{}'.format(self.workload_name, self.shard_id)
        else:
            self.instance_name = self.workload_name

    def __repr__(self):
        return '{}/{}/{}/bs{}'.format(self.suite_name, self.workload_name, self.platform_name,
                                      self.batch_size)

    def label(self, inc_engine=True):
        label_parts = [self.platform.gpu, self.workload_name]
        if self.batch_size:
            label_parts += [str(self.batch_size)]
        if inc_engine:
            label_parts += [self.platform.engine]
        return '-'.join(label_parts)

    def path(self, root):
        batch_size = 'BATCH_SIZE={}'.format(self.batch_size)
        return root / self.suite_name / self.workload_name / self.platform_name / batch_size


def iterate_tests(plan, pipeline):
    gpu_flops = plan['CONST']['gpu_flops']
    for skey, suite in plan['SUITES'].items():
        for pkey, platform in suite['platforms'].items():
            pinfo = plan['PLATFORMS'][pkey]
            variant = pinfo['variant']
            if pipeline not in platform['pipelines']:
                continue
            for wkey, workload in suite['workloads'].items():
                popt = PlanOption(suite, workload, pkey)
                skip = workload.get('skip_platforms', [])
                if pkey in skip:
                    continue
                shards = popt.get('shards', 1)
                for shard_id in range(shards):
                    for batch_size in suite['params'][pipeline]['batch_sizes']:
                        yield TestInfo(
                            suite=(skey, suite),
                            workload=(wkey, workload),
                            platform=(pkey, Platform(pkey, gpu_flops)),
                            batch_size=batch_size,
                            variant=variant,
                            popt=popt,
                            shard_id=shard_id,
                            shards=shards,
                        )
