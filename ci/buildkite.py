#!/usr/bin/env python

import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import harness
import pystache
import report
import util
import yaml

# Artifacts are stored with buildkite using the following scheme:
#
# $ROOT/tmp/output/$SUITE/$WORKLOAD/$PLATFORM/{params}/[result.json, result.npy]

DEFAULT_PIPELINE = 'plaidml'

PIPELINE = os.getenv('PIPELINE', os.getenv('BUILDKITE_PIPELINE_NAME', DEFAULT_PIPELINE))
BUILD_ID = os.getenv('BUILDKITE_BUILD_NUMBER', '0')

cli = argparse.ArgumentParser()
cli.add_argument('--pipeline', default=PIPELINE)
subparsers = cli.add_subparsers(dest="subcommand")


def load_template(name):
    this_dir = Path(__file__).parent
    template_path = this_dir / name
    return template_path.read_text()


def get_emoji(variant):
    if variant == 'windows_x86_64':
        return ':windows:'
    if variant == 'macos_x86_64':
        return ':darwin:'
    if variant == 'macos_x86_64_dbg':
        return ':darwin::sleuth_or_spy:'
    if variant == 'linux_x86_64_dbg':
        return ':linux::sleuth_or_spy:'
    return ':linux:'


def get_engine(pkey):
    if 'stripe-ocl' in pkey:
        return ':barber::cl:'
    if 'stripe-mtl' in pkey:
        return ':barber::metal:'
    if 'plaid-mtl' in pkey:
        return ':black_square_button::metal:'
    if 'plaid-ocl' in pkey:
        return ':black_square_button::cl:'
    if 'llvm-cpu' in pkey:
        return ':crown:'
    if 'opencl-cpu' in pkey:
        return ':crown::cl:'
    if 'tf-' in pkey:
        return ':tensorflow:'
    if 'ocl-gen' in pkey:
        return ':information_source::cl:'
    if 'vk-gen' in pkey:
        return ':information_source:'
    return ':small_blue_diamond:'


def get_python(variant):
    if variant == 'windows_x86_64':
        return 'python'
    return 'python3'


@util.subcommand(subparsers, util.argument('--count', action='store_true'))
def cmd_pipeline(args, remainder):
    with open('ci/plan.yml') as file_:
        plan = yaml.safe_load(file_)

    variants = []
    for variant in plan['VARIANTS'].keys():
        variants.append(
            dict(
                name=variant,
                python=get_python(variant),
                emoji=get_emoji(variant),
                artifacts='dbg' not in variant,
            ))

    tests = []
    for test in util.iterate_tests(plan, args.pipeline):
        tests.append(
            dict(
                suite=test.suite_name,
                workload=test.workload_name,
                platform=test.platform_name,
                batch_size=test.batch_size,
                variant=test.variant,
                timeout=test.timeout,
                retry=test.retry,
                soft_fail=test.soft_fail,
                python=get_python(test.variant),
                emoji=get_emoji(test.variant),
                engine=get_engine(test.platform_name),
            ))

    if args.count:
        util.printf('variants: {}'.format(len(variants)))
        util.printf('tests   : {}'.format(len(tests)))
        util.printf('total   : {}'.format(len(variants) + len(tests)))
    else:
        ctx = dict(variants=variants, tests=tests)
        yml = pystache.render(load_template('pipeline.yml'), ctx)
        util.printf(yml)


@util.subcommand(subparsers, util.argument('variant'))
def cmd_build(args, remainder):
    with open('ci/plan.yml') as file_:
        plan = yaml.safe_load(file_)

    env = os.environ.copy()
    variant = plan['VARIANTS'][args.variant]
    for key, value in variant.get('env', {}).items():
        env[key] = str(value)

    build_root = variant.get('build_root', 'build-x86_64')
    build_type = variant.get('build_type', 'Release')
    check = variant.get('check', 'smoke')
    system = variant.get('system', 'Linux')

    temp_dir = Path('/tmp') / os.getenv('BUILDKITE_AGENT_NAME')
    build_dir = Path(build_root) / build_type
    logs_dir = Path('logs').resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    util.printf('--- :building_construction: configure')
    configure_log = logs_dir / 'configure.log'
    with configure_log.open('wb') as fp:
        util.check_call(
            ['python', 'configure', '--ci', f'--temp={temp_dir}', f'--type={build_type}'],
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT)

    util.printf('--- :hammer_and_wrench: ninja')
    util.check_call(['ninja', '-C', build_dir], env=env)

    util.printf('--- :hammer_and_wrench: ninja package')
    util.check_call(['ninja', '-C', build_dir, 'package'], env=env)

    util.printf(f'--- :hammer_and_wrench: ninja check-{check}')
    check_log = logs_dir / f'check-{check}.log'
    with check_log.open('wb') as fp:
        util.check_call(['ninja', '-C', build_dir, f'check-{check}'],
                        env=env,
                        stdout=fp,
                        stderr=subprocess.STDOUT)

    util.printf('--- Test devkit')
    devkit_dir = build_dir / '_CPack_Packages' / system / 'TGZ' / f'PlaidML-1.0.0-{system}' / 'devkit'
    devkit_build_dir = devkit_dir / 'build'
    cmd = ['cmake']
    cmd += ['-S', devkit_dir]
    cmd += ['-B', devkit_build_dir]
    cmd += ['-G', 'Ninja']
    util.check_call(cmd, env=env)
    util.check_call(['ninja', '-C', devkit_build_dir], env=env)
    util.check_call([devkit_build_dir / 'edsl_test'], env=env)

    if 'dbg' not in args.variant:
        util.buildkite_upload(build_dir / '*.whl')
        util.buildkite_upload(build_dir / '*.tar.gz')


@util.subcommand(
    subparsers,
    util.argument('platform'),
    util.argument('suite'),
    util.argument('workload'),
    util.argument('batch_size'),
    util.argument('--local', action='store_true'),
)
def cmd_test(args, remainder):
    harness.run(args, remainder)


def download_test_artifacts(pattern):
    util.buildkite_download(pattern, '.', check=False)
    util.buildkite_download(pattern.replace('/', '\\'), '.', check=False)
    for path in glob.glob(pattern):
        src = Path(path)
        tgt = Path(path.replace('\\', '/'))
        tgt.parent.mkdir(parents=True, exist_ok=True)
        src.rename(tgt)


@util.subcommand(
    subparsers,
    util.argument('--local', action='store_true'),
)
def cmd_report(args, remainder):
    args.root = Path('tmp').resolve()
    if not args.local:
        download_test_artifacts('tmp/test/**/*')
    report.run(args, remainder)


def main():
    args, remainder = cli.parse_known_args()
    if args.subcommand:
        args.func(args, remainder)
    else:
        cli.print_help()


if __name__ == '__main__':
    main()
