import base64
import csv
import json
import mimetypes
import pathlib
import subprocess
import sys

import pystache
import util
import yaml
from asq.initiators import query
from asq.record import new

DEFAULT_BUILD_URL = 'https://buildkite.com/plaidml'
PLAN_PATH = pathlib.Path('ci/plan.yml')


def printf(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def check_call(cmd, **kwargs):
    printf(cmd)
    subprocess.check_call(cmd, **kwargs)


def collect_results(root, pipeline):
    with open(PLAN_PATH) as file_:
        plan = yaml.safe_load(file_)
    gpu_flops = plan['CONST']['gpu_flops']
    baseline_name = plan['CONST']['efficiency_baseline']
    for info in util.iterate_tests(plan, pipeline):
        if info.platform_name == baseline_name:
            continue
        path = info.path(root) / 'report.json'
        print(path)
        if path.exists():
            with path.open() as fp:
                data = json.load(fp)
        else:
            data = {
                'compare': False,
                'ratio': None,
                'compile_duration': None,
                'cur.execution_duration': None,
                'ref.execution_duration': None,
                'status': 'ERROR',
                'failures': [],
                'errors': [],
                'reason': 'Result not found',
                'build_url': DEFAULT_BUILD_URL,
            }
        data['info'] = info
        yield data


CSS_MAP = {
    'ERROR': 'background-color: red; color: white',
    'FAIL': 'background-color: red; color: white',
    'SKIP': 'background-color: yellow',
    'PASS': 'background-color: green; color: white',
}


def load_template(name):
    this_dir = pathlib.Path(__file__).parent
    template_path = this_dir / 'templates' / name
    with open(template_path, 'r') as file_:
        return file_.read()


def ratio_plot(path, labels, values, title):
    import matplotlib
    matplotlib.use('Agg')  # this must come before importing pyplot
    import matplotlib.pyplot as plt

    y_pos = range(len(labels))[::-1]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, len(values) / 4)

    # Vertical lines with major at 1.0
    ax.xaxis.grid(True, color='#666666')
    ax.axvline(1.0, color='gray')

    # Horizontal bar chart, labeled on y axis with test config
    plt.title(title)
    plt.barh(y_pos, values, zorder=10)
    plt.yticks(y_pos, labels)

    for patch, value in zip(ax.patches, values):
        ax.text(patch.get_width() - 0.06,
                patch.get_y() + 0.01,
                '%.2f' % round(value, 2),
                ha='center',
                va='bottom',
                color='white',
                zorder=20)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def generate_ratio_chart(results, report_dir):
    data = query(results) \
        .where(lambda x: x['compare']) \
        .where(lambda x: x['ratio']) \
        .order_by(lambda x: x['info'].label()) \
        .select(lambda x: new(label=x['info'].label(), value=x['ratio'])) \
        .to_list()
    labels = query(data).select(lambda x: x.label).to_list()
    values = query(data).select(lambda x: x.value).to_list()
    if len(labels):
        filename = report_dir / 'ratios.png'
        ratio_plot(filename, labels, values, 'Throughput compared to golden')
        return Image(filename)
    return None


def generate_efficiency_chart(results, report_dir):
    data = query(results) \
        .where(lambda x: x['compare']) \
        .where(lambda x: x['efficiency']) \
        .order_by(lambda x: x['info'].label()) \
        .select(lambda x: new(label=x['info'].label(), value=x['efficiency'])) \
        .to_list()
    labels = query(data).select(lambda x: x.label).to_list()
    values = query(data).select(lambda x: x.value).to_list()
    if len(values):
        filename = report_dir / 'efficiency.png'
        ratio_plot(filename, labels, values, 'Efficiency relative to TF/GP100')
        return Image(filename)
    return None


def render_float(value):
    if value:
        return '{0:.3f}'.format(value)
    return 'N/A'


def make_html_results(results):

    def _make_one_result(x):
        info = x['info']
        return new(
            label=info.label(),
            status_css=CSS_MAP.get(x['status']),
            status=x['status'],
            gpu=info.platform.gpu,
            engine=info.platform.engine,
            workload=info.instance_name,
            batch_size=info.batch_size,
            cur_com=render_float(x['compile_duration']),
            cur_run=render_float(x['cur.execution_duration']),
            ref_run=render_float(x['ref.execution_duration']),
            ratio=render_float(x['ratio']),
            log=x['build_url'],
            reason=x['reason'],
        )

    return query(results) \
        .select(_make_one_result) \
        .order_by(lambda x: x.label) \
        .to_list()


def make_html_suites(results):
    return query(results) \
        .group_by(
            lambda x: x['info'].suite_name,
            result_selector=lambda k, g: new(name=k, results=make_html_results(g))) \
        .order_by(lambda x: x.name) \
        .to_list()


def make_html_summary(results):
    counts = query(results) \
        .group_by(lambda x: x['status']) \
        .to_dictionary(lambda x: x.key, len)

    errors = counts.get('ERROR', 0)
    failures = counts.get('FAIL', 0)
    skipped = counts.get('SKIP', 0)
    passed = counts.get('PASS', 0)
    total = errors + failures + skipped + passed

    if errors:
        status = 'ERROR'
    elif failures:
        status = 'FAIL'
    else:
        status = 'PASS'

    return new(
        status=status,
        css=CSS_MAP.get(status),
        errors_count=errors,
        failures_count=failures,
        skipped_count=skipped,
        passed_count=passed,
        total_count=total,
    )


def make_html_failures(results, status):
    failures = query(results) \
        .where(lambda x: x['status'] == status) \
        .select(lambda x: new(
            name=x['info'].label(),
            body=x['reason'],
            job_url=x['build_url'])) \
        .order_by(lambda x: x.name) \
        .to_list()
    if len(failures):
        return {'count': len(failures), 'items': failures}
    return None


def is_skipped(record):
    return record['status'] == 'SKIP'


def make_junit_failure(record):
    if record['status'] == 'FAIL':
        msg = '; '.join(record['failures'])
        return new(message=msg)
    return None


def make_junit_error(record):
    if record['status'] == 'ERROR':
        msg = '; '.join(record['errors'])
        return new(message=msg)
    return None


def make_junit_stdout(record):
    reason = record['reason']
    if reason:
        return new(text=reason)
    return None


def make_junit_context(results):
    testcases = query(results) \
        .select(lambda x: new(
            classname=x['info'].platform.full,
            name='{}-{}'.format(x['info'].workload_name, x['info'].batch_size),
            time=x['cur.execution_duration'],
            skipped=is_skipped(x),
            failure=make_junit_failure(x),
            error=make_junit_error(x),
            stdout=make_junit_stdout(x))) \
        .to_list()
    return dict(testcases=testcases)


def make_csv_results(results):

    def _make_one_result(x):
        info = x['info']
        return dict(
            label=info.label(),
            status=x['status'],
            gpu=info.platform.gpu,
            engine=info.platform.engine,
            workload=info.instance_name,
            batch_size=info.batch_size,
            cur_com=x['compile_duration'],
            cur_run=x['cur.execution_duration'],
            ref_run=x['ref.execution_duration'],
            ratio=x['ratio'],
            reason=x['reason'],
        )

    return query(results) \
        .select(_make_one_result) \
        .order_by(lambda x: x['label']) \
        .to_list()


class Image(object):

    def __init__(self, path):
        self.path = path

    def artifact_url(self):
        return 'artifact://report/{}'.format(self.path.name)

    def data_url(self):
        mime, _ = mimetypes.guess_type(str(self.path))
        with open(self.path, 'rb') as fp:
            data = fp.read()
        data64 = base64.b64encode(data).decode()
        return 'data:{};base64,{}'.format(mime, data64)


def write_file(filename, content):
    printf('Writing:', filename)
    with open(filename, 'w') as file_:
        file_.write(content)


def buildkite_annotate(root, style, html):
    printf('--- :buildkite: Uploading artifacts and adding annotations')
    check_call(['buildkite-agent', 'artifact', 'upload', 'report/**/*'], cwd=root)

    cmd = ['buildkite-agent', 'annotate', '--style', style]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(html.encode())


def run(args, remainder):
    printf('--- :bar_chart: Analyzing test results')

    test_dir = args.root / 'test'
    report_dir = args.root / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)

    results = list(collect_results(test_dir, args.pipeline))

    csv_path = report_dir / 'report.csv'
    csv_results = make_csv_results(results)
    field_names = list(csv_results[0].keys())
    with csv_path.open('w') as csv_file:
        printf('Writing:', csv_path)
        writer = csv.DictWriter(csv_file, field_names)
        writer.writeheader()
        writer.writerows(csv_results)

    xml = pystache.render(load_template('junit.xml'), make_junit_context(results))
    write_file(report_dir / 'junit.xml', xml)

    summary = make_html_summary(results)
    context = {
        'suites': make_html_suites(results),
        'summary': summary,
    }

    ratio_png = generate_ratio_chart(results, report_dir)
    if ratio_png:
        context['ratio_png'] = ratio_png.data_url()

    efficiency_png = generate_efficiency_chart(results, report_dir)
    if efficiency_png:
        context['efficiency_png'] = efficiency_png.data_url()

    html = pystache.render(load_template('report.html'), context)
    write_file(report_dir / 'report.html', html)

    if summary.status == 'PASS':
        style = 'success'
    else:
        style = 'error'
    write_file(report_dir / 'status.txt', style)

    context = {
        'summary': summary,
        'errors': make_html_failures(results, 'ERROR'),
        'failures': make_html_failures(results, 'FAIL'),
        'report_url': 'artifact://report/report.html',
        'wheel_url': 'artifact://all_wheels.tar.gz'
    }
    html = pystache.render(load_template('annotate.html'), context)
    write_file(report_dir / 'annotate.html', html)

    if not args.local:
        buildkite_annotate(args.root, style, html)
