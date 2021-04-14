# Copyright 2018 Intel Corporation.

import argparse
import functools
import logging
import operator
import os
import sys
import unittest
import warnings
from collections import OrderedDict

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['KERAS_BACKEND'] = 'plaidml.bridge.keras'

import numpy as np
import numpy.testing as npt

# Make sure we win the race with TF to load libstdc++...
import plaidml
# Tensorflow needs some code called directly
import tensorflow
# Theano breaks on convolution if given a default optimizer
import theano
from keras.backend import floatx
from keras.backend import tensorflow_backend as tf
from keras.backend import theano_backend as th
from plaidml.bridge import keras as pkb

# Removes (almost) all tensorflow deprecation warnings
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

theano.config.optimizer = "None"
logger = logging.getLogger(__name__)

# We have to set_floatx before the interpreter encounters any of the test
# functions, because it will execute the 'opTest' decorator as it processes
# each test function, which will execute the value-generation functions, which
# will use the value of floatx() to create test data. Changing floatx later
# will have inconsistent effects.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--shard', type=int, default=0, help='Which shard to run')
    parser.add_argument('--shard-count',
                        type=int,
                        default=0,
                        help='Run sharded test split into this many shards. ' +
                        'Does not forward additional arguments to unittest. ' +
                        '0 (default) to not shard.')
    args, remainder = parser.parse_known_args()

    # plaidml._internal_set_vlog(args.verbose)
    if args.fp16:
        pkb.set_floatx('float16')
        DEFAULT_TOL = 1e-2
        DEFAULT_ATOL = 1e-5
    else:
        pkb.set_floatx('float32')
        DEFAULT_TOL = 1e-3
        DEFAULT_ATOL = 1e-8


def m(*args, **kwargs):
    dtype = kwargs.get('dtype', floatx())
    """Makes a test matrix whose dimensions are the supplied arguments."""
    total = functools.reduce(operator.mul, args, 1)
    arr = np.array(range(-2, total - 2), dtype=dtype) / 2.
    arr = np.reshape(arr, args)
    return arr


def n(*args):
    """Makes a test matrix whose dimensions are the supplied arguments.

    Differs from m only in what values it has."""
    total = functools.reduce(operator.mul, args, 1)
    arr = np.array(range(-11, total - 11), dtype=floatx())
    arr = np.reshape(arr, args)
    for i in range(5):
        if len(args) > i + 1:
            np.swapaxes(arr, 0, i + 1)
    arr = np.reshape(arr, args)
    return arr


def r(*args):
    """Makes a test matrix whose dimensions are the supplied arguments. Uniform random values"""

    return np.random.uniform(0, 1.0, args)


def _conv_inp(IN, IC, OC, IS, KS, data_format=None):
    kernel_mat_np = m(*(KS + [IC, OC]))
    if data_format == 'channels_first':
        input_mat_np = m(*([IN] + [IC] + IS))
    else:
        input_mat_np = m(*([IN] + IS + [IC]))
    inputMat = input_mat_np
    kernelMat = kernel_mat_np
    return [inputMat, kernelMat, data_format]


def _separable_conv_inp(IN, IC, OC, CM, IS, KS, data_format=None):
    depth_kernel_mat = m(*(KS + [IC, CM]))
    point_kernel_mat = m(*([1] * len(KS) + [CM * IC, OC]))
    if data_format == 'channels_first':
        input_mat = m(*([IN] + [IC] + IS))
    else:
        input_mat = m(*([IN] + IS + [IC]))
    return [input_mat, depth_kernel_mat, point_kernel_mat, data_format]


def compareForwardExact(skip_theano=True, skip_tensorflow=False):
    """Decorates test methods, checking equality under multiple backends."""

    def decorator(test_func):

        def compare(self, *args):
            if not skip_theano:
                theano_result = test_func(self, th, *args).eval()
            if not skip_tensorflow:
                tf_session = tensorflow.Session()
                tf.set_session(tf_session)
                tensorflow_result_intermediate = test_func(self, tf, *args)
                tf_session.run(tensorflow.global_variables_initializer())
                tensorflow_result = tensorflow_result_intermediate.eval(session=tf_session)
            plaidml_result = test_func(self, pkb, *args).eval()
            if not skip_theano:
                npt.assert_array_equal(plaidml_result,
                                       theano_result,
                                       err_msg='x=plaidml, y=theano')
            if not skip_tensorflow:
                npt.assert_array_equal(plaidml_result,
                                       tensorflow_result,
                                       err_msg='x=plaidml, y=tensorflow')
                tf_session.close()

        return compare

    return decorator


def compareForwardClose(epsilon=DEFAULT_TOL,
                        atol=DEFAULT_ATOL,
                        skip_theano=True,
                        skip_tensorflow=False):
    """Decorates test methods, checking near-equality under multiple backends."""

    def decorator(test_func):

        def compare(self, *args):
            if not skip_theano:
                theano_result = test_func(self, th, *args).eval()
            if not skip_tensorflow:
                tf_session = tensorflow.Session()
                tf.set_session(tf_session)
                tensorflow_result_intermediate = test_func(self, tf, *args)
                tf_session.run(tensorflow.global_variables_initializer())
                tensorflow_result = tensorflow_result_intermediate.eval(session=tf_session)
            plaidml_result = test_func(self, pkb, *args).eval()
            if not skip_theano:
                npt.assert_allclose(plaidml_result,
                                    theano_result,
                                    rtol=epsilon,
                                    atol=atol,
                                    err_msg='x=plaidml, y=theano')
            if not skip_tensorflow:
                npt.assert_allclose(plaidml_result,
                                    tensorflow_result,
                                    rtol=epsilon,
                                    atol=atol,
                                    err_msg='x=plaidml, y=tensorflow')
                tf_session.close()

        return compare

    return decorator


def compareMultiple(arguments):

    def decorator(test_func):

        def compare(*args):
            for test_arguments in arguments:
                test_func(*(args + tuple(test_arguments)))

        return compare

    return decorator


def opTest(in_data,
           tol=DEFAULT_TOL,
           atol=DEFAULT_ATOL,
           do_grads=False,
           skip_theano=True,
           skip_tensorflow=False,
           verbose=False,
           input_shapes=None):
    # If using with non-tensor parameters, all tensor params must appear before
    # all non-tensor params
    # input_shapes should be None or an iterable containing tuples
    # defining the shape for each input.
    # Use this to define symbolic shapes (None).
    def run_one_backend(self, data, test_func, b, shapes=None, *run_args):
        tf_session = tensorflow.Session()
        tf.set_session(tf_session)
        results = []
        with tf_session.as_default():
            if shapes:
                x = [b.placeholder(shape=t) for t in shapes]
            else:
                x = [b.placeholder(shape=t.shape) for t in data if hasattr(t, 'shape')]
            xv = [b.variable(t, dtype=floatx()) for t in data if hasattr(t, 'shape')]
            ps = [t for t in data if not hasattr(t, 'shape')]
            grad_funcs = test_func(self, b, *(x + ps + list(run_args)))
            funcs = test_func(self, b, *(xv + ps + list(run_args)))
            tf_session.run(tensorflow.global_variables_initializer())
            for gf, f in zip(grad_funcs, funcs):
                fr = f.eval()
                if do_grads:
                    df = b.gradients(b.mean(gf), x)
                    gfn = b.function(x, df, updates=[])
                    gr = gfn([t for t in data if hasattr(t, 'shape')])
                    try:
                        values = gr[0].values
                        dense_shape = gr[0].dense_shape
                        indices = gr[0].indices
                        result = np.zeros(dense_shape)
                        for i in indices:
                            result[i] += values[i]
                        gr[0] = result
                    except AttributeError:
                        # This wasn't an IndexedSlices object, do nothing
                        pass
                if args.verbose > 2 or verbose:
                    print('backend: ', b)
                    print('data: {}'.format(data))
                    print('fr: {}'.format(fr))
                    if do_grads:
                        print('gr: {}'.format(gr))
                if do_grads:
                    results.append((fr, gr))
                else:
                    results.append((fr,))
        return results

    def apply(test_func):

        def output(self, *args):
            for didx, data in enumerate(in_data):
                shapes = None
                if input_shapes:
                    shapes = input_shapes[didx]
                if not skip_theano:
                    theano_results = run_one_backend(self,
                                                     data,
                                                     test_func,
                                                     th,
                                                     *args,
                                                     shapes=shapes)
                if not skip_tensorflow:
                    tensorflow_results = run_one_backend(self,
                                                         data,
                                                         test_func,
                                                         tf,
                                                         *args,
                                                         shapes=shapes)
                plaidml_results = run_one_backend(self, data, test_func, pkb, *args, shapes=shapes)
                if not skip_theano:
                    for idx, (pmlr, thr) in enumerate(zip(plaidml_results, theano_results)):
                        idx = idx + 1
                        npt.assert_allclose(
                            pmlr[0],
                            thr[0],
                            rtol=tol,
                            atol=atol,
                            err_msg='ERR: datum={}, test={}, x=plaidml, y=theano'.format(
                                didx, idx))
                        if do_grads:
                            for x in range(0, len(pmlr[1])):
                                npt.assert_allclose(
                                    pmlr[1][x],
                                    thr[1][x],
                                    rtol=tol,
                                    atol=atol,
                                    err_msg='ERR: datum={}, test={}, grad, x=plaidml, y=theano'.
                                    format(didx, idx))
                if not skip_tensorflow:
                    for idx, (pmlr, tfr) in enumerate(zip(plaidml_results, tensorflow_results)):
                        idx = idx + 1
                        npt.assert_allclose(
                            pmlr[0],
                            tfr[0],
                            rtol=tol,
                            atol=atol,
                            err_msg='ERR: datum={}, test={}, x=plaidml, y=tensorflow'.format(
                                didx, idx))
                        if do_grads:
                            for x in range(0, len(pmlr[1])):
                                npt.assert_allclose(
                                    pmlr[1][x],
                                    tfr[1][x],
                                    rtol=tol,
                                    atol=atol,
                                    err_msg='ERR: datum={}, test={}, grad, x=plaidml, y=tensorflow'
                                    .format(didx, idx))

        return output

    return apply


class TestBackendOps(unittest.TestCase):
    """Tests PlaidML Keras operation definitions"""

    def a_testLearningPhase(self):
        # Test name prefixed with 'a_' because this needs to run before other tests
        assert isinstance(pkb.learning_phase()._plaidml_val(), plaidml.Placeholder)
        pkb.set_learning_phase(1)
        assert isinstance(pkb.learning_phase(), int)
        npt.assert_equal(pkb.learning_phase(), 1)
        pkb.set_learning_phase(0)
        assert isinstance(pkb.learning_phase(), int)
        npt.assert_equal(pkb.learning_phase(), 0)

    @unittest.skip("gradient is not yet implemented")
    def testReverseGradient(self):
        x = m(2, 2, 3) + 3.
        pl = pkb.placeholder(shape=x.shape)
        y = pkb.reverse_gradient(pl)
        z = pkb.sqrt(y) + pl
        df = pkb.gradients(pkb.mean(z), [pl])
        gfn = pkb.function([pl], df, updates=[])
        gr = gfn([x])
        npt.assert_allclose(gr[0], (1. / x.size) * (-1. / (2 * np.sqrt(x)) + 1.),
                            rtol=DEFAULT_TOL,
                            atol=DEFAULT_ATOL)

    @opTest([
        # Don't use exactly 0 (inconsistent gradient behavior between frameworks at ReLU cusp)
        [
            m(4, 7, 3) + .000001,
            n(4, 3) + .000001,
            m(3, 3) + .000001,
            n(3, 3) + .000001,
            False,
        ],
        [
            m(4, 7, 3) + .000001,
            n(4, 3) + .000001,
            m(3, 3) + .000001,
            n(3, 3) + .000001,
            True,
        ],
    ])
    @unittest.skip("Unknown instability issues")
    def testRNN(self, b, inp, init_state, ker, r_ker, go_back):

        def step_function(inputs, states):
            prev_out = states[0]
            activation = b.relu
            h = b.dot(inputs, ker)
            output = h + b.dot(prev_out, r_ker)
            output = activation(output)
            return output, [output]

        initial_states = [init_state]
        go_backwards = go_back
        mask = None
        constants = None
        unroll = False
        input_length = None
        out_val, all_out, all_states = b.rnn(step_function=step_function,
                                             inputs=inp,
                                             initial_states=initial_states,
                                             go_backwards=go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=unroll,
                                             input_length=input_length)
        result = [out_val, all_out] + list(all_states)
        return result

    @compareForwardExact()
    def testShape(self, b):
        return b.shape(b.variable(m(3, 3, 3, 3, 4)))

    def testClearSession(self):
        # If this test is run as anything other than the first test, there will
        # already be a context and device. pkb.clear_session() blows those away;
        # if a (new) context & device work after that, this test will pass, if
        # not it will fail (and probably other tests will too).
        data = m(3, 3)
        pkb.clear_session()
        x = pkb.identity(pkb.variable(data))
        npt.assert_array_equal(x.eval(), data, "x=plaidml, y=input")

    @compareForwardExact()
    def testPassthrough(self, b):
        return b.variable(m(3, 3))

    @compareForwardExact()
    def testArgmax(self, b):
        return b.argmax(b.variable(m(3, 3)))

    @compareForwardExact()
    def testArgmaxWithAxis(self, b):
        return b.argmax(b.variable(n(2, 3, 4) % 3), axis=-2)

    @compareForwardExact()
    def testArgmaxUnequal(self, b):
        x = b.variable(m(3, 2))
        y = b.variable(np.array([[2, 4], [5, -1], [3, 0]]))
        return b.equal(b.argmax(x, axis=0), b.argmax(y, axis=0))

    @compareForwardExact()
    def testArgmin(self, b):
        return b.argmin(b.variable(m(7, 2)))

    @compareForwardExact()
    def testArgminUnequal(self, b):
        x = b.variable(m(3, 2))
        y = b.variable(np.array([[2, 4], [5, -1], [3, 0]]))
        return b.equal(b.argmin(x, axis=0), b.argmin(y, axis=0))

    def testDropoutManual(self):
        # Note: Due to the probabilistic nature of dropout, and since we can't
        # just use seeds as different backends have different PRNGs, this tests
        # that we stay within expected tolerances.
        shape = (17, 22, 13, 13)
        noise_shape = (17, 22, 13, 1)
        level = 0.8
        x = pkb.variable(np.ones(shape))
        result = pkb.dropout(x, level, noise_shape).eval()
        mean = np.mean(result)
        if mean < 0.9 or mean > 1.1:
            raise RuntimeError("Unexpectedly large deviation from expected dropout level")
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if np.any(result[i, j, k]):
                        if not np.all(result[i, j, k]):
                            raise RuntimeError("Noise incorrectly shaped")
                        if abs(result[i, j, k, 0] - 1. / (1. - level)) > 0.0001:
                            raise RuntimeError("Bad normalization")

    @compareMultiple([[10], [-2, 5, 2, 'float32']])
    @compareForwardExact()
    def testArange(self, b, *args):
        return b.arange(*args)

    @opTest([
        [m(3, 3), m(3, 3)],
        [m(2, 3, 4, 5), m(2, 3, 5, 2)],
    ])
    def testDot(self, b, x, y):
        return [b.dot(x, y)]

    @unittest.skip(
        "In TF calling set_value and then evaluating doesn't seem to produce the new value. "
        "If we want to test this, we will need a more complex test.")
    @compareMultiple([[3, 4, 2], [7]])
    @compareForwardExact()
    def testSetValue(self, b, *args):  # TODO: Not yet correct
        np_old_x = m(*args)
        np_new_x = n(*args)
        print("old: {}, new: {}".format(np_old_x, np_new_x))
        x = b.variable(m(*args))
        b.set_value(x, n(*args))
        return x

    @opTest([
        [m(1, 2), m(1, 3, 2), (1, 2)],
        [m(2, 5), m(2, 5), 1],
        [m(2, 4, 5), m(2, 5, 1), None],
    ],
            skip_tensorflow=not bool(os.getenv('PLAIDML_BATCHDOT_TF_BEHAVIOR')),
            skip_theano=True)
    def testBatchDot(self, b, x, y, ax):
        if ax is None:
            return [b.batch_dot(x, y)]
        else:
            return [b.batch_dot(x, y, axes=ax)]

    """
    @opTest([[m(2, 3, 4, 5)]], skip_tensorflow=False)
    def testBatchDot2(self, b, x):
        return [
            b.batch_dot(x, b.variable(m(2, 3, 5, 2))),
            b.batch_dot(x, b.variable(m(2, 6, 5, 3)), axes=(0,2))
        ]
    """

    @opTest([[m(2, 5)]])
    def testBatchDot3(self, b, x):
        return [b.batch_dot(x, b.variable(m(2, 5)), axes=1)]

    """
    @opTest([[m(2, 4, 5)]])
    def testBatchDot4(self, b, x):
        return [b.batch_dot(x, b.variable(m(2, 5, 2)))]
    """

    @opTest([[m(2, 4, 5)], [m(4)], [m(2, 3)]])
    def testBatchFlatten(self, b, x):
        return [b.batch_flatten(x)]

    @opTest([[m(2, 4, 7)]])
    def testFlatten(self, b, x):
        return [b.flatten(x)]

    @compareMultiple([[10], [3, 'int8']])
    @compareForwardExact()
    def testEye(self, b, *args):
        return b.eye(*args)

    @opTest([[m(3, 3), m(3, 3)]])
    def testAddElements(self, b, x, y):
        return [x + y]

    @opTest([[m(3, 3), m(3, 3), m(3, 3)]])
    def testAddElementsRepeated(self, b, x, y, z):
        return [x + y + z]

    @opTest([
        [m(3, 3), 1.0],
        [m(3, 3), -3.4],
    ])
    def testAddConstant(self, b, x, c):
        return [
            x + c,
            c + x,
        ]

    @opTest([
        [m(3, 3), m(3, 3)],
        [m(2, 3), m(3)],
    ], do_grads=False)  # TODO: fix gradients
    def testSubElements(self, b, x, y):
        return [x - y]

    @opTest([
        [m(3, 3), 1.0],
        [m(3, 3), -3.4],
    ])
    def testSubConstant(self, b, x, c):
        return [
            x - c,
            c - x,
        ]

    @opTest([
        [m(3, 3), m(3, 3)],
        [m(2, 4), m(2, 4)],
    ])
    def testMulElements(self, b, x, y):
        return [x * y]

    @opTest([
        [m(3, 3), 2.0],
        [m(3, 3), -3.4],
    ])
    def testMulConstant(self, b, x, c):
        return [
            x * c,
            c * x,
        ]

    @opTest([
        [m(3, 3), m(3, 3)],
        [m(2, 1, 1), m(1)],
        [m(2), m(1)],
    ],
            skip_theano=True,
            do_grads=False)  # TODO: fix gradients
    def testDivElements(self, b, x, y):
        return [x / y]

    @opTest([
        [m(3, 3), 2.0],
        [m(3, 3), -3.4],
    ])
    def testDivConstant(self, b, x, c):
        return [
            x / c,
            c / x,
        ]

    @opTest([[m(2, 3, 2, 4)], [m(1, 1, 2, 1) + 1], [m(1, 2, 3, 1) + 4]], do_grads=False)
    def testAll(self, b, x):
        return [
            b.all(x),
            b.all(x, keepdims=True),
            b.all(x, axis=[1, 3]),
            b.all(x, axis=-1),
        ]

    @opTest([[m(2, 3, 2, 4)], [m(1, 1, 2, 1) + 1]], do_grads=False)
    def testAny(self, b, x):
        return [
            b.any(x),
            b.any(x, keepdims=True),
            b.any(x, axis=[1, 3]),
            b.any(x, axis=-1),
        ]

    @opTest([
        [m(3, 3)],
        [m(3, 3), None, True],
        [m(2, 3, 4, 5), [1, 3]],
        [m(3, 4, 5), -1],
        [m(2, 3, 4), 0],
    ])
    def testSum(self, b, x, ax=None, kd=False):
        return [b.sum(x, axis=ax, keepdims=kd)]

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardExact()
    def testProd(self, b):
        return b.prod(b.variable(m(3, 3)))

    @compareForwardClose()
    def testProdOfShape(self, b):
        return b.prod(b.shape(b.variable(m(2, 3, 4))))

    @compareForwardClose()
    def testReshapeShape(self, b):
        x = b.variable(m(5, 2, 3))
        y = b.variable(m(5, 6))
        return b.reshape(x, b.shape(y))

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardExact()
    def testProdKeepdims(self, b):
        return b.prod(b.variable(m(3, 3)), keepdims=True)

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardClose()
    def testProdAxis(self, b):
        return b.prod(b.variable(m(2, 3, 4, 5)), axis=[1, 3])

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardClose()
    def testProdAxisNumpy(self, b):
        return b.prod(b.variable(m(2, 3, 4, 5)), axis=np.array((1, 3)))

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivs
    @compareForwardClose()
    def testProdNegAxis(self, b):
        return b.prod(b.variable(m(3, 4, 5)), axis=-1)

    @opTest([
        [m(3, 4)],
        [m(3, 3, 2), None, True],
        [m(4, 3, 2, 1), [1, 3]],
    ])
    def testMax(self, b, x, ax=None, kd=False):
        return [b.max(x, axis=ax, keepdims=kd)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(3, 4) - 3.3, m(3, 4) / 2.0]])
    def testMaximum(self, b, x, y):
        return [b.maximum(x, y)]

    @opTest([[m(2, 4, 3)]])
    def testMin(self, b, x):
        return [b.min(x)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(4, 3) - 3.1, m(4, 3) / 2.0]])
    def testMinimum(self, b, x, y):
        return [b.minimum(x, y)]

    @opTest([
        [m(3, 3)],
        [m(3, 3), None, True],
        [m(2, 3, 4, 5), [1, 3]],
        [m(1, 2, 3, 4, 5), [-2, 2]],  # Note: axis -2 means next to last axis
    ])
    def testMean(self, b, x, ax=None, kd=False):
        return [b.mean(x, axis=ax, keepdims=kd)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(3, 3), 2.0001, 5.0001]])
    def testClip(self, b, x, lo, hi):
        return [b.clip(x, lo, hi)]

    @opTest([
        [m(3, 2) + 0.0001, -2],
        [m(3, 3) - 0.0001, 0.5],
        [m(3, 4) + 0.0001, 0.1],
    ])
    def testElu(self, b, x, a=1.0):
        return [b.elu(x), b.elu(x, alpha=a)]

    # T1031: This doesn't match TF/Theano on corner
    @opTest([
        [m(3, 3) - 0.0001, 0.5, 3, 0.0],
        [m(3, 4) + 0.0001, 0.1, 5, 0.0],
        [m(3, 4) + 0.0001, 0.1, 5, 2.5],
    ])
    def testRelu(self, b, x, a=0.0, m=None, threshold=0.0):
        return [
            b.relu(x),
            b.relu(x, alpha=a),
            b.relu(x, max_value=m),
            b.relu(x, alpha=a, max_value=m),
            b.relu(x, threshold=threshold),
            b.relu(x, alpha=a, threshold=threshold),
            b.relu(x, max_value=m, threshold=threshold),
            b.relu(x, alpha=a, max_value=m, threshold=threshold),
        ]

    @opTest([[m(3, 6)]])
    def testHardSigmoid(self, b, x):
        return [b.hard_sigmoid(x)]

    @compareForwardExact()
    def testEqual(self, b):
        return b.equal(b.variable(m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testEqualVersusNumeric(self, b):
        return b.equal(b.variable(m(3, 3)), m(3, 3))

    @compareForwardExact()
    def testNotEqual(self, b):
        return b.not_equal(b.variable(m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testNotEqualVersusNumeric(self, b):
        return b.not_equal(b.variable(m(3, 3)), m(3, 3))

    @compareForwardExact()
    def testLess(self, b):
        return b.less(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testLessVersusNumeric(self, b):
        return b.less(b.variable(2 * m(3, 3)), m(3, 3))

    @compareForwardExact()
    def testLessEqual(self, b):
        return b.less_equal(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testLessEqualVersusNumeric(self, b):
        return b.less_equal(b.variable(2 * m(3, 3)), m(3, 3))

    @compareForwardExact()
    def testGreater(self, b):
        return b.greater(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testGreaterVersusNumeric(self, b):
        return b.greater(b.variable(2 * m(3, 3)), m(3, 3))

    @compareForwardExact()
    def testGreaterEqual(self, b):
        return b.greater_equal(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testGreaterEqualVersusNumeric(self, b):
        return b.greater_equal(b.variable(2 * m(3, 3)), m(3, 3))

    @opTest([[m(3, 3) - 0.0001]])
    def testAbs(self, b, x):
        return [b.abs(x)]

    @opTest([[m(3, 3)]])
    def testSquare(self, b, x):
        return [b.square(x)]

    @opTest([[m(2, 3) + 3], [m(2, 3, 4) + 3]])
    def testSqrt(self, b, x):
        return [b.sqrt(x)]

    @opTest([
        [np.log(m(2, 4) + 1.5)],
        [m(1, 2, 1)],
        [np.sqrt(m(5, 5, 10) + 2) - 3],
        [np.sin(m(4, 3, 2, 1, 6))],
    ],
            1e-02,
            skip_theano=True)
    def testSoftmax(self, b, x):
        return [
            -b.log(b.softmax(x)),
            -b.log(b.softmax(x, axis=-1)),
            -b.log(b.softmax(x, axis=1)),
        ]

    @opTest([[m(1, 3, 2)]])
    def testBranchWithSoftmax(self, b, x):
        return [
            b.softmax(x) + b.mean(x),
            b.log(b.softmax(x)) + b.log(b.mean(x)),
            b.softmax(x, axis=1) * b.mean(x, axis=1, keepdims=True),
        ]

    @opTest([[m(1, 3, 4)], [m(7, 19) - 10.]])
    def testSoftsign(self, b, x):
        return [b.softsign(x)]

    @opTest([[m(2, 6)], [m(2, 9, 9) - 3.1]])
    def testSoftplus(self, b, x):
        return [b.softplus(x)]

    # TODO: Enable gradients again after we fix the Stripe bug
    @opTest([[m(10, 10)]], do_grads=False)
    def testSign(self, b, x):
        return [b.sign(x)]

    @opTest([[m(10, 10)]], skip_theano=True)
    def testSigmoid(self, b, x):
        return [b.sigmoid(x)]

    @opTest([
        [np.array([[0, 1], [1, 0]]), m(2, 2)],
        [np.array([[0.3, 0.7], [0.1, 0.9]]), m(2, 2)],
        [np.array([[0, 0.7], [1, .3]]), m(2, 2)],
    ],
            skip_theano=True,
            atol=1e-7)
    def testBinaryCrossentropy(self, b, x, y):
        return [
            b.binary_crossentropy(x, y, from_logits=True),
            b.binary_crossentropy(x, y, from_logits=False),
            b.binary_crossentropy(x, b.sigmoid(y))
        ]

    @opTest([[np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), (m(3, 3) + 3) / 15.0],
             [np.array([0, 0, 1, 0, 0, 0]), (m(6) + 7) / 11.0]])
    def testCategoricalCrossentropy(self, b, x, y):
        return [b.categorical_crossentropy(x, y)]

    @opTest([[np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]]), (m(3, 3) + 3)]])
    def testSoftCat(self, b, x, y):
        return [b.categorical_crossentropy(x, b.softmax(y))]

    @unittest.skip(
        "Doesn't need to agree b/c what we do with garbage input is implementation detail")
    @opTest([[(m(2, 2) + 3) / 10.0, np.array([[0., 0.], [1., 2.]])]])
    def testCategoricalCrossentropyGarbageIn(self, b, x, y):
        return [b.categorical_crossentropy(x, y)]

    #TODO: Merge with general cat xentropy if we can resolve the TF problems
    @opTest([[np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), (m(3, 3) + 3) / 15.0]],
            atol=1e-7,
            skip_tensorflow=True)
    def testCategoricalCrossentropyLogits(self, b, x, y):
        return [b.categorical_crossentropy(x, y, from_logits=True)]

    @opTest([[m(3, 3, 10)]], skip_theano=True, tol=0.01)
    def testSparseCategoricalCrossentropy(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(np.array([[7, 8, 5], [9, 3, 8], [0, 7, 6]]))
        return [
            b.sparse_categorical_crossentropy(sbest, smax),
            b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)
        ]

    @opTest([[m(1, 3, 10)]], skip_theano=True, tol=0.01)
    def testSparseCategoricalCrossentropyUnbalanced(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(np.array([[7, 8, 5]]))
        return [
            b.sparse_categorical_crossentropy(sbest, smax),
            b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)
        ]

    @opTest([[m(3, 10)]], skip_theano=True, tol=0.001)
    def testSparseCategoricalCrossentropyShort(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(np.array([7, 8, 5]))
        return [
            b.sparse_categorical_crossentropy(sbest, smax),
            b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)
        ]

    @opTest([[m(3, 3, 2, 10)]], skip_theano=True, tol=0.01)
    def testSparseCategoricalCrossentropyLong(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(
            np.array([[[1, 7], [2, 8], [9, 5]], [[4, 9], [0, 3], [9, 8]], [[0, 0], [6, 7], [6,
                                                                                            6]]]))
        return [
            b.sparse_categorical_crossentropy(sbest, smax),
            b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)
        ]

    @opTest([[m(3, 3, 2, 1, 10)]], skip_theano=True, tol=0.01)
    def testSparseCategoricalCrossentropyXLong(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(
            np.array([[[[1], [7]], [[2], [8]], [[9], [5]]], [[[4], [9]], [[0], [3]], [[9], [8]]],
                      [[[0], [0]], [[6], [7]], [[6], [6]]]]))
        return [
            b.sparse_categorical_crossentropy(sbest, smax),
            b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)
        ]

    @compareForwardExact(skip_theano=True)
    def testOneHot(self, b):
        A = b.variable(np.array([[0, 1, 2], [2, 4, 0], [0, 2, 7]]), dtype='int32')
        return b.one_hot(A, 20)

    @opTest([[m(20)], [m(7, 3)]])
    def testExp(self, b, x):
        return [b.exp(x)]

    @opTest([[m(20)], [m(2, 2, 2)]])
    def testPow(self, b, x):
        return [b.pow(x, 5)]

    @opTest([[m(20) + 3], [m(10, 3)]])
    def testLog(self, b, x):
        return [b.log(x)]

    @opTest([
        [m(3, 3)],
        [m(3, 3), None, True],
        [m(2, 3, 4, 5), [1, 3]],
        [m(3, 4, 5), -1],
        [m(2, 3, 4), 0],
    ])
    def testLogSumExp(self, b, x, ax=None, kd=False):
        return [b.logsumexp(x, axis=ax, keepdims=kd)]

    @opTest([[m(10)], [m(2, 2, 2, 3)]], 1e-2)
    def testTanh(self, b, x):
        return [b.tanh(x)]

    @compareForwardClose(.1)
    def testRandomUniformMean(self, b):
        rand = b.random_uniform((1000, 1000))
        return b.mean(rand)

    @compareForwardClose(.1)
    def testRandomUniformDev(self, b):
        rand = b.random_uniform((1000, 1000))
        mean = b.mean(rand)
        diffs = rand - mean
        return b.mean(b.square(diffs))

    @compareForwardClose(.1)
    def testRandomUniformVariableMean(self, b):
        rand = b.random_uniform_variable((1000, 1000), low=0.0, high=1.0)
        return b.mean(rand)

    @compareForwardClose(.1)
    def testRandomUniformVariableDev(self, b):
        rand = b.random_uniform_variable((1000, 1000), low=0.0, high=1.0)
        mean = b.mean(rand)
        diffs = rand - mean
        return b.mean(b.square(diffs))

    @compareForwardClose(.1)
    def testRandomNormalMean(self, b):
        rand = b.random_normal((1000, 1000), mean=42.0, stddev=0.1)
        return b.mean(rand)

    @compareForwardClose(.1)
    def testRandomNormalDev(self, b):
        rand = b.random_normal((1000, 1000), mean=42.0, stddev=0.1)
        mean = b.mean(rand)
        diffs = rand - mean
        return b.mean(b.square(diffs))

    @compareMultiple([
        [[100, 100], 5, 2],
        [[50, 50], 5, 2, 'float16'],
    ])
    @compareForwardClose(epsilon=0.2)
    def testRandomNormalVariableMean(self, b, *args):
        return b.mean(b.random_normal_variable(*args))

    @compareForwardClose(.1)
    def testTruncatedNormalMean(self, b):
        rand = b.truncated_normal((1000, 1000), mean=42.0, stddev=0.1)
        return b.mean(rand)

    @compareForwardClose(.1, skip_theano=True)
    def testTruncatedNormalDev(self, b):
        rand = b.truncated_normal((1000, 1000), mean=42.0, stddev=0.1)
        mean = b.mean(rand)
        diffs = rand - mean
        return b.mean(b.square(diffs))

    @opTest([
        _conv_inp(IN=1, IC=16, OC=16, IS=[6, 5], KS=[3, 3]),
    ], 1e-04, skip_theano=True)
    def testWinograd(self, b, im, km, df):
        return [
            b.conv2d(im, km, padding='same') if b == pkb else b.conv2d(im, km, padding='same'),
        ]

    # Asymmetric stride examples not included for separable convolutions b/c they
    # aren't implemented in tensorflow (and theano doesn't do separable convs)
    @unittest.skip("Broken on some drivers. TODO: periodically check if this is resolved")
    @opTest(
        [
            _separable_conv_inp(IN=1, IC=2, OC=6, CM=3, IS=[8, 8], KS=[3, 3]),
            _separable_conv_inp(IN=4, IC=3, OC=6, CM=2, IS=[7, 9], KS=[3, 4]),
            _separable_conv_inp(IN=1, IC=2, OC=5, CM=1, IS=[10, 12], KS=[2, 5]),
            _separable_conv_inp(
                IN=2, IC=4, OC=8, CM=2, IS=[12, 12], KS=[3, 3], data_format='channels_first'),
        ],
        atol=1e-5,  # TF separable conv math is really loose, and ends up with
        # values like 2.59e-6 where a 0 should be.
        skip_theano=True)
    def testSeparableConv2d(self, b, im, dkm, pkm, df):
        return [
            b.separable_conv2d(im, dkm, pkm, padding='valid', strides=(2, 2), data_format=df),
            b.separable_conv2d(im, dkm, pkm, padding='valid', strides=(1, 1), data_format=df),
            b.separable_conv2d(im, dkm, pkm, padding='same', strides=(3, 3), data_format=df),
            b.separable_conv2d(im, dkm, pkm, padding='valid', dilation_rate=(2, 1),
                               data_format=df),
        ]

    @opTest([
        [m(1, 4, 4, 3), m(3, 3, 3, 3), 'channels_last'],
        [m(4, 8, 6, 3), m(3, 4, 3, 6), 'channels_last'],
        [m(1, 7, 11, 2), m(2, 3, 2, 5), 'channels_last'],
        [m(1, 5, 5, 1), m(3, 3, 1, 1), 'channels_last'],
        [m(1, 1, 5, 7), m(3, 3, 1, 1), 'channels_first'],
    ],
            do_grads=False,
            verbose=False,
            skip_theano=True)
    def testDepthwiseConv2d(self, b, im, km, df):
        # A number of tests are turned off due to bizarre TF behavior that seems like it must be bugged
        # (e.g. increasing spatial size of input decreasing spatial size of output)
        return [
            b.depthwise_conv2d(im,
                               km,
                               strides=(1, 1),
                               padding='same',
                               data_format=df,
                               dilation_rate=(1, 1)),
            #b.depthwise_conv2d(im, km, strides=(3, 2), padding='same', data_format=df, dilation_rate=(1, 1)),  # TF unhappy with mixed strides
            b.depthwise_conv2d(im,
                               km,
                               strides=(2, 2),
                               padding='same',
                               data_format=df,
                               dilation_rate=(1, 1)),
            b.depthwise_conv2d(im,
                               km,
                               strides=(1, 1),
                               padding='same',
                               data_format=df,
                               dilation_rate=(2, 3)),
            #b.depthwise_conv2d(im, km, strides=(2, 2), padding='same', data_format=df, dilation_rate=(2, 3)),  # TF unhappy with strides + dilation
            #b.depthwise_conv2d(im, km, strides=(2, 2), padding='same', data_format=df, dilation_rate=(2, 2)),  # TF unhappy with strides + dilation
            #b.depthwise_conv2d(im, km, strides=(2, 2), padding='valid', data_format=df, dilation_rate=(2, 2)),  # TF unhappy with strides + dilation
        ]

    @opTest(
        [
            _conv_inp(IN=1, IC=3, OC=1, IS=[8], KS=[2], data_format='channels_last'),
            _conv_inp(IN=2, IC=1, OC=4, IS=[8], KS=[3], data_format='channels_last')
        ],
        # Tensorflow doesn't support 1d convos in this order yet
        #_conv_inp(IN=4, IC=1, OC=5, IS=[9], KS=[4], data_format='channels_first')],
        1e-04,
        skip_theano=True)
    def testConv1d(self, b, im, km, df):
        return [
            b.conv1d(im, km, padding='same', data_format=df),
            b.conv1d(im, km, padding='valid', data_format=df),
            b.conv1d(im, km, padding='valid', strides=(2), data_format=df),
            b.conv1d(im, km, padding='valid', dilation_rate=3, data_format=df),
            b.conv1d(im, km, padding='same', dilation_rate=2, data_format=df),
            b.conv1d(im, km, padding='causal', dilation_rate=2, data_format=df),
        ]

    @opTest([
        _conv_inp(IN=2, IC=2, OC=4, IS=[4, 7], KS=[3, 3]),
        _conv_inp(IN=3, IC=3, OC=1, IS=[9, 8], KS=[2, 2], data_format='channels_last'),
        _conv_inp(IN=1, IC=1, OC=3, IS=[5, 4], KS=[3, 3], data_format='channels_first'),
        _conv_inp(IN=2, IC=4, OC=2, IS=[5, 5], KS=[2, 2], data_format='channels_first'),
    ],
            1e-04,
            skip_theano=True)
    def testConv2d(self, b, im, km, df):
        return [
            b.conv2d(im, km, padding='same', data_format=df),
            b.conv2d(im, km, padding='valid', data_format=df),
            b.conv2d(im, km, padding='same', strides=(2, 2), data_format=df),
            b.conv2d(im, km, padding='valid', strides=(3, 1), data_format=df),
            b.conv2d(im, km, padding='same', dilation_rate=(2, 2), data_format=df),
        ]

    @opTest(
        [[m(1, 1, 3, 1),
          m(1, 4, 1, 1), (1, 1, 9, 1), (1, 4), 'same', 'channels_last', (1, 1)],
         [m(1, 3, 3, 1),
          m(3, 3, 1, 1), (1, 5, 5, 1), (2, 2), 'same', 'channels_last', (1, 1)],
         [m(1, 2, 2, 1),
          m(3, 3, 1, 1), (1, 5, 5, 1), (2, 2), 'valid', 'channels_last', (1, 1)],
         [m(1, 5, 3, 7),
          m(5, 5, 4, 7), (1, 9, 10, 4), (1, 2), 'valid', 'channels_last', (1, 1)],
         [m(1, 5, 3, 7),
          m(5, 5, 4, 7), (1, 9, 9, 4), (1, 2), 'valid', 'channels_last', (1, 1)],
         [m(4, 8, 5, 5),
          m(3, 3, 2, 8), (4, 2, 9, 9), (2, 2), 'same', 'channels_first', (1, 1)],
         [m(4, 3, 5, 8),
          m(3, 3, 2, 8), (4, 9, 9, 2), (3, 2), 'same', 'channels_last', (1, 1)],
         [m(1, 1, 6, 1),
          m(7, 1, 1, 1), (1, 1, 22, 1), (4, 1), 'same', 'channels_first', (1, 1)],
         [m(1, 1, 4, 1),
          m(7, 1, 1, 1), (1, 1, 22, 1), (4, 1), 'valid', 'channels_first', (1, 1)],
         [m(1, 8, 9, 3),
          m(3, 2, 3, 3), (1, 8, 9, 3), (1, 1), 'same', 'channels_last', (2, 2)]],
        verbose=False,
        skip_theano=True,
    )
    def testConv2dTranspose(self, b, x, k, os, st, pd, df, dr):
        return [
            b.conv2d_transpose(x, k, os, strides=st, padding=pd, data_format=df, dilation_rate=dr)
        ]

    @opTest([_conv_inp(IN=1, IC=1, OC=1, IS=[1, 6], KS=[1, 1], data_format='channels_last')],
            1e-04,
            skip_theano=True)
    def testConv2dSpecial(self, b, im, km, df):
        '''A simplified example highlighting a bug in Keras 2.0.8 TF

        If we're not concerned with Keras 2.0.8 we probably don't need to retain this.'''
        return [b.conv2d(im, km, padding='same', strides=(2, 3), data_format=df)]

    @opTest([
        _conv_inp(IN=3, IC=1, OC=3, IS=[4, 7, 5], KS=[3, 3, 3]),
        _conv_inp(IN=3, IC=4, OC=2, IS=[3, 6, 3], KS=[2, 1, 2], data_format='channels_last'),
        _conv_inp(IN=2, IC=3, OC=1, IS=[5, 5, 3], KS=[3, 2, 2], data_format='channels_first'),
    ],
            1e-04,
            skip_theano=True)
    def testConv3d(self, b, im, km, df):
        return [
            b.conv3d(im, km, padding='same', data_format=df),
            b.conv3d(im, km, padding='same', strides=(2, 3, 3), data_format=df),
            b.conv3d(im, km, padding='valid', strides=(2, 1, 2), data_format=df),
            b.conv3d(im, km, padding='valid', dilation_rate=(1, 3, 2), data_format=df),
        ]

    @opTest(
        [
            [
                m(1, 3, 3, 3, 1),
                m(3, 3, 4, 2, 1), (1, 5, 7, 9, 2), (2, 3, 4), 'same', 'channels_last'
            ],
            [
                m(1, 1, 2, 8, 4),
                m(4, 1, 2, 1, 1), (1, 1, 7, 8, 14), (3, 1, 4), 'valid', 'channels_first'
            ],
        ],
        verbose=False,
        skip_theano=True,
    )
    def testConv3dTranspose(self, b, x, k, os, st, pd, df):
        return [b.conv3d_transpose(x, k, os, strides=st, padding=pd, data_format=df)]

    @opTest([[m(1, 4, 4, 1)], [m(1, 7, 5, 1)], [m(2, 11, 13, 3)]], skip_theano=True)
    def testAvgPool(self, b, x):
        return [
            b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='avg'),
            b.pool2d(x, (3, 3), strides=(1, 1), pool_mode='avg', padding='same'),
            b.pool2d(x, (3, 4), strides=(2, 3), pool_mode='avg', padding='valid'),
        ]

    @opTest([
        [m(1, 4, 4, 1) - 33.],
        [m(1, 9, 9, 1)],
        [m(1, 8, 10, 1)],
        [m(2, 9, 11, 3)],
    ],
            skip_theano=True)
    def testMaxPool(self, b, x):
        return [
            b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='max'),
            b.pool2d(x, (3, 3), strides=(1, 1), pool_mode='max'),
            b.pool2d(x, (3, 3), strides=(2, 2), pool_mode='max'),
            b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='max', padding='same'),
            b.pool2d(x, (3, 3), strides=(2, 2), pool_mode='max', padding='same')
        ]

    @opTest([
        [m(3, 3, 4, 5, 2)],
        [m(1, 5, 4, 7, 1)],
    ], skip_theano=True)
    def testPool3D(self, b, x):
        return [
            b.pool3d(x, (1, 2, 2), strides=(2, 1, 2), pool_mode='max', padding='valid'),
            b.pool3d(x, (2, 2, 3), strides=(2, 3, 1), pool_mode='avg', padding='same'),
        ]

    @opTest([
        [m(1, 1, 60), (60,)],
        [m(4, 3, 70, 2), (14, 10, 6, 2)],
        [m(7, 3, 2, 4), (-1,)],
        [m(4, 4), (-1,)],
    ])
    def testReshape(self, b, x, s):
        return [b.reshape(x, s)]

    def testReshapeMatchDim(self):
        a = pkb.variable(m(1, 1, 60))
        output = pkb.reshape(a, (2, 0, 30))
        self.assertEqual(pkb.int_shape(output), (2, 1, 30))
        output = pkb.reshape(a, (2, 0, -1))
        self.assertEqual(pkb.int_shape(output), (2, 1, 30))
        output = pkb.reshape(a, (0, 0, 0))
        self.assertEqual(pkb.int_shape(output), (1, 1, 60))
        # expect runtime exceptions
        with self.assertRaises(plaidml.Error) as cm:
            pkb.reshape(a, (-1, -1))
        self.assertTrue("at most one dimension's size may be inferred" in str(cm.exception))
        with self.assertRaises(plaidml.Error) as cm:
            pkb.reshape(a, (1, 1, 1, 0))
        self.assertTrue(
            "matching dimension requested at 4 from 3-dimensional tensor" in str(cm.exception))

    @opTest([
        [m(1, 1, 60), (60,)],
        [m(4, 3, 70, 2), (14, 10, 6, 2)],
        [m(7, 3, 2, 4), (-1,)],
        [m(4, 4), (-1,)],
    ],
            input_shapes=[((1, 1, None),), ((4, None, 70, 2),), ((None, 3, 2, 4),),
                          ((None, None),)],
            verbose=False)
    def testReshapeSymbolic(self, b, x, s):
        return [b.reshape(x, s)]

    @opTest([
        [m(3)],
        #[m()],  # TODO: Need to support empty shapes for placeholders
        [m(4, 7)],
        [m(6, 3, 2, 4, 7, 1, 5)],
    ])
    def testTranspose(self, b, x):
        return [b.transpose(x)]

    @opTest([
        [m(1, 1, 60), (60,)],
        [m(4, 3, 70, 2), (14, 10, 6, 2)],
        [m(7, 3, 2, 4), (-1,)],
        [m(4, 4), (-1,)],
    ])
    def testTransposeReshape(self, b, x, s):
        return [b.reshape(b.transpose(x), s)]

    @opTest([
        [m(3), None],
        #[m(), tuple()],  # TODO: Need to support empty shapes for placeholders
        [m(4, 7), (1, 0)],
        [m(3, 6, 2, 4, 7, 1, 5), (5, 2, 0, 3, 6, 1, 4)],
    ])
    def testPermuteDimensions(self, b, x, s):
        return [b.permute_dimensions(x, pattern=s)]

    @opTest([
        [m(4, 2, 1, 3, 2), 2],
        [m(5, 3, 2, 1), -1],
    ])
    def testSqueeze(self, b, x, ax):
        return [b.squeeze(x, ax)]

    @opTest([[m(10, 10), n(10, 10), 0], [m(10, 10), n(10, 10), -1]])
    def testStack(self, b, *args):
        return [b.stack(args[:-1])]

    @compareForwardExact()
    def testZeros(self, b):
        a = b.zeros(shape=(10,))
        return a

    @compareForwardExact()
    def testOnes(self, b):
        a = b.ones(shape=(10,))
        return a

    @compareForwardExact()
    def testConstant(self, b):
        a = b.constant(5, shape=(10,))
        return a

    def testIsPlaceholder(self):
        x = pkb.placeholder((4, 3))
        self.assertTrue(pkb.is_placeholder(x))
        y = pkb.variable(m(2, 2))
        self.assertFalse(pkb.is_placeholder(y))
        z = pkb.exp(x)
        self.assertFalse(pkb.is_placeholder(z))

    # Note: we skip tensorflow since init_global must be called in the middle of this function
    # for correct semantics, and Theano is sufficient.
    @compareForwardExact(skip_tensorflow=True, skip_theano=False)
    def testUpdate(self, b):
        a = b.variable(m(10, 10))
        a2 = a * a
        up = b.update(a, a2)
        f = b.function([], [], updates=[up])
        f([])
        f([])
        return a

    @unittest.expectedFailure
    @compareForwardExact()
    def testRandomChanges(self, b):
        a = b.random_uniform((3, 3))
        f = b.function([], [a])
        out1 = np.copy(f([])[0])
        logger.debug('out1:\n{}'.format(out1))
        out2 = f([])[0]
        logger.debug('out2:\n{}'.format(out2))
        diff = np.abs(out1 - out2).max()
        logger.debug('diff:\n{}'.format(diff))
        self.assertLess(diff, .01)
        return b.constant(0)

    # Note: This test assumes that our update code matches Theano's, and
    # that testing the second component of the returned update tuple is
    # sufficient. It may be worthwhile to make this test more resilient
    # to refactoring and make it test that the update portion is working
    # as expected.
    @compareForwardClose(skip_tensorflow=True, skip_theano=False)
    def testMovingAverageUpdate(self, b):
        return b.moving_average_update(b.variable(m(5, 4, 9, 3, 2)), b.variable(n(5, 4, 9, 3, 2)),
                                       0.95)[1]

    @compareForwardClose(skip_tensorflow=True, skip_theano=False, atol=1e-6)
    def testBatchNormAndUpdate(self, b):
        b.set_learning_phase(1)
        x = b.variable(n(4, 7))
        moving_mean = b.variable(m(4, 4))
        moving_var = b.variable(m(4, 4))
        beta = b.zeros([4, 1])
        gamma = b.ones([4, 1])
        normed, mean, var = b.normalize_batch_in_training(x, gamma, beta, reduction_axes=[1])
        mean_update = b.moving_average_update(moving_mean, mean, 0.01)
        var_update = b.moving_average_update(moving_var, var, 0.01)
        f = b.function([], [], updates=[mean_update, var_update])
        f([])
        return moving_var

    @opTest([[
        m(2, 3, 5),
        m(2, 3, 1) + 3,
        m(2, 3, 1) + 4,
    ]], atol=1e-7)
    def testNormalizeBatchInTrainingSimple(self, b, x, mov_avg, mov_var):
        return [(b.normalize_batch_in_training(x, mov_avg, mov_var, [2]))[0]]

    @opTest([
        [n(2, 3), np.array([3., 4., .7]),
         np.array([1.44, .99, .98])],
        [n(3, 2), None, None],
        [n(1, 3), None, np.array([2., 3., .55])],
        [n(1, 2), np.array([-2., 0.]), None],
    ],
            skip_theano=True,
            skip_tensorflow=True)
    def testNormalizeBatchInTraining(self, b, x, beta, gamma):
        return [b.normalize_batch_in_training(x, gamma, beta, [1])[0]]

    @compareForwardClose()
    def testNormalizeBatchInTrainingWeirdAxis(self, b):
        return b.normalize_batch_in_training(
            b.variable(n(5, 4, 7, 3)),
            b.constant(0.8, shape=(5, 1, 7, 3)),
            b.constant(-5, shape=(5, 1, 7, 3)),
            [1],
        )[1]

    @compareForwardClose()
    @unittest.skip("Behavior mismatch with TF on bizarre shapes")
    def testNormalizeBatchInTrainingWeirdMultiAxis(self, b):
        # These shapes are pretty much nonsense, but TF figures it out (via reshape) so we should too
        return b.normalize_batch_in_training(
            b.variable(n(2, 3, 5, 7)),
            b.constant(11, shape=(3, 1, 1, 1, 1, 1, 1, 1)),
            b.constant(0, shape=(3, 1)),
            [0, 2, 3],
        )[2]

    @compareForwardClose()
    def testNormalizeBatchInTrainingMultiAxis(self, b):
        return b.normalize_batch_in_training(
            b.variable(n(2, 3, 5, 7, 11)),
            b.constant(11, shape=(1, 3, 1, 1, 11)),
            b.constant(0, shape=(1, 3, 1, 1, 11)),
            [0, 2, 3],
        )[2]

    @opTest([[
        n(4, 3),
        np.array([0.0, 0.1, 0.1]),
        np.array([100., 101., 50.]),
        np.array([3., 4., .7]),
        np.array([1.44, .99, .98])
    ]],
            do_grads=False)  # TODO: fix gradients
    def testBatchNormalization(self, b, x, mean, var, beta, gamma):
        return [b.batch_normalization(x, mean, var, beta, gamma)]

    @opTest([[np.array([100])]], skip_theano=True, do_grads=False)  # TODO: fix gradients
    def testBatchNormalizationVar(self, b, var):
        return [
            b.batch_normalization(
                b.variable(n(1, 1, 2)),
                b.variable(np.array([15])),
                var,
                None,
                None,
            ),
            b.batch_normalization(
                b.variable(n(2, 1, 1)),
                b.variable(np.array([15])),
                var,
                None,
                None,
            )
        ]

    @opTest([[np.array([15])]], skip_theano=True, do_grads=False)  # TODO: fix gradients
    def testBatchNormalizationMean(self, b, mean):
        return [
            b.batch_normalization(
                b.variable(n(3, 4, 5)),
                mean,
                b.variable(np.array([100])),
                None,
                None,
            )
        ]

    @compareForwardClose()
    def testBatchNormalizationOneElement(self, b):
        x = b.variable(n(1, 4, 5))
        return b.batch_normalization(
            b.variable(n(1, 4, 5)),
            b.variable(np.array([15])),
            b.variable(np.array([100])),
            b.variable(np.array([3])),
            b.variable(np.array([1.44])),
        )

    @compareForwardClose()
    def testBatchNormalizationNoBeta(self, b):
        return b.batch_normalization(
            b.variable(n(3, 4, 5)),
            b.variable(np.array([15])),
            b.variable(np.array([100])),
            None,
            b.variable(np.array([1.44])),
        )

    @compareForwardClose()
    def testBatchNormalizationNoGamma(self, b):
        return b.batch_normalization(
            b.variable(n(3, 4, 5)),
            b.variable(np.array([15])),
            b.variable(np.array([100])),
            b.variable(np.array([3])),
            None,
        )

    @opTest([
        [m(4, 6)],
        [m(4, 3, 5)],
        [m(3, 7), None, True],
        [m(2, 5, 4, 7, 3), 1],
    ], atol=1e-7)
    def testVarSimple(self, b, x, ax=None, kd=False):
        return [b.var(x, axis=ax, keepdims=kd)]

    @opTest([
        [m(3, 4)],
        [m(1, 5, 2)],
        [m(7, 2), None, True],
        [m(2, 1, 5, 7, 3), 4],
    ], atol=1e-7)
    def testStd(self, b, x, ax=None, kd=False):
        return [b.std(x, axis=ax, keepdims=kd)]

    @opTest([[m(3, 3)]])
    def testSelfMult(self, b, x):
        A = x
        return [b.dot(A, A)]

    @opTest([
        [m(3, 4), 0],
        [m(1, 3, 2, 4), [0, 2]],
        [m(1, 2, 2, 2), 3],
    ])
    def testReverse(self, b, x, ax):
        return [b.reverse(x, ax)]

    # Test vs. Theano not TF b/c TF seems to be doing something weird (perhaps
    # returning a pre-sparse-to-dense-conversion version?) with the gradient
    # and it doesn't match Theano & us.
    @opTest(
        [[np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]])],
         [
             np.array([[[3., 2., 4.], [1., 0., -1.], [1.4, 2.5, 3.4], [2.4, 3.6, 4.4]],
                       [[-3., 1.1, 4.1], [3.2, -0.4, -4.], [-1.5, 2.2, 3.99], [2.114, -3.2, -4.]],
                       [[4.1, -1.2, .1234], [4.2, .943, 9.21], [43.4, 47.1, 22.],
                        [0.0, -3434., -2.4]]])
         ]],
        skip_theano=False,
        skip_tensorflow=True)
    def testGather(self, b, v):
        I = b.variable(np.array([0, 2, 1, 0], dtype='int32'), dtype='int32')
        return [b.gather(v, I)]

    @compareForwardClose()
    def testGather2(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[2, 1], [0, 1], [1, 0], [2, 1], [0, 0]], dtype='int32'),
                       dtype='int32')
        return b.gather(V, I)

    @compareForwardClose()
    def testGatherLong(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]], dtype='int32'),
                       dtype='int32')
        return b.gather(V, I)

    @compareForwardClose()
    def testGatherWithA1Dim(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[0], [1], [0]], dtype='int32'), dtype='int32')
        return b.gather(V, I)

    @compareForwardClose()
    def testGatherLong2(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[[0, 1, 1, 0], [1, 0, 0, 1]], [[1, 0, 1, 0], [0, 0, 1, 1]]],
                                dtype='int32'),
                       dtype='int32')
        return b.gather(V, I)

    @opTest([[m(2, 3)]])
    def testRepeat(self, b, x):
        return [b.repeat(x, 4)]

    @opTest([[m(3, 2, 4, 5, 6)]])
    def testRepeatElements(self, b, x):
        return [b.repeat_elements(x, 3, 4)]

    @opTest([
        [m(4, 6, 9, 3), 3, 1, 'channels_last'],
        [m(2, 3, 12, 12), 2, 3, 'channels_first'],
    ])
    def testResizeImages(self, b, x, h, w, df):
        return [b.resize_images(x, h, w, df)]

    @opTest([
        [m(3, 2, 5, 11), 3, 1, 'channels_last'],
        [m(1, 3, 7, 5), 2, 3, 'channels_first'],
        [m(1, 1, 2, 3), 3, 4, 'channels_first'],
    ])
    def testResizeImagesBilinear(self, b, x, h, w, df):
        # Tested without ends b/c of different padding behavior from TF
        resized = b.resize_images(x, h, w, df, interpolation='bilinear')
        shp = b.get_variable_shape(x)
        if df == 'channels_first':
            return [resized[:, :, :h * shp[2] - h, :w * shp[3] - w]]
        elif df == 'channels_last':
            return [resized[:, :h * shp[1] - h, :w * shp[2] - w, :]]
        else:
            raise ValueError('Bad data format requested for test')

    @opTest([
        [m(4, 6, 5, 2, 3), 3, 1, 2, 'channels_last'],
        [m(1, 3, 2, 5, 4), 2, 3, 4, 'channels_first'],
    ])
    def testResizeVolumes(self, b, x, d, h, w, df):
        return [b.resize_volumes(x, d, h, w, df)]

    @opTest([[m(3, 5)]])
    def testL2Normalize(self, b, x):
        return [b.l2_normalize(x, axis=1)]

    @opTest([
        [m(2, 3, 4), m(2, 3, 3), m(2, 3, 1)],
        [m(3, 2, 4), m(3, 1, 4), m(3, 3, 4), 1],
    ])
    def testConcatenate(self, b, x, y, z, ax=-1):
        return [b.concatenate([x, y, z], axis=ax)]

    @opTest([
        [m(2, 3, 4), m(2, 3, 3), m(2, 3, 1)],
        [m(3, 2, 4), m(3, 1, 4), m(3, 3, 4), 1],
    ],
            input_shapes=[((2, 3, None), (2, 3, None), (2, 3, None)),
                          ((3, None, 4), (3, None, 4), (3, None, 4))],
            verbose=False)
    def testConcatenateSymbolic(self, b, x, y, z, ax=-1):
        return [b.concatenate([x, y, z], axis=ax)]

    @compareForwardExact()
    def testZerosLike(self, b):
        A = b.variable(m(3, 2, 4))
        return b.zeros_like(A)

    @compareForwardExact()
    def testOnesLike(self, b):
        A = b.variable(m(3, 2, 4))
        return b.ones_like(A)

    @opTest([
        [m(3, 2, 4), 1],
        [m(2, 5, 3)],
    ])
    def testExpandDims(self, b, x, ax=-1):
        return [b.expand_dims(x, ax)]

    @opTest([
        [m(3, 2, 4), [1, 7, 3]],
        [m(2, 3, 1), [2, 1, 4]],
    ])
    def testTile(self, b, x, n):
        return [b.tile(x, n)]

    @opTest([[m(34)]])
    def testSliceBasic(self, b, x):
        return [b.exp(x[2:30]), b.log(x[:5]), b.tanh(x[-4:]), b.sqrt(x[-1])]

    @opTest([
        [m(4, 3, 3, 2, 5)],
        [m(5, 4, 4, 3, 6)],
        [m(6, 5, 5, 4, 7)],
    ])
    def testSliceMessy(self, b, x):
        return [
            x[-1::-3, :2:2, -3:-2, ::-1, -1:-6:-2],
            x[2::-1, 0:2:1, 1, :1, 1:5:2],
        ]

    @opTest([[m(2, 3, 2)]])
    def testSliceShort(self, b, x):
        return [x[1]]

    @opTest([[m(2, 3, 4, 5)], [m(2, 1, 2)]])
    def testSliceEllipsis(self, b, x):
        return [x[..., 1], x[-1, ..., 0, ::-1], x[...]]

    def testConvParameterRankExceptions(self):
        A = pkb.variable(m(2, 3, 1))
        B = pkb.variable(m(1, 2, 1))
        C = pkb.variable(m(2, 2, 2, 1))
        with self.assertRaises(plaidml.Error):
            pkb.conv(A, C)
        with self.assertRaises(plaidml.Error):
            pkb.conv(A, B, strides=(2, 3))
        with self.assertRaises(plaidml.Error):
            pkb.conv(A, B, dilation_rate=(1, 1))

    @compareForwardExact()
    def testCastToInt(self, b):
        A = b.variable(m(3, 2, 4))
        return b.cast(A, dtype='int16')

    @compareForwardExact()
    def testCastToFloat(self, b):
        A = b.variable(m(3, 2, 4))
        A2 = b.cast(A, dtype='int32')
        return b.cast(A, dtype='float32')

    @compareForwardExact()
    def testCastToUInt(self, b):
        A = b.variable(m(3, 2, 4))
        return b.cast(A + 2, dtype='uint8')

    # Th/TF disagree w/ us about negative zeros and I think the even/odd rounding
    # direction for numbers ending in *.5, so we'll settle for a close match.
    @compareForwardClose()
    def testRound(self, b):
        vals = np.array([[1.7, 0.8, 1.5], [0.9, -0.3, -0.8], [0, 1.7, 0.6]])
        return b.round(b.variable(vals))

    def testCeil(self):
        npt.assert_allclose(pkb.ceil(pkb.variable(m(6, 2, 3))).eval(), np.ceil(m(6, 2, 3)))

    def testFloor(self):
        npt.assert_allclose(pkb.floor(pkb.variable(m(6, 2, 3))).eval(), np.floor(m(6, 2, 3)))

    @opTest([
        [m(3, 2, 4), n(3, 2, 4), 0],
        [m(2, 3), n(2, 3), 1],
    ])
    def testSwitch(self, b, e, t, c):
        c_tensor = b.variable(c)
        return [
            b.switch(c_tensor, e, t),
        ]

    @opTest([
        [m(1, 2, 4), 0],
        [m(2, 3), 1],
        [m(2, 3), 0],
        [m(2, 4, 5), 2],
    ])
    def testCumSum(self, b, t, a):
        return [b.cumsum(t, a)]

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardExact()
    def testCumProd(self, b):
        t = b.constant(m(5, 3))
        return b.cumprod(t, 1)

    @opTest([[m(4, 7, 1)], [m(2, 8, 3), (1, 3)], [m(2, 5, 7), (0, 0)]])
    def testTemporalPadding(self, b, x, p=(1, 1)):
        return [b.temporal_padding(x, padding=p)]

    @opTest([[m(4, 6, 7, 1)], [m(2, 4, 8, 3), ((1, 3), (0, 2))],
             [m(2, 5, 3, 1), ((4, 0), (0, 0)), 'channels_first']])
    def testSpatial2DPadding(self, b, x, p=((1, 1), (1, 1)), d=None):
        return [b.spatial_2d_padding(x, padding=p, data_format=d)]

    @opTest([[m(4, 5, 3, 1, 2)], [m(3, 8, 1, 5, 6), ((0, 2), (3, 5), (1, 4))],
             [m(4, 3, 7, 2, 2), ((0, 0), (1, 1), (0, 0)), 'channels_first']])
    def testSpatial3DPadding(self, b, x, p=((1, 1), (1, 1), (1, 1)), d=None):
        return [b.spatial_3d_padding(x, padding=p, data_format=d)]

    # Big rollup
    @opTest([[m(1000, 1000)]], do_grads=False)
    def testBigRollup(self, b, x):
        return [b.sum(x, axis=1)]

    # Resnet sized tests
    @opTest([[m(1, 224, 224, 3), m(7, 7, 3, 64)]], do_grads=False)
    def resnetLayer1(self, b, x, k):
        return [b.conv2d(x, k, strides=(2, 2), padding='valid')]

    @opTest([[m(1, 56, 56, 64), m(3, 3, 64, 64)]], do_grads=False)
    def resnetLayer2(self, b, x, k):
        c = b.conv2d(x, k, padding='same')
        o = b.relu(c)
        return [o]

    @opTest([[m(1, 56, 56, 256), m(3, 3, 256, 128)]], do_grads=False)
    def res3a_branch2a(self, b, x, k):
        return [b.conv2d(x, k, strides=(2, 2), padding='same')]

    @opTest([[m(1, 56, 56, 64), m(1, 1, 64, 64)]], do_grads=False)
    def res2a_branch2a(self, b, x, k):
        return [b.conv2d(x, k, strides=(1, 1), padding='same')]

    @opTest([[m(1, 56, 56, 64), m(3, 3, 64, 64)]], do_grads=False)
    def res2a_branch2b(self, b, x, k):
        return [b.conv2d(x, k, strides=(1, 1), padding='same')]

    @opTest([[m(1, 2048), m(2048, 1000)]], do_grads=False)
    def fc_1000(self, b, x, k):
        return [b.dot(x, k)]

    @opTest([[m(1024, 1024), m(1024, 1024)]], do_grads=False)
    def bigMatMul(self, b, A, B):
        return [b.dot(A, B)]

    @opTest([[m(1, 2048), m(2048, 1000)]], do_grads=False)
    def resnetDense(self, b, A, B):
        return [b.dot(A, B)]

    def testDupOutputs(self):

        def model(b):
            A = b.variable(m(10, 20), name='A')
            B = b.variable(m(20, 30), name='B')
            C = b.dot(A, B)
            fn = b.function([], [C, C, C])
            return fn([])

        tf_session = tensorflow.Session()
        tf.set_session(tf_session)
        tensorflow_result = model(tf)
        plaidml_result = model(pkb)

        for result in zip(plaidml_result, tensorflow_result):
            npt.assert_allclose(result[0],
                                result[1],
                                rtol=DEFAULT_TOL,
                                atol=DEFAULT_ATOL,
                                err_msg='x=plaidml, y=tensorflow')


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    if args.shard_count:
        print('Running shard {} of {}'.format(args.shard, args.shard_count))
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_num, test in enumerate(loader.loadTestsFromTestCase(TestBackendOps)):
            if test_num % args.shard_count == args.shard:
                print("test_num: {}, test: {}".format(test_num, test))
                suite.addTest(test)
        runner = unittest.TextTestRunner()
        exit(not runner.run(suite).wasSuccessful())
    else:
        unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
