# -*- coding:utf-8 -*-

"""
Tensorflowから内実を隠蔽したグラフをつくるための関数定義群
"""

import tensorflow as tf
import numpy as np

def f(x):
    def f_imp(x):
        return x + 1

    y = py_func(f_imp,
                [x],
                [tf.float32],
                grad=f_grad,
                stateful=True, name="f")[0]
    return y

def f_grad(op, out_grad):
    def f_grad_imp(x):
        return x + 1

    out_grad = py_func(f_grad_imp,
                       list(op.inputs) + [out_grad],
                       [tf.float32],
                       grad=f_grad_grad,
                       stateful=True, name="f_grad")

    print ("len(f_grad_out) = {}".format(len(out_grad)))
    for g in out_grad:
        print (g.op.name)

    return out_grad

def f_grad_grad(op, grad):
    def f_grad_grad_imp(x):
        return x + 1

    out_grads = py_func(f_grad_grad_imp,
                   list(op.inputs)+[grad],
                   [tf.float32, tf.float32],
                   grad=None,
                   stateful=True, name="f_grad_grad")

    g1 = tf.identity(out_grads[0], "grad_in")
    g2 = tf.identity(out_grads[1], "grad_grad")
    out_grads = [g1, g2]


    print ("len(f_grad_grad_out) = {}".format(len(out_grads)))
    for g in out_grads:
        print (g.op.name)

    return out_grads


def g(x):
    def g_imp(x):
        return x + 1

    y = py_func(g_imp,
                [x],
                [tf.float32],
                grad=g_grad,
                stateful=True, name="g")[0]
    return y

def g_grad(op, out_grad):
    def g_grad_imp(x):
        return x + 1

    out_grad = py_func(g_grad_imp,
                       list(op.inputs) + [out_grad],
                       [tf.float32],
                       grad=g_grad_grad,
                       stateful=True, name="g_grad")

    print ("len(g_grad_out) = {}".format(len(out_grad)))
    for g in out_grad:
        print (g.op.name)

    return out_grad

def g_grad_grad(op, grad):
    def g_grad_grad_imp(x):
        return x + 1

    out_grads= py_func(g_grad_grad_imp,
                   list(op.inputs)+[grad],
                   [tf.float32, tf.float32],
                   grad=None,
                   stateful=True, name="g_grad_grad")

    g1 = tf.identity(out_grads[0], "grad_in")
    g2 = tf.identity(out_grads[1], "grad_grad")
    out_grads = [g1, g2]

    print ("len(g_grad_grad_out) = {}".format(len(out_grads)))
    for g in out_grads:
        print (g.op.name)

    return out_grads

def h(x):
    def h_imp(x):
        return x + 1

    y = py_func(h_imp,
                [x],
                [tf.float32],
                grad=g_grad,
                stateful=True, name="h")[0]
    return y

def h_grad(op, grad):
    def h_grad_imp(x):
        return x + 1

    grad = py_func(h_grad_imp,
                   list(op.inputs)+[grad],
                   [tf.float32],
                   grad=None,
                   stateful=True, name="h_grad")[0]
    return grad


# Define custom py_func which takes also a grad op as argument:
# https://stackoverflow.com/questions/41535347/how-gradient-passed-by-tf-py-func
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

