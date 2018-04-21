# -*- coding:utf-8 -*-

"""
2階微分を用いたropの計算グラフの一般系を示すためのグラフ
Tensorboardでグラフを見ると、ropの計算の流れがわかる

ただし、name_scopeのネストが深くてわかりづらいので、tf.gradientsの定義をいじって、
gradientsにname_scopeをはらせないように変更してからグラフを見たほうが良い。
"""

import tensorflow as tf
import func

M, K, N = 3, 4, 5

with tf.Graph().as_default() as graph:
    x = tf.placeholder(shape=[M, 1], dtype=tf.float32, name="x")
    f = func.f(x)
    g = func.g(f)
    tf.nn.fused_batch_norm()

    w = tf.ones([4], name="w")
    v = tf.ones([M, 1], name="v")
    lop    = tf.gradients(g, x, grad_ys=w, name="lop")[0]
    lop_log = tf.identity(lop, "lop")
    rop = tf.gradients(lop, w, grad_ys=v, name="rop")[0]

    grad_x = tf.identity(rop, "rop")

    writer = tf.summary.FileWriter(logdir="./graph", graph=graph)
    writer.flush()

