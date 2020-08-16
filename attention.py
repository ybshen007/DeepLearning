# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
tf.set_random_seed(777)

# (batch_size, query_len, dim) = (2, 3, 4)
queries = tf.constant(list(range(24)), shape=(2, 3, 4), dtype=tf.float32)

# (batch_size, key_len, dim) = (2, 4, 4)
keys = tf.constant(list(range(32)), shape=(2, 4, 4), dtype =tf.float32)

'''
mask
query_masks = 
[[ True  True False]
 [ True  True  True]]
'''
ql = tf.constant([[2], [3]], dtype=tf.int32)
query_masks = tf.sequence_mask(ql, 3)
kl = tf.constant([[3], [4]], dtype=tf.int32)
key_masks = tf.sequence_mask(kl, 4)

# attention hidden dimension
num_units = 8
# attention output dimension: 输出序列中每一维大小
num_output_units = 8
# number of multi head
num_heads = 4
keep_prob = 0.7

# variables collections
attn_variables_collections = 'multihead_attention_hidden'
attn_outputs_collections = 'multihead_attention_output'
ffn_variables_collections = 'ffn_attention_hidden'
ffn_outputs_collections = 'ffn_attention_output'

# use layer normalizer and residual connection
residual_connection = False
attention_normalize = True

# Q.shape = (batch_size, seq_len, num_units)
Q = layers.fully_connected(queries,
                            num_units,
                            variables_collections=[attn_variables_collections],
                            outputs_collections=[attn_outputs_collections], scope="Q")  
# K.shape = (batch_size, seq_len, num_units)
K = layers.fully_connected(keys,
                            num_units,
                            variables_collections=[attn_variables_collections],
                            outputs_collections=[attn_outputs_collections], scope="K")  
# V.shape = (batch_size, seq_len, num_output_units)
V = layers.fully_connected(keys,
                            num_output_units,
                            variables_collections=[attn_variables_collections],
                            outputs_collections=[attn_outputs_collections], scope="V")  

def split_last_dimension_then_transpose(tensor, num_heads):
    # 这个方法比较trick, 如果Q/K/V的dim无法被num_heads整除, 会发生不符合预期的情况
    t_shape = tensor.get_shape().as_list()
    tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
    return tf.transpose(tensor, [0, 2, 1, 3])  

# Q_.shape = (batch_size, num_heads, seq_len, dim//num_heads)
Q_ = split_last_dimension_then_transpose(Q, num_heads)  
K_ = split_last_dimension_then_transpose(K, num_heads)  
V_ = split_last_dimension_then_transpose(V, num_heads)  


# Q, K求点积
# outputs.shape=(batch_size, num_heads, query_len, key_len)
# 最后两维度 (query_len, key_len) 组成 Q 与 K 的点击矩阵
outputs = tf.matmul(Q_, K_, transpose_b=True)

# Scale
outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

query_len = queries.get_shape().as_list()[1]
key_len = keys.get_shape().as_list()[1]

# key_masks.shape = (batch_size, num_heads, query_len, key_len)
# 和 outputs 对应
key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]), [1, num_heads, query_len, 1])

# key序列中长度不足key_len的部分, 用-2 ** 32 + 1填充, 使得后续计算softmax的时候为0
paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
outputs = tf.where(key_masks, outputs, paddings)

# outputs_sm.shape = (batch_size, num_heads, query_len, key_len)
# 最后两维度 (query, key_len) 组成 Q 与 K 的权重矩阵, 即 q_i 对 k_j 的注意力大小
outputs_sm = tf.nn.softmax(outputs) 

# query_masks.shape = (batch_size, num_heads, query_len, key_len)
# 和 outputs 对应
query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]), [1, num_heads, 1, key_len])

# query序列中长度不足query_len的部分, 用0填充去对应的权重矩阵
paddings = tf.fill(tf.shape(outputs_sm), tf.constant(0, dtype=tf.float32))
outputs = tf.where(query_masks, outputs_sm, paddings)

# Attention vector
att_vec = outputs

# Weighted sum
# outputs.shape = (batch_size, num_heads, query_len, dim // num_heads)
#               = (batch_size, num_heads, query_len, key_len) * (batch_size, num_heads, key_len, dim//num_heads)
outputs = tf.matmul(outputs, V_) 

def transpose_then_concat_last_two_dimenstion(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3]) 
    t_shape = tensor.get_shape().as_list()
    num_heads, dim = t_shape[-2:]
    return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

# Restore shape
# outputs.shape = (batch_size, seq_len, output_units)
# 恢复到和 V 一样的shape
outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

# Line projection
outputs = layers.fully_connected(outputs,
                                 num_output_units,
                                 variables_collections=[attn_variables_collections],
                                 outputs_collections=[attn_outputs_collections],
                                 scope='O')

# dropout
outputs = layers.dropout(outputs, keep_prob=keep_prob)

# Residual connection
# 如果要使用残差连接, 需保证 queries.get_shape().as_list()[-1] 和 V 的output_units相等 
if residual_connection:
    outputs += queries

# Normalize
if attention_normalize:
    outputs = layers.layer_norm(outputs)

# FFN
inputs = outputs
num_units = [inputs.get_shape().as_list()[-1] * 4, inputs.get_shape().as_list()[-1]]
outputs = layers.fully_connected(inputs,
                                num_units[0],
                                activation_fn=tf.nn.relu,
                                variables_collections=[ffn_variables_collections],
                                outputs_collections=[ffn_outputs_collections])

outputs = layers.fully_connected(outputs,
                                num_units[1],
                                activation_fn=None,
                                variables_collections=[ffn_variables_collections],
                                outputs_collections=[ffn_outputs_collections])

outputs = layers.dropout(outputs, keep_prob=keep_prob)
outputs = layers.layer_norm(outputs)
outputs += inputs

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs))