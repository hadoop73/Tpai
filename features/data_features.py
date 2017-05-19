#!/home/hadoop/env2.7/bin/python
# coding:utf-8


import argparse
import pandas as pd




"""
    特征产生思路：1) 先用 train21all.csv 产生所有数据的特征
                2) 再用 train21.csv 的数据与 train21all.csv 的特征进行 join 合并
                3) 对 valid test 同样可以通过步骤 2) 获得特征
"""


"""
1 获得 appID 的激活情况
"""




