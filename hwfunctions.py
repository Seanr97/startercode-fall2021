#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

def fun_inc(x):
    return x+1


def fun_factor(x):
    """ counts the number of factors of x including 1 and x """
    total = 0
    for i in range(1, x+1):
        if x % i == 0:
            total = total + 1
    time.sleep(0.25)
    return total


