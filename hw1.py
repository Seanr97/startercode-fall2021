#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys

from dask import delayed 
from dask.distributed import Client, wait, progress, as_completed

from hwfunctions import fun_factor, fun_inc



#A helper function that aids the main delayed_increment to running in smaller chuncks is directly below

@delayed
def helper_inc(start, end):
    return sum([fun_inc(_) for _ in range(start , end)])

def delayed_increment(c, start, end):
    n = 16                                                                                   #sets how chunks are to be similarly separated and since this function is simple more workers could get it done faster.
                                                                                             #These n values were tickered with a decent bit but found the values given throughout this assignment provided sufficicient results for workers
    dist = end - start                                                                                       
    incr_len =  dist // n                                                                    #There need to be a way to break the list down in smaller chunks for computation
                                                                                             #end - start will find the distance from the beginning to end and // n will find the amount that equally separates them for the workers
    
    start_place = [start + incr_len * _ for _ in range(n)]                                   #Finds the start of each chunk for the helper function
    end_place = [start + incr_len * (_ + 1) for _ in range(n)]                               #This line finds the end of each chunk by basically identifing the start of the next chuck
    end_place[-1] = end                                                                      #Due to how range operates and how floor division could affect chunks, we need to ensure the final value is included

    chunk_totals = [helper_inc(start_place[_], end_place[_]) for _ in range(n)]              #applies the helper function to the chunks and keeps their totals as delayed objects 
    full_total = delayed(sum)(chunk_totals)                                                  #computes the actual full total
    return full_total

#Another helper function was made for factor for the same reason as increment above directly below

@delayed 
def helper_fact(start, end):
    return sum([fun_factor(_) for _ in range(start , end)])

def delayed_factor(c, start, end):
    n = 4 
    dist = end - start                                                                                   #The reasoning behind the code in this function follow that of the increment function
    fact_len =  dist // n                                                                                #Factor is more complex so I went with less chunks 
    start_place = [start + fact_len * _ for _ in range(n)]
    end_place = [start + fact_len * (_ + 1) for _ in range(n)]
    end_place[-1] = end
    
    chuck_totals = [helper_fact(start_place[_], end_place[_]) for _ in range(n)]
    full_total = delayed(sum)(chuck_totals)
    return full_total

#For both futures functions I wanted to make sure that delayed wasn't used at all and due to the previous helper functions having the @delayed operator, they were rewritten without them

def helper_inc_fut(start, end):
    return sum([fun_inc(_) for _ in range(start , end)])

def future_increment(c, start, end):
    n = 16                                                                                   #same chuck amount as the delayed function
    dist = end - start
    incr_len = dist // n      
    start_place = [start + incr_len * _ for _ in range(n)]                                   #Chunk separation was used just as it was in the previous functions that used delayed
    end_place = [start + incr_len * (_ + 1) for _ in range(n)]
    end_place[-1] = end
    
    fut_totals = [c.submit(helper_inc_fut, start_place[_], end_place[_]) for _ in range(n)]    #c.submit is used to create futures objects for chucks and is done so using a for loop like in the the delayed functions
                                                                                               #this was similarly done in the delayed functions only in this it creates 4 futures objects
    full_total = c.submit(sum, fut_totals)                                                     #futures objects will then be combined using sum and be a futures object themselves
    return c.compute(full_total)                                                               #c.compute will wait for the c.submit objects to complete and then apply what the computation is doing
                                                                                               #note that anything involving .result() raised an error because the return must give a futures object and not a result

#See line 48 for helper function below

def helper_fact_fut(start, end):
    return sum([fun_factor(_) for _ in range(start , end)])

def future_factor(c, start, end):
    n = 4                                                                                      #same chunk amount as the delayed function                               
    dist = end - start
    fact_len = dist // n                               
    start_place = [start + fact_len * _ for _ in range(n)]
    end_place = [start + fact_len * (_ + 1) for _ in range(n)]
    end_place[-1] = end
    
    fut_totals = [c.submit(helper_fact_fut, start_place[_], end_place[_]) for _ in range(n)]         #See above comments for reasoning behind c.submit and c.compute use
    full_total = c.submit(sum, fut_totals)
    return c.compute(full_total)
    
    