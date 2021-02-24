import re
from collections import Counter
import numpy as np
import pandas as pd
# DO NOT add any library
stop_chars = ['?' , '!' , ',' , ' ', '.' , '1', '2' , '3' , '4', '5','6' , '7' , '8' , '9' , '10' , ';' , ':']
def replace_all(text,rep, repl):
    for c in rep:
        text = text.replace(c, repl)
    return text

# WRITE as many function as you want for clean code
COSTS = {}
def reset():
    COSTS.clear()
def addCost(word , i ,base=0):
    if word in COSTS:
        #print("wanna update : {} , i:{} , base : {} , cost : {}".format(word , i , base , COSTS[word]))
        if base+i<COSTS[word]:
            COSTS[word] = base+i
    else:
        COSTS[word]=base+i

def delete_letter(word):
    base =  0 if word not in COSTS else COSTS[word]
    delete_l = [word[:i] + word[i+1:] for i ,w in enumerate(word)]
    #WRITE your code here
    for w in delete_l:
        addCost(w , 1, base=base) 
    return delete_l
    
def replace_letter(word  ):
    base =  0 if word not in COSTS else COSTS[word]
    english_letters = [chr(a) for a in range(ord('a') , ord('z')+1)]
    replace_l=[word[:i] +x+ word[i+1:] for x in english_letters for i ,w in enumerate(word)]
    #WRITE your code here   
    for w in replace_l:
        addCost(w , 2, base=base) 
    return replace_l
    
def insert_letter(word):
    base =  0 if word not in COSTS else COSTS[word]
    english_letters = [chr(a) for a in range(ord('a') , ord('z')+1)]
    insert_l = [word[:i] +x+ word[i:] for x in english_letters for i  in range(len(word)+1)]
    #WRITE your code here
    for w in insert_l:
        addCost(w , 1 , base=base) 
    return insert_l

def edit_one_letter(word):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    edit_one_set.update(insert_letter(word))
    edit_one_set.update(replace_letter(word))

    return edit_one_set
    
def edit_two_letters(word):
    #Write your code here
    edit_two_set = set()
    edit_one_set = edit_one_letter(word)
    for w in edit_one_set:
        edit_two_set.update(edit_one_letter(w))
    return edit_two_set  

def autoCorrect(word):
    # we call this function in judge
    # Write your main code here and generate correct word as result.
    # please do not modify parameters or this function name
#     COSTS = {}
    reset()
    edit1 = edit_one_letter(word)
    edit2 = edit_two_letters(word)
    words = list(edit1.union(edit2))
    with open('shakespeare.txt') as f:
        data = f.readlines()
    freq={}
    for l in data:
        for w in l.split():
            w = w.lower()
            w = replace_all(w , stop_chars,'')
            if "'" in w :
                w = w.split("'")[0]
            if w in freq:
                freq[w]+=1
            else:
                freq[w]=1
#     print('hi')
    l = [w for w in words if w in freq.keys()]
    l = sorted(l , key=lambda x : (COSTS[x] , -freq[x]))
    # WRITE YOUR CODE HERE
    result=l[0] if len(l)>0 else ""
    return result