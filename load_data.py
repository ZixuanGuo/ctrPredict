# -*- coding: utf-8 -*-

'''
August 2017 by She Changlue. 
snakepointid@sina.com.
https://github.com/snakepointid
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import pickle
import tensorflow as tf
import numpy as np
import codecs
import sys
#-----------------------------------------------------------------------------------------------------------------------------
#define reader func
def file_reader(file,sep = '\t'):
    with codecs.open(file,'r','utf-8') as inStream:
        for line in inStream:
                yield line.strip().split(sep)
#-----------------------------------------------------------------------------------------------------------------------------
def update_vocab(file):
    try:
            token2code = pickle.load(open("../data/other/token2code.pkl","rb"))
            tokenFreq  = pickle.load(open("../data/other/tokenFreq.pkl","rb"))
    except:
            print("initial the token2code")
            token2code = {"inFreq":1,"Freq":2,"#Number#":3}
            tokenFreq  = {"#allTitleNum$":0}
    #-------------------------------------------------------------
    reader = file_reader(file)#load raw data
    for _, seg, _ in reader: 
        tokens = set(seg.split())
        for token in tokens:
            tokenFreq.setdefault(token,0)
            tokenFreq[token]+=1
        tokenFreq["#allTitleNum$"]+=1
    #-------------------------------------------------------------
    newCode = max(token2code.values())
    #print(newCode)
    for token in tokenFreq:
        if token2code.get(token,0)<3:
            freq = tokenFreq[token]*1.0/tokenFreq["#allTitleNum$"]
            if freq < hp.inFreqToken:
                token2code[token] = 1
            elif freq > hp.tooFreqToken:
                token2code[token] = 2
            else:
                print(newCode,token)
                token2code[token] = newCode
                newCode+=1
    #-------------------------------------------------------------
    #pickle.dump(token2code,open("../data/other/token2code.pkl","wb"))
    #pickle.dump(tokenFreq,open("../data/other/tokenFreq.pkl","wb"))
    return token2code
#-----------------------------------------------------------------------------------------------------------------------------
def create_data(file):  
    token2code = update_vocab(file)
    try:
            tag2code = pickle.load(open("../data/other/tag2code.pkl","rb"))
    except:
            print("initial the tag2code")
            tag2code = {}
    #-------------------------------------------------------------
    reader = file_reader(file)#load raw data
    #-------------------------------------------------------------
    # Index
    tag_list, seg_list, ctr_list,seqLen_list = [], [], [] ,[]             
    for tag, seg, ctr in reader:
        seg    = [token  for token in seg.split() if token not in ',” 的了和呢吧…、.<>．▶【@/;:"[]{}-_=+|`~；：，。《》“' ]
        seg    = [token2code.get(token,1) if token.isalpha() else 3 for token in seg]
        seqLen = len(seg)
        seg    = seg[:hp.maxTitleLen]+[0]*(hp.maxTitleLen-seqLen)

        tag    = tag2code.setdefault(tag,len(tag2code))

        tag_list.append(tag),seg_list.append(seg),ctr_list.append(ctr),seqLen_list.append(seqLen)
    #-------------------------------------------------------------
    #pickle.dump(tag2code,open("../data/other/tag2code.pkl","wb"))
    return np.array(tag_list,dtype=np.int32), np.array(seg_list,dtype=np.int32), np.array(ctr_list,dtype=np.float32), np.array(seqLen_list,dtype=np.int32)
#-----------------------------------------------------------------------------------------------------------------------------
def get_batch_data(file):
    # Load data
    TAG,SEG,CTR,SEQLEN = create_data(file)
    
    # calc total batch count
    num_batch = len(TAG) // hp.batch_size
    
    # Convert to tensor
    TAG,SEG,CTR,SEQLEN = tf.convert_to_tensor(TAG, tf.int32),tf.convert_to_tensor(SEG, tf.int32),tf.convert_to_tensor(CTR, tf.float32),tf.convert_to_tensor(SEQLEN, tf.int32)
 
    # Create Queues
    input_queues = tf.train.slice_input_producer([TAG,SEG,CTR,SEQLEN])
            
    # create batch queues
    tag,seg,ctr,seqLen = tf.train.shuffle_batch(input_queues,
                                num_threads=hp.num_threads,
                                batch_size =hp.batch_size, 
                                capacity   =hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return tag,seg,ctr,seqLen,num_batch # (N, T), (N, T), ()

if __name__ == '__main__': 
    tag,seg,ctr,seqLen = create_data(sys.argv[1])
    print(seg[:10])