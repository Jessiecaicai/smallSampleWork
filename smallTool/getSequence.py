# -*- coding: utf-8 -*-
# @Author  : Jessie
# @Time    : 2021/11/17 3:58 下午
# @Function:
import random
from Bio import SeqIO
import os

from tqdm import tqdm


def getSequenceInturn():
    '''
    顺序获得fasta文件中的sequence
    :return:
    '''
    with open('/home/guo/data/datacluster/uniref50/db/uniref50.fasta') as txt:
        txtData = txt.readlines()
        # print(txtData)

    resultFile = open('resultFile.fasta', 'w')

    count = 1
    number = 50000
    sumLength = 0


    for sequence in txtData:
        if count <= 2 * number:
            count = count + 1
            sumLength = sumLength + len(sequence)
            resultFile.write(sequence)
        else:
            break

    resultFile.close()
    print("sequenceAvaLength = " + str(sumLength/number))

#getSequenceInturn()

def get_sequence_number():
    count = 0
    with open('/home/guo/data/datacluster/uniref50/db/uniref50_256.txt') as txt:
        txtData = txt.readlines()
    for sequence in txtData:
        # if sequence[0] == ">":
        #     count += 1
        count += 1
    print("共有" + str(count) + "条序列")

# get_sequence_number()

def cut_toolong_sequence():
    '''
    截断最长序列长度为256
    :return:
    '''
    with open('/home/guo/data/datacluster/uniref50/db/uniref50.txt') as txt:
        txtData = txt.readlines()
    result_file = open('/home/guo/data/datacluster/uniref50/db/uniref50_256.txt', 'a')
    for sequence in txtData:
        if len(sequence) > 254:
            cut_sequence = sequence[0:254]
            result_file.write(cut_sequence + '\n')
        if len(sequence) <= 254:
            result_file.write(sequence)
    result_file.close()

# cut_toolong_sequence()
def getSequencetoTxt():
    '''
    顺序获取uniref50.fasta中的序列，并且转成txt格式
    :return:
    '''

    # dataset_path = "/home/guo/data/datacluster/uniref50/db"
    # os.chdir(dataset_path)
    count = 0

    # with open('/home/guo/data/datacluster/uniref50/db/uniref50.fasta') as txt:
    with open('/research/zqg/dataset/uniref50.fasta') as txt:
        txtData = txt.readlines()

    result_file = open('/research/zqg/dataset/uniref50_train.txt', 'a')

    index_dict = []
    for i, sequence in enumerate(txtData):
        if sequence[0] == '>':
            index_dict.append(i)
            count += 1
    print(count)
    for j, number in enumerate(index_dict):
        little_sequenct_list = []
        if j + 1 == count:
            for k in range(300326066 - number ):
                little_sequence = txtData[number + k + 1].strip('\n')
                little_sequenct_list.append(little_sequence)
            write_sequence = "".join(little_sequenct_list)
            result_file.write(write_sequence)
        else:
            little_sequence_count = index_dict[j + 1] - index_dict[j] - 1
            for k in range(little_sequence_count):
                little_sequence = txtData[number + k + 1].strip('\n')
                little_sequenct_list.append(little_sequence)
                k += 1
            write_sequence = "".join(little_sequenct_list)
            result_file.write(write_sequence + '\n')
    result_file.close()
getSequencetoTxt()

def getFullSequence(i, sequenceCut):
    '''
    传入序列号和接下来几行的片段序列，返回此序列开始的几行组成完整sequence以及接下来的序列号
    :return:
    '''
    little_sequence_count = 0
    for littleSequence in sequenceCut:
        littleSequence += 1
    full_sequence = "".join(sequenceCut)
    next_index = i + little_sequence_count
    return_dict = {full_sequence:full_sequence, next:next}

    return return_dict

def getSequencetoList():
    '''
    把uniref50的序列读出来放到list里测速
    :return:
    '''
    unire50SequenceList = []
    with open('/home/guo/data/datacluster/uniref50/db/uniref50.fasta')as txt:
        txtData = txt.readlines()
    for sequence in txtData:
        unire50SequenceList.append(sequence)
    print("finish")

# getSequencetoList()

def getSequenceRandom():
    '''
    随机获取特定条数的fasta文件里的sequence(未完成
    :return:
    '''
    with open('/home/guo/data/datacluster/uniref50/db/uniref50.fasta')as txt:
        txtData = txt.readlines()
    #print(txtData)
    resultFile = open('resultSequence.fasta','w')

    count = 1
    number = 5000 # count为所取sequence条数

    while count <= number:
        count = count + 1
        randomNumber = random.randint(1,len(txtData))
        print("len(txtData)为：" + str(len(txtData)))
        resultFile.write(txtData[randomNumber])

    resultFile.close()

#getSequenceRandom()

def getSequenceInturnBio():
    '''
    bio解析+顺序获取特定条数fasta文件里的sequence
    :return:
    '''

    count = 1
    number = 90 #获取sequence条数
    resultFile = open('resultBioInturnSequence.fasta','w')

    for seq_record in SeqIO.parse('/home/guo/data/datacluster/uniref50/db/uniref50.fasta','fasta'):
        count = count + 1
        if(count < number):
            record = seq_record
            resultFile.write('>' + record.description + '\r\n')
            resultFile.write(str(record.seq) + '\r\n')

    resultFile.close()

#getSequenceInturnBio()

def getSequenceRandomBio():
    '''
    bio解析+随机获取特定条数fasta文件里的sequence
    :return:
    '''



def getSequencefromffdata():
    '''
    获得乱码ffdata里的sequence数据
    :return:
    '''
    #with open('/home/data/datacluster/uniref50/dbsearch500Sequence/msaDB_ca3m.ffdata') as txt:
    with open('/home/data/datacluster/uniclust30_2018_08/uniclust30_2018_08_a3m.ffdata') as txt:
        txtData = txt.readlines()
        print(txtData)

#getSequencefromffdata()

def getSequencefromPPITxt():
    '''
    获得PPItxt格式文件里的sequence序列
    :return:
    '''
    with open('/home/guo/data/zxb_data/finally_train.txt') as txt:
    #with open('/home/guo/data/zxb_data/testSequence.txt') as txt:
        txtData = txt.readlines()
        #print(txtData)

    i = 1
    for sequence in txtData:
        realSequence = sequence.split('\t')
        num = len(realSequence)
        j = 0
        while j < 2:
            resultFile = open('/home/guo/data/zxb_data/sequences/sequence{}-{}.fasta'.format(i, j), 'w')
            resultFile.write(realSequence[j])
            resultFile.close()
            j += 1
        i += 1
    print("共计行数" + str(i))

#getSequencefromPPITxt()
