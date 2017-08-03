cd /home/hdp-reader-tag/shechanglue/ctrPredict/code
#--------------------------------------------------------------------------------------------------
#set the batch day
batchDate=`date +%Y%m%d -d"-1 days"`
#--------------------------------------------------------------------------------------------------
#set relevant dir
PYTHON=/home/hdp-reader-tag/shechanglue/software/anaconda2/bin
HADOOP=/usr/bin/hadoop/software/hadoop/bin
INPUT_FILE=hdfs://w-namenode.qss.zzbc2.qihoo.net:9000/home/nlp/shechanglue/TagUrlStat_SEG/$batchDate
#--------------------------------------------------------------------------------------------------
#delete original file
rm -rf ../data/raw/*
#--------------------------------------------------------------------------------------------------
#wait until the input file exist
while [ 1 ]; do
                $HADOOP/hadoop fs -test -e $INPUT_FILE/_SUCCESS
                if [ $? -ne 0 ]; then
                                echo "`date`, waiting for input file"
                                sleep 1800
                                date_now_2daybefore=`date -d "2 day ago" +"%Y%m%d"`
                                if [ ${date_now_2daybefore} -eq ${batchDate} ]; then
                                                echo "`date`, 88 for ${batchDate}"
                                                exit;
                                fi
                else
                        echo "INPUT FILE check OK"
                        break;
                fi
done
#--------------------------------------------------------------------------------------------------
#pull the rawdata in the hadoop
$HADOOP/hadoop fs -getmerge $INPUT_FILE ../data/raw/$batchDate\.gz
#--------------------------------------------------------------------------------------------------
#unzip the data
gunzip ../data/raw/$batchDate\.gz
#--------------------------------------------------------------------------------------------------
# run the preprocess file 
source $PYTHON/activate tfEnvi
$PYTHON/python preprocess.py $batchDate
source $PYTHON/deactivate tfEnvi
#--------------------------------------------------------------------------------------------------
#delete the raw data 
rm -rf ../data/raw/*

