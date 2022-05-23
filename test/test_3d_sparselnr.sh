#!/bin/bash

shfile=`realpath $0`
testdir=`dirname $shfile`
rootdir=`dirname $testdir`

echo "Directory is $shfile"
echo "Directory is $testdir"
echo "Directory is $rootdir"

matfiles=("/home/min/a/kadhitha/ispc-examples/data/tns/matmul_5-5-5.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/delicious-3d.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/flickr-3d.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/nell-2.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/nell-1.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/vast-2015-mc1-3d.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/darpa1998.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_music.tns" "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_sampled.tns")
matfilenum=(0)

mkdir -p $testdir/stats
statfiles=("/stats/mttkrp-gemm.txt" "/stats/ttm-ttm.txt")
testcases=("scheduling_eval.mttkrpFused" "scheduling_eval.ttmFused")
test_case_num=(1)

export TACO_CFLAGS=" -O3 -ffast-math -std=c99 -fopenmp "

for j in ${!test_case_num[@]}; do

    statfile="$testdir${statfiles[test_case_num[j]]}"
    test_case="${testcases[test_case_num[j]]}"

    # rm -f $statfile
    touch -a $statfile

    export STAT_FILE=$statfile

    # if [[ -z "${STAT_FILE}" ]]; then #check if zero
    #     echo "stat file not found"
    #     export STAT_FILE=$statfile
    # else 
    #     echo "STAT_FILE ${STAT_FILE}"
    # fi

    echo "" >> $statfile
    echo "" >> $statfile
    echo "" >> $statfile
    echo "Dataset, SparseLNR, TACO-Separate, TACO-Original" >> $statfile

    for i in ${!matfilenum[@]}; do
        printf "%s %s : %s %s %s\n" $j $statfile $i ${matfilenum[i]} ${matfiles[matfilenum[i]]}
        export TENSOR_FILE=${matfiles[matfilenum[i]]}
        echo "$rootdir/build/bin/taco-test --gtest_filter=$test_case"

        $rootdir/build/bin/taco-test --gtest_filter=$test_case

    done

done


unset TENSOR_FILE
unset STAT_FILE

# $rootdir/build/bin/taco-test --gtest_filter=scheduling_eval.spmmFusedWithSyntheticData