#!/bin/bash

shfile=`realpath $0`
testdir=`dirname $shfile`
rootdir=`dirname $testdir`

# replace with absolute paths to tensor files with tns format
matfiles=("/absolute/path/to/file1.mtx" "/absolute/path/to/file2.mtx" "/absolute/path/to/filen.mtx")
matfilenum=(0)

mkdir -p $testdir/stats
statfiles=("/stats/spmm-gemm.txt" "/stats/sddmm-spmm.txt" "/stats/hadamard-gemm.txt" "/stats/sddmm-spmm-gemm.txt")
testcases=("scheduling_eval.spmmFused" "scheduling_eval.sddmmFused" "scheduling_eval.hadamardFused" "scheduling_eval.sddmmSpmmFused")
test_case_num=(3)

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