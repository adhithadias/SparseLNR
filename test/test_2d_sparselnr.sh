#!/bin/bash

shfile=`realpath $0`
testdir=`dirname $shfile`
rootdir=`dirname $testdir`

echo "Directory is $shfile"
echo "Directory is $testdir"
echo "Directory is $rootdir"

matfiles=("/home/min/a/kadhitha/workspace/my_taco/taco/net-repo-graph/cora.mtx" "/home/min/a/kadhitha/workspace/my_taco/taco/net-repo-graph/amazon.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx" "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx" "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx" "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k.mtx")
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