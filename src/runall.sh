# Run all .py targets in this directory.
#
# Note that this script uses -c opt on the bazel command-line.
# This may cause problems on some OS'es but it builds the accelerated C++
# xgates library. Code will run without it, just about 10x+ slower.
#
# The -c opt option can be removed, things will just run a little slower.

bazel build -c opt lib/... || exit 1
bazel run -c opt lib/circuit_test || exit 1

for algo in `ls -1 *.py | sort`
do
    echo
    echo "--- [$algo] ------------------------"
    # Bazel would be the default way to run these:
    #   testcase=`echo $algo | sed s@\.py@@g`
    #   bazel run -c opt $testcase || exit 1
    #
    # But it is also possible, perhaps easier, to run directly via Python:
    python $algo || exit 1
done
