# Run all .py targets in this directory.
#
# This first command build the accelerated libxgates.so.
#
# The script uses -c opt on the bazel command-line,
# which may cause problems on some OS'es. It can be removed.
#
# All code will run without the library, just about 10x+ slower.
#
bazel build -c opt lib:libxgates.so
if [[ $? != 0 ]]; then
    echo "*** Building libxgates failed. ***"
    echo "*** Try building manually with script 'make_libxgates'"
fi

#
# Now we just iterate over all Python files and run them.
#
for algo in `ls -1 *.py | sort`
do
    echo
    echo "--- [$algo] ------------------------"
    python3 $algo || exit 1
done
