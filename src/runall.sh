# Run all .py targets in this directory.
#
# Note that this script uses -c opt on the bazel command-line.
# This can cause problems in some OS'es (MacOS).
#
# The option can be removed, things will just run a little
# slower.

bazel run lib/circuit_test || exit 1

for algo in `ls -1 *.py | sort`
do
  if [ "$algo" = "__init__.py" ]; then
    continue
  fi
  testcase=`echo $algo | sed s@\.py@@g`
  echo ""
  echo "--- [$testcase] ------------------------"
  bazel run -c opt $testcase || exit 1
done

