# Run all .py targets in this directory.

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
