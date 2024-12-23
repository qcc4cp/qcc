for t in `ls *test.py`;
do
    echo
    echo "Running $t"
    python3 $t || exit 1
done
