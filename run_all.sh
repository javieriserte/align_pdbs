files=$(ls data2/input/*_gapstrip)
for x in ${files}
do
    echo $x
    base=$(sed 's/data2.input.//g'<<<"${x}")
    cp $x data2/input/aln.fa
    echo " "
    echo "######### $base ###########"
    python src/align_pdb.py create-pdb-mappings data2
    python src/align_pdb.py align-kinases data2 $base
done