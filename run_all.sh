files=$(ls data3/input/*_fullidentity*)
for x in ${files}
do
    base=$(sed 's/data3.input.\(.\+\)_clustalO.\+$/\1/g'<<<"${x}")
    echo $base
    cp $x data3/input/aln.fa
    echo " "
    echo "######### $base ###########"
    python src/align_pdb.py create-pdb-mappings data3
    python src/align_pdb.py align-kinases data3 $base
done