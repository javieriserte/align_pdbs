files=$(ls data3/input/*_clustalO_fullidentity_60.fasta.aln)
for x in ${files}
do
    base=$(sed 's/data3.input.\(.\+\)_clustalO.\+$/\1/g'<<<"${x}")
    echo $x
    echo $base
    cp $x data3/input/aln.fa
    echo " "
    echo "######### $base ###########"
    python src/align_pdb.py create-pdb-mappings data3
    python src/align_pdb.py calc-kin-cre-distances --up ${base} --folder data3
    # exit
done