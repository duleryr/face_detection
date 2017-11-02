for i in *List.txt
do 
    grep -o img $i | wc -l
done;
