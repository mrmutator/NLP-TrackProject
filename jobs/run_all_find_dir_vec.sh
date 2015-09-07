mkdir jobs
for i in {1..200}
do
 mkdir output$i
 sed "s/INPUT/candidates_index.p.$i/g" generic_find_dir_vec.sh | sed "s/OUTPUT/output$i/g" > jobs/job$i.sh
 qsub jobs/job$i.sh  || echo "$i failed" >> failed.txt
done


