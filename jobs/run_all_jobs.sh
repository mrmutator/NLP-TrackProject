mkdir jobs
for i in {2..200}
do
 mkdir output$i
 sed "s/INPUT/candidates_index.p.$i/g" generic_eval_cand.sh | sed "s/OUTPUT/output$i/g" > jobs/job$i.sh
 qsub jobs/job$i.sh
done


