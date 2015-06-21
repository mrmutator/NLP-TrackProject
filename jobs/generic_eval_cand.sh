# argument1 candidate_index_file argument2 output_directory
# shell for the job:
#PBS -S /bin/bash
# use one node with 16 cores:
#PBS -lnodes=1:cores16:mem64gb
# job requires at most 4 hours, 0 minutes
#     and 0 seconds wallclock time
#PBS -lwalltime=72:00:00
# cd to the directory where the program is to be called:
cd "$TMPDIR"
cp $HOME/NLP-TrackProject/models/tree.ann.gz tree.ann.gz
cp $HOME/NLP-TrackProject/models/candidates/INPUT INPUT
cp $HOME/NLP-TrackProject/evaluate_candidates.py ./
cp $HOME/NLP-TrackProject/models/mono_500_de.bin.gz ./
cp -r $HOME/venv ./
gunzip tree.ann.gz
gunzip mono_500_de.bin.gz
# venv
module load python  2>err_py
source venv/bin/activate 2>err_venv
# call the programs
python evaluate_candidates.py -w mono_500_de.bin -d 500 -t tree.ann -c INPUT -o results.p -p 16 -s 500  -r 100 > out 2>&1    # start prog1
cp out $HOME/jobs/evaluate_candidates/OUTPUT/
cp err_* $HOME/jobs/evaluate_candidates/OUTPUT/
cp results.p $HOME/jobs/evaluate_candidates/OUTPUT/
wait          # wait until programs are finished
