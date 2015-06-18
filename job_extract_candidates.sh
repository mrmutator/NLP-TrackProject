# shell for the job:
#PBS -S /bin/bash
# use one node with 16 cores:
#PBS -lnodes=1:cores16:mem64gb
# job requires at most 4 hours, 0 minutes
#     and 0 seconds wallclock time
#PBS -lwalltime=48:00:00
# cd to the directory where the program is to be called:
cd "$TMPDIR"
mkdir output
cp -r $HOME/NLP-TrackProject/models/mono_500_de.bin ./
cp $HOME/NLP-TrackProject/extract_candidates.py ./
cp -r $HOME/venv ./
# venv
module load python  2>err_py
source venv/bin/activate 2>err_venv
# call the programs
python extract_candidates.py -w mono_500_de.bin -b output/dawg -c output/candidates.p -o output/tree.ann -i output/candidates_index.p -l 4 -n 100 > out1 2>&1    # start prog1
cp out* $HOME/jobs/extract_candidates/
cp err_* $HOME/jobs/extract_candidates/
cp -r output/ $HOME/jobs/extract_candidates/
wait          # wait until programs are finished
