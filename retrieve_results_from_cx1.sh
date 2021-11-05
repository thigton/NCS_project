
if [ $# -eq 0 ]; then
    echo "No folder to retrieve supplied"
else

if [ $2 = "log_only" ]; then
scp -r tdh17@login.hpc.ic.ac.uk:/rds/general/user/tdh17/home/$1/results/model_log.csv  results/$1_model_log.csv
# # run some python scipt to
python merge_model_log.py $1
rm results/$1_model_log.csv
else
scp -r tdh17@login.hpc.ic.ac.uk:/rds/general/user/tdh17/home/$1/results  results/$1
# move the contents of the results to a results pool
mv results/$1/model_log.csv results/$1_model_log.csv
mv  results/$1/* results
# # run some python scipt to
python merge_model_log.py $1
rm results/$1_model_log.csv
rm -r results/$1
fi
fi
