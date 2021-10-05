# create folder to send to ssh
# content is classes folder, the script which you want to run, contam model

if [if [ $# -eq 0 ]
  then
    echo "No folder to move supplied"
else
echo $1
mkdir -v $1
mkdir -v $1/results
cp -R classes $1
cp -R util $1
cp -R contam_files $1
cp $2 $1/run.py
cp job $1
cp savepdf_tex.py $1
scp -r $1 tdh17@login.hpc.ic.ac.uk:/rds/general/user/tdh17/home
rm -r $1

fi
