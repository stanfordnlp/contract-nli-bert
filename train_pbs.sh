#!/bin/bash

set -eu

script_dir="$(dirname "$(readlink -f "$0")")"
cd $script_dir

date_id=`date "+%m%d-%H%M%S"`

if [ "$#" -eq 3 ]; then
    logdir=$script_dir/work/log_$date_id
elif [ "$#" -eq 4 ]; then
    logdir="$4"
else
    echo "usage: train_pbs.sh GROUP_ID CONF MULTI_GPU [LOG_DIR]";
fi

if [ "$3" -eq 1 ]; then
    COMMAND="python -m torch.distributed.launch --nproc_per_node=4 train.py $2"
    RESOURCE="rt_G.large=1"
elif [ "$3" -eq 0 ]; then
    COMMAND="python train.py $2"
    RESOURCE="rt_G.small=1"
else
    echo "MULTI_GPU must be either 1 (multi-gpu training) or 0 (single GPU training)."
fi

tmppath=${logdir}/tmp
mkdir -p $tmppath

echo "Copying all files from $script_dir/ to $tmppath/files"
rsync -ar --exclude='work/' --exclude='.git/' $script_dir/ $tmppath/files/
# ln -s `realpath --relative-to=$tmppath/files/ ${script_dir}/data` $tmppath/files/data
ln -s `realpath --relative-to=$tmppath/files/ ${script_dir}/work` $tmppath/files/work

touch ${logdir}/waiting

{
    printf '\n\n##### Git diff ##############\n'
    git --no-pager diff
    printf '\n\n##### Git status ############\n'
    git status -s
    printf '\n\n##### Other info ############\n'
    printf "disptach date: $date_id\n"
    printf "git hash: `git rev-parse HEAD`\n"
    printf "project dir: $script_dir\n"
} 2>&1 > $logdir/screen.log

cat <<'__EOF__' |
#!/bin/bash
#$ -l __RESOURCE__
#$ -l h_rt=48:00:00
#$ -j y

set -eu

source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 nccl/2.6/2.6.4-1

export PYENV_VIRTUALENV_DISABLE_PROMPT=1;
export PYENV_ROOT="$HOME/.pyenv";
export PATH="$PYENV_ROOT/bin:$PATH";
export PYENV_ROOT="$HOME/.pyenv";
eval "$(pyenv init -)";
eval "$(pyenv virtualenv-init -)";

cd __TMPDIR__/files
export PYTHONPATH=`pwd`

logdir=__LOGDIR__
logfile=$logdir/screen.log

trap "touch ${logdir}/failed; rm -f ${logdir}/running" ERR
mv ${logdir}/waiting ${logdir}/running

{
    printf "temp workdir: `pwd`\n"
    printf "JOB_ID: $JOB_ID\n"
    printf "PE_HOSTFILE: $PE_HOSTFILE\n"
    printf "started date: `date '+%m%d-%H%M%S'`\n"
    printf "HOST: `hostname`\n"
    printf "USER: $USER\n"
    printf "log file: $logfile\n"
    printf "COMMAND: __COMMAND__ $logdir"
    printf '\n\n\n\n\n\n\n-----------------------------\n\n'
    __COMMAND__ $logdir
} 2>&1 >> $logfile

mv ${logdir}/running ${logdir}/succeeded

cd ~
rm -r __TMPDIR__
__EOF__
sed -e "s|__COMMAND__|$COMMAND|g" \
    -e "s|__LOGDIR__|${logdir}|g" \
    -e "s|__TMPDIR__|${tmppath}|g" \
    -e "s|__RESOURCE__|${RESOURCE}|g" \
    > $tmppath/command.pbs
printf "Making temporary qsub script to ${tmppath}/command.pbs\n"
printf "Logging to $logdir with JOB id t${date_id}\n"
qsub -g "$1" -N t${date_id} $tmppath/command.pbs
