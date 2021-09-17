#!/bin/bash
# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###################################################################################
#### Parsing arguments
###################################################################################
function usage()
{
    echo "Run random parameter search"
    echo ""
    echo "usage: ./run_tuning.sh [--start IDX] [--num NUM] GROUP_ID PARAM OUT"
    echo "    -h     --help"
    echo "    -s     --start         Index of parameter to start from. (default: 1)"
    echo "    -n     --num           Number of parameters to run. (default: 0)"
    echo "                           Specify 0 to use the specified parameter file."
    echo "                           If larger than 0, PARAM will be interpretted as a directory."
    echo "    -m     --multiplicity  Number of instances to run for each parameter. (default: 1)"
    echo ""
    echo "Example"
    echo " ./run_tuning.sh -m 5 gca00000 data/conf_base.yml outdir/"
    echo "    Run five instances of train.py with data/conf_base.yml and different seeds,"
    echo "    and save them to outdir/params_0001/000X/ ."
    echo " ./run_tuning.sh -s 2 -n 10 gca00000 params/ outdir/"
    echo "    Run train.py with params/conf_0002.yml to params/conf_0011.yml and "
    echo "    and save them to outdir/params_00XX/0001/ ."
    echo ""
}

# (c) Robert Siemer http://stackoverflow.com/a/29754866
# modified by Yuta Koreedaa

getopt --test > /dev/null
if [[ $? -ne 4 ]]; then
    echo "I’m sorry, `getopt --test` failed in this environment."
    exit 1
fi


if [[ ! "`dirname $0`" -ef "`pwd`" ]]; then
    usage
    echo "You must be in the same path as $0 to run $0."
    exit 1
fi

# adding : means it accepts extra argument
SHORT=hs:n:m:
LONG=help,start:,num:,multiplicity:

# -temporarily store output to be able to check for errors
# -activate advanced mode getopt quoting e.g. via “--options”
# -pass arguments only via   -- "$@"   to separate them correctly
PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # e.g. $? == 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# use eval with "$PARSED" to properly handle the quoting
eval set -- "$PARSED"

arg_start="1"
arg_num="0"
arg_mul="1"

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
	-h|--help)
	    usage
	    exit 0
	    ;;
        -s|--start)
            arg_start="$2"
            shift 2
            ;;
	-n|--num)
            arg_num="$2"
            shift 2
            ;;
        -m|--multiplicity)
            arg_mul="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# handle non-option arguments
if [[ $# -ne 3 ]]; then
    usage
    echo "$0: SCRIPT GROUP_ID PARAM OUT are required"
    exit 4
fi

############# End of obtained code ####################

group_id="$1"
conf_path="`readlink -f $2`"
out_dir="`readlink -f $3`"

initial_seed_str="seed: 42"

###################################################################################
#### End of argument parsing
###################################################################################

convert_conf () {
    local input_conf="$1"
    local output_path="$2"
    local seed="$3"
    if ! `grep -q "$initial_seed_str" "$input_conf"`; then
        echo "Seed info "'"'"$initial_seed_str"'"'" did not appear in $input_conf . Aborting ...."
        exit 5
    fi
    cat $input_conf | sed "s/${initial_seed_str}/seed: ${seed}/" > $output_path
}

set -eu

if [[ $arg_num -gt 0 ]]; then
    end_idx=$(($arg_start + $arg_num - 1))
else
    end_idx="$arg_start"
fi

mkdir -p $out_dir/params/

for i in $(seq -f "%04g" $arg_start $end_idx); do
    for seed in $(seq 1 $arg_mul); do
        conf_name="conf_${i}_`printf %04g $seed`"
        copied_conf_path=$out_dir/params/${conf_name}.yml
        if [[ $arg_num -gt 0 ]]; then
            input_conf_path="${conf_path}/conf_${i}.yml"
        else
            input_conf_path="$conf_path"
        fi
        convert_conf $input_conf_path $copied_conf_path $seed
        out_path=${out_dir}/$conf_name
        echo "bash train_pbs.sh $group_id $copied_conf_path 1 $out_path"
        bash train_pbs.sh $group_id $copied_conf_path 1 $out_path
        sleep 2
    done
done
