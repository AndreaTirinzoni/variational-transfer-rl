if [ "$#" -ne 3 ]; then
    echo "USAGE: sh launch.sh script_file start_core_id end_core_id"
else

script_file=$1
start_core=$2
end_core=$3
new=1
idx=0

while read line 
do
	if [ -z "$line" ]; then
		new=1
	else
		if [ $new -eq 1 ]; then
			tmux new-session -d -s $idx "export OMP_NUM_THREADS=1; taskset -c ${start_core}-${end_core} ${line}; bash -i"
			idx=$((idx + 1))
		else
			tmux split-window -d "export OMP_NUM_THREADS=1; taskset -c ${start_core}-${end_core} ${line}; bash -i"
		fi
		new=0
	fi
done < $script_file

fi
