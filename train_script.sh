for x in {F}
do
	echo "Start finetuning model on ROOM$x"
	cp ../models/full_model_1.npz_bestll.npz ../models/ROOM$x.npz

	THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/ROOM$x.npz'], ['reload', True], ['save-per-epoch', True]]"
	THEANORC_PID=$!
#	sleep 3900
	kill -9 ${THEANORC_PID}

	echo "Finish finetuning model on ROOM$x"
done
