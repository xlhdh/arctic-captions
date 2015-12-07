for x in J
do
	echo "Start validation for model $x on ROOM"
	rm ../scores/ROOM${x}.dev.scores.log

	for y in {1..10}
	do
		echo "model $x epoch $y ..."
		cp ../models/ROOM$x.npz_epoch_$y.npz ../models/ROOM$x.npz
		python generate_caps_origin.py ../models/ROOM$x.npz ROOM$x -p 8 -d 'dev' 2> /dev/null
		python metrics.py ROOM${x}.dev.txt ROOM${x}gold1.dev.txt ROOM${x}gold2.dev.txt ROOM${x}gold3.dev.txt ROOM${x}gold4.dev.txt >> ../scores/ROOM${x}.dev.scores.log
	done

	echo "Finish validation for model $x on ROOM"
done
