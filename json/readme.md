Usage: 
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', 'full_model_1.npz'], ['reload', True]]"
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', 'full_model_2_stoch.npz'], ['attn_type', 'stochastic']]"


python generate_caps.py ../models/full_model_1.npz_bestll.npz neweval -p 8 -d 'test' -pkl_name ../models/full_model_1.npz

python generate_caps.py ../models/animal_from_1.npz animal_eval -p 8 -d 'test' 

python metrics.py animal_eval.test.txt animal_evalgold0.test.txt animal_evalgold1.test.txt animal_evalgold2.test.txt animal_evalgold3.test.txt animal_evalgold4.test.txt
