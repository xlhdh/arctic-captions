# train models
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/ROOMB.npz'], ['reload', True]]"
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', 'full_model_2_stoch.npz'], ['attn_type', 'stochastic']]"


python generate_caps.py ../models/full_model_1.npz_bestll.npz neweval -p 8 -d 'test' -pkl_name ../models/full_model_1.npz

python generate_caps.py ../models/animal_from_1.npz animal_eval -p 8 -d 'test' 

python metrics.py animal_eval.test.txt animal_evalgold0.test.txt animal_evalgold1.test.txt animal_evalgold2.test.txt animal_evalgold3.test.txt animal_evalgold4.test.txt


python generate_caps.py ../models/full_model_1.npz ROOMA -p 6 -d 'test' 
python metrics.py ROOMAgold0.test.txt ROOMAgold1.test.txt ROOMAgold2.test.txt ROOMAgold3.test.txt ROOMAgold4.test.txt
python metrics.py ROOMA.test.txt ROOMAgold1.test.txt ROOMAgold2.test.txt ROOMAgold3.test.txt ROOMAgold4.test.txt

python generate_caps.py ../models/full_model_1.npz_bestll.npz ROOMA -p 6 -d 'test' -pkl_name ../models/full_model_1.npz
python metrics.py ROOMA.test.txt ROOMgold1.test.txt ROOMgold2.test.txt ROOMgold3.test.txt ROOMgold4.test.txt

# test finetuned ROOMA
python generate_caps.py ../models/ROOMA.npz ROOMA -p 6 -d 'test' 
python metrics.py ROOMA.test.txt ROOMgold1.test.txt ROOMgold2.test.txt ROOMgold3.test.txt ROOMgold4.test.txt

# test finetuned ROOMB
python generate_caps.py ../models/ROOMB.npz ROOMB -p 6 -d 'test' 
python metrics.py ROOMB.test.txt ROOMgold1.test.txt ROOMgold2.test.txt ROOMgold3.test.txt ROOMgold4.test.txt

# test finetuned ROOMC
python generate_caps.py ../models/ROOMC.npz ROOMC -p 6 -d 'test' 
python metrics.py ROOMC.test.txt ROOMgold1.test.txt ROOMgold2.test.txt ROOMgold3.test.txt ROOMgold4.test.txt

python generate_caps.py ../models/animal_from_1.npz_bestll.npz animal_bll -p 8 -d 'test' -pkl_name ../models/animal_from_1.npz 
python metrics.py animal_bll.test.txt animal_bllgold0.test.txt animal_bllgold1.test.txt animal_bllgold2.test.txt animal_bllgold3.test.txt animal_bllgold4.test.txt
