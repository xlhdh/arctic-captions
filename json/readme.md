# Train Models

## Train models for ROOM
### train models for ROOMA
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/ROOMA.npz'], ['reload', True], ['save-per-epoch', True]]"

### train models for ROOMB
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/ROOMB.npz'], ['reload', True], ['save-per-epoch', True]]"

### train models for ROOMC
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/ROOMC.npz'], ['reload', True], ['save-per-epoch', True]]"

## Train models for animal
### train models for animalA
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/animalA.npz'], ['reload', True], ['save-per-epoch', True]]"

### train models for animalB
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/animalB.npz'], ['reload', True], ['save-per-epoch', True]]"

### train models for animalC
THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', '../models/animalC.npz'], ['reload', True], ['save-per-epoch', True]]"

### other training commands
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

# Test Models

## Test finetuned models for ROOM
### test finetuned model for ROOMA
python generate_caps_origin.py ../models/ROOMA.npz ROOMA -p 6 -d 'dev'

python metrics.py ROOMA.dev.txt ROOMAgold1.dev.txt ROOMAgold2.dev.txt ROOMAgold3.dev.txt ROOMAgold4.dev.txt

### test finetuned model for ROOMB
python generate_caps_origin.py ../models/ROOMB.npz ROOMB -p 6 -d 'dev' 

python metrics.py ROOMB.dev.txt ROOMBgold1.dev.txt ROOMBgold2.dev.txt ROOMBgold3.dev.txt ROOMBgold4.dev.txt

### test finetuned model for ROOMC
python generate_caps_origin.py ../models/ROOMC.npz ROOMC -p 6 -d 'dev' 

python metrics.py ROOMC.dev.txt ROOMCgold1.dev.txt ROOMCgold2.dev.txt ROOMCgold3.dev.txt ROOMCgold4.dev.txt

## Test finetuned models for animal
### test finetuned models for animalA
python generate_caps.py ../models/animalA.npz animalA -p 6 -d 'test'

python metrics.py animalA.test.txt animalgold1.test.txt animalgold2.test.txt animalgold3.test.txt animalgold4.test.txt

### test finetuned models for animalB
python generate_caps.py ../models/animalB.npz animalB -p 6 -d 'test'

python metrics.py animalB.test.txt animalgold1.test.txt animalgold2.test.txt animalgold3.test.txt animalgold4.test.txt

### test finetuned models for animalC
python generate_caps.py ../models/animalC.npz animalC -p 6 -d 'test'

python metrics.py animalC.test.txt animalgold1.test.txt animalgold2.test.txt animalgold3.test.txt animalgold4.test.txt






## Other test finetuned commands
python generate_caps.py ../models/animal_from_1.npz_bestll.npz animal_bll -p 8 -d 'test' -pkl_name ../models/animal_from_1.npz 
python metrics.py animal_bll.test.txt animal_bllgold0.test.txt animal_bllgold1.test.txt animal_bllgold2.test.txt animal_bllgold3.test.txt animal_bllgold4.test.txt
