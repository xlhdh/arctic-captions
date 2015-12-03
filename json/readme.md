Usage: THEANORC=theanorc-gpu.rc nohup python evaluate_coco.py "[['model', 'full_model_1.npz'], ['reload', True]]"

python generate_caps.py my_caption_model.npz neweval -p 8 -d 'test

python generate_caps.py ../models/full_model_1.npz_bestll.npz neweval -p 8 -d 'test' -pkl_name ../models/full_model_1.npz
