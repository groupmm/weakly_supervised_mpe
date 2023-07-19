import glob
import os.path

import numpy as np

os.makedirs("data/Schubert_Winterreise/cqt_hs512", exist_ok=True)

for wav_file_name in glob.glob("data/Schubert_Winterreise/hcqt_hs512_o6_h5_s1/*.npy"):
    base_name = os.path.basename(wav_file_name)
    output_file_path = f"data/Schubert_Winterreise/cqt_hs512/{base_name}"

    a = np.load(wav_file_name)
    a = np.abs(a[1::3, :, 1])
    np.save(output_file_path, a.T)
