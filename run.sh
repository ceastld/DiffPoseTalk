CUDA_VISIBLE_DEVICES="3" python demo.py --exp_name SA-hubert-WM --iter 100000 \
    -a demo/input/audio/cxk.mp3 -c demo/input/coef/man.npy \
    -s demo/input/style/smile.npy -o cxk-man-smile.mp4 \
    -n 3 -ss 3 -sa 1.15 -dtr 0.99