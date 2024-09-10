# CUDA_VISIBLE_DEVICES="3" python demo.py --exp_name head-SA-hubert-WM --iter 110000 \
#     -a demo/input/audio/song.mp3 -c demo/input/coef/TH050.npy \
#     -s demo/input/style/pitch.npy -o song-head-pitch.mp4 \
#     -n 3 -ss 3 -sa 2 -dtr 0.99 \
#     # --save_coef

# CUDA_VISIBLE_DEVICES="4" python demo.py --exp_name head-SA-hubert-WM --iter 110000 \
#     -a demo/input/audio/hutao.wav -c demo/input/coef/TH050.npy \
#     -s demo/input/style/pitch.npy -o hutao-head-pitch-ss1.mp4 \
#     -n 3 -ss 1 -sa 1.15 -dtr 0.99 \
#     --save_coef


# CUDA_VISIBLE_DEVICES="4" python demo.py --exp_name head-SA-hubert-WM --iter 110000 \
#     -a demo/input/audio/talk.mp3 -c demo/input/coef/TH050.npy \
#     -s demo/input/style/pitch.npy -o talk-head-pitch-ss1.mp4 \
#     -n 3 -ss 1 -sa 1.15 -dtr 0.99 \
#     --save_coef

CUDA_VISIBLE_DEVICES="4" python demo.py --exp_name SA-hubert-WM --iter 100000 \
    -a demo/input/audio/talk.mp3 -c demo/input/coef/TH050.npy \
    -s demo/input/style/smile.npy -o talk-head-smile-ss1.mp4 \
    -n 3 -ss 1 -sa 1.15 -dtr 0.99 \
    --save_coef