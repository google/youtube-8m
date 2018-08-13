This is code of PhoenixLin(ranked5th/394) for the 2nd Youtube-8M Video Understanding Challenge

All the experiments are ran on 2 NVIDIA GTX 1080TI GPUs with a batch size of 160 (80 on each).
Local experiments could run at a speed of 400+ example/sec with SSD and finish in less than 10 hours.
The final submission will takes about 3 days to finish if you want to get the best results.
To reproduce the local experiments:
```bash
bash scripts/train_nextvlad_local.sh
```

To reproduce the final submission experiments:
```bash
bash scripts/train_mix_nextvlad_final_submission.sh
```

The parameters of the model can be adjusted by modifying the bash scripts. 

### Don't forget to change the data paths!

I have some modification on the train.py, eval.py and utils.py. If you plan to incorporate the solution 
into your own framework, don't forget those either.

The solution is based on the Youtube-8M Tensorflow Starter Code. Please refer to YOUTUBE8M_README.md or https://github.com/google/youtube-8m
for more details!

Our implementation is based on the code from Willow, who is the winner of fist Youtube8M competition
(https://github.com/antoine77340/Youtube-8M-WILLOW)


















