DATA=RedPajama
for size in "70M" "160M" "305M" "410M"
do
    python3 steplaw.py --train_data $DATA --savefig ../figs/${DATA}_${size}_steps.png --model_size ${size} --fit_step_range 10000,30000 --tie_alpha valid --ratios ../sl_pipeline/ratios/all_RP.txt
done

python sizelaw.py --train_data $DATA --savefig ../figs/${DATA}_sizes.png --fit_sizes 70M,160M,305M,410M --target_size 1B --step 100000 --tie_alpha all --ratios ../sl_pipeline/ratios/all_RP.txt # --variable flops
python mixlaw.py --size 1B --step 100000 --train_data $DATA --ratios ratios.txt

python find_opt.py --size 1B --step 100000 --savefig ../figs/opt.png
