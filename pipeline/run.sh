DATA=RedPajama
mkdir ../figs
for size in "70M" "160M" "305M" "410M"
do
    python3 steplaw.py --train_data $DATA --savefig ../figs/${DATA}_${size}_steps.png --model_size ${size} --fit_step_range 10000,30000 --tie_alpha valid --ratios ../data/${DATA}/proportions.txt
done

python sizelaw.py --train_data $DATA --savefig ../figs/${DATA}_sizes.png --fit_sizes 70M,160M,305M,410M --target_size 1B --step 100000 --tie_alpha all --ratios ../data/${DATA}/proportions.txt 
python mixlaw.py --size 1B --step 100000 --train_data $DATA --ratios ../data/${DATA}/proportions.txt 

python find_opt.py --size 1B --step 100000 --savefig ../figs/opt.png
