BATCH_SIZE="8"
LR="0.002"

for b in $BATCH_SIZE
do
    for l in $LR
    do
        python main.py --learning_rate $l --batch_size $b --epochs 10
    done
done

