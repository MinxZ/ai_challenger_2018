rm -r pred
mkdir pred
mv *.txt pred
tar -czvf pred.tar.gz pred

scp z@192.168.3.2:~/data/zl/pred.tar.gz ~/zl/pred.tar.gz
tar -xzvf pred.tar.gz

scp z@192.168.3.2:~/data/zl/pred_all.txt ~/zl/


rm -r /data/zl/img_test
tar -czvf /data/zl/img_test.tar.gz /data/zl/img_test
scp z@192.168.3.2:~/data/zl/img_test.tar.gz ~/zl/img_test.tar.gz
tar -xzvf ~/zl/img_test.tar.gz

mkdir ~/zl/fruits/
mkdir ~/zl/animals/
scp z@192.168.3.2:~/data/zl/fruits/features_test.npy   ~/zl/fruits/
scp z@192.168.3.2:~/data/zl/fruits/features_train.npy   ~/zl/fruits/
scp z@192.168.3.2:~/data/zl/fruits/class_a.npy   ~/zl/fruits/
scp z@192.168.3.2:~/data/zl/animals/features_test.npy   ~/zl/animals/
scp z@192.168.3.2:~/data/zl/animals/features_train.npy   ~/zl/animals/
scp z@192.168.3.2:~/data/zl/animals/images_test.npy   ~/zl/animals/

scp z@192.168.3.2:~/data/ai_challenger_zsl2018_train_test_a_20180321.zip   ~/zl/

scp ~/zl/fruits_dataset/X.npy z@192.168.3.2:~/data/zl/fruits_test/
scp ~/zl/fruits_dataset/y.npy z@192.168.3.2:~/data/zl/fruits_test/
