rm -r pred
mkdir pred
mv *.txt pred
tar -czvf pred.tar.gz pred

scp z@192.168.3.2:~/data/zl/pred.tar.gz ~/zl/pred.tar.gz
tar -xzvf pred.tar.gz


rm -r /data/zl/img_test
tar -czvf /data/zl/img_test.tar.gz /data/zl/img_test
scp z@192.168.3.2:~/data/zl/img_test.tar.gz ~/zl/img_test.tar.gz
tar -xzvf ~/zl/img_test.tar.gz

<<<<<<< HEAD
scp z@192.168.3.2:~/data/zl/fruits/ans_fruits_true.txt   ~/zl/fruits_attr/ans_fruits_true.txt
=======
scp z@192.168.3.2:~/data/zl/fruits/fruits_test.txt   ~/zl/fruits_test.txt
scp z@192.168.3.2:~/data/ai_challenger_zsl2018_train_test_a_20180321.zip   ~/zl/
>>>>>>> e175d16ce9cdbf3714812d18158c6e86fbad4b59

scp ~/zl/fruits_dataset/X.npy z@192.168.3.2:~/data/zl/fruits_test/
scp ~/zl/fruits_dataset/y.npy z@192.168.3.2:~/data/zl/fruits_test/
