rm -r pred
mkdir pred
mv *.txt pred
tar -czvf pred.tar.gz pred

scp z@192.168.3.2:~/data/zl/pred.tar.gz ~/zl/pred.tar.gz
tar -xzvf pred.tar.gz


tar -czvf img_test.tar.gz img_test
scp z@192.168.3.2:~/data/zl/img_test.tar.gz ~/zl/img_test.tar.gz
tar -xzvf img_test.tar.gz

scp z@192.168.3.2:~/data/zl/fruits/fruits_test.txt   ~/zl/fruits_test.txt
scp z@192.168.3.2:~/data/ai_challenger_zsl2018_train_test_a_20180321.zip   ~/zl/
<<<<<<< HEAD

scp ~/zl/fruits_dataset/X.npy z@192.168.3.2:~/data/zl/fruits_test/
scp ~/zl/fruits_dataset/y.npy z@192.168.3.2:~/data/zl/fruits_test/
=======
>>>>>>> dcbc0cbe647a4dce94b5eb2601e215466d7f0a79
