Ubuntu:
rm -r pred
mkdir pred
mv *.txt pred
tar -czvf pred.tar.gz pred

MacOS:
scp z@192.168.3.2:~/data/zl/pred.tar.gz ~/zl/pred.tar.gz
tar -xzvf pred.tar.gz

scp z@192.168.3.2:~/data/zl/attr_animals.txt   ~/zl/attr_animals.txt

tar -czvf img_test.tar.gz img_test
scp z@192.168.3.2:~/data/zl/img_test.tar.gz ~/zl/img_test.tar.gz
tar -xzvf img_test.tar.gz
