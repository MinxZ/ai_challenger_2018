Ubuntu:
rm -r pred
mkdir pred
mv *.txt pred
tar -czvf pred.tar.gz pred

MacOS:
scp z@192.168.3.2:~/data/zl/pred.tar.gz ~/zl/pred.tar.gz
tar -xzvf pred.tar.gz
