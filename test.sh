./darknet detector test cfg/bdci2017.data cfg/bdci2017_test.cfg backup/bdci2017_train.backup $1 
echo ""
echo "----------test finished------------"
echo ""
baidupcs upload -f predictions.png /prefictions/$1
echo ""
echo "---------upload finished------------"
echo ""

