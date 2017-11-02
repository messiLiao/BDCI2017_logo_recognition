./darknet detector valid cfg/bdci2017.data cfg/bdci2017_test.cfg backup/bdci2017_train_900.backup 
echo ""
echo "----------valid finished------------"
echo ""
tar -czvf results.tar.gz ./results/
echo ""
echo "----------tar dirctory results finished"
echo ""
baidupcs upload -f results.tar.gz results.tar.gz
echo ""
echo "-------upload results.tar.gz finised-----"
echo ""


