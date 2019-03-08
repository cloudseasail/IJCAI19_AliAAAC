$tag="n2.3.7"
$folder="..\NonTargetAttack"
# $folder="..\TargetAttack"
# $folder="..\Defense"

cp ..\..\IJCAI19 .\IJCAI19 -Force -Recurse
cp $folder\run.sh .\run.sh -Force 
cp $folder\attack.py .\attack.py -Force 

docker build -f Dockerfile -t registry.cn-shanghai.aliyuncs.com/haifan_deep/ijcai2019:$tag .\

rm .\IJCAI19 -Recurse
rm .\run.sh -Force 
rm .\attack.py -Force 

docker push registry.cn-shanghai.aliyuncs.com/haifan_deep/ijcai2019:$tag