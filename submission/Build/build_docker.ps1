$tag="d2.2.2"
#$folder="..\NonTargetAttack"
# $folder="..\TargetAttack"
$folder="..\Defense"

cp ..\..\IJCAI19 .\IJCAI19 -Force -Recurse
cp $folder\run.sh .\run.sh -Force 
cp $folder\attack.py .\attack.py -Force 

docker build -f Dockerfile -t registry.cn-hangzhou.aliyuncs.com/haifan_deep/ijcai2019_pilot:$tag .\

rm .\IJCAI19 -Recurse
rm .\run.sh -Force 
rm .\attack.py -Force 