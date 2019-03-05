$tag="u2.1.4"
cp ..\..\IJCAI19 .\IJCAI19 -Force -Recurse
docker build -f Dockerfile -t registry.cn-hangzhou.aliyuncs.com/haifan_deep/ijcai2019_pilot:$tag .\
rm .\IJCAI19 -Recurse