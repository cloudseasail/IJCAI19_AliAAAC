cp ..\..\IJCAI19 .\IJCAI19 -Force -Recurse
docker build -f Dockerfile -t registry.cn-hangzhou.aliyuncs.com/haifan_deep/ijcai2019_pilot:u1.0.3 .\
rm .\IJCAI19 -Recurse