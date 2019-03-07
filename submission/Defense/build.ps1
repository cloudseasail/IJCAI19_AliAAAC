$tag="d2.3.4"

cp ..\..\IJCAI19 .\IJCAI19 -Force -Recurse

docker build -f Dockerfile -t registry.cn-shanghai.aliyuncs.com/haifan_deep/ijcai2019:$tag .\

rm .\IJCAI19 -Recurse

docker push registry.cn-shanghai.aliyuncs.com/haifan_deep/ijcai2019:$tag