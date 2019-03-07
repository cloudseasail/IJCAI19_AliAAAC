
FROM registry.cn-shanghai.aliyuncs.com/haifan_deep/ijcai2019:weight2.3.0
#MAINTAINER
MAINTAINER cloudseasail

COPY ./IJCAI19/weight/xception_19 /competition/IJCAI19/weight/xception_19

COPY ./IJCAI19/model /competition/IJCAI19/model
COPY ./IJCAI19/module /competition/IJCAI19/module
COPY ./IJCAI19/__init__.py /competition/IJCAI19/__init__.py
COPY ./run.sh /competition/run.sh
COPY ./attack.py /competition/attack.py

WORKDIR /competition
