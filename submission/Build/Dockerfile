
FROM registry.cn-hangzhou.aliyuncs.com/haifan_deep/ijcai2019_pilot:weight2.2.0
#MAINTAINER
MAINTAINER cloudseasail

COPY ./IJCAI19/model /competition/IJCAI19/model
COPY ./IJCAI19/module /competition/IJCAI19/module
COPY ./IJCAI19/__init__.py /competition/IJCAI19/__init__.py
COPY ./run.sh /competition/run.sh
COPY ./attack.py /competition/attack.py

WORKDIR /competition
