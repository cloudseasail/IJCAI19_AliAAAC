
FROM registry.cn-hangzhou.aliyuncs.com/haifan_deep/ijcai2019_pilot:2.0.1
#MAINTAINER
MAINTAINER cloudseasail

COPY . /competition
COPY ../../IJCAI19 /competition/
WORKDIR /competition
RUN pip --no-cache-dir install  -r requirements.txt
# INSTALL cleverhans foolboxW
