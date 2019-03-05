set IJCAI19_ROOT=\\rfsw-bj3\Develop\IJCAI2019
set TARGET_CONTAINER=
set SUBMISSION_FOLDER=NonTargetAttack

docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\submission\%SUBMISSION_FOLDER%\run.sh %TARGET_CONTAINER%:/competition
docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\submission\%SUBMISSION_FOLDER%\attack.py %TARGET_CONTAINER%:/competition
docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\submission\%SUBMISSION_FOLDER%\requirements.txt %TARGET_CONTAINER%:/competition

docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\IJCAI19\ %TARGET_CONTAINER%:/competition/

REM docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\IJCAI19\model %TARGET_CONTAINER%:/competition/IJCAI19
REM docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\IJCAI19\module %TARGET_CONTAINER%:/competition/IJCAI19
REM docker cp %IJCAI19_ROOT%\IJCAI19_AliAAAC\IJCAI19\weight %TARGET_CONTAINER%:/competition/IJCAI19

docker cp %IJCAI19_ROOT%\official_data\dev_data %TARGET_CONTAINER%:/competition
docker cp %IJCAI19_ROOT%\official_data\out_data %TARGET_CONTAINER%:/competition