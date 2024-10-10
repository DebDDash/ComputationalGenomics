@echo off
REM Batch script to reproduce results

set image_name=typistmerfish
set k=60
set tcn=8
set epochs=2500
set a=0.005
set d=256

call conda activate CytoCommunity
python Step1_ConstructCellularSpatialGraphs.py -k %k%
echo Step 1 Done at %date% %time% with k=%k%

call conda activate CytoCommunity
python Step2_TCNLearning_Unsupervised.py -i %image_name% -a %a% -t %tcn% -e %epochs% -d %d% -r 4
echo Step 2 Done at %date% %time% with alpha=%a% and dim=%d%

call conda deactivate
Rscript Step3_TCNEnsemble.R %image_name%
echo Step 3 Done at %date% %time%

call conda activate CytoCommunity
python Step4_ResultVisualization.py -i %image_name%
echo Step 4 Done at %date% %time%

call conda activate genomics
python postprocessing.py -i %image_name%


set image_name=osmfish
set k=80
set tcn=11
set epochs=2500
set a=0.001
set d=128

call conda activate CytoCommunity
python Step1_ConstructCellularSpatialGraphs.py -k %k%
echo Step 1 Done at %date% %time% with k=%k%

call conda activate CytoCommunity
python Step2_TCNLearning_Unsupervised.py -i %image_name% -a %a% -t %tcn% -e %epochs% -d %d% -r 4
echo Step 2 Done at %date% %time% with alpha=%a% and dim=%d%

call conda deactivate
Rscript Step3_TCNEnsemble.R %image_name%
echo Step 3 Done at %date% %time%

call conda activate genomics
python Step4_ResultVisualization.py -i %image_name%
echo Step 4 Done at %date% %time%

call conda activate genomics
python postprocessing.py -i %image_name%

echo All scripts have been executed.
pause