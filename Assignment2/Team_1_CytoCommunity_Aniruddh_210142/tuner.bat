@echo off
REM Batch script to run the sequence of scripts

set image_name=typistosmfish
set tcn=11
set epochs=2500

for %%k in (80 100) do (
    call conda activate CytoCommunity
    python Step1_ConstructCellularSpatialGraphs.py -k %%k
    echo Step 1 Done at %date% %time% with k=%%k

    for %%a in (0.005 0.001 0.0005) do (
        for %%d in (128 256) do (
            call conda activate CytoCommunity
            python Step2_TCNLearning_Unsupervised.py -i %image_name% -a %%a -t %tcn% -e %epochs% -d %%d -r 3
            echo Step 2 Done at %date% %time% with alpha=%%a and dim=%%d
        )
    )
)
call conda deactivate
Rscript Step3_TCNEnsemble.R %image_name%
echo Step 3 Done at %date% %time%

call conda activate genomics
python Step4_ResultVisualization.py -i %image_name%
echo Step 4 Done at %date% %time%

echo All scripts have been executed.
pause