@echo off
REM Navigate back to project root if needed
cd /d "%~dp0"

set variants=baseline word_bigram word_trigram word_ngram_1_2 word_ngram_1_3 char_ngram_3_5 char_ngram_2_4 hybrid_word_char hybrid_extended

for %%v in (%variants%) do (
    echo =========================================
    echo Training and evaluating variant: %%v
    echo =========================================
    REM Use proper Hydra override syntax: +output_file or override existing
    python main.py mode=stats active_defences=[svm] train=true variant=%%v output_file=results/results_%%v.txt
    
    REM Check if training was successful
    if errorlevel 1 (
        echo Error training variant %%v
        pause
    ) else (
        echo Variant %%v completed successfully
    )
    echo.
)

echo =========================================
echo All variants completed!
echo Results saved in results/ directory
echo =========================================
pause
