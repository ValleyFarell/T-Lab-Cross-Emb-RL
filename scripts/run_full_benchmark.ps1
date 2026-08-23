# Полный воспроизводимый прогон исходного метода и проекции подцели.
param(
    [string]$ResultsRoot = "results_benchmark_all_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int[]]$Tasks = (1..5),
    [int[]]$Seeds = (0..20),
    [string]$Python = "python",
    [switch]$SkipBaseline,
    [switch]$SkipDeepProjection
)

# Останавливаем всю серию сразу после первой ошибки запуска.
$ErrorActionPreference = "Stop"

function Assert-LastRunSucceeded {
    param([string]$Description)

    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка запуска: $Description"
    }
}

# Исходный метод получает ту же сетку задач и инициализаций, что и сравниваемая гипотеза.
if (-not $SkipBaseline) {
    $baselineDir = Join-Path $ResultsRoot "baseline"

    foreach ($taskId in $Tasks) {
        foreach ($seed in $Seeds) {
            Write-Host "Запуск исходного метода: задача=$taskId, инициализация=$seed"
            & $Python -m scripts.run_baseline `
                --controller baseline `
                --task-id $taskId `
                --environment-seed $seed `
                --controller-seed 0 `
                --temperature 0 `
                --results-dir $baselineDir

            Assert-LastRunSucceeded "controller=baseline task=$taskId seed=$seed"
        }
    }
}

# Метод проекции запускается на идентичных сценариях без изменения чекпоинта.
if (-not $SkipDeepProjection) {
    $deepDir = Join-Path $ResultsRoot "deep_projection_vmax_finish_r1"

    foreach ($taskId in $Tasks) {
        foreach ($seed in $Seeds) {
            Write-Host "Запуск проекции подцели: задача=$taskId, инициализация=$seed"
            & $Python -m scripts.run_decoded_dataset_subgoal `
                --decoder artifacts/intention_xy_decoder_deep `
                --task-id $taskId `
                --environment-seed $seed `
                --controller-seed 0 `
                --temperature 0 `
                --candidate-radius 0.5 `
                --max-candidates 64 `
                --disagreement-penalty 0.5 `
                --selection-mode max-v `
                --finish-mode dynamic-v-max `
                --replan-interval 1 `
                --results-dir $deepDir

            Assert-LastRunSucceeded "controller=deep_projection task=$taskId seed=$seed"
        }
    }
}

Write-Host "Все результаты сохранены в каталоге: $ResultsRoot"
