# --- 配置区 ---
$CONDA_ENV_NAME = "unet-pytorch"  # 你的 conda 环境名称
$WAIT_TIME = 14400               # 等待时间（秒）

# 1. 等待虚拟数据训练结束
Write-Host "开始等待虚拟训练结束，计时 6 小时..." -ForegroundColor Cyan
Start-Sleep -Seconds $WAIT_TIME

# 2. 备份最佳权重
Write-Host "正在备份最佳权重文件..." -ForegroundColor Green
$weightPath = "logs/best_epoch_weights.pth"
$backupPath = "logs/best_unet_30epoch.pth"

if (Test-Path $weightPath) {
    Copy-Item $weightPath $backupPath -Force
    Write-Host "成功备份权重至: $backupPath"
} else {
    Write-Host "错误: 未找到 $weightPath，请检查路径！" -ForegroundColor Red
    exit
}

# 3. 运行评估 (使用 conda run 确保环境正确)
Write-Host "开始运行评估 (Environment: $CONDA_ENV_NAME)..." -ForegroundColor Yellow
conda run -n $CONDA_ENV_NAME python get_miou.py

# 4. 切换数据集文件夹 (增加安全检查)
Write-Host "正在切换数据集路径..." -ForegroundColor Green
$dir2007 = "VOCdevkit\VOC2007"
$dirReal = "VOCdevkit\VOC2007-real"
$dirVirt = "VOCdevkit\VOC2007-virtual"

if (Test-Path $dir2007) {
    Rename-Item $dir2007 "VOC2007-virtual" -ErrorAction SilentlyContinue
    Write-Host "已将当前数据集重命名为 VOC2007-virtual"
}

if (Test-Path $dirReal) {
    Rename-Item $dirReal "VOC2007" -ErrorAction SilentlyContinue
    Write-Host "已将 VOC2007-real 启用为当前数据集 VOC2007"
} else {
    Write-Host "警告: 未找到 VOC2007-real 文件夹，train2.py 可能会报错！" -ForegroundColor Red
}

# 5. 开始现实数据微调训练
Write-Host "启动 train2.py 进行现实微调..." -ForegroundColor Cyan
conda run --no-capture-output -n $CONDA_ENV_NAME python -u train2.py | Tee-Object -FilePath "logs/real_finetune_log.txt"

Write-Host "所有流程已完成！" -ForegroundColor Magenta