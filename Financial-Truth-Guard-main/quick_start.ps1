# Financial Truth Guard 快速启动脚本
# 使用uv进行环境管理

Write-Host "=== Financial Truth Guard 快速启动 ===" -ForegroundColor Green

# 检查uv是否安装
Write-Host "`n[1/5] 检查uv安装..." -ForegroundColor Yellow
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到uv，正在安装..." -ForegroundColor Red
    pip install uv
}

# 创建虚拟环境
Write-Host "`n[2/5] 创建虚拟环境..." -ForegroundColor Yellow
if (!(Test-Path ".venv")) {
    uv venv
    Write-Host "虚拟环境已创建" -ForegroundColor Green
} else {
    Write-Host "虚拟环境已存在" -ForegroundColor Green
}

# 激活虚拟环境
Write-Host "`n[3/5] 激活虚拟环境..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 安装依赖
Write-Host "`n[4/5] 安装依赖包..." -ForegroundColor Yellow
Write-Host "使用精简版依赖（避免不兼容的包）..." -ForegroundColor Cyan
uv pip install -r requirements_minimal.txt

# 下载NLTK数据
Write-Host "`n[5/5] 下载NLTK数据..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 启动应用
Write-Host "`n=== 环境准备完成！===" -ForegroundColor Green
Write-Host "`n选择要执行的操作：" -ForegroundColor Cyan
Write-Host "1. 启动Web应用"
Write-Host "2. 训练朴素贝叶斯模型 (Model_v2)"
Write-Host "3. 训练CNN模型 (Model_v3)"
Write-Host "4. 训练Pilot金融模型"
Write-Host "5. 退出"

$choice = Read-Host "`n请输入选项 (1-5)"

switch ($choice) {
    "1" {
        Write-Host "`n正在启动Web应用..." -ForegroundColor Green
        Set-Location Financial_Truth_Guard_Web
        python manage.py runserver
    }
    "2" {
        Write-Host "`n正在训练朴素贝叶斯模型..." -ForegroundColor Green
        Set-Location Model_v2\models
        python naive_bayes.py
    }
    "3" {
        Write-Host "`n正在训练CNN模型..." -ForegroundColor Green
        Set-Location Model_v3\models
        python cnn_model.py
    }
    "4" {
        Write-Host "`n正在训练Pilot金融模型..." -ForegroundColor Green
        Set-Location Pilot\model
        python pilot_model.py
    }
    "5" {
        Write-Host "再见！" -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host "无效选项" -ForegroundColor Red
    }
}
