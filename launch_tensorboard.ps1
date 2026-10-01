# =====================================================
# TensorBoard 即時展示啟動腳本
# 用途：專題審查時，讓評審親自操作 TensorBoard 動態圖表
# =====================================================
# 使用方式：在 PowerShell 中執行此腳本，然後開啟瀏覽器 http://localhost:6006

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TensorBoard - 肝臟超音波 LoRA 訓練紀錄" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Log 目錄: C:\Users\AlanLab\Downloads\liver\liver_lora_output_medical_512\logs_medical_512"
Write-Host "Checkpoint 總數: 240 個 (每 500 步自動儲存)"
Write-Host "訓練時間跨度: 2026-09-10 16:03 ~ 2026-09-12 16:16 (約 48 小時)"
Write-Host ""
Write-Host "啟動 TensorBoard 伺服器中..." -ForegroundColor Yellow
Write-Host "請在瀏覽器開啟: http://localhost:6006" -ForegroundColor Green
Write-Host "按 Ctrl+C 可停止伺服器" -ForegroundColor Gray
Write-Host ""

C:\Users\AlanLab\Miniconda3\envs\pocus\python.exe -m tensorboard.main --logdir="C:\Users\AlanLab\Downloads\liver\liver_lora_output_medical_512\logs_medical_512" --bind_all --port=6006
