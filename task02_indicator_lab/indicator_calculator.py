"""
技术指标计算与可视化脚本
========================

功能：
1. 加载已存储的股价数据
2. 计算RSI、MACD、布林带指标
3. 绘制可视化图形

数据：贵州茅台（600519.SH）近一年每日交易数据
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ========== 配置参数 ==========
DATA_PATH = "output/600519_1y.csv"  # 数据文件路径
OUTPUT_DIR = "output"               # 输出目录
STOCK_NAME = "贵州茅台"
STOCK_CODE = "600519.SH"

# 中文字体配置（macOS）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 使用非交互式后端，避免显示窗口
plt.switch_backend('Agg')


# ========== 1. 加载已存储的股价数据 ==========
def load_stock_data(file_path):
    """
    加载股票数据
    
    参数：
    - file_path: CSV文件路径
    
    返回：
    - df: 处理后的DataFrame
    """
    print(f"📂 正在加载数据：{file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 数据预处理
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    print(f"✅ 数据加载成功")
    print(f"   股票：{STOCK_NAME} ({STOCK_CODE})")
    print(f"   日期范围：{df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   交易日数量：{len(df)}")
    print(f"   数据字段：{', '.join(df.columns.tolist())}")
    print()
    
    return df


# ========== 2. 计算技术指标 ==========

def calculate_rsi(df, n=14):
    """
    计算RSI（相对强弱指标）
    
    参数：
    - df: DataFrame，包含'close'列
    - n: 计算周期，默认14
    
    返回：
    - rsi: Series，RSI值
    """
    # 计算每日涨跌幅
    delta = df['close'].diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)  # 上涨部分
    loss = -delta.where(delta < 0, 0)  # 下跌部分（取正值）
    
    # 计算平均上涨和平均下跌（使用简单移动平均）
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()
    
    # 计算RS和RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD（移动平均收敛/发散指标）
    
    参数：
    - df: DataFrame，包含'close'列
    - fast: 快线周期，默认12
    - slow: 慢线周期，默认26
    - signal: 信号线周期，默认9
    
    返回：
    - macd: Series，MACD线
    - signal_line: Series，信号线
    - histogram: Series，柱状图
    """
    # 计算快慢EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # 计算MACD线
    macd = ema_fast - ema_slow
    
    # 计算信号线
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # 计算柱状图
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_bollinger_bands(df, n=20, k=2):
    """
    计算布林带（Bollinger Bands）
    
    参数：
    - df: DataFrame，包含'close'列
    - n: 移动平均周期，默认20
    - k: 标准差倍数，默认2
    
    返回：
    - middle_band: Series，中轨
    - upper_band: Series，上轨
    - lower_band: Series，下轨
    """
    # 计算中轨（SMA）
    middle_band = df['close'].rolling(window=n).mean()
    
    # 计算标准差
    std = df['close'].rolling(window=n).std()
    
    # 计算上轨和下轨
    upper_band = middle_band + (k * std)
    lower_band = middle_band - (k * std)
    
    return middle_band, upper_band, lower_band


def calculate_atr(df, n=14):
    """
    计算ATR（平均真实波幅）
    
    参数：
    - df: DataFrame，包含'high', 'low', 'close'列
    - n: ATR周期，默认14
    
    返回：
    - atr: Series，ATR值
    """
    # 计算真实波幅（TR）
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    # 取三者最大值
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # 计算ATR（使用指数移动平均）
    atr = tr.ewm(span=n, adjust=False).mean()
    
    return atr


# ========== 3. 绘制可视化图形 ==========

def plot_close_price(df):
    """
    绘制收盘价走势图
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 收盘价曲线（中国风格：红线表示收盘价）
    ax.plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8, label='收盘价')
    
    # 标注最高价和最低价
    max_idx = df['close'].idxmax()
    min_idx = df['close'].idxmin()
    ax.scatter(df.loc[max_idx, 'trade_date'], df.loc[max_idx, 'close'],
               color='#ff4d4f', s=80, zorder=5, label=f"最高 {df.loc[max_idx, 'close']:.2f}")
    ax.scatter(df.loc[min_idx, 'trade_date'], df.loc[min_idx, 'close'],
               color='#3fb950', s=80, zorder=5, label=f"最低 {df.loc[min_idx, 'close']:.2f}")
    
    ax.set_title(f"{STOCK_NAME} ({STOCK_CODE})  近一年收盘价走势",
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("收盘价（元）", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # 优化X轴日期显示
    step = max(1, len(df) // 10)
    ax.set_xticks(df['trade_date'].iloc[::step])
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in df['trade_date'].iloc[::step]],
                       rotation=30, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_close_price.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 收盘价走势图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


def plot_rsi(df):
    """
    绘制RSI指标图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 上图：收盘价
    ax1.plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8, label='收盘价')
    ax1.set_ylabel('收盘价（元）', fontsize=12)
    ax1.set_title(f'{STOCK_NAME} - 收盘价 vs RSI指标', fontsize=15, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best')
    
    # 下图：RSI
    ax2.plot(df['trade_date'], df['RSI_14'], color='#1f77b4', linewidth=1.5, label='RSI(14)')
    ax2.axhline(y=70, color='#ff4d4f', linestyle='--', linewidth=1, alpha=0.7, label='超买线(70)')
    ax2.axhline(y=30, color='#3fb950', linestyle='--', linewidth=1, alpha=0.7, label='超卖线(30)')
    ax2.axhline(y=50, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5, label='中轴线(50)')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best')
    
    # 填充超买超卖区域
    ax2.fill_between(df['trade_date'], 70, 100, alpha=0.1, color='#ff4d4f')
    ax2.fill_between(df['trade_date'], 0, 30, alpha=0.1, color='#3fb950')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_RSI.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 RSI指标图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


def plot_macd(df):
    """
    绘制MACD指标图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 上图：收盘价
    ax1.plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8, label='收盘价')
    ax1.set_ylabel('收盘价（元）', fontsize=12)
    ax1.set_title(f'{STOCK_NAME} - 收盘价 vs MACD指标', fontsize=15, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best')
    
    # 下图：MACD
    ax2.plot(df['trade_date'], df['MACD'], color='#1f77b4', linewidth=1.5, label='MACD线(12-26)')
    ax2.plot(df['trade_date'], df['MACD_signal'], color='#ff7f0e', linewidth=1.5, label='信号线(9)')
    ax2.axhline(y=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 绘制柱状图
    colors = ['#ff4d4f' if x < 0 else '#3fb950' for x in df['MACD_hist']]
    ax2.bar(df['trade_date'], df['MACD_hist'], color=colors, alpha=0.5, label='柱状图', width=0.8)
    
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_MACD.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 MACD指标图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


def plot_bollinger_bands(df):
    """
    绘制布林带指标图
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制收盘价
    ax.plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8, label='收盘价', zorder=3)
    
    # 绘制布林带
    ax.plot(df['trade_date'], df['BB_upper'], color='#999999', linewidth=1, linestyle='--', alpha=0.7, label='上轨(+2STD)')
    ax.plot(df['trade_date'], df['BB_middle'], color='#1f77b4', linewidth=1.2, linestyle='-', alpha=0.8, label='中轨(SMA20)')
    ax.plot(df['trade_date'], df['BB_lower'], color='#999999', linewidth=1, linestyle='--', alpha=0.7, label='下轨(-2STD)')
    
    # 填充布林带区域
    ax.fill_between(df['trade_date'], df['BB_upper'], df['BB_lower'], alpha=0.1, color='#1f77b4')
    
    ax.set_title(f'{STOCK_NAME} - 布林带指标', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格（元）', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_Bollinger.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 布林带指标图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


def plot_atr(df):
    """
    绘制ATR指标图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 上图：收盘价 + ATR通道
    ax1.plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8, label='收盘价')
    
    # 绘制ATR通道（收盘价±ATR）
    ax1.fill_between(df['trade_date'], 
                     df['close'] - df['ATR_14'], 
                     df['close'] + df['ATR_14'], 
                     alpha=0.2, color='#1f77b4', label='±1倍ATR通道')
    
    ax1.set_ylabel('收盘价（元）', fontsize=12)
    ax1.set_title(f'{STOCK_NAME} - 收盘价 vs ATR波动性', fontsize=15, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best')
    
    # 下图：ATR
    ax2.plot(df['trade_date'], df['ATR_14'], color='#1f77b4', linewidth=1.5, label='ATR(14)')
    ax2.axhline(y=df['ATR_14'].mean(), color='#999999', linestyle='--', linewidth=1, alpha=0.7, 
                label=f'平均值({df["ATR_14"].mean():.2f})')
    ax2.set_ylabel('ATR（元）', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_ATR.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 ATR指标图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


def plot_all_indicators(df):
    """
    绘制所有指标的综合图
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    # 图1：收盘价
    axes[0].plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.8)
    axes[0].set_ylabel('收盘价（元）', fontsize=11)
    axes[0].set_title(f'{STOCK_NAME} ({STOCK_CODE}) - 技术指标综合展示', fontsize=15, fontweight='bold', pad=12)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # 图2：RSI
    axes[1].plot(df['trade_date'], df['RSI_14'], color='#1f77b4', linewidth=1.5)
    axes[1].axhline(y=70, color='#ff4d4f', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axhline(y=30, color='#3fb950', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].fill_between(df['trade_date'], 70, 100, alpha=0.1, color='#ff4d4f')
    axes[1].fill_between(df['trade_date'], 0, 30, alpha=0.1, color='#3fb950')
    axes[1].set_ylabel('RSI', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # 图3：MACD
    axes[2].plot(df['trade_date'], df['MACD'], color='#1f77b4', linewidth=1.5)
    axes[2].plot(df['trade_date'], df['MACD_signal'], color='#ff7f0e', linewidth=1.5)
    axes[2].axhline(y=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5)
    colors = ['#ff4d4f' if x < 0 else '#3fb950' for x in df['MACD_hist']]
    axes[2].bar(df['trade_date'], df['MACD_hist'], color=colors, alpha=0.5, width=0.8)
    axes[2].set_ylabel('MACD', fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    # 图4：布林带
    axes[3].plot(df['trade_date'], df['close'], color='#e64340', linewidth=1.5)
    axes[3].plot(df['trade_date'], df['BB_upper'], color='#999999', linewidth=1, linestyle='--', alpha=0.7)
    axes[3].plot(df['trade_date'], df['BB_middle'], color='#1f77b4', linewidth=1.2)
    axes[3].plot(df['trade_date'], df['BB_lower'], color='#999999', linewidth=1, linestyle='--', alpha=0.7)
    axes[3].fill_between(df['trade_date'], df['BB_upper'], df['BB_lower'], alpha=0.1, color='#1f77b4')
    axes[3].set_ylabel('价格（元）', fontsize=11)
    axes[3].grid(True, linestyle='--', alpha=0.3)
    
    # 图5：ATR
    axes[4].plot(df['trade_date'], df['ATR_14'], color='#1f77b4', linewidth=1.5)
    axes[4].axhline(y=df['ATR_14'].mean(), color='#999999', linestyle='--', linewidth=1, alpha=0.7)
    axes[4].set_ylabel('ATR（元）', fontsize=11)
    axes[4].set_xlabel('日期', fontsize=12)
    axes[4].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_all_indicators.png")
    plt.savefig(output_path, dpi=150)
    print(f"📊 综合指标图已保存：{output_path}")
    
    # 关闭图表，释放内存
    plt.close()


# ========== 主函数 ==========
def main():
    """
    主函数：执行完整的技术指标计算与可视化流程
    """
    print("=" * 60)
    print("📈 技术指标计算与可视化系统")
    print("=" * 60)
    print()
    
    # 1. 加载数据
    df = load_stock_data(DATA_PATH)
    
    # 2. 计算技术指标
    print("📊 正在计算技术指标...")
    df['RSI_14'] = calculate_rsi(df, n=14)
    print(f"   ✅ RSI计算完成（周期=14）")
    
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
    print(f"   ✅ MACD计算完成（12, 26, 9）")
    
    df['BB_middle'], df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df, n=20, k=2)
    print(f"   ✅ 布林带计算完成（20, 2）")
    
    df['ATR_14'] = calculate_atr(df, n=14)
    print(f"   ✅ ATR计算完成（周期=14）")
    print()
    
    # 显示计算结果统计
    print("📈 指标计算结果统计")
    print("-" * 40)
    print(f"   RSI范围：{df['RSI_14'].min():.2f} ~ {df['RSI_14'].max():.2f}")
    print(f"   最新RSI：{df['RSI_14'].iloc[-1]:.2f}")
    print(f"   最新MACD：{df['MACD'].iloc[-1]:.2f}")
    print(f"   布林带上轨：{df['BB_upper'].iloc[-1]:.2f}")
    print(f"   布林带中轨：{df['BB_middle'].iloc[-1]:.2f}")
    print(f"   布林带下轨：{df['BB_lower'].iloc[-1]:.2f}")
    print(f"   最新ATR：{df['ATR_14'].iloc[-1]:.2f}")
    print()
    
    # 3. 绘制可视化图形
    print("🎨 正在生成可视化图形...")
    print()
    
    # 收盘价走势图
    plot_close_price(df)
    
    # RSI指标图
    plot_rsi(df)
    
    # MACD指标图
    plot_macd(df)
    
    # 布林带指标图
    plot_bollinger_bands(df)
    
    # ATR指标图
    plot_atr(df)
    
    # 综合指标图
    plot_all_indicators(df)
    
    # 4. 保存计算结果到CSV
    output_csv = os.path.join(OUTPUT_DIR, f"{STOCK_CODE.split('.')[0]}_with_indicators.csv")
    output_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 
                   'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 
                   'BB_upper', 'BB_middle', 'BB_lower',
                   'ATR_14']
    df[output_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"💾 计算结果已保存：{output_csv}")
    
    print()
    print("=" * 60)
    print("✅ 所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
