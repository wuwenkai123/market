import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import argparse
import os
from tqdm import tqdm
import talib
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class BTCTrendAnalyzer:
    """
    比特币多时间周期趋势分析工具
    
    分析不同时间周期(1小时、8小时、24小时、1周)的比特币价格趋势
    并提供技术指标和趋势判断
    """
    
    def __init__(self, exchange='binance', symbol='BTC/USDT', cache_dir='btc_data'):
        """
        初始化比特币趋势分析器
        
        参数:
            exchange: 交易所名称，默认为 'binance'
            symbol: 交易对，默认为 'BTC/USDT'
            cache_dir: 数据缓存目录
        """
        self.exchange_id = exchange
        self.symbol = symbol
        self.cache_dir = cache_dir
        self.timeframes = {
            '1h': {'name': '1小时', 'limit': 200},
            '8h': {'name': '8小时', 'limit': 100},
            '1d': {'name': '24小时', 'limit': 90},
            '1w': {'name': '1周', 'limit': 52}
        }
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 连接交易所
        try:
            self.exchange = getattr(ccxt, self.exchange_id)({
                'enableRateLimit': True,  # 启用频率限制
                'timeout': 30000,  # 设置超时时间为30秒
            })
            print(f"已连接到交易所：{self.exchange_id}")
        except Exception as e:
            print(f"连接交易所失败: {e}")
            self.exchange = None
    
    def fetch_ohlcv_data(self, timeframe, limit=None, refresh=False):
        """
        获取历史K线数据
        
        参数:
            timeframe: 时间周期，如 '1h', '1d'
            limit: 获取多少根K线
            refresh: 是否强制刷新数据
            
        返回:
            DataFrame: 包含OHLCV数据的DataFrame
        """
        # 构建缓存文件路径
        cache_file = os.path.join(self.cache_dir, f"{self.exchange_id}_{self.symbol.replace('/', '_')}_{timeframe}.csv")
        cache_file = cache_file.replace(':', '')  # 去掉可能含有的冒号
        
        now = datetime.now()
        
        # 检查缓存文件是否存在且为今天的
        if os.path.exists(cache_file) and not refresh:
            try:
                # 读取缓存数据
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 检查最新数据时间
                last_timestamp = df['timestamp'].max()
                elapsed_hours = (now - last_timestamp).total_seconds() / 3600
                
                # 如果数据较新（小于8小时前），直接使用缓存
                if elapsed_hours < 8:
                    print(f"使用缓存的{timeframe}数据，最后更新时间: {last_timestamp}")
                    return df
            except Exception as e:
                print(f"读取缓存数据出错: {e}")
        
        # 如果没有缓存或需要刷新，则从交易所获取数据
        if self.exchange is None:
            print("交易所连接不可用，无法获取数据")
            return None
        
        try:
            # 获取历史K线数据
            limit = limit or self.timeframes[timeframe]['limit']
            print(f"从{self.exchange_id}获取{self.symbol} {timeframe}数据...")
            
            # 获取OHLCV数据
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol, 
                timeframe=timeframe,
                limit=limit
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 保存到缓存
            df.to_csv(cache_file, index=False)
            print(f"已保存{timeframe}数据到缓存: {cache_file}")
            
            return df
        
        except Exception as e:
            print(f"获取{timeframe}数据出错: {e}")
            
            # 如果出错但缓存存在，尝试使用缓存
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    print(f"使用备用缓存数据: {cache_file}")
                    return df
                except Exception as e2:
                    print(f"读取备用缓存数据出错: {e2}")
            
            return None

    def calculate_indicators(self, df):
        """
        计算技术指标
        
        参数:
            df: 包含OHLCV数据的DataFrame
            
        返回:
            DataFrame: 添加了技术指标的DataFrame
        """
        if df is None or len(df) < 20:
            return None
        
        # 复制DataFrame，避免修改原始数据
        df = df.copy()
        
        # 添加常用技术指标
        
        # 1. 移动平均线
        df['MA5'] = talib.SMA(df['close'].values, timeperiod=5)
        df['MA10'] = talib.SMA(df['close'].values, timeperiod=10)
        df['MA20'] = talib.SMA(df['close'].values, timeperiod=20)
        df['MA50'] = talib.SMA(df['close'].values, timeperiod=50)
        
        # 2. MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # 3. RSI
        df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # 4. 布林带
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
        
        # 5. 随机震荡指标
        slowk, slowd = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        
        # 6. ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        return df

    def analyze_trend(self, df, timeframe):
        """
        分析价格趋势
        
        参数:
            df: 包含OHLCV和技术指标的DataFrame
            timeframe: 时间周期
            
        返回:
            dict: 趋势分析结果
        """
        if df is None or len(df) < 20:
            return {'trend': 'unknown', 'strength': 0, 'summary': '数据不足，无法分析'}
        
        # 获取最新数据点
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 判断趋势的多个因素
        factors = {}
        
        # 1. 价格相对于移动平均线
        factors['price_vs_ma20'] = 1 if current['close'] > current['MA20'] else -1
        factors['price_vs_ma50'] = 1 if current['close'] > current['MA50'] else -1
        
        # 2. 移动平均线排列
        factors['ma_alignment'] = 1 if current['MA5'] > current['MA10'] > current['MA20'] else \
                                  -1 if current['MA5'] < current['MA10'] < current['MA20'] else 0
        
        # 3. MACD
        factors['macd_signal'] = 1 if current['MACD'] > current['MACD_signal'] else -1
        factors['macd_hist'] = 1 if current['MACD_hist'] > 0 else -1
        factors['macd_hist_change'] = 1 if current['MACD_hist'] > prev['MACD_hist'] else -1
        
        # 4. RSI
        factors['rsi'] = 1 if current['RSI'] > 50 else -1
        factors['rsi_trend'] = 1 if current['RSI'] > prev['RSI'] else -1
        
        # 5. 价格趋势
        recent_prices = df['close'].tail(5).values
        factors['price_trend'] = 1 if recent_prices[-1] > recent_prices[0] else -1
        
        # 6. 布林带位置
        if current['close'] > current['BB_upper']:
            factors['bb_position'] = 2  # 超买
        elif current['close'] < current['BB_lower']:
            factors['bb_position'] = -2  # 超卖
        elif current['close'] > current['BB_middle']:
            factors['bb_position'] = 1  # 偏强
        else:
            factors['bb_position'] = -1  # 偏弱
        
        # 7. ADX趋势强度
        if current['ADX'] > 25:
            # 强趋势
            factors['trend_strength'] = 2 if factors['price_trend'] == 1 else -2
        else:
            # 弱趋势
            factors['trend_strength'] = 1 if factors['price_trend'] == 1 else -1
        
        # 汇总趋势得分
        trend_score = sum(factors.values())
        
        # 根据得分判断趋势
        if trend_score >= 6:
            trend = '强烈上涨'
            strength = 3
        elif trend_score >= 3:
            trend = '上涨'
            strength = 2
        elif trend_score > 0:
            trend = '小幅上涨'
            strength = 1
        elif trend_score == 0:
            trend = '盘整'
            strength = 0
        elif trend_score > -3:
            trend = '小幅下跌'
            strength = -1
        elif trend_score > -6:
            trend = '下跌'
            strength = -2
        else:
            trend = '强烈下跌'
            strength = -3
        
        # 获取价格统计
        price_change = ((current['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
        price_volatility = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-5] * 100
        
        # 生成总结
        timeframe_name = self.timeframes[timeframe]['name']
        summary = f"BTC {timeframe_name}趋势: {trend}\n"
        summary += f"价格: {current['close']:.2f} USDT\n"
        summary += f"近5个{timeframe_name}涨跌: {price_change:.2f}%\n"
        summary += f"波动幅度: {price_volatility:.2f}%\n"
        
        # 添加技术指标状态
        macd_status = "金叉" if factors['macd_signal'] == 1 and factors['macd_hist_change'] == 1 else \
                     "死叉" if factors['macd_signal'] == -1 and factors['macd_hist_change'] == -1 else \
                     "多头" if factors['macd_signal'] == 1 else "空头"
        
        rsi_status = "超买" if current['RSI'] > 70 else \
                    "超卖" if current['RSI'] < 30 else \
                    "多头" if current['RSI'] > 50 else "空头"
        
        summary += f"MACD: {macd_status}\n"
        summary += f"RSI({current['RSI']:.1f}): {rsi_status}\n"
        
        # 返回分析结果
        return {
            'trend': trend,
            'strength': strength,
            'summary': summary,
            'price': current['close'],
            'change': price_change,
            'volatility': price_volatility,
            'factors': factors,
            'macd_status': macd_status,
            'rsi': current['RSI'],
            'rsi_status': rsi_status,
            'adx': current['ADX']
        }

    def plot_trend(self, dfs, analysis_results, output_file=None):
        """
        绘制多个时间周期的趋势图
        
        参数:
            dfs: 不同时间周期的DataFrame字典
            analysis_results: 不同时间周期的分析结果字典
            output_file: 输出文件路径，None则显示图表
        """
        timeframes = list(dfs.keys())
        num_timeframes = len(timeframes)
        
        # 创建子图
        fig, axs = plt.subplots(num_timeframes, 1, figsize=(14, 4 * num_timeframes), dpi=100)
        if num_timeframes == 1:
            axs = [axs]
        
        # 设置标题
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title = f"BTC多周期趋势分析 - {current_time}"
        fig.suptitle(title, fontsize=16)
        
        for i, timeframe in enumerate(timeframes):
            df = dfs[timeframe]
            result = analysis_results[timeframe]
            
            if df is None or len(df) < 20:
                axs[i].text(0.5, 0.5, f"{self.timeframes[timeframe]['name']}数据不足", 
                            ha='center', va='center', transform=axs[i].transAxes)
                continue
                
            # 绘制价格和均线
            axs[i].plot(df['timestamp'], df['close'], label='价格', linewidth=1.5)
            axs[i].plot(df['timestamp'], df['MA20'], label='MA20', linewidth=1)
            axs[i].plot(df['timestamp'], df['MA50'], label='MA50', linewidth=1)
            
            # 绘制布林带
            axs[i].plot(df['timestamp'], df['BB_upper'], 'g--', alpha=0.3, linewidth=0.8)
            axs[i].plot(df['timestamp'], df['BB_lower'], 'g--', alpha=0.3, linewidth=0.8)
            axs[i].fill_between(df['timestamp'], df['BB_upper'], df['BB_lower'], alpha=0.1, color='green')
            
            # 设置标题和标签
            trend_color = 'green' if result['strength'] > 0 else 'red' if result['strength'] < 0 else 'black'
            
            axs[i].set_title(f"{self.timeframes[timeframe]['name']} 趋势: {result['trend']}", 
                            fontsize=14, color=trend_color)
            axs[i].set_ylabel('价格 (USDT)', fontsize=12)
            
            # 格式化X轴日期
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d' if timeframe in ['1d', '1w'] else '%m-%d %H:%M'))
            axs[i].xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 添加网格
            axs[i].grid(True, alpha=0.3)
            
            # 旋转X轴标签
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)
            
            # 添加技术指标标注
            y_pos = df['close'].min() + (df['close'].max() - df['close'].min()) * 0.05
            axs[i].annotate(f"RSI: {result['rsi']:.1f} ({result['rsi_status']})\n"
                           f"MACD: {result['macd_status']}\n"
                           f"变化: {result['change']:.2f}%",
                          xy=(df['timestamp'].iloc[-1], y_pos),
                          xytext=(-100, 10),
                          textcoords='offset points',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            
            # 图例
            axs[i].legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存或显示
        if output_file:
            plt.savefig(output_file)
            print(f"图表已保存至: {output_file}")
        else:
            plt.show()
        
        # 关闭图表，释放内存
        plt.close()

    def analyze_and_plot_all(self, timeframes=None, refresh=False, output_file=None):
        """
        分析并绘制所有指定时间周期的趋势
        
        参数:
            timeframes: 要分析的时间周期列表，默认为所有
            refresh: 是否刷新数据
            output_file: 输出文件路径
        
        返回:
            dict: 分析结果
        """
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
        
        # 获取数据并计算指标
        dfs = {}
        analysis_results = {}
        
        for timeframe in timeframes:
            print(f"\n分析 {self.timeframes[timeframe]['name']} 趋势...")
            
            # 获取数据
            df = self.fetch_ohlcv_data(timeframe, refresh=refresh)
            if df is not None and len(df) >= 20:
                # 计算指标
                df_with_indicators = self.calculate_indicators(df)
                dfs[timeframe] = df_with_indicators
                
                # 分析趋势
                result = self.analyze_trend(df_with_indicators, timeframe)
                analysis_results[timeframe] = result
                
                # 打印结果
                print(result['summary'])
            else:
                print(f"数据不足，无法分析 {self.timeframes[timeframe]['name']} 趋势")
                dfs[timeframe] = None
                analysis_results[timeframe] = {
                    'trend': '未知', 
                    'strength': 0, 
                    'summary': f"数据不足，无法分析 {self.timeframes[timeframe]['name']} 趋势"
                }
        
        # 绘制趋势图
        self.plot_trend(dfs, analysis_results, output_file)
        
        # 返回分析结果
        return analysis_results

    def generate_comprehensive_report(self, analysis_results=None, output_file=None):
        """
        生成综合分析报告
        
        参数:
            analysis_results: 分析结果字典，如果为None则重新分析
            output_file: 输出文件路径
        """
        if analysis_results is None:
            analysis_results = self.analyze_and_plot_all()
        
        # 获取当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 创建报告
        report = []
        report.append("=" * 50)
        report.append(f"BTC多时间周期趋势分析报告 - {current_time}")
        report.append("=" * 50)
        report.append("")
        
        # 添加各时间周期分析结果
        for timeframe, result in analysis_results.items():
            if 'summary' in result:
                report.append(f"【{self.timeframes[timeframe]['name']}】")
                report.append("-" * 30)
                report.append(result['summary'])
                report.append("")
        
        # 综合趋势分析
        report.append("=" * 50)
        report.append("综合趋势分析")
        report.append("-" * 30)
        
        # 获取各时间周期的趋势强度
        strengths = {tf: result.get('strength', 0) for tf, result in analysis_results.items()}
        
        # 短期趋势 (1小时)
        short_term = strengths.get('1h', 0)
        # 中期趋势 (8小时 + 1天)
        mid_term = (strengths.get('8h', 0) + strengths.get('1d', 0)) / 2 if '8h' in strengths and '1d' in strengths else 0
        # 长期趋势 (1周)
        long_term = strengths.get('1w', 0)
        
        # 趋势描述
        def strength_to_desc(strength):
            if strength >= 2.5:
                return "强烈看涨"
            elif strength >= 1.5:
                return "看涨"
            elif strength >= 0.5:
                return "略微看涨"
            elif strength > -0.5:
                return "中性"
            elif strength > -1.5:
                return "略微看跌"
            elif strength > -2.5:
                return "看跌"
            else:
                return "强烈看跌"
        
        report.append(f"短期趋势(1小时): {strength_to_desc(short_term)}")
        report.append(f"中期趋势(8小时+日线): {strength_to_desc(mid_term)}")
        report.append(f"长期趋势(周线): {strength_to_desc(long_term)}")
        report.append("")
        
        # 交易建议
        report.append("-" * 30)
        report.append("交易建议")
        report.append("-" * 30)
        
        # 根据不同时间周期的趋势强度给出建议
        if short_term >= 1 and mid_term >= 1:
            if long_term >= 1:
                advice = "多头趋势明确，可考虑逢低买入并持有"
            else:
                advice = "中短期看涨，长期不明确，可考虑短线买入，注意设置止盈"
        elif short_term <= -1 and mid_term <= -1:
            if long_term <= -1:
                advice = "空头趋势明确，可考虑逢高卖出或持币观望"
            else:
                advice = "中短期看跌，长期不明确，可考虑减仓或观望"
        elif abs(short_term) <= 1 and abs(mid_term) <= 1:
            advice = "市场处于盘整状态，建议耐心等待明确信号"
        elif short_term * mid_term < 0:  # 短期和中期趋势相反
            if abs(short_term) > abs(mid_term):
                advice = "短期反转信号，但需确认中期趋势配合，建议谨慎"
            else:
                advice = "关注短期是否会跟随中期趋势，暂时观望"
        else:
            advice = "市场信号混杂，建议等待更明确的趋势确认"
        
        report.append(advice)
        report.append("")
        
        # 风险提示
        report.append("-" * 30)
        report.append("风险提示")
        report.append("-" * 30)
        report.append("1. 本分析仅供参考，不构成投资建议")
        report.append("2. 加密货币市场波动剧烈，请控制仓位和风险")
        report.append("3. 技术分析不能预测突发事件的影响")
        report.append("4. 始终使用止损单保护您的资金安全")
        report.append("=" * 50)
        
        # 转换为文本
        report_text = "\n".join(report)
        
        # 保存或打印报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存至: {output_file}")
        else:
            print(report_text)
        
        return report_text


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='BTC多时间周期趋势分析工具')
    parser.add_argument('--exchange', type=str, default='binance', help='交易所 (默认: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--timeframes', type=str, default='1h,8h,1d,1w', 
                        help='要分析的时间周期，逗号分隔 (默认: 1h,8h,1d,1w)')
    parser.add_argument('--refresh', action='store_true', help='强制刷新数据')
    parser.add_argument('--report', type=str, default=None, help='保存报告的文件路径')
    parser.add_argument('--chart', type=str, default=None, help='保存图表的文件路径')
    
    args = parser.parse_args()
    
    # 解析时间周期
    timeframes = args.timeframes.split(',')
    
    # 创建分析器
    analyzer = BTCTrendAnalyzer(exchange=args.exchange, symbol=args.symbol)
    
    # 分析并生成图表
    results = analyzer.analyze_and_plot_all(timeframes=timeframes, refresh=args.refresh, output_file=args.chart)
    
    # 生成报告
    analyzer.generate_comprehensive_report(results, args.report)


if __name__ == "__main__":
    main()
